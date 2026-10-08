// ============================================================================
// bench.cu — fp8xfp8 MoE Mega Kernel 基准 harness
//
// 内容：
//   M0：rank 对称区分配 + IPC + 空 kernel graph 链基准
//   M1：前序（ROUTE/PUBLISH/XBAR/TASK_SETUP）三形态正确性 + graph 链计时
//
// 性能口径：CUDA Graph 链（单图 nodes 个同构节点、预热 2 次、5 次 replay
// 取最小值，replay 前跨 rank 对齐）。详见 docs/02_测试流程与性能标准.md。
//
// 编译：bash scripts/build.sh    运行：bash scripts/run.sh
// ============================================================================

#include <mcr/mc_runtime.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#include <mpi.h>   // MPI 模式（每进程一 rank/一卡）

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "common.cuh"
#include "flags_defs.cuh"
#include "host_layout.cuh"
#include "bench_check.cuh"
#include "megakernel.h"   // host C++ 接口（链接 megakernel.so）；不直接发射 kernel

// bar flag 索引（生产布局常量）
static constexpr int64_t kXreleaseIdx = deep_gemm::layout::kBarXreleaseIdx;
static constexpr int64_t kSliceFlagsIdx = deep_gemm::layout::kBarSliceFlagsIdx;
static constexpr int64_t kL1ProgIdx = deep_gemm::layout::kBarL1ProgressIdx;
static constexpr int64_t kL2DoneIdx = deep_gemm::layout::kBarL2DoneIdx;
static constexpr int64_t kTaskReadyIdx = deep_gemm::layout::kBarTaskReadyIdx;
static constexpr int64_t kTaskCounterIdx = deep_gemm::layout::kBarTaskCounterIdx;

// [R32D] 诊断：graph 链内 REDUCE 形态（仅计时诊断；未设置 env 时恒为 0=完整）
//   0 = 完整 REDUCE；1 = 不发射 REDUCE 节点；2 = 同几何空核（纯 launch 成本）
static int red_mode() {
    const char* s = std::getenv("BENCH_RED_MODE");
    return s ? std::atoi(s) : 0;
}
// [R32D] 诊断：单 launch 路径分别夹住 mega/reduce（事件）并打印
static bool red_timing() {
    const char* s = std::getenv("BENCH_RED_TIMING");
    return s && std::atoi(s) != 0;
}
// 说明：kernel 发射已全部下沉到 src/megakernel.cu（megakernel.so），bench 只调用
// `fp8mega::launch_*` / `fp8mega::fill_weights_lane`。原 [R33] fused-check 诊断
// （shadow CombineTail 实例）随接口化移除；需要时在接口层增加对应 launch 变体。

// ---------------------------------------------------------------- 运行态 --
struct RankRes {
    int rank = -1;
    int dev = -1;
    uint8_t* sym = nullptr;
    local_flags* lf = nullptr;      // 本 rank 私有旗区（普通显存）
    ranks_flags* rf = nullptr;      // 跨 rank 旗区（FG 显存）
    ranks_flags** d_rf_ptrs = nullptr;
    mcIpcMemHandle_t rf_handle;
    std::vector<ranks_flags*> peer_rf;
    // M2 计算链
    uint8_t* w1 = nullptr;          // [EPR][2I][H] lane 序 packed tile
    uint8_t* w2 = nullptr;          // [EPR][H][I] lane 序 packed tile
    uint8_t* ffn_hidden = nullptr;  // [EPR][ring][win_cap][I] FP8
    uint8_t* pool_rwk = nullptr;    // [A-RWK] L1 A pool 独立普通显存缓冲（实验）
    uint8_t* frag_a = nullptr;      // [OPT-AFRAG] A 输入 fragment 缓冲
    uint8_t* frag2 = nullptr;       // [OPT-AFRAG] L1 输出 fragment 缓冲
    uint16_t* l2_out = nullptr;     // [kGrid][win_cap][H] BF16
    float* y = nullptr;             // M3：REDUCE 输出 [max_tokens][H] FP32
    uint8_t** d_base_ptrs = nullptr;
    mcIpcMemHandle_t handle;
    std::vector<uint8_t*> peer_sym;
};

// 跨 rank 主机线程对齐（每次 replay 前调用；落在 event 窗口之外）
static std::atomic<int> g_rb_cnt{0};
static std::atomic<int> g_rb_gen{0};
static int g_rb_n = 0;
static int g_mpi_rank = 0;   // MPI rank（mpirun 下 = 本进程负责的 rank）
static int g_mpi_size = 1;   // MPI size（>1 走单进程单 rank 路径）

static void rank_barrier() {
    if (g_mpi_size > 1) { MPI_Barrier(MPI_COMM_WORLD); return; }
    const int n = g_rb_n;
    if (n <= 1) return;
    const int gen = g_rb_gen.load(std::memory_order_acquire);
    if (g_rb_cnt.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
        g_rb_cnt.store(0, std::memory_order_relaxed);
        g_rb_gen.fetch_add(1, std::memory_order_release);
    } else {
        while (g_rb_gen.load(std::memory_order_acquire) == gen) {}
    }
}

// ---------------------------------------------------- CUDA Graph 链测量 --
// 单张 graph 内 N 个同构节点；host 只发 1 次/replay；per-kernel = 总时间/N。
// 预热 2 次不入测量；每个 replay 单独用 event 夹住；取 measure_replays 次最小值。
//
// 降级语义（与 mega bench 的 [graph 降级] 一致）：capture / instantiate 失败时
// 不直接判失败，而是退到 eager 顺序发射同一批节点再测一遍，口径回填
// "eager(降级: 原因)"，这样扇形仍能给出该档耗时（只是口径不同）。
// captured（非空）回填是否真的走了 graph。
// [扇形] 诊断：置 1 强制模拟「graph 捕获不可用」，用来验证 [graph 降级] 分支
// 确实会退到 eager 口径并打明原因（与 mega 的 MXBENCH_GRAPH_FAIL 同款）。
static bool graph_fail_injected() {
    const char* s = std::getenv("BENCH_GRAPH_FAIL");
    return s && std::atoi(s) != 0;
}
template <typename LaunchFn>
static std::string chain_measure(mcStream_t st, int nodes, int measure_replays,
                                 LaunchFn&& launch_nodes, double* out_us_per_kernel,
                                 bool* captured = nullptr) {
    mcGraph_t g = nullptr;
    mcGraphExec_t ex = nullptr;
    std::string reason;
    bool cap = !graph_fail_injected() &&
               mcStreamBeginCapture(st, mcStreamCaptureModeThreadLocal) == mcSuccess;
    if (graph_fail_injected()) reason = "BENCH_GRAPH_FAIL=1（人为注入）";
    else if (!cap) reason = "BeginCapture 失败";
    if (cap) {
        for (int k = 0; k < nodes; ++k) launch_nodes(st, k);
        cap = mcStreamEndCapture(st, &g) == mcSuccess;
        if (!cap) reason = "EndCapture 失败";
    } else {
        reason = "BeginCapture 失败";
    }
    if (cap) {
        const mcError_t e = mcGraphInstantiate(&ex, g, nullptr, nullptr, 0);
        mcGraphDestroy(g);
        if (e != mcSuccess || ex == nullptr) {
            cap = false;
            reason = "GraphInstantiate 失败(err=" + std::to_string((int)e) + ")";
        }
    }
    if (captured) *captured = cap;
    if (!cap) {
        // eager 降级口径：顺序发射 nodes 个节点，pre-kernel 换算口径不变
        for (int i = 0; i < 2; ++i) {
            for (int k = 0; k < nodes; ++k) launch_nodes(st, k);
            CHECK(mcStreamSynchronize(st));
        }
        rank_barrier();
        std::vector<mcEvent_t> c0(measure_replays, nullptr),
            c1(measure_replays, nullptr);
        std::vector<double> per(measure_replays, 0.0);
        for (int i = 0; i < measure_replays; ++i) {
            rank_barrier();
            CHECK(mcEventCreate(&c0[i]));
            CHECK(mcEventCreate(&c1[i]));
            CHECK(mcEventRecord(c0[i], st));
            for (int k = 0; k < nodes; ++k) launch_nodes(st, k);
            CHECK(mcEventRecord(c1[i], st));
            CHECK(mcStreamSynchronize(st));
            float ms = 0;
            CHECK(mcEventElapsedTime(&ms, c0[i], c1[i]));
            per[i] = (double)ms * 1000.0 / (double)nodes;
            mcEventDestroy(c0[i]);
            mcEventDestroy(c1[i]);
        }
        std::sort(per.begin(), per.end());
        *out_us_per_kernel = per.front();
        return "eager(降级: " + reason + ")";
    }
    for (int i = 0; i < 2; ++i) CHECK(mcGraphLaunch(ex, st));
    CHECK(mcStreamSynchronize(st));
    rank_barrier();
    std::vector<mcEvent_t> c0(measure_replays, nullptr), c1(measure_replays, nullptr);
    for (int i = 0; i < measure_replays; ++i) {
        rank_barrier();
        CHECK(mcEventCreate(&c0[i]));
        CHECK(mcEventCreate(&c1[i]));
        CHECK(mcEventRecord(c0[i], st));
        CHECK(mcGraphLaunch(ex, st));
        CHECK(mcEventRecord(c1[i], st));
    }
    CHECK(mcStreamSynchronize(st));
    std::vector<double> per(measure_replays, 0.0);
    for (int i = 0; i < measure_replays; ++i) {
        float ms = 0;
        CHECK(mcEventElapsedTime(&ms, c0[i], c1[i]));
        per[i] = (double)ms * 1000.0 / (double)nodes;
    }
    std::sort(per.begin(), per.end());
    *out_us_per_kernel = per.front();
    for (int i = 0; i < measure_replays; ++i) {
        mcEventDestroy(c0[i]);
        mcEventDestroy(c1[i]);
    }
    mcGraphExecDestroy(ex);
    return "graph";
}

// 旧接口（保持既有调用点不变）：只关心成败，不关心口径
template <typename LaunchFn, typename AfterWarmupFn>
static bool graph_chain_measure(mcStream_t st, int nodes, int measure_replays,
                                LaunchFn&& launch_nodes,
                                AfterWarmupFn&& after_warmup,
                                double* out_us_per_kernel) {
    double us = 0.0;
    bool cap = false;
    chain_measure(st, nodes, measure_replays, launch_nodes, &us, &cap);
    after_warmup();
    *out_us_per_kernel = us;
    return cap;
}

// ---------------------------------------------------------------- 审计 --
static const char* mem_type_name(mcMemoryType t) {
    switch (t) {
        case mcMemoryTypeUnregistered: return "Unregistered";
        case mcMemoryTypeHost: return "Host";
        case mcMemoryTypeDevice: return "Device";
        case mcMemoryTypeManaged: return "Managed";
        case mcMemoryTypeArray: return "Array";
        case mcMemoryTypeUnified: return "Unified";
        default: return "Unknown";
    }
}

static bool audit_sym(const ProdLayout& L, const std::vector<RankRes>& res,
                      int out_rank) {
    const RankRes& r = res[out_rank];
    CHECK(mcSetDevice(r.dev));
    bool ok = true;
    if (out_rank == 0) {
        std::printf("\n---- 显存审计 rank0 (sym=%.2fMB mega=%.2fMB scratch=%.2fMB) ----\n",
                    L.sym_bytes / 1048576.0, L.mega_bytes / 1048576.0,
                    L.scratch_total / 1048576.0);
    }
    struct Probe {
        const char* name;
        const void* ptr;
        bool want_fine;
    };
    const Probe probes[] = {
        // 对称区用 fine-grained（生产 memory.hpp:19 同款）；普通显存跨 rank 写会丢写。
        {"sym_buffer", r.sym, true},
        {"x(view)", r.sym + L.x_off, true},
        {"topk_idx(view)", r.sym + L.topk_idx_off, true},
        {"topk_weights(view)", r.sym + L.topk_weights_off, true},
        {"local_flags", r.lf, false},
        {"ranks_flags", r.rf, true},
    };
    for (const auto& p : probes) {
        mcPointerAttribute_t a;
        std::memset(&a, 0, sizeof(a));
        const mcError_t e = mcPointerGetAttributes(&a, p.ptr);
        const bool fine = (a.type == mcMemoryTypeUnified && a.isManaged);
        const bool ord = (a.type == mcMemoryTypeDevice && !a.isManaged);
        const bool pass = (e == mcSuccess) && (p.want_fine ? fine : ord);
        ok &= pass;
        if (out_rank == 0 || !pass)
            std::printf("  [%s] %-19s ptr=%p type=%-8s managed=%d\n",
                        pass ? "PASS" : "FAIL", p.name, p.ptr,
                        mem_type_name(a.type), (int)a.isManaged);
    }
    const bool x_in = L.x_off >= 0 && L.x_off + L.input_token_bytes <= L.sym_bytes;
    const bool idx_in =
        L.topk_idx_off >= 0 &&
        L.topk_idx_off + (int64_t)L.slot_capacity * kTopk * 8 <= L.sym_bytes;
    const bool w_in =
        L.topk_weights_off >= 0 &&
        L.topk_weights_off + (int64_t)L.slot_capacity * kTopk * 4 <= L.sym_bytes;
    ok &= x_in && idx_in && w_in;
    if (out_rank == 0)
        std::printf("  [%s] x/topk_idx/topk_weights 视图在对称区内\n",
                    (x_in && idx_in && w_in) ? "PASS" : "FAIL");
    const bool page_align =
        (reinterpret_cast<int64_t>(r.sym) + L.barrier_slots_off) %
            deep_gemm::layout::kMegaMoePageAlign == 0;
    ok &= page_align;
    if (out_rank == 0)
        std::printf("  [%s] barrier_slots 4KB 页对齐\n", page_align ? "PASS" : "FAIL");
    int mapped = 0;
    for (size_t d = 0; d < res.size(); ++d)
        if ((int)d != out_rank && r.peer_sym[d] != nullptr) ++mapped;
    ok &= mapped == (int)res.size() - 1;
    if (out_rank == 0)
        std::printf("  [%s] IPC peer 映射 %d/%d\n",
                    (mapped == (int)res.size() - 1) ? "PASS" : "FAIL", mapped,
                    (int)res.size() - 1);
    return ok;
}

// ------------------------------------------------------------- 参数 --
struct RunOpts {
    SceneOpts scene;
    bool graph = true;     // 空 kernel graph 基准
    bool prelude = true;   // M1 前序正确性 + 计时
    int pattern = -1;      // -1=三形态全测；>=0 只跑指定形态（0/1/2）
    int perf_pattern = -1; // 诊断：单launch/graph链路由形态（-1=balanced）
    bool task_trace = false;  // 诊断：dbg[7] 哨兵 + pre_ticks 第二 bank
    int measure_replays = 5;
    bool compute = false;  // M2：启用任务循环（dispatch/L1/L2）

    // ---- 扇形（--sweep 1）相关 ----
    bool sweep = false;    // TPR 扇形模式（逐档起子进程跑，再汇总成表）
    bool quiet = false;    // 静默：只打逐档汇总行 + 最终 [汇总]
    std::vector<int> tpr_list;  // 扇形档位（空 = 用默认表）
    int tpr_timeout = 0;   // 单档超时秒数（0=不限，由外层 timeout 控制）
    int check_rows = 16;   // 数值校验：每专家最多核验的行数（控 CPU 参考成本）
    bool compute_chain = true;  // 额外测一条"计算链"graph（去 REDUCE 收尾）
    bool compute_set = false;   // --compute 是否被显式指定（扇形缺省改 1）
    bool nodes_set = false;     // --nodes 是否被显式指定（扇形缺省改 50）
    bool replays_set = false;   // --replays 是否被显式指定（扇形缺省改 3）
    // 扇形内部用：正确性失败也继续计时（否则该档没有耗时可比）
    bool force_perf = false;
    // run_scene 回填：计时口径（graph / eager[降级]）
    std::string caliber = "graph";

    // ---- 对齐 mega bench_mega_fp8.cu 的口径（默认开）----
    // 单进程、按 R 一次性分配（权重/池子/IPC 复用），对 TPR 列表逐档：
    //   balanced 路由 -> graph 链 nodes 个同构节点（每节点 = mega_kernel + REDUCE）
    //   -> min-of-replays、replay 总时间/nodes -> 跨 rank 取 min。
    // 默认不做正确性校验；输出 R/TPR/us-kernel/口径 表。
    bool aligned = true;
    bool max_tokens_set = false;  // --max-tokens 是否显式给出（缺省 512）
    bool tokens_set = false;      // --tokens 是否显式给出（旧单档口径）
    bool ranks_set = false;       // --ranks 是否显式给出（对齐缺省 32）
    // 不均衡测例：从 TK_INFO 日志复现某次 launch 的每专家收行分布
    std::string tk_info;          // --tk-info <path>
    int tk_launch = 0;            // --tk-launch <N>（0=自动选最不均衡的全 rank launch）
    bool tk_all = false;          // --tk-all：单进程扫描日志全部 launch 出性能表
    int tk_first = 0;             // --tk-first N：只跑前 N 个（按 launch 升序）
};

// 16B/lane 配对 packed tile 权重填充（layout16b 契约，与 megakernel.cu 设备侧同式）：
//   supertile 基址 = W + ((out_col/32)*kIters + (k/32))*1024；lane L 低 8B = 逻辑行
//   (out_col/32)*32 + L%16 的 K 段 [(L/16)*8, +8)，高 8B = 逻辑行 +16 的同段。
//   值函数 WPROBE_VAL 见 megakernel.h（host 参考与设备生成必须逐字节一致；
//   -DWPROBE=1 切非周期小值探针）。
// 本函数仅生成 host CPU 参考缓冲；设备侧生成走 fp8mega::fill_weights_lane。
static void fill_w_lane(std::vector<uint8_t>& buf, int n_rows, int k_cols,
                        uint32_t seed) {
    buf.assign((size_t)n_rows * k_cols, 0);
    const int k_iters = k_cols / 32;
    const int npairs = n_rows / 32;
    // 多线程并行（300MB 单线程生成是 R=32 测试超时主因之一）
    const int nthreads = std::max(1, std::min(16, npairs));
    std::vector<std::thread> th;
    th.reserve(nthreads);
    for (int w = 0; w < nthreads; ++w) {
        th.emplace_back([&, w]() {
            for (int np = w; np < npairs; np += nthreads) {
                for (int kt = 0; kt < k_iters; ++kt) {
                    uint8_t* tile =
                        buf.data() + ((size_t)np * k_iters + kt) * 1024;
                    for (int L = 0; L < 64; ++L) {
                        const int row0 = np * 32 + (L % 16);
                        const int row1 = row0 + 16;
                        const int k0 = kt * 32 + (L / 16) * 8;
                        for (int b = 0; b < 8; ++b) {
                            tile[L * 16 + b] = WPROBE_VAL(seed, row0, k0 + b);
                            tile[L * 16 + 8 + b] = WPROBE_VAL(seed, row1, k0 + b);
                        }
                    }
                }
            }
        });
    }
    for (auto& t : th) t.join();
}

// 构造接口入参（M1：只前序；M3：combine 区偏移由调用方传入）
static fp8mega::MegaLaunchArgs make_mega_args(
    const RunOpts& o, const ProdLayout& L, const RankRes& r, int win_cap,
    int64_t combine_rows_off) {
    fp8mega::MegaLaunchArgs a{};
    a.topk_idx = reinterpret_cast<const int64_t*>(r.sym + L.topk_idx_off);
    a.topk_idx_off = L.topk_idx_off;
    a.self = r.sym;
    a.peer_bases = r.d_base_ptrs;
    a.R = o.scene.ranks;
    a.rank = r.rank;
    a.tokens = o.scene.tokens;
    a.slot_capacity = o.scene.max_tokens;
    a.win_cap = win_cap;
    a.pool_stride = win_cap + o.scene.win_pad;
    a.tbl_cap = o.scene.max_tokens;
    a.k_in = kHidden;
    a.topk = kTopk;
    a.ring = kRing;
    a.src_off = L.src_index_table_off;
    a.rsc_off = L.route_slot_counts_off;
    a.recv_off = L.expert_recv_counts_off;
    a.prefix_off = L.prefix_tbl_off;
    a.barrier_off = L.barrier_slots_off;
    a.lf = r.lf;
    a.rf = r.rf;
    a.rf_peers = r.d_rf_ptrs;
    a.xrelease_idx = kXreleaseIdx;
    a.task_ready_idx = kTaskReadyIdx;
    a.task_counter_idx = kTaskCounterIdx;
    a.l2_done_idx = kL2DoneIdx;
    a.slice_flags_idx = kSliceFlagsIdx;
    a.l1_prog_idx = kL1ProgIdx;
    a.x_off = L.x_off;
    a.weights_off = L.topk_weights_off;
    a.pool_off = L.expert_tokens_off;
    a.pool_ptr = r.pool_rwk;   // [A-RWK] 非空 → pool 走独立普通显存
    a.w_off = L.token_route_weights_off;
    a.meta_off = L.token_src_meta_off;
    a.max_spins = (int)kSpinLimit;
    a.publish_epoch = kPublishEpoch;
    a.recv_stride = (int)L.recv_stride;
    a.prefix_ptr = nullptr;
    // 计算链（M2：--compute 1 时有效）
    a.w1 = r.w1;
    a.l1_n_total = kIntermediate * 2;
    a.l1_block_n = 256;
    a.l1_k_out = kIntermediate;
    a.ffn_hidden = r.ffn_hidden;
    a.frag_a = r.frag_a;
    a.frag2 = r.frag2;
    a.w2 = r.w2;
    a.l2_out = r.l2_out;
    a.l2_k_out = kHidden;
    a.combine_rows_off = combine_rows_off;
    a.combine_cap = o.scene.max_tokens;
    a.sums_stride = deep_gemm::layout::kMegaMoeMaxCombineSlices;
    // REDUCE 输出 + 发射几何
    a.y = r.y;
    a.grid = kGrid;
    a.merge_grid = kGrid;   // 缺省：任务粗化到 ≤ 单波（原行为）
    // [TEST] env 开关：启用 merge=1（不粗化，多波）用于验证 GEN 修复
    if (std::getenv("MX_MERGE1") != nullptr) a.merge_grid = 0x3fffffff;
    const int block_n = prod_block_n(o.scene.block_n);
    a.threads = block_n == 512 ? 1024 : 512;
    a.smem = prod_smem(o.scene.ranks, o.scene.window, block_n);
    return a;
}

static const int kTraceSentinel = 0x517A;   // 诊断哨兵（compute trace 变体识别）

// 单档（TPR 固定）的完整执行结果；扇形汇总用
struct SceneResult {
    bool audit_ok = false;
    bool graph_ok = false;    // 空 kernel graph 链基准
    bool prelude_ok = false;  // 前序三形态正确性
    bool num_ok = false;      // 数值校验（L1/L2/REDUCE）
    bool perf_ok = false;     // 计时成功
    bool check_ran = false;   // 本档是否跑了数据正确性
    bool check_ok = false;    // 数据正确性总判定
    double full_us = -1.0;    // 全链 us/kernel（含 REDUCE）
    double compute_us = -1.0; // 计算链 us/kernel（去 REDUCE 收尾）
    int nodes = 0;
    std::string caliber_full = "-";
    std::string caliber_compute = "-";
    double wall_ms = 0.0;
    bool overall() const {
        return audit_ok && graph_ok && prelude_ok && num_ok && perf_ok;
    }
};

// ------------------------------------------------------------ 扇形档位表 --
// 需求指定的 R=32 均衡路由档位（2 的幂边界及其 +1，暴露跨对齐边界的跳变），
// 用 `--tpr-list all` 全跑。实测（R=32、nodes=50、replays=3、202 机）单档墙钟
// 随 TPR 快速增长：TPR≤4 约 2.2s、TPR=16/17 约 8.3s、TPR≥32 远超 10s 单档预算。
// 故 bench 缺省只给小档子集（1,4）以满足 AGENTS.md 的 10s/档 硬约束；
// 其余档位用 --tpr-list 显式给出（配合 --tpr-timeout 防单档卡死）。
static const std::vector<int>& default_tpr_sweep() {
    static const std::vector<int> kSweep = {1, 4};
    return kSweep;
}
// 需求指定的完整档位（--tpr-list all 或 run.sh 的 TPR_LIST=full）
static const std::vector<int>& full_tpr_sweep() {
    static const std::vector<int> kFull = {1,  4,  16,  17,  32,  33,  64,  65,
                                           128, 129, 256, 257, 512, 513};
    return kFull;
}
// 对齐 mega bench_mega_fp8.cu 的默认档位（R=32 均衡场景）
static const std::vector<int>& aligned_tpr_sweep() {
    static const std::vector<int> kTprs = {1, 4, 16, 32, 33, 64, 128, 256};
    return kTprs;
}

static void usage(const char* argv0) {
    std::printf(
        "用法: %s [选项]\n"
        "  [对齐口径] 默认模式：单进程、均衡路由、graph 链、默认不校验，\n"
        "             输出 R/TPR/us-kernel 表（对齐 mega bench_mega_fp8.cu）。\n"
        "  -R, --ranks 2|8|32   rank 数（对齐模式默认 32；legacy 默认 8）\n"
        "  --tprs A,B,C        TPR 档位（别名 --tpr/--tpr-list；'all'=14 档）\n"
        "                      缺省 = 1,4,16,32,33,64,128,256\n"
        "  --tk-info FILE      不均衡测例：从 TK_INFO 日志复现某 launch 的每专家\n"
        "                      收行分布（覆盖 --tprs，tpk/tokens 由日志推导）\n"
        "  --tk-launch N       指定 TK_INFO 中的 launch 号（缺省=自动选最不均衡者）\n"
        "  --tk-all            扫描 TK_INFO 全部 launch 出性能表（单进程，需 --tk-info）\n"
        "  --nodes N           graph 链节点数（对齐默认 20；legacy 默认 2000）\n"
        "  --replays R         graph replay 次数，取 min（对齐默认 2；legacy 默认 5）\n"
        "  --balanced          对齐模式强制均衡路由（默认即均衡）\n"
        "  --aligned 0|1       0=退回旧单档/扇形口径（默认 1）\n"
        "  --tokens T          旧单档：本 rank token 数（TPR），<= --max-tokens（默认 16）\n"
        "  --max-tokens M      slot 容量，需 >= T（默认 16）\n"
        "  --win-cap N         专家窗口容量，-1=与 max-tokens 相同（默认 -1）\n"
        "  --window N          生产 smem 计算用窗口（默认 8）\n"
        "  --bn 512|256|128    N tile（默认 256）\n"
        "  --graph 0|1         空 kernel graph 链基准（默认 1）\n"
        "  --prelude 0|1       前序正确性 + 计时（默认 1）\n"
        "  --pattern -1|0|1|2  正确性路由形态，-1=三形态全测（默认 -1）\n"
        "  --perf-pattern -1|0|1|2  计时路由形态，-1=balanced（默认 -1）\n"
        "  --compute 0|1       任务循环（dispatch/L1/L2）（默认 0）\n"
        "  --audit 0|1         显存审计（默认 1）\n"
        "  --perf-only 0|1     跳过数据正确性（默认 0）\n"
        "  --task-trace 0|1    诊断：dbg[7] 哨兵 + 第二 bank tick（默认 0）\n"
        "  --sweep 0|1         TPR 扇形模式（默认 0；不带时是老的单档行为）\n"
        "  --tpr-list A,B,C    扇形档位（逗号分隔）；'all' = 需求指定的 14 档\n"
        "  --quiet 0|1         扇形静默：只打逐档一行 + [汇总]（默认 0）\n"
        "  --tpr-timeout S     扇形单档超时秒数（0=不限；被超时杀 -> 该档 FAILED）\n"
        "  --check-rows N      数值校验每专家最多核验行数（默认 16，控 CPU 参考成本）\n"
        "  --compute-chain 0|1 额外测一条「去 REDUCE 收尾」的 graph 链作为 compute 列\n"
        "                     （默认 1，仅扇形路径使用）\n"
        "  --help, -h          显示本帮助\n"
        "\n"
        "退出码: 0=总体 PASS，1=总体 FAIL，2=参数错误。\n"
        "环境变量: MACA_VISIBLE_DEVICES 选卡（逗号分隔，本平台不认 0-7 范围语法）\n"
        "          BENCH_RED_MODE / BENCH_RED_TIMING  诊断开关（见源码注释）\n",
        argv0);
}

// 解析逗号分隔整数列表；spec == "all" 时用需求指定的完整档位表。
static bool parse_int_list(const std::string& spec, std::vector<int>* out,
                           std::string* err) {
    out->clear();
    if (spec == "all") {
        *out = full_tpr_sweep();
        return true;
    }
    size_t pos = 0;
    while (pos <= spec.size()) {
        const size_t c = spec.find(',', pos);
        const std::string tok =
            spec.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
        if (!tok.empty()) {
            const int v = std::atoi(tok.c_str());
            if (v <= 0) {
                *err = "TPR 档位必须为正整数: " + tok;
                return false;
            }
            out->push_back(v);
        }
        if (c == std::string::npos) break;
        pos = c + 1;
    }
    if (out->empty()) {
        *err = "TPR 列表为空";
        return false;
    }
    return true;
}

// ------------------------------------------------------------ 参数解析 --
static bool parse_args(int argc, char** argv, RunOpts* o, std::string* err) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&]() -> int {
            if (++i >= argc) {
                *err = "missing value for " + a;
                return -1;
            }
            return std::atoi(argv[i]);
        };
        if (a == "--help" || a == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else if (a == "--ranks" || a == "-R") {
            o->scene.ranks = value();
            o->ranks_set = true;
        }
        else if (a == "--max-tokens") {
            o->scene.max_tokens = value();
            o->max_tokens_set = true;
        }
        else if (a == "--win-cap") o->scene.win_cap = value();
        else if (a == "--win-pad") o->scene.win_pad = value();
        else if (a == "--tokens") {
            o->scene.tokens = value();
            o->tokens_set = true;
        }
        else if (a == "--window") o->scene.window = value();
        else if (a == "--bn") o->scene.block_n = value();
        else if (a == "--nodes") {
            o->scene.nodes = value();
            o->nodes_set = true;
        }
        else if (a == "--replays") {
            o->measure_replays = value();
            o->replays_set = true;
        }
        else if (a == "--audit") o->scene.audit = value() != 0;
        else if (a == "--perf-only") o->scene.perf_only = value() != 0;
        else if (a == "--graph") o->graph = value() != 0;
        else if (a == "--prelude") o->prelude = value() != 0;
        else if (a == "--pattern") o->pattern = value();
        else if (a == "--perf-pattern") o->perf_pattern = value();
        else if (a == "--task-trace") o->task_trace = value() != 0;
        else if (a == "--compute") {
            o->compute = value() != 0;
            o->compute_set = true;
        }
        else if (a == "--sweep") o->sweep = value() != 0;
        else if (a == "--quiet") o->quiet = value() != 0;
        else if (a == "--tpr-timeout") o->tpr_timeout = value();
        else if (a == "--check-rows") o->check_rows = value();
        else if (a == "--compute-chain") o->compute_chain = value() != 0;
        else if (a == "--aligned") o->aligned = value() != 0;
        else if (a == "--balanced") o->perf_pattern = 0;   // 参考口径：均衡路由
        else if (a == "--iters" || a == "--trials") (void)value();  // 参考兼容，忽略
        else if (a == "--tpr-list" || a == "--tprs" || a == "--tpr") {
            if (++i >= argc) {
                *err = "missing value for " + a;
                return false;
            }
            if (!parse_int_list(argv[i], &o->tpr_list, err)) return false;
        } else if (a == "--tk-info") {
            if (++i >= argc) {
                *err = "missing value for --tk-info";
                return false;
            }
            o->tk_info = argv[i];
        } else if (a == "--tk-launch") o->tk_launch = value();
        else if (a == "--tk-all") o->tk_all = true;
        else if (a == "--tk-first") { o->tk_first = value(); o->tk_all = true; }
        else {
            *err = "未知参数: " + a;
            return false;
        }
        if (!err->empty()) return false;
    }
    if (o->scene.ranks != 2 && o->scene.ranks != 8 && o->scene.ranks != 32) {
        *err = "ranks must be 2/8/32 (got " + std::to_string(o->scene.ranks) + ")";
        return false;
    }
    if (o->scene.tokens > o->scene.max_tokens) {
        *err = "--tokens 不能大于 --max-tokens";
        return false;
    }
    if (o->measure_replays <= 0 || o->scene.nodes <= 0) {
        *err = "--replays 与 --nodes 必须为正";
        return false;
    }
    if (o->check_rows <= 0) o->check_rows = 1;
    // 旧单档调用（显式 --tokens 且未给 TPR 列表）自动退回 legacy，不改其语义。
    if (o->aligned && o->tokens_set && o->tpr_list.empty()) o->aligned = false;
    // 对齐模式缺省 R=32（参考基准场景），显式 --ranks 时尊重。
    if (o->aligned && !o->ranks_set) o->scene.ranks = 32;
    return true;
}

// ------------------------------------------------------------- 单档执行 --
// 原 main 的主体：固定一个 TPR 跑完整流程（分配 → 审计 → 正确性 → 计时）。
// 扇形模式下每档由 fork 出的子进程独立调用一次（显存/IPC 不跨档复用）。
static int run_scene(RunOpts o, SceneResult* out) {
    int ndev = 0;
    CHECK(mcGetDeviceCount(&ndev));
    if (ndev < o.scene.ranks) {
        std::fprintf(stderr, "need %d visible devices, got %d\n", o.scene.ranks,
                     ndev);
        return 1;
    }
    int sms = 0, clock_khz = 0;
    CHECK(mcSetDevice(0));
    CHECK(mcDeviceGetAttribute(&sms, mcDeviceAttributeMultiProcessorCount, 0));
    CHECK(mcDeviceGetAttribute(&clock_khz, mcDeviceAttributeClockRate, 0));

    // win_cap 是物理专家窗口容量（决定 scratch/寻址/barrier 序列几何），
    // 可独立于 --max-tokens 设置：--win-cap < max-tokens 时容量收紧（尾块/多波），
    // --win-cap > max-tokens 时只是分配富余（无额外 kernel 代价）。
    // 因此取 max(win_cap, max_tokens)，同时对 smem 预算做显式校验。
    const int win_cap = (o.scene.win_cap < 0)
                            ? o.scene.max_tokens
                            : (o.scene.win_cap > o.scene.max_tokens
                                   ? o.scene.win_cap
                                   : o.scene.max_tokens);
    ProdLayout L =
        compute_layout(o.scene.ranks, o.scene.max_tokens, kTopk, kHidden,
                       kIntermediate, win_cap + o.scene.win_pad);
    L.win_capacity = win_cap;  // 逻辑窗口容量（layout 参数含 win_pad 地址 padding）
    const int block_n = prod_block_n(o.scene.block_n);
    const int threads = block_n == 512 ? 1024 : 512;
    const int64_t smem = prod_smem(o.scene.ranks, o.scene.window, block_n);
    if (smem > 227 * 1024) {
        std::fprintf(stderr,
                     "win_cap=%d (max_tokens=%d) 需要 smem=%lldB > 227KB；"
                     "请减小 --window\n",
                     win_cap, o.scene.max_tokens, (long long)smem);
        return 2;
    }

    // 测试扩展区（sym 尾部）：COMBINE BF16 行 + 完成标记（M3 使用）
    const int64_t barrier_pad_off = ((L.sym_bytes + 255) / 256) * 256;
    const int64_t recv_pad_off =
        ((barrier_pad_off + (int64_t)o.scene.ranks * 64 * 4 + 255) / 256) * 256;
    const int64_t combine_f32_off =
        ((recv_pad_off + (int64_t)o.scene.ranks * 64 * 4 + 255) / 256) * 256;
    // combine_rows（BF16）末端；combine_sums 已迁入 ranks_flags，不再占 sym。
    const int64_t combine_rows_end =
        ((combine_f32_off + (int64_t)kTopk * o.scene.max_tokens * kHidden * 2 + 255) /
         256) * 256;
    const int64_t sym_alloc = combine_rows_end;

    std::printf("============================================================\n");
    std::printf("fp8xfp8 MoE Mega Kernel 基准（M1 前序）\n");
    std::printf("ranks=%d EPR=%d E=%d topk=%d H=%d I=%d TPR=%d\n", o.scene.ranks,
                kEPR, o.scene.ranks * kEPR, kTopk, kHidden, kIntermediate,
                o.scene.tokens);
    std::printf("launch: grid=%d threads=%d bn=%d smem=%lld (SMs=%d clock=%dMHz)\n",
                kGrid, threads, block_n, (long long)smem, sms, clock_khz / 1000);
    std::printf("sym: alloc=%.2fMB (layout=%.2fMB) win_cap=%d win_pad=%d nodes=%d\n",
                sym_alloc / 1048576.0, L.sym_bytes / 1048576.0, win_cap,
                o.scene.win_pad, o.scene.nodes);
    std::printf("============================================================\n");

    std::vector<RankRes> res(o.scene.ranks);
    std::atomic<int> alloc_ready{0};
    std::atomic<int> handle_ready{0};
    g_rb_n = o.scene.ranks;

    // M2：权重在 host 生成一次（lane 序 packed tile），各 rank memcpy
    std::vector<uint8_t> w1_host, w2_host;
    if (o.compute) {
        fill_w_lane(w1_host, kEPR * 2 * kIntermediate, kHidden, 17u);
        fill_w_lane(w2_host, kEPR * kHidden, kIntermediate, 91u);
        std::printf("  [M2] 权重生成: W1=%.0fMB W2=%.0fMB（lane 序 packed tile）\n",
                    w1_host.size() / 1048576.0, w2_host.size() / 1048576.0);
    }

    // ---- 分配 ----
    {
        std::vector<std::thread> pool;
        for (int rank = 0; rank < o.scene.ranks; ++rank) {
            pool.emplace_back([&, rank]() {
                RankRes& r = res[rank];
                r.rank = rank;
                r.dev = rank;
                CHECK(mcSetDevice(rank));
                void* p = nullptr;
                // 对称区必须用 fine-grained（生产 csrc/apis/memory.hpp:19 同款）：
                // 普通 mcMalloc 在跨 rank peer 写下会丢写（O-D1 探针：写方回读可见、
                // 收方本地读永远为 0）→ xbar/recv 收齐超时。FG 是唯一可靠选择。
                const mcError_t me = mcExtMallocWithFlags(
                    &p, (size_t)sym_alloc, mcDeviceMallocFinegrained);
                if (me != mcSuccess || p == nullptr) {
                    std::fprintf(stderr,
                                 "rank%d FG 分配失败(%d)，回退普通显存"
                                 "（跨 rank 写不可靠，仅诊断用）\n",
                                 rank, (int)me);
                    CHECK(mcMalloc(&p, (size_t)sym_alloc));
                }
                r.sym = static_cast<uint8_t*>(p);
                CHECK(mcMemset(r.sym, 0, (size_t)sym_alloc));
                // local_flags：本 rank 私有，普通显存
                CHECK(mcMalloc((void**)&r.lf, sizeof(local_flags)));
                CHECK(mcMemset(r.lf, 0, sizeof(local_flags)));
                // ranks_flags：跨 rank，FG 显存（与 sym 同款；普通显存跨 rank 写丢写）
                {
                    void* rfp = nullptr;
                    const mcError_t rme = mcExtMallocWithFlags(
                        &rfp, sizeof(ranks_flags), mcDeviceMallocFinegrained);
                    if (rme != mcSuccess || rfp == nullptr) {
                        std::fprintf(stderr,
                                     "rank%d ranks_flags FG 分配失败(%d)，回退普通显存\n",
                                     rank, (int)rme);
                        CHECK(mcMalloc(&rfp, sizeof(ranks_flags)));
                    }
                    r.rf = static_cast<ranks_flags*>(rfp);
                    CHECK(mcMemset(r.rf, 0, sizeof(ranks_flags)));
                }
                // out_total/out_processed/dbg/ck_dbg/pre_ticks/l1_chk/l2_chk
                // 已迁入 local_flags，由上面的 lf 清零覆盖。
                // M2 计算链资源（--compute 1）
                if (!w1_host.empty()) {
                    CHECK(mcMalloc((void**)&r.w1, w1_host.size()));
                    CHECK(mcMalloc((void**)&r.w2, w2_host.size()));
                    // 设备侧生成（替代 host→device 9.6GB 上传；host 缓冲仅作数值参考）
                    fp8mega::fill_weights_lane(r.w1, kEPR * 2 * kIntermediate,
                                               kHidden, 17u, 0);
                    fp8mega::fill_weights_lane(r.w2, kEPR * kHidden,
                                               kIntermediate, 91u, 0);
                    CHECK(mcMalloc((void**)&r.ffn_hidden,
                                   (size_t)kEPR * kRing * win_cap * kIntermediate));
                    CHECK(mcMemset(r.ffn_hidden, 0,
                                   (size_t)kEPR * kRing * win_cap * kIntermediate));
                    CHECK(mcMalloc((void**)&r.l2_out,
                                   (size_t)kGrid * win_cap * kHidden * 2));
                    CHECK(mcMemset(r.l2_out, 0,
                                   (size_t)kGrid * win_cap * kHidden * 2));
                    // [A-RWK] 实验：L1 A pool 独立普通显存（env 开关，默认关）
                    if (std::getenv("FP8MEGA_POOL_RWK") != nullptr) {
                        const size_t pool_bytes = (size_t)kEPR * kRing *
                            (size_t)(win_cap + o.scene.win_pad) * kHidden;
                        CHECK(mcMalloc((void**)&r.pool_rwk, pool_bytes));
                        CHECK(mcMemset(r.pool_rwk, 0, pool_bytes));
                    }
                    // M3 REDUCE 输出 y [max_tokens][H] FP32
                    CHECK(mcMalloc((void**)&r.y,
                                   (size_t)o.scene.max_tokens * kHidden *
                                       sizeof(float)));
                    CHECK(mcMemset(r.y, 0,
                                   (size_t)o.scene.max_tokens * kHidden *
                                       sizeof(float)));
                }
                // DISPATCH 校验用：确定性填充 x 视图与 topk_weights 视图
                {
                    std::vector<uint8_t> xb((size_t)o.scene.max_tokens * kHidden);
                    for (int t = 0; t < o.scene.max_tokens; ++t)
                        for (int c = 0; c < kHidden; ++c)
                            // &0x3f：E4M3 正有限值（0x80+ 是负数/NaN，会让 L1 出 NaN）
                            xb[(size_t)t * kHidden + c] =
                                (uint8_t)((rank * 37 + t * 7 + c * 3) & 0x3f);
                    CHECK(mcMemcpy(r.sym + L.x_off, xb.data(), xb.size(),
                                   mcMemcpyHostToDevice));
                    std::vector<float> wv((size_t)o.scene.max_tokens * kTopk);
                    for (int i = 0; i < o.scene.max_tokens * kTopk; ++i)
                        wv[i] = 1.0f + rank * 0.25f + (i % 7) * 0.01f;
                    CHECK(mcMemcpy(r.sym + L.topk_weights_off, wv.data(),
                                   wv.size() * sizeof(float),
                                   mcMemcpyHostToDevice));
                }
                r.peer_sym.assign(o.scene.ranks, nullptr);
                CHECK(mcIpcGetMemHandle(&r.handle, r.sym));
                r.peer_rf.assign(o.scene.ranks, nullptr);
                CHECK(mcIpcGetMemHandle(&r.rf_handle, r.rf));
                alloc_ready.fetch_add(1);
            });
        }
        while (alloc_ready.load() < o.scene.ranks) {}
        for (auto& t : pool) t.join();
    }

    // ---- IPC open + peer access ----
    {
        std::vector<std::thread> pool;
        std::vector<int> peer_enabled(o.scene.ranks, 0),
            peer_unavail(o.scene.ranks, 0);
        for (int rank = 0; rank < o.scene.ranks; ++rank) {
            pool.emplace_back([&, rank]() {
                RankRes& r = res[rank];
                CHECK(mcSetDevice(rank));
                for (int d = 0; d < o.scene.ranks; ++d) {
                    if (d == rank) continue;
                    void* q = nullptr;
                    const mcError_t e = mcIpcOpenMemHandle(
                        &q, res[d].handle, mcIpcMemLazyEnablePeerAccess);
                    if (e != mcSuccess) {
                        std::fprintf(stderr, "rank%d open peer%d failed: %d\n",
                                     rank, d, (int)e);
                        std::exit(1);
                    }
                    r.peer_sym[d] = static_cast<uint8_t*>(q);
                    void* qr = nullptr;
                    const mcError_t er = mcIpcOpenMemHandle(
                        &qr, res[d].rf_handle, mcIpcMemLazyEnablePeerAccess);
                    if (er != mcSuccess) {
                        std::fprintf(stderr, "rank%d open peer%d rf failed: %d\n",
                                     rank, d, (int)er);
                        std::exit(1);
                    }
                    r.peer_rf[d] = static_cast<ranks_flags*>(qr);
                }
                int enabled = 0, unavailable = 0;
                for (int d = 0; d < o.scene.ranks; ++d) {
                    if (d == rank) continue;
                    int can = 0;
                    if (mcDeviceCanAccessPeer(&can, rank, d) == mcSuccess && can) {
                        const mcError_t e = mcDeviceEnablePeerAccess(d, 0);
                        if (e == mcSuccess || e == mcErrorPeerAccessAlreadyEnabled)
                            ++enabled;
                        else
                            ++unavailable;
                    }
                }
                std::vector<uint8_t*> base_ptrs(o.scene.ranks);
                std::vector<ranks_flags*> rf_ptrs(o.scene.ranks);
                for (int d = 0; d < o.scene.ranks; ++d) {
                    base_ptrs[d] = (d == rank) ? r.sym : r.peer_sym[d];
                    rf_ptrs[d] = (d == rank) ? r.rf : r.peer_rf[d];
                }
                CHECK(mcMalloc((void**)&r.d_base_ptrs,
                               o.scene.ranks * sizeof(uint8_t*)));
                CHECK(mcMemcpy(r.d_base_ptrs, base_ptrs.data(),
                               o.scene.ranks * sizeof(uint8_t*),
                               mcMemcpyHostToDevice));
                CHECK(mcMalloc((void**)&r.d_rf_ptrs,
                               o.scene.ranks * sizeof(ranks_flags*)));
                CHECK(mcMemcpy(r.d_rf_ptrs, rf_ptrs.data(),
                               o.scene.ranks * sizeof(ranks_flags*),
                               mcMemcpyHostToDevice));
                peer_enabled[rank] = enabled;
                peer_unavail[rank] = unavailable;
                handle_ready.fetch_add(1);
            });
        }
        while (handle_ready.load() < o.scene.ranks) {}
        for (auto& t : pool) t.join();
        int sum_en = 0, sum_un = 0;
        for (int rank = 0; rank < o.scene.ranks; ++rank) {
            sum_en += peer_enabled[rank];
            sum_un += peer_unavail[rank];
        }
        std::printf("  peer_access: enabled=%d unavailable=%d（R=%d）\n", sum_en,
                    sum_un, o.scene.ranks);
    }

    // ---- 显存审计 ----
    bool audit_ok = true;
    if (o.scene.audit) {
        int pass = 0, fail = 0;
        for (int rank = 0; rank < o.scene.ranks; ++rank) {
            const bool ok = audit_sym(L, res, rank);
            (ok ? pass : fail) += 1;
            audit_ok &= ok;
        }
        std::printf("[结论-A 显存审计] R=%d PASS=%d FAIL=%d %s\n",
                    o.scene.ranks, pass, fail, audit_ok ? "PASS" : "FAIL");
    }

    // ---- 空 kernel graph 链基准 ----
    bool graph_ok = true;
    double empty_prod_us = 0.0;
    if (o.graph) {
        struct GeomCase {
            const char* name;
            int grid, threads;
            int64_t smem;
        };
        const std::vector<GeomCase> cases = {
            {"G0 1x32 smem0", 1, 32, 0},
            {"G1 32x512 smem0", kGrid, threads, 0},
            {"G2 生产几何", kGrid, threads, smem},
        };
        std::vector<double> geom_us;
        for (const auto& c : cases) {
            std::vector<std::thread> th(o.scene.ranks);
            std::vector<double> us(o.scene.ranks, 0.0);
            std::vector<int> fail(o.scene.ranks, 0);
            std::atomic<int> go{0};
            for (int rank = 0; rank < o.scene.ranks; ++rank) {
                th[rank] = std::thread([&, rank]() {
                    CHECK(mcSetDevice(rank));
                    mcStream_t st = nullptr;
                    CHECK(mcStreamCreate(&st));
                    while (go.load(std::memory_order_acquire) == 0) {}
                    const bool ok = graph_chain_measure(
                        st, o.scene.nodes, o.measure_replays,
                        [&](mcStream_t s, int) {
                            fp8mega::launch_empty(c.grid, c.threads, c.smem, s);
                        },
                        []() {}, &us[rank]);
                    fail[rank] = ok ? 0 : 1;
                    mcStreamDestroy(st);
                });
            }
            go.store(1, std::memory_order_release);
            for (auto& t : th) t.join();
            for (int rank = 0; rank < o.scene.ranks; ++rank)
                graph_ok &= fail[rank] == 0;
            double worst = 0;
            for (int rank = 0; rank < o.scene.ranks; ++rank)
                worst = std::max(worst, us[rank]);
            if (std::string(c.name).rfind("G2", 0) == 0) empty_prod_us = worst;
            geom_us.push_back(worst);
        }
        if (graph_ok) {
            std::printf("[空kernel][graph] R=%d nodes=%d: G0=%.3f | G1=%.3f | G2=%.3f us/kernel\n",
                        o.scene.ranks, o.scene.nodes, geom_us[0], geom_us[1],
                        geom_us[2]);
            std::printf("[结论-B 空kernel] R=%d 生产几何 graph 链=%.3f us/kernel\n",
                        o.scene.ranks, empty_prod_us);
        } else {
            std::printf("[空kernel][graph] capture 失败\n");
        }
    }

    // ---- M1 前序：正确性（三形态）+ graph 链计时 ----
    bool prelude_ok = true;
    double prelude_us = 0.0;
    bool m2_num_ok = true;   // M2 数值校验（L1/L2 参考对比）
    bool m3_num_ok = true;   // M3 数值校验（REDUCE y 参考对比）
    if (o.prelude) {
        const RoutePattern pats[3] = {RoutePattern::Balanced, RoutePattern::Hot,
                                      RoutePattern::MixedInvalid};
        const int npat = (o.pattern >= 0 && o.pattern < 3) ? 1 : 3;
        const int pat0 = (o.pattern >= 0 && o.pattern < 3) ? o.pattern : 0;
        std::vector<std::vector<int64_t>> routes(o.scene.ranks);
        // 计时用 balanced 路由（先上传）
        for (int rank = 0; rank < o.scene.ranks; ++rank)
            routes[rank] = make_routes(RoutePattern::Balanced, rank,
                                       o.scene.ranks, o.scene.tokens);
        if (!o.scene.perf_only) {
            for (int pi = 0; pi < npat; ++pi) {
                const RoutePattern pat = pats[pat0 + pi];
                for (int rank = 0; rank < o.scene.ranks; ++rank)
                    routes[rank] = make_routes(pat, rank, o.scene.ranks,
                                               o.scene.tokens);
                // 清零 + 上传路由。
                // 两遍清零：第一遍清本形态残留；间隔 50ms 后第二遍清除前一形态
                // 可能迟到到达的 __stwt posted 写（跨 rank 写无全局顺序，多形态
                // 连测时旧数据可能自洽通过 checksum 而污染本形态）。
                for (int pass = 0; pass < 2; ++pass) {
                    if (pass == 1)
                        std::this_thread::sleep_for(std::chrono::milliseconds(50));
                    std::vector<std::thread> th(o.scene.ranks);
                    for (int rank = 0; rank < o.scene.ranks; ++rank) {
                        th[rank] = std::thread([&, rank]() {
                            RankRes& r = res[rank];
                            CHECK(mcSetDevice(rank));
                            CHECK(mcMemcpy(r.sym + L.topk_idx_off,
                                           routes[rank].data(),
                                           routes[rank].size() * sizeof(int64_t),
                                           mcMemcpyHostToDevice));
                            CHECK(mcMemset(r.sym + L.src_index_table_off, 0,
                                           (size_t)L.src_index_table_bytes));
                            CHECK(mcMemset(r.sym + L.route_slot_counts_off, 0,
                                           (size_t)o.scene.ranks * kEPR * sizeof(int)));
                            CHECK(mcMemset(r.sym + L.expert_recv_counts_off, 0,
                                           (size_t)o.scene.ranks * L.recv_stride * 4));
                            CHECK(mcMemset(r.sym + L.prefix_tbl_off, 0,
                                           (size_t)L.prefix_tbl_bytes));
                            CHECK(mcMemset(r.sym + L.barrier_slots_off, 0,
                                           (size_t)o.scene.ranks * o.scene.ranks * 4));
                            CHECK(mcMemset(r.lf, 0, sizeof(local_flags)));
                            CHECK(mcMemset(r.rf, 0, sizeof(ranks_flags)));
                            // 所有 rank 的第二遍清零完成后才允许各自 launch：
                            // 避免某 rank 提前 launch 的 publish 写被其他 rank 覆盖
                            if (pass == 1) rank_barrier();
                        });
                    }
                    for (auto& t : th) t.join();
                }
                // 单 launch（前序）
                {
                    std::vector<std::thread> th(o.scene.ranks);
                    for (int rank = 0; rank < o.scene.ranks; ++rank) {
                        th[rank] = std::thread([&, rank]() {
                            CHECK(mcSetDevice(rank));
                            fp8mega::MegaLaunchArgs args = make_mega_args(
                                o, L, res[rank], win_cap, combine_f32_off);
                            fp8mega::launch_prelude(args, 0);
                            CHECK(mcDeviceSynchronize());
                        });
                    }
                    for (auto& t : th) t.join();
                }
                // 全 rank 校验
                int pass = 0, fail = 0;
                std::string first_msg;
                for (int rank = 0; rank < o.scene.ranks; ++rank) {
                    std::string msg;
                    bool ok = true;
                    if (!validate_route(L, o.scene.ranks, o.scene.tokens, rank,
                                        routes, res[rank].sym, &msg))
                        ok = false;
                    if (ok && !validate_publish(L, o.scene.ranks, o.scene.tokens,
                                                rank, routes, res[rank].sym,
                                                kPublishEpoch, &msg,
                                                (int)L.recv_stride,
                                                L.expert_recv_counts_off))
                        ok = false;
                    std::vector<int> ts_cpu(kGrid, 0);
                    if (ok) {
                        CHECK(mcSetDevice(rank));
                        CHECK(mcMemcpy(ts_cpu.data(), res[rank].lf->out_total,
                                       kGrid * sizeof(int),
                                       mcMemcpyDeviceToHost));
                        if (!validate_task_setup(L, o.scene.ranks, rank, routes,
                                                 res[rank].sym,
                                                 /*bar_flags=*/nullptr,
                                                 ts_cpu.data(), &msg))
                            ok = false;
                    }
                    (ok ? pass : fail) += 1;
                    if (!ok && first_msg.empty()) {
                        std::vector<int> dbg_cpu(8, 0);
                        std::vector<int> pfx((size_t)kEPR * (o.scene.ranks + 1), 0);
                        std::vector<uint32_t> ck((size_t)kEPR * 16, 0);
                        CHECK(mcSetDevice(rank));
                        CHECK(mcMemcpy(dbg_cpu.data(), res[rank].lf->dbg,
                                       8 * sizeof(int), mcMemcpyDeviceToHost));
                        CHECK(mcMemcpy(pfx.data(),
                                       res[rank].sym + L.prefix_tbl_off,
                                       pfx.size() * sizeof(int),
                                       mcMemcpyDeviceToHost));
                        CHECK(mcMemcpy(ck.data(), res[rank].lf->ck_dbg,
                                       ck.size() * sizeof(uint32_t),
                                       mcMemcpyDeviceToHost));
                        first_msg = "rank" + std::to_string(rank) + ": " + msg +
                                    " dbg=[";
                        for (int d = 0; d < 8; ++d)
                            first_msg += std::to_string(dbg_cpu[d]) + ",";
                        first_msg += "] prefix=";
                        for (int e = 0; e < kEPR; ++e) {
                            first_msg += "[";
                            for (int rr = 0; rr <= o.scene.ranks; ++rr)
                                first_msg +=
                                    std::to_string(pfx[e * (o.scene.ranks + 1) + rr]) + ",";
                            first_msg += "]";
                        }
                        first_msg += " ck=";
                        for (int e = 0; e < kEPR; ++e) {
                            first_msg += "[";
                            for (int c = 0; c < 10; ++c)
                                first_msg += std::to_string(ck[e * 16 + c]) + ",";
                            first_msg += "]";
                        }
                    }
                }
                prelude_ok &= fail == 0;
                std::printf("[前序正确性] %-13s R=%d PASS=%d FAIL=%d %s%s\n",
                            pattern_name(pat), o.scene.ranks, pass, fail,
                            fail == 0 ? "PASS" : "FAIL",
                            first_msg.empty() ? "" : ("  " + first_msg).c_str());
            }
        }
        // ---- 单 launch 相位计时（clock64 诊断口径；不受 graph 跨 rank 同步问题影响）----
        if (prelude_ok || o.scene.perf_only) {
            const RoutePattern pp = (o.perf_pattern >= 0 && o.perf_pattern < 3)
                ? (RoutePattern)o.perf_pattern : RoutePattern::Balanced;
            for (int rank = 0; rank < o.scene.ranks; ++rank)
                routes[rank] = make_routes(pp, rank,
                                           o.scene.ranks, o.scene.tokens);
            {
                std::vector<std::thread> th(o.scene.ranks);
                for (int rank = 0; rank < o.scene.ranks; ++rank) {
                    th[rank] = std::thread([&, rank]() {
                        RankRes& r = res[rank];
                        CHECK(mcSetDevice(rank));
                        CHECK(mcMemcpy(r.sym + L.topk_idx_off, routes[rank].data(),
                                       routes[rank].size() * sizeof(int64_t),
                                       mcMemcpyHostToDevice));
                        CHECK(mcMemset(r.sym + L.src_index_table_off, 0,
                                       (size_t)L.src_index_table_bytes));
                        CHECK(mcMemset(r.sym + L.expert_recv_counts_off, 0,
                                       (size_t)o.scene.ranks * L.recv_stride * 4));
                        CHECK(mcMemset(r.sym + L.barrier_slots_off, 0,
                                       (size_t)o.scene.ranks * o.scene.ranks * 4));
                        CHECK(mcMemset(r.lf, 0, sizeof(local_flags)));
                        CHECK(mcMemset(r.rf, 0, sizeof(ranks_flags)));
                        if (o.task_trace)
                            CHECK(mcMemcpy(r.lf->dbg + 7, &kTraceSentinel, sizeof(int),
                                           mcMemcpyHostToDevice));
                        CHECK(mcMemset(r.lf->pre_ticks, 0,
                                       (size_t)kGrid * 16 *
                                           sizeof(unsigned long long)));
                        // M3：COMBINE 目标区清零（combine_sums 标记在 rf，已随 rf 清零）
                        CHECK(mcMemset(r.sym + combine_f32_off, 0,
                                       (size_t)kTopk * o.scene.max_tokens *
                                           kHidden * 2));
                        fp8mega::MegaLaunchArgs args = make_mega_args(
                            o, L, r, win_cap, combine_f32_off);
                        // 所有 rank 清零完成后统一 launch（FG 下跨 rank 写可靠）
                        rank_barrier();
                        const bool rt = red_timing();
                        mcEvent_t e0 = nullptr, e1 = nullptr, e2 = nullptr;
                        if (rt) {
                            CHECK(mcEventCreate(&e0));
                            CHECK(mcEventCreate(&e1));
                            CHECK(mcEventCreate(&e2));
                            CHECK(mcEventRecord(e0, 0));
                        }
                        if (o.compute) {
                            fp8mega::launch_compute(args, 0);
                            // [R32D] 诊断 rt==2：再跑一次热 mega（覆盖 pre_ticks 得暖态相位）
                            if (red_timing() == 2)
                                fp8mega::launch_compute(args, 0);
                            if (rt) CHECK(mcEventRecord(e1, 0));
                            // M3 REDUCE：任务循环结束后独立 kernel 收尾（同流有序）。
                            // grid = 2×tokens（每 token 2 block 列分半）；TPR=16 时恰为 32。
                            fp8mega::launch_reduce(args, 0);
                        } else {
                            fp8mega::launch_prelude(args, 0);
                        }
                        if (rt) CHECK(mcEventRecord(e2, 0));
                        CHECK(mcDeviceSynchronize());
                        if (rt) {
                            // 第二次 REDUCE：标记已全部就绪（纯工作量，无等待）
                            mcEvent_t e3 = nullptr;
                            CHECK(mcEventCreate(&e3));
                            if (o.compute) fp8mega::launch_reduce(args, 0);
                            CHECK(mcEventRecord(e3, 0));
                            CHECK(mcDeviceSynchronize());
                            float m1 = 0, m2 = 0, m3 = 0;
                            CHECK(mcEventElapsedTime(&m1, e0, e1));
                            CHECK(mcEventElapsedTime(&m2, e1, e2));
                            CHECK(mcEventElapsedTime(&m3, e2, e3));
                            std::printf("[REDUCE计时] rank=%d mega=%.1f "
                                        "reduce=%.1f reduce2(warm)=%.1f us\n",
                                        r.rank, (double)m1 * 1000.0,
                                        (double)m2 * 1000.0,
                                        (double)m3 * 1000.0);
                            mcEventDestroy(e0);
                            mcEventDestroy(e1);
                            mcEventDestroy(e2);
                            mcEventDestroy(e3);
                        }
                    });
                }
                for (auto& t : th) t.join();
            }
            std::vector<unsigned long long> pt(8, 0);
            CHECK(mcSetDevice(0));
            CHECK(mcMemcpy(pt.data(), res[0].lf->pre_ticks,
                           8 * sizeof(unsigned long long), mcMemcpyDeviceToHost));
            if (o.task_trace) {
                // 第二 bank：pt + kGrid*8；每块 8 槽
                std::vector<unsigned long long> tr((size_t)kGrid * 8, 0);
                CHECK(mcMemcpy(tr.data(), res[0].lf->pre_ticks + (size_t)kGrid * 8,
                               tr.size() * sizeof(unsigned long long),
                               mcMemcpyDeviceToHost));
                std::printf(
                    "[trace] 相位累计周期（1600MHz）：claim/gate pull pub+waitall "
                    "L1 l1pwait L2 tail+done tasks\n");
                for (int b = 0; b < kGrid; ++b) {
                    const unsigned long long* t = tr.data() + (size_t)b * 8;
                    std::printf("[trace] b=%02d %llu %llu %llu %llu %llu %llu "
                                "%llu n=%llu\n",
                                b, t[0], t[1], t[2], t[3], t[4], t[5], t[6],
                                t[7]);
                }
            }
            const double clk = 2000.0;   // MHz（XCORE_CLK，骨架同口径）
            std::printf("[相位-单launch] R=%d %s: route=%.1f publish=%.1f "
                        "xbar=%.1f setup=%.1f | 前序=%.1f us\n",
                        o.scene.ranks, pattern_name(pp),
                        (double)(pt[0] - pt[6]) / clk,
                        (double)(pt[1] - pt[0]) / clk,
                        (double)(pt[2] - pt[1]) / clk,
                        (double)(pt[3] - pt[2]) / clk,
                        (double)(pt[3] - pt[6]) / clk);
            if (o.compute) {
                std::vector<float> l1c(kGrid, 0), l2c(kGrid, 0);
                std::vector<int> proc(kGrid, 0);
                CHECK(mcSetDevice(0));
                CHECK(mcMemcpy(l1c.data(), res[0].lf->l1_chk, kGrid * sizeof(float),
                               mcMemcpyDeviceToHost));
                CHECK(mcMemcpy(l2c.data(), res[0].lf->l2_chk, kGrid * sizeof(float),
                               mcMemcpyDeviceToHost));
                CHECK(mcMemcpy(proc.data(), res[0].lf->out_processed,
                               kGrid * sizeof(int), mcMemcpyDeviceToHost));
                int sum_proc = 0, min_proc = 1 << 30, max_proc = 0;
                for (int b = 0; b < kGrid; ++b) {
                    sum_proc += proc[b];
                    min_proc = std::min(min_proc, proc[b]);
                    max_proc = std::max(max_proc, proc[b]);
                }
                std::printf("[M2] l1_chk[0]=%.1f l2_chk[0]=%.1f（全 block 累加） | "
                            "out_processed 合计=%d 范围=[%d,%d]\n",
                            l1c[0], l2c[0], sum_proc, min_proc, max_proc);
                if (pt[7] > pt[6]) {
                    std::printf("[计算链相位] dispatch=%.1f L1=%.1f "
                                "L2+COMBINE+尾=%.1f | 计算链合计=%.1f us\n",
                                (double)(pt[4] - pt[3]) / clk,
                                (double)(pt[5] - pt[4]) / clk,
                                (double)(pt[7] - pt[5]) / clk,
                                (double)(pt[7] - pt[3]) / clk);
                }
                // ---- 数值正确性（L1/L2）：设备字节 vs host 参考（C 子agent 提供）----
                if (!w1_host.empty()) {
                    const int e0 = 0, slice0 = 0;
                    const int pool_slot = e0 * kRing;   // w=0
                    // 核验行数 = min(本档实际行数, --check-rows)。
                    // 实际行数由 balanced 路由模型算出（专家 e0 收到的对数，按
                    // cap 截断）；TPR<16 的档位实际行数不足 16，固定按 16 行读
                    // 会读到未写过的垃圾行 —— 那是 bench 校验器的历史缺陷，
                    // 不是 kernel 缺陷（TPR=1 的 L2 FAIL 由此而来）。
                    const int E = o.scene.ranks * kEPR;
                    const int cap = o.scene.win_cap < 0 ? o.scene.max_tokens
                                                        : o.scene.win_cap;
                    int real_rows = 0;
                    for (int src = 0; src < o.scene.ranks; ++src) {
                        int c = 0;
                        for (int64_t v : routes[src])
                            if ((uint64_t)v == (uint64_t)e0) ++c;
                        real_rows += std::min(c, cap);
                    }
                    const int rows = std::max(1, std::min(real_rows, o.check_rows));
                    if (real_rows < o.check_rows)
                        std::printf(
                            "[校验口径] 专家 e0 实际行数=%d < --check-rows=%d，"
                            "L1/L2 按 %d 行核验\n",
                            real_rows, o.check_rows, rows);
                    (void)E;
                    const int pool_stride = win_cap + o.scene.win_pad;
                    std::vector<uint8_t> pool_d((size_t)rows * kHidden);
                    const uint8_t* pool_src =
                        res[0].pool_rwk != nullptr
                            ? res[0].pool_rwk +
                                  (size_t)pool_slot * pool_stride * kHidden
                            : res[0].sym + L.expert_tokens_off +
                                  (size_t)pool_slot * pool_stride * kHidden;
                    CHECK(mcMemcpy(pool_d.data(), pool_src, pool_d.size(),
                                   mcMemcpyDeviceToHost));
                    std::vector<float> ffn_exp((size_t)rows * kIntermediate);
                    ref_l1(pool_d.data(), w1_host.data(), e0, rows, kHidden,
                           kIntermediate, ffn_exp.data());
                    // 参考量化到 E4M3（RNE+饱和，与设备量化语义一致）再解码比较：
                    // 测试数据量级大（K=6144 累加）会超出 FP8 范围，设备饱和到 448，
                    // 直接与量化前 FP32 期望比会全错。
                    std::vector<uint8_t> ffn_exp_q(ffn_exp.size());
                    ref_quantize_fp8(ffn_exp.data(), (long long)ffn_exp.size(),
                                     ffn_exp_q.data());
                    std::vector<float> ffn_exp_f(ffn_exp.size());
                    for (size_t i = 0; i < ffn_exp.size(); ++i)
                        ffn_exp_f[i] = fp8_e4m3_decode(ffn_exp_q[i]);
                    std::vector<uint8_t> ffn_d((size_t)rows * kIntermediate);
                    CHECK(mcMemcpy(ffn_d.data(),
                                   res[0].ffn_hidden +
                                       (size_t)pool_slot * win_cap * kIntermediate,
                                   ffn_d.size(), mcMemcpyDeviceToHost));
                    const CheckStats st1 = compare_fp8_rows(
                        ffn_d.data(), ffn_exp_f.data(), rows, kIntermediate,
                        kIntermediate, slice0 * (kIntermediate / kNSlice),
                        (slice0 + 1) * (kIntermediate / kNSlice), 7e-2, 2e-3);
                    print_check_stats("L1 ffn_hidden(slice0)", st1);
                    std::vector<float> out_exp((size_t)rows * kHidden);
                    ref_l2(ffn_d.data(), w2_host.data(), e0, rows, kIntermediate,
                           kHidden, out_exp.data());
                    std::vector<uint16_t> out_d((size_t)rows * kHidden);
                    CHECK(mcMemcpy(out_d.data(), res[0].l2_out,
                                   out_d.size() * 2, mcMemcpyDeviceToHost));
                    const CheckStats st2 = compare_bf16_rows(
                        out_d.data(), out_exp.data(), rows, kHidden, kHidden,
                        slice0 * (kHidden / kNSlice),
                        (slice0 + 1) * (kHidden / kNSlice), 1e-2, 1e-3);
                    print_check_stats("L2 out(slice0)", st2);
                    m2_num_ok = st1.pass() && st2.pass();
                }
                // ---- 数值正确性（M3 REDUCE）：设备 y vs host ref_reduce ----
                // 每 rank 抽样前 4 个 token（全 token 参考在 R=32 时超 10s 测试约束）
                {
                    const int E = o.scene.ranks * kEPR;
                    const int t_end = std::min(o.scene.tokens, 4);
                    // R>8 时只查 rank0（其余 rank 逻辑相同；保证单测 ≤10s 约束的余量）
                    const int rank_step = (o.scene.ranks > 8) ? o.scene.ranks : 1;
                    CheckStats st3;
                    for (int rank = 0; rank < o.scene.ranks; rank += rank_step) {
                        std::vector<uint8_t> rows_h(
                            (size_t)kTopk * o.scene.max_tokens * kHidden * 2);
                        std::vector<float> w_h((size_t)o.scene.tokens * kTopk);
                        std::vector<float> y_h((size_t)o.scene.tokens * kHidden);
                        std::vector<float> y_ref((size_t)o.scene.tokens * kHidden);
                        CHECK(mcSetDevice(rank));
                        CHECK(mcMemcpy(rows_h.data(),
                                       res[rank].sym + combine_f32_off,
                                       rows_h.size(), mcMemcpyDeviceToHost));
                        CHECK(mcMemcpy(w_h.data(),
                                       res[rank].sym + L.topk_weights_off,
                                       w_h.size() * sizeof(float),
                                       mcMemcpyDeviceToHost));
                        CHECK(mcMemcpy(y_h.data(), res[rank].y,
                                       y_h.size() * sizeof(float),
                                       mcMemcpyDeviceToHost));
                        for (int t = 0; t < t_end; ++t)
                            ref_reduce(rows_h.data(), routes[rank].data(),
                                       w_h.data(), t, kTopk, E,
                                       o.scene.max_tokens, kHidden,
                                       y_ref.data() + (size_t)t * kHidden);
                        const CheckStats s = compare_f32(
                            y_h.data(), y_ref.data(),
                            t_end * kHidden, 1e-5, 1e-6);
                        st3.n += s.n;
                        st3.bad += s.bad;
                        st3.nan += s.nan;
                        if (s.max_abs > st3.max_abs) {
                            st3.max_abs = s.max_abs;
                            st3.worst =
                                s.worst + (long long)rank * o.scene.tokens * kHidden;
                            st3.worst_got = s.worst_got;
                            st3.worst_want = s.worst_want;
                        }
                        if (s.max_rel > st3.max_rel) st3.max_rel = s.max_rel;
                    }
                    print_check_stats("REDUCE y(每rank前4token)", st3);
                    m3_num_ok = st3.pass();
                }
            }
        }
        // graph 链计时（balanced）
        // force_perf（扇形）：即使正确性失败也要给出该档耗时，便于对比
        if (prelude_ok || o.scene.perf_only || o.force_perf) {
            const RoutePattern pp = (o.perf_pattern >= 0 && o.perf_pattern < 3)
                ? (RoutePattern)o.perf_pattern : RoutePattern::Balanced;
            for (int rank = 0; rank < o.scene.ranks; ++rank)
                routes[rank] = make_routes(pp, rank,
                                           o.scene.ranks, o.scene.tokens);
            // 计时前清零（两遍，间隔 50ms：清前一形态迟到写；计时链同路由无污染）
            for (int pass = 0; pass < 2; ++pass) {
                if (pass == 1)
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                std::vector<std::thread> th(o.scene.ranks);
                for (int rank = 0; rank < o.scene.ranks; ++rank) {
                    th[rank] = std::thread([&, rank]() {
                        RankRes& r = res[rank];
                        CHECK(mcSetDevice(rank));
                        CHECK(mcMemcpy(r.sym + L.topk_idx_off, routes[rank].data(),
                                       routes[rank].size() * sizeof(int64_t),
                                       mcMemcpyHostToDevice));
                        CHECK(mcMemset(r.sym + L.src_index_table_off, 0,
                                       (size_t)L.src_index_table_bytes));
                        CHECK(mcMemset(r.sym + L.barrier_slots_off, 0,
                                       (size_t)o.scene.ranks * o.scene.ranks * 4));
                        CHECK(mcMemset(r.lf, 0, sizeof(local_flags)));
                        CHECK(mcMemset(r.rf, 0, sizeof(ranks_flags)));
                        if (pass == 1) rank_barrier();
                    });
                }
                for (auto& t : th) t.join();
            }
            // 节点体：with_reduce=true 全链（mega+REDUCE）；false 计算链（仅 mega）
            auto make_launch = [&](int rank, bool with_reduce) {
                return [&, rank, with_reduce](mcStream_t s) {
                    RankRes& r = res[rank];
                    const fp8mega::MegaLaunchArgs a2 = make_mega_args(
                        o, L, r, win_cap, combine_f32_off);
                    if (!o.compute) {
                        fp8mega::launch_prelude(a2, s);
                        return;
                    }
                    // 端到端口径：graph 链内追加 REDUCE（同流有序，依赖 combine_sums 标记）
                    if (with_reduce) fp8mega::launch_full(a2, s);
                    else fp8mega::launch_compute(a2, s);
                };
            };
            // 口径 1：全链（含 REDUCE 收尾）；口径 2：计算链（仅 mega 任务循环）
            const int n_cal = (o.compute && o.compute_chain) ? 2 : 1;
            std::vector<std::vector<double>> us(
                n_cal, std::vector<double>(o.scene.ranks, 0.0));
            std::vector<std::string> cal(n_cal, "-");
            std::atomic<int> go{0};
            for (int ci = 0; ci < n_cal; ++ci) {
                std::vector<std::thread> th(o.scene.ranks);
                for (int rank = 0; rank < o.scene.ranks; ++rank) {
                    th[rank] = std::thread([&, rank, ci]() {
                        CHECK(mcSetDevice(rank));
                        mcStream_t st = nullptr;
                        CHECK(mcStreamCreate(&st));
                        auto launch = make_launch(rank, ci == 0);
                        while (go.load(std::memory_order_acquire) == 0) {}
                        cal[ci] = chain_measure(
                            st, o.scene.nodes, o.measure_replays,
                            [&](mcStream_t s, int) { launch(s); }, &us[ci][rank]);
                        mcStreamDestroy(st);
                    });
                }
                go.store(1, std::memory_order_release);
                for (auto& t : th) t.join();
            }
            bool tok = true;
            for (int ci = 0; ci < n_cal; ++ci)
                if (cal[ci].rfind("eager", 0) == 0) tok = false;  // graph 不可用
            double worst = 0;
            for (int rank = 0; rank < o.scene.ranks; ++rank)
                worst = std::max(worst, us[0][rank]);
            prelude_us = worst;
            for (int ci = 0; ci < n_cal; ++ci) {
                if (cal[ci].rfind("eager", 0) == 0)
                    std::printf("[graph 降级] %s 口径降级为 eager（%s）\n",
                                ci == 0 ? "全链" : "计算链", cal[ci].c_str());
            }
            // 无论 graph 还是降级 eager，都照常输出耗时行（口径列单独标注），
            // 这样扇形仍能拿到该档数字，只是口径列写 eager(degraded)。
            for (int ci = 0; ci < n_cal; ++ci) {
                double w = 0;
                for (int rank = 0; rank < o.scene.ranks; ++rank)
                    w = std::max(w, us[ci][rank]);
                std::printf("[%s计时] R=%d %s nodes=%d: %s=%.3f us/kernel 口径=%s\n",
                            ci == 0 ? (o.compute ? "全链" : "前序") : "计算链",
                            o.scene.ranks, pattern_name(pp), o.scene.nodes,
                            ci == 0 ? (o.compute ? "full" : "prelude") : "compute", w,
                            cal[ci].c_str());
            }
            out->full_us = prelude_us;
            out->caliber_full = tok ? "graph" : "eager(degraded)";
            out->perf_ok = true;
            out->nodes = o.scene.nodes;
            if (n_cal == 2) {
                double wc = 0;
                for (int rank = 0; rank < o.scene.ranks; ++rank)
                    wc = std::max(wc, us[1][rank]);
                out->compute_us = wc;
                out->caliber_compute = tok ? "graph" : "eager(degraded)";
            } else {
                out->compute_us = -1.0;
                out->caliber_compute = "-";
            }
            if (!tok) prelude_ok = false;   // graph 不可用：单档模式仍判 FAIL
            {
                // 诊断：dbg 码 dump（0=xbar 1=release 2=checksum 3=task_ready 4=l1 5=l2）
                for (int rank = 0; rank < o.scene.ranks; ++rank) {
                    std::vector<int> d8(8, 0);
                    CHECK(mcSetDevice(rank));
                    CHECK(mcMemcpy(d8.data(), res[rank].lf->dbg, 8 * sizeof(int),
                                   mcMemcpyDeviceToHost));
                    std::printf("  [dbg] rank%d=[", rank);
                    for (int d = 0; d < 8; ++d)
                        std::printf("%d%s", d8[d], d == 7 ? "" : ",");
                    std::printf("]\n");
                }
            }
        }
    }
    out->audit_ok = audit_ok;
    out->graph_ok = graph_ok;
    out->prelude_ok = prelude_ok;
    out->num_ok = m2_num_ok && m3_num_ok;
    out->check_ran = !o.scene.perf_only;
    out->check_ok = audit_ok && graph_ok && prelude_ok && m2_num_ok && m3_num_ok;
    out->nodes = o.scene.nodes;

    const bool overall = out->check_ok && out->perf_ok;
    std::printf("[总体] R=%d %s\n", o.scene.ranks, overall ? "PASS" : "FAIL");
    return overall ? 0 : 1;
}

// ------------------------------------------------------------ 扇形驱动 --
// 在子进程里跑一个档位：stdout/stderr 走管道，父进程收集并按行解析。
// 用子进程而不是同进程循环的原因：
//   1) 显存不跨档复用（R=32 大档 sym 数百 MB × 32 卡，循环同进程会 OOM）；
//   2) 某档崩溃/超时不影响后续档（需求：任一档失败不中断整轮扇形）；
//   3) 驱动解析用的判据行（[总体] / [前序正确性] / [*计时]）本来就都在 stdout。
struct ChildOut {
    std::string text;
    int rc = 0;          // 子进程退出码
    bool timed_out = false;
};

static ChildOut run_tpr_child(const RunOpts& o) {
    ChildOut out;
    int fds[2] = {-1, -1};
    if (pipe(fds) != 0) {
        std::perror("pipe");
        out.rc = -1;
        return out;
    }
    std::fflush(stdout);
    std::fflush(stderr);
    const pid_t pid = fork();
    if (pid < 0) {
        std::perror("fork");
        close(fds[0]);
        close(fds[1]);
        out.rc = -1;
        return out;
    }
    if (pid == 0) {
        // 子进程：stdout/stderr 重定向到管道；超时由父进程 kill
        close(fds[0]);
        dup2(fds[1], STDOUT_FILENO);
        dup2(fds[1], STDERR_FILENO);
        close(fds[1]);
        SceneResult dummy;
        const int rc = run_scene(o, &dummy);
        std::fflush(stdout);
        std::fflush(stderr);
        _exit(rc);
    }
    close(fds[1]);
    // 父进程：边读边等，超时则 kill
    const auto t0 = std::chrono::steady_clock::now();
    bool killed = false;
    char buf[4096];
    while (true) {
        if (o.tpr_timeout > 0 && !killed) {
            const double el = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - t0)
                                  .count();
            if (el > (double)o.tpr_timeout) {
                out.text += "\n[扇形] TPR=" + std::to_string(o.scene.tokens) +
                            " 超过 --tpr-timeout=" + std::to_string(o.tpr_timeout) +
                            "s，已终止该档子进程\n";
                kill(pid, SIGKILL);
                killed = true;
            }
        }
        // 非阻塞读 + 短暂 sleep，避免 read 阻塞导致无法及时 kill
        const int flags = fcntl(fds[0], F_GETFL, 0);
        fcntl(fds[0], F_SETFL, flags | O_NONBLOCK);
        const ssize_t n = read(fds[0], buf, sizeof(buf));
        if (n > 0) {
            out.text.append(buf, (size_t)n);
            continue;
        }
        if (n == 0) break;   // EOF：子进程已关管道
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            int status = 0;
            const pid_t w = waitpid(pid, &status, WNOHANG);
            if (w == pid) {
                // 子进程已退出：把管道里剩余的读干净
                ssize_t m = 0;
                while ((m = read(fds[0], buf, sizeof(buf))) > 0)
                    out.text.append(buf, (size_t)m);
                out.rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
                out.timed_out = killed;
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }
        break;  // 读错误
    }
    close(fds[0]);
    if (out.rc == 0 && !killed) {
        int status = 0;
        if (waitpid(pid, &status, 0) == pid)
            out.rc = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }
    out.timed_out = killed;
    return out;
}

// 从子进程输出里抽关键判据（前序正确性 / 总体 / 两条计时）
struct TprVerdict {
    int prelude_pass = 0, prelude_fail = 0;
    bool overall_pass = false;
    bool saw_overall = false;
    bool saw_full = false, saw_compute = false;
    double full_us = -1.0, compute_us = -1.0;
    int nodes = 0;
    bool degraded = false;
    bool calib_note = false;   // 只是行数提示，不是降级
    std::string calib_full = "-", calib_compute = "-";
};

static bool line_has(const std::string& s, const char* key) {
    return s.find(key) != std::string::npos;
}

static TprVerdict parse_child(const std::string& text) {
    TprVerdict v;
    size_t pos = 0;
    while (pos <= text.size()) {
        const size_t nl = text.find('\n', pos);
        const std::string line =
            text.substr(pos, nl == std::string::npos ? std::string::npos : nl - pos);
        // [前序正确性] <pat> R=.. PASS=n FAIL=m PASS|FAIL
        if (line_has(line, "[前序正确性]")) {
            const size_t p = line.find("PASS=");
            const size_t f = line.find("FAIL=");
            if (p != std::string::npos) v.prelude_pass = std::atoi(line.c_str() + p + 5);
            if (f != std::string::npos) v.prelude_fail = std::atoi(line.c_str() + f + 5);
        }
        if (line_has(line, "[总体]")) {
            v.saw_overall = true;
            v.overall_pass = line_has(line, "PASS");
        }
        if (line_has(line, "[graph 降级]")) v.degraded = true;
        if (line_has(line, "[全链计时]") || line_has(line, "[前序计时]")) {
            // 先剥掉尾部 "口径=xxx"，避免 rfind('=') 命中口径里的等号
            std::string body = line;
            size_t cal = body.find("口径=");
            if (cal != std::string::npos) {
                v.calib_full = body.substr(cal + 7);  // "口径=" = 2 汉字(3B each)+1B
                body = body.substr(0, cal);
            }
            const size_t c = body.rfind('=');
            const size_t u = body.find("us/kernel");
            if (c != std::string::npos && u != std::string::npos && c < u) {
                v.full_us = std::atof(body.c_str() + c + 1);
                v.saw_full = true;
            }
            const size_t nn = body.find("nodes=");
            if (nn != std::string::npos) v.nodes = std::atoi(body.c_str() + nn + 6);
        }
        if (line_has(line, "[计算链计时]")) {
            std::string body = line;
            const size_t cal = body.find("口径=");
            if (cal != std::string::npos) {
                v.calib_compute = body.substr(cal + 7);
                body = body.substr(0, cal);
            }
            const size_t c = body.rfind('=');
            const size_t u = body.find("us/kernel");
            if (c != std::string::npos && u != std::string::npos && c < u) {
                v.compute_us = std::atof(body.c_str() + c + 1);
                v.saw_compute = true;
            }
        }
        if (line_has(line, "[校验口径]")) v.calib_note = true;
        if (nl == std::string::npos) break;
        pos = nl + 1;
    }
    if (v.calib_full == "-") v.calib_full = v.saw_full ? "graph" : "-";
    if (v.calib_compute == "-")
        v.calib_compute = v.saw_compute ? "graph" : "-";
    return v;
}

static int run_sweep(RunOpts o) {
    std::vector<int> tprs = o.tpr_list.empty() ? default_tpr_sweep() : o.tpr_list;
    // 扇形固定口径：每档 tokens = max_tokens = win_cap = TPR（容量紧贴 TPR）
    o.force_perf = true;
    // 扇形缺省含计算链（与 mega 的 --compute 缺省 1 一致）；显式给了就尊重
    if (!o.compute_set) o.compute = true;
    // 扇形缺省 nodes=50（需求指定）；显式给了就尊重
    if (!o.nodes_set) o.scene.nodes = 50;
    // 扇形缺省 replays=3（需求指定 min-of-3）；显式给了就尊重
    if (!o.replays_set) o.measure_replays = 3;
    std::printf("配置: R=%d 模式=sweep 档位数=%zu nodes=%d replays=%d "
                "compute=%d compute-chain=%d check-rows=%d pattern=balanced\n",
                o.scene.ranks, tprs.size(), o.scene.nodes, o.measure_replays,
                (int)o.compute, (int)o.compute_chain, o.check_rows);
    std::printf("档位(TPR):");
    for (int t : tprs) std::printf(" %d", t);
    std::printf("\n");
    if (o.tpr_timeout > 0)
        std::printf("单档超时: %ds（超时档标 FAILED 并继续）\n", o.tpr_timeout);

    struct Row {
        int tpr;
        bool check_ok, check_ran, perf_ok;
        std::string calib;
        int nodes;
        double full_us, compute_us, wall_s;
    };
    std::vector<Row> rows;
    int n_pass = 0;
    for (size_t i = 0; i < tprs.size(); ++i) {
        const int tpr = tprs[i];
        RunOpts oo = o;
        oo.scene.tokens = tpr;
        oo.scene.max_tokens = tpr;
        oo.scene.win_cap = tpr;
        oo.sweep = false;
        if (!o.quiet)
            std::printf("\n>>> [%zu/%zu] TPR=%d 开始（nodes=%d）\n", i + 1,
                        tprs.size(), tpr, o.scene.nodes);
        const auto t0 = std::chrono::steady_clock::now();
        const ChildOut co = run_tpr_child(oo);
        const double wall = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - t0)
                                .count();
        if (!o.quiet) std::fputs(co.text.c_str(), stdout);
        const TprVerdict v = parse_child(co.text);
        Row r;
        r.tpr = tpr;
        r.check_ran = !oo.scene.perf_only;
        // 正确性判定：复用子进程 [总体]（含显存审计 / 空核 graph / 前序三形态 /
        // 数值校验），与单档模式的判定口径完全一致。
        r.check_ok = v.saw_overall && v.overall_pass && !co.timed_out;
        r.perf_ok = v.saw_full && !co.timed_out;
        r.calib = v.calib_full;
        r.nodes = v.nodes > 0 ? v.nodes : o.scene.nodes;
        r.full_us = v.saw_full ? v.full_us : -1.0;
        r.compute_us = v.saw_compute ? v.compute_us : -1.0;
        r.wall_s = wall;
        const bool ok = r.check_ok && r.perf_ok;
        if (ok) ++n_pass;
        char full_s[32] = "-", comp_s[32] = "-";
        if (r.full_us > 0) std::snprintf(full_s, sizeof(full_s), "%.3f", r.full_us);
        if (r.compute_us > 0)
            std::snprintf(comp_s, sizeof(comp_s), "%.3f", r.compute_us);
        std::printf("TPR=%-5d %-8s %-16s nodes=%-4d full=%-11s compute=%-11s "
                    "%.2fs%s\n",
                    r.tpr, r.check_ran ? (r.check_ok ? "PASS" : "FAILED") : "SKIPPED",
                    r.calib.c_str(), r.nodes, full_s, comp_s, r.wall_s,
                    ok ? "" : "  <-- FAILED");
        rows.push_back(r);
    }

    std::printf("\n=========== 扇形汇总 ===========\n");
    std::printf("%-7s %-9s %-16s %6s %12s %12s %9s\n", "TPR", "正确性", "口径",
                "nodes", "full(us/k)", "compute(us/k)", "档位耗时");
    for (const Row& r : rows) {
        char full_s[32] = "-", comp_s[32] = "-", wall_s[24];
        if (r.full_us > 0) std::snprintf(full_s, sizeof(full_s), "%.3f", r.full_us);
        if (r.compute_us > 0)
            std::snprintf(comp_s, sizeof(comp_s), "%.3f", r.compute_us);
        std::snprintf(wall_s, sizeof(wall_s), "%.2fs", r.wall_s);
        std::printf("%-7d %-9s %-16s %6d %12s %12s %9s\n", r.tpr,
                    r.check_ran ? (r.check_ok ? "PASS" : "FAILED") : "SKIPPED",
                    r.calib.c_str(), r.nodes, full_s, comp_s, wall_s);
    }
    std::printf(
        "\n说明: full = graph 链 mega_kernel(+REDUCE) 的 us/kernel（每 replay 单独\n"
        "      event 夹住、min-of-%d、replay 总时间 / nodes）；compute = 同一 nodes\n"
        "      的 graph 链去掉 REDUCE 收尾节点（其余完全同构），两者之差即 REDUCE。\n"
        "      口径 graph=捕获成功；eager(degraded)=捕获不可用、已降级 eager 复刻。\n"
        "      正确性口径复用单档模式的 [总体]（显存审计 + 空核 graph + 前序三形态\n"
        "      route/publish/xbar/task_setup + L1/L2/REDUCE 数值校验）。\n",
        o.measure_replays);
    std::printf("\n[汇总] %zu 档：%d PASS / %zu FAILED\n", rows.size(), n_pass,
                rows.size() - (size_t)n_pass);
    return n_pass == (int)rows.size() ? 0 : 1;
}

// ============================================================================
// 不均衡测例：从 TK_INFO 日志复现一次真实 launch 的每专家收行分布
//
// 日志格式（MXMOEMega fp8xfp8 内核 DG_TK_INFO_EVERY）：
//   [TK_INFO] rank=<r> launch=<n> n=<EPR> : c0 c1 ... c_{EPR-1}
// 其中 ci = 该 rank 物理专家 i **收到**的 route pair 数（全网源 rank 汇总）。
//
// 复现方式：把「全局专家 e=r*EPR+le 需要 count[r][le] 个 pair」展开成一个
// pool（按专家分组），再按 `j % R` round-robin 分给 R 个源 rank —— 每个源 rank
// 恰好 tokens*topk 个 pair，且各专家收行数精确等于日志值。round-robin 让每个
// 专家的一份行摊到各源 rank（每 (专家,源) ≤ ceil(count/R)），不超 slot 容量。
// ============================================================================
struct TkCase {
    int tokens = 0;            // 每 rank token 数（= total/(R*topk)）
    int launch = 0;            // 选中的 launch 号
    int max_expert = 0;        // 单专家最大收行数（最不均衡处）
    int nz_expert = 0;         // 非零收行的专家数
    int ranks = 0;             // 日志中出现的 rank 数（<R 表示抽样/部分）
    bool imbalanced = false;   // 该例是否判为长尾不均衡
    int win_cap = 0;           // [P50-D] 该例自适应窗口容量（逐例计算）
    std::vector<std::vector<int64_t>> routes;  // [R][tokens*topk] 全局专家号
};

// 由 cnt[R][epr] 构造路由（round-robin 分给源 rank）+ 自检。total 须整除 R*topk。
static bool build_tk_case(const std::vector<std::vector<int>>& cnt, int launch,
                          int nranks, int R, int topk, int epr, TkCase* out,
                          std::string* err) {
    long long total = 0;
    int max_expert = 0, nz = 0;
    for (int r = 0; r < R; ++r)
        for (int e = 0; e < epr; ++e) {
            total += cnt[r][e];
            if (cnt[r][e] > max_expert) max_expert = cnt[r][e];
            if (cnt[r][e] > 0) ++nz;
        }
    const int per = R * topk;
    if (total <= 0 || total % per != 0) {
        *err = "launch=" + std::to_string(launch) + " 总 pair=" +
               std::to_string(total) + " 不能被 R*topk=" + std::to_string(per) +
               " 整除";
        return false;
    }
    const int tokens = (int)(total / per);
    if (tokens <= 0) {
        *err = "推导出 tokens<=0";
        return false;
    }
    out->routes.assign(R, std::vector<int64_t>((size_t)tokens * topk, 0));
    long long j = 0;
    for (int r = 0; r < R; ++r)
        for (int e = 0; e < epr; ++e) {
            const int64_t ge = (int64_t)r * epr + e;
            for (int c = 0; c < cnt[r][e]; ++c, ++j) {
                const int src = (int)(j % R);
                const int idx = (int)(j / R);
                out->routes[src][(size_t)(idx / topk) * topk + (idx % topk)] = ge;
            }
        }
    std::vector<std::vector<int>> got(R, std::vector<int>(epr, 0));
    for (int s = 0; s < R; ++s)
        for (int64_t e : out->routes[s])
            if (e >= 0 && e < (int64_t)R * epr)
                ++got[(int)(e / epr)][(int)(e % epr)];
    for (int r = 0; r < R; ++r)
        for (int e = 0; e < epr; ++e)
            if (got[r][e] != cnt[r][e]) {
                *err = "路由自检失败 rank=" + std::to_string(r) + " expert=" +
                       std::to_string(e);
                return false;
            }
    out->tokens = tokens;
    out->launch = launch;
    out->max_expert = max_expert;
    out->nz_expert = nz;
    out->ranks = nranks;
    return true;
}

static bool load_tk_case(const std::string& path, int want_launch, int R,
                         int topk, int epr, TkCase* out, std::string* err) {
    std::ifstream f(path);
    if (!f) {
        *err = "无法打开 TK_INFO 文件: " + path;
        return false;
    }
    std::map<int, std::map<int, std::vector<int>>> data;  // launch -> rank -> vals
    std::string line;
    while (std::getline(f, line)) {
        const size_t p = line.find("[TK_INFO]");
        if (p == std::string::npos) continue;
        int rank = -1, launch = -1, n = -1;
        if (std::sscanf(line.c_str() + p, "[TK_INFO] rank=%d launch=%d n=%d :",
                        &rank, &launch, &n) != 3)
            continue;
        const size_t colon = line.find(':', p);
        if (colon == std::string::npos) continue;
        std::istringstream is(line.substr(colon + 1));
        std::vector<int> vals;
        int v = 0;
        while (is >> v) vals.push_back(v);
        if ((int)vals.size() != n || n != epr) continue;
        data[launch][rank] = vals;
    }
    if (data.empty()) {
        *err = "TK_INFO 中未解析到任何合法行: " + path;
        return false;
    }
    // 选 launch：指定优先；否则在「含 R 个 rank」的 launch 里取单专家最大值最大者
    int sel = -1, sel_max = -1;
    if (want_launch > 0) {
        if (!data.count(want_launch)) {
            *err = "TK_INFO 中无 launch=" + std::to_string(want_launch);
            return false;
        }
        sel = want_launch;
    } else {
        for (const auto& kv : data) {
            if ((int)kv.second.size() != R) continue;
            int mx = 0;
            for (const auto& rk : kv.second)
                for (int c : rk.second) mx = std::max(mx, c);
            if (mx > sel_max) {
                sel_max = mx;
                sel = kv.first;
            }
        }
        if (sel < 0) {
            *err = "TK_INFO 中没有含 " + std::to_string(R) + " 个 rank 的 launch";
            return false;
        }
    }

    std::vector<std::vector<int>> cnt(R, std::vector<int>(epr, 0));
    for (const auto& rk : data[sel]) {
        if (rk.first < 0 || rk.first >= R) continue;
        cnt[rk.first] = rk.second;
    }
    return build_tk_case(cnt, sel, (int)data[sel].size(), R, topk, epr, out, err);
}

// [TK-ALL] 解析全部 launch（跳过非法/空的），返回按 launch 升序的测例表。
static bool load_tk_all(const std::string& path, int R, int topk, int epr,
                        std::vector<TkCase>* out, std::string* err) {
    std::ifstream f(path);
    if (!f) {
        *err = "无法打开 TK_INFO 文件: " + path;
        return false;
    }
    std::map<int, std::map<int, std::vector<int>>> data;
    std::string line;
    while (std::getline(f, line)) {
        const size_t p = line.find("[TK_INFO]");
        if (p == std::string::npos) continue;
        int rank = -1, launch = -1, n = -1;
        if (std::sscanf(line.c_str() + p, "[TK_INFO] rank=%d launch=%d n=%d :",
                        &rank, &launch, &n) != 3)
            continue;
        const size_t colon = line.find(':', p);
        if (colon == std::string::npos) continue;
        std::istringstream is(line.substr(colon + 1));
        std::vector<int> vals;
        int v = 0;
        while (is >> v) vals.push_back(v);
        if ((int)vals.size() != n || n != epr) continue;
        data[launch][rank] = vals;
    }
    out->clear();
    int skipped = 0;
    for (const auto& kv : data) {
        std::vector<std::vector<int>> cnt(R, std::vector<int>(epr, 0));
        for (const auto& rk : kv.second)
            if (rk.first >= 0 && rk.first < R) cnt[rk.first] = rk.second;
        TkCase c;
        std::string e;
        if (!build_tk_case(cnt, kv.first, (int)kv.second.size(), R, topk, epr,
                           &c, &e)) {
            ++skipped;
            continue;
        }
        out->push_back(std::move(c));
    }
    if (out->empty()) {
        *err = "TK_INFO 中没有可用 launch（skipped=" + std::to_string(skipped) +
               "）";
        return false;
    }
    std::fprintf(stderr, "[tk-all] 解析 %zu 个可用 launch（跳过 %d）\n",
                 out->size(), skipped);
    return true;
}

// ============================================================================
// 对齐 mega bench_mega_fp8.cu 的口径（run_aligned）
//
// 与 run_scene 的差异：单进程、按 R **只分配一次**资源（权重/池子/IPC 全复用），
// 对 TPR 列表逐档跑 full-chain graph（mega_kernel + REDUCE），默认不做正确性校验。
//
// 测量口径（对齐参考 measure_scene/chain_measure）：
//   每 rank 一条 stream；graph 内 nodes 个同构节点（每节点 = mega + REDUCE）；
//   预热 2 次；replay 2 次（可调），每次独立 event 夹住，per-kernel = replay 时间
//   / nodes，取 min-of-replays；**跨 rank 取 min**。
//
// 缺省口径对齐 perf.md 基线：R=32、TPR={1,4,16,32,33,64,128,256}、nodes=20、
// replays=2、max_tokens=512（win_cap 同 run_scene 规则抬到 512）、均衡路由。
// ============================================================================
static int run_aligned(RunOpts o) {
    int ndev = 0;
    CHECK(mcGetDeviceCount(&ndev));
    const int R = o.scene.ranks;
    if (ndev < R) {
        std::fprintf(stderr, "need %d visible devices, got %d\n", R, ndev);
        return 1;
    }

    // ---- 构造测例（tokens + 每 rank 路由）----
    // 缺省：均衡路由 × TPR 列表；--tk-info：从日志复现一次真实（不均衡）launch。
    const bool tk_mode = !o.tk_info.empty();
    std::vector<TkCase> cases;
    if (o.tk_all) {
        std::string e;
        if (!load_tk_all(o.tk_info, R, kTopk, kEPR, &cases, &e)) {
            std::fprintf(stderr, "[bench] %s\n", e.c_str());
            return 2;
        }
        if (o.tk_first > 0 && (int)cases.size() > o.tk_first)
            cases.resize(o.tk_first);   // 只跑前 N 个（map 已按 launch 升序）
    } else if (tk_mode) {
        TkCase c;
        std::string e;
        if (!load_tk_case(o.tk_info, o.tk_launch, R, kTopk, kEPR, &c, &e)) {
            std::fprintf(stderr, "[bench] %s\n", e.c_str());
            return 2;
        }
        cases.push_back(c);
    } else {
        const std::vector<int> tprs =
            o.tpr_list.empty() ? aligned_tpr_sweep() : o.tpr_list;
        for (int t : tprs) {
            TkCase c;
            c.tokens = t;
            c.max_expert = t * kTopk / kEPR;  // 均衡：每专家恰好 t*topk/EPR 对
            c.routes.resize(R);
            for (int rank = 0; rank < R; ++rank)
                c.routes[rank] = make_routes(RoutePattern::Balanced, rank, R, t);
            cases.push_back(c);
        }
    }
    int tmax = 0, cap_need = 0;
    for (const TkCase& c : cases) {
        tmax = std::max(tmax, c.tokens);
        cap_need = std::max(cap_need, c.max_expert);
    }
    const int max_tokens =
        o.max_tokens_set
            ? std::max(std::max(o.scene.max_tokens, tmax), cap_need)
            : std::max(std::max(512, tmax), cap_need);
    // win_cap 策略（不均衡路由专项）：
    //   背景：单专家收行数远大于其余（TK_INFO 真实路由，如一个专家 512 行、
    //   其余 16~100），窗口数 = ceil(count/win_cap)，win_cap 大时大专家只有
    //   1 个任务 → 整个长尾串行落在一个 block（4700µs）。把 win_cap 压到
    //   ≈cap_need/kRing，让大专家切成 ~kRing 个窗口 → 多任务摊到多 block。
    //   边界：win_cap 过小（>>kRing 个窗口）会令池槽 ring 复用 + 任务数暴涨，
    //   反而变慢（实测 64 最优、32 变差）；均衡分布（各专家行数相近）下压
    //   win_cap 反而把短任务切碎、回归显著（TPR=256 1914→2946µs），故只在
    //   检测到「长尾不均衡」（max_expert 远大于非零中位数）时收缩。
    //   显式 --win-cap 时始终按值取（下限 kM）。
    // 逐测例判「长尾不均衡」：该例 max_expert 是否 >= 4×该例非零中位数。
    // 逐例判断（不能跨例合并中位数：均衡多档列表 16/64/256 合并后会误判）。
    // [P50-D] 逐例自适应 win_cap：原实现把「是否存在长尾」与「用多大窗」都
    //   按整个测例集合聚合（imb_med 取全体长尾例中位数最大者），单例集合下每例
    //   各自最优，聚合后却被一个「中等专家」离群例（如 806600 med=142）抬高到 64，
    //   把其余 49 个「小中位数」例一并拖慢。改为逐例判定 + 逐例取窗：
    //     - 长尾例：中位小用 48（多切大专家）；中位大用 ~max_expert/kRing（减少任务）
    //     - 均衡例：max(128,max_tokens)（与单例行为一致）
    //   分配/布局仍用全集合最大值（各例 win_cap ≤ 全局，缓冲与 scratch 按全局分配
    //   即可覆盖；各区域寻址均走 make_mega_args 的逐例 win_cap/pool_stride）。
    bool imbalanced = false;
    for (TkCase& c : cases) {
        std::map<int, int> cnt;
        for (const auto& r : c.routes)
            for (int64_t e : r)
                if (e >= 0 && e < (int64_t)R * kEPR) ++cnt[(int)e];
        int med = 1;
        if (!cnt.empty()) {
            std::vector<int> nz;
            nz.reserve(cnt.size());
            for (const auto& kv : cnt) nz.push_back(kv.second);
            std::sort(nz.begin(), nz.end());
            med = nz[nz.size() / 2];
        }
        c.imbalanced = (c.max_expert >= 3 * std::max(1, med));
        if (c.imbalanced) imbalanced = true;
        int wc;
        if (o.scene.win_cap > 0) {
            wc = o.scene.win_cap;
        } else if (c.imbalanced) {
            int tgt = (med >= 32)
                          ? (c.max_expert + kRing - 1) / kRing
                          : std::min((int)48, c.max_expert);
            tgt = ((tgt + kM - 1) / kM) * kM;   // 对齐 M tile
            wc = std::min(max_tokens, std::max((int)kM, tgt));
        } else {
            wc = std::max(128, max_tokens);
        }
        c.win_cap = std::max((int)kM, wc);
    }
    int win_cap = kM;   // 全局分配窗口容量 = max(逐例 win_cap)
    for (const TkCase& c : cases) win_cap = std::max(win_cap, c.win_cap);
    if (o.scene.win_cap > 0) win_cap = o.scene.win_cap;
    const int nodes = o.nodes_set ? o.scene.nodes : 20;
    const int replays = o.replays_set ? o.measure_replays : 2;
    o.scene.tokens = tmax;
    o.scene.max_tokens = max_tokens;
    o.scene.win_cap = win_cap;

    ProdLayout L = compute_layout(R, max_tokens, kTopk, kHidden, kIntermediate,
                                  win_cap + o.scene.win_pad);
    L.win_capacity = win_cap;
    const int block_n = prod_block_n(o.scene.block_n);
    const int threads = block_n == 512 ? 1024 : 512;
    const int64_t smem = prod_smem(R, o.scene.window, block_n);
    if (smem > 227 * 1024) {
        std::fprintf(stderr, "win_cap=%d 需要 smem=%lldB > 227KB\n", win_cap,
                     (long long)smem);
        return 2;
    }

    // combine 区偏移（与 run_scene 同式）
    const int64_t barrier_pad_off = ((L.sym_bytes + 255) / 256) * 256;
    const int64_t recv_pad_off =
        ((barrier_pad_off + (int64_t)R * 64 * 4 + 255) / 256) * 256;
    const int64_t combine_f32_off =
        ((recv_pad_off + (int64_t)R * 64 * 4 + 255) / 256) * 256;
    // combine_rows（BF16）末端；combine_sums 已迁入 ranks_flags，不再占 sym。
    const int64_t combine_rows_end =
        ((combine_f32_off + (int64_t)kTopk * max_tokens * kHidden * 2 + 255) / 256) *
        256;
    const int64_t sym_alloc = combine_rows_end;

    std::printf(
        "设备数=%d  topk=%d  路由=%s  graph: nodes=%d replays=%d（取最小）\n",
        ndev, kTopk, o.tk_all ? "TK_ALL" : (tk_mode ? "不均衡(TK_INFO)" : "均衡"),
        nodes, replays);
    if (tk_mode && !o.tk_all) {
        const TkCase& c = cases[0];
        std::printf(
            "[TK_INFO] launch=%d tokens=%d 单专家最大收行=%d 非零专家=%d/%d\n",
            c.launch, c.tokens, c.max_expert, c.nz_expert, R * kEPR);
    }
    if (o.tk_all)
        std::printf("%-9s %-5s %-6s %-7s %-6s %12s %s\n", "launch", "ranks",
                    "tokens", "maxexp", "nz", "us/kernel", "cal");
    else
        std::printf("%-4s %-6s %12s %14s\n", "R", "TPR", "us/kernel", "口径");

    std::vector<RankRes> res(R);
    g_rb_n = R;

    // 权重 host 生成一次（lane 序 packed tile）；设备侧按同式生成
    std::vector<uint8_t> w1_host, w2_host;
    fill_w_lane(w1_host, kEPR * 2 * kIntermediate, kHidden, 17u);
    fill_w_lane(w2_host, kEPR * kHidden, kIntermediate, 91u);

    // ---- 分配（镜像 run_scene）----
    {
        std::atomic<int> ready{0};
        std::vector<std::thread> pool;
        for (int rank = 0; rank < R; ++rank)
            pool.emplace_back([&, rank]() {
                RankRes& r = res[rank];
                r.rank = rank;
                r.dev = rank;
                CHECK(mcSetDevice(rank));
                void* p = nullptr;
                const mcError_t me = mcExtMallocWithFlags(
                    &p, (size_t)sym_alloc, mcDeviceMallocFinegrained);
                if (me != mcSuccess || p == nullptr)
                    CHECK(mcMalloc(&p, (size_t)sym_alloc));
                r.sym = static_cast<uint8_t*>(p);
                CHECK(mcMemset(r.sym, 0, (size_t)sym_alloc));
                // local_flags：本 rank 私有，普通显存
                CHECK(mcMalloc((void**)&r.lf, sizeof(local_flags)));
                CHECK(mcMemset(r.lf, 0, sizeof(local_flags)));
                // ranks_flags：跨 rank，FG 显存（与 sym 同款；普通显存跨 rank 写丢写）
                {
                    void* rfp = nullptr;
                    const mcError_t rme = mcExtMallocWithFlags(
                        &rfp, sizeof(ranks_flags), mcDeviceMallocFinegrained);
                    if (rme != mcSuccess || rfp == nullptr) {
                        std::fprintf(stderr,
                                     "rank%d ranks_flags FG 分配失败(%d)，回退普通显存\n",
                                     rank, (int)rme);
                        CHECK(mcMalloc(&rfp, sizeof(ranks_flags)));
                    }
                    r.rf = static_cast<ranks_flags*>(rfp);
                    CHECK(mcMemset(r.rf, 0, sizeof(ranks_flags)));
                }
                // out_total/out_processed/dbg/ck_dbg/pre_ticks/l1_chk/l2_chk
                // 已迁入 local_flags，由上面的 lf 清零覆盖。
                CHECK(mcMalloc((void**)&r.w1, w1_host.size()));
                CHECK(mcMalloc((void**)&r.w2, w2_host.size()));
                fp8mega::fill_weights_lane(r.w1, kEPR * 2 * kIntermediate,
                                           kHidden, 17u, 0);
                fp8mega::fill_weights_lane(r.w2, kEPR * kHidden,
                                           kIntermediate, 91u, 0);
                CHECK(mcMalloc((void**)&r.ffn_hidden,
                               (size_t)kEPR * kRing * win_cap * kIntermediate));
                CHECK(mcMemset(r.ffn_hidden, 0,
                               (size_t)kEPR * kRing * win_cap * kIntermediate));
                 CHECK(mcMalloc((void**)&r.l2_out,
                                (size_t)kGrid * win_cap * kHidden * 2));
                 CHECK(mcMemset(r.l2_out, 0,
                                (size_t)kGrid * win_cap * kHidden * 2));
                 // [A-RWK] 实验：L1 A pool 独立普通显存（env 开关，默认关）
                 if (std::getenv("FP8MEGA_POOL_RWK") != nullptr) {
                     const size_t pool_bytes = (size_t)kEPR * kRing *
                         (size_t)(win_cap + o.scene.win_pad) * kHidden;
                     CHECK(mcMalloc((void**)&r.pool_rwk, pool_bytes));
                     CHECK(mcMemset(r.pool_rwk, 0, pool_bytes));
                 }
                 CHECK(mcMalloc((void**)&r.y,
                                (size_t)max_tokens * kHidden * sizeof(float)));
                CHECK(mcMemset(r.y, 0,
                               (size_t)max_tokens * kHidden * sizeof(float)));
                // [OPT-AFRAG] 多窗口 A/f2 fragment 专用缓冲（每 (expert,窗槽) 一份）；
                //   仅不均衡长尾场景启用（均衡场景走精简内核，避免 collateral 回退）
                r.frag_a = nullptr;
                r.frag2 = nullptr;
                if (imbalanced) {
                    const int mt_total = (win_cap + 15) >> 4;
                    const size_t asz =
                        (size_t)(kHidden / 32) * mt_total * 512;
                    const size_t f2sz =
                        (size_t)(kIntermediate / 32) * mt_total * 512;
                    if (mt_total > 0) {
                        CHECK(mcMalloc((void**)&r.frag_a,
                                       (size_t)kEPR * kRing * asz));
                        CHECK(mcMalloc((void**)&r.frag2,
                                       (size_t)kEPR * kRing * f2sz));
                    }
                }
                // x 视图 + topk_weights 视图（确定性填充）
                {
                    std::vector<uint8_t> xb((size_t)max_tokens * kHidden);
                    for (int t = 0; t < max_tokens; ++t)
                        for (int c = 0; c < kHidden; ++c)
                            xb[(size_t)t * kHidden + c] =
                                (uint8_t)((rank * 37 + t * 7 + c * 3) & 0x3f);
                    CHECK(mcMemcpy(r.sym + L.x_off, xb.data(), xb.size(),
                                   mcMemcpyHostToDevice));
                    std::vector<float> wv((size_t)max_tokens * kTopk);
                    for (int i = 0; i < max_tokens * kTopk; ++i)
                        wv[i] = 1.0f + rank * 0.25f + (i % 7) * 0.01f;
                    CHECK(mcMemcpy(r.sym + L.topk_weights_off, wv.data(),
                                   wv.size() * sizeof(float),
                                   mcMemcpyHostToDevice));
                }
                r.peer_sym.assign(R, nullptr);
                CHECK(mcIpcGetMemHandle(&r.handle, r.sym));
                r.peer_rf.assign(R, nullptr);
                CHECK(mcIpcGetMemHandle(&r.rf_handle, r.rf));
                ready.fetch_add(1);
            });
        while (ready.load() < R) {}
        for (auto& t : pool) t.join();
    }

    // ---- IPC open + peer access ----
    {
        std::atomic<int> ready{0};
        std::vector<std::thread> pool;
        for (int rank = 0; rank < R; ++rank)
            pool.emplace_back([&, rank]() {
                RankRes& r = res[rank];
                CHECK(mcSetDevice(rank));
                for (int d = 0; d < R; ++d) {
                    if (d == rank) continue;
                    void* q = nullptr;
                    const mcError_t e = mcIpcOpenMemHandle(
                        &q, res[d].handle, mcIpcMemLazyEnablePeerAccess);
                    if (e != mcSuccess) {
                        std::fprintf(stderr, "rank%d open peer%d failed: %d\n",
                                     rank, d, (int)e);
                        std::exit(1);
                    }
                    r.peer_sym[d] = static_cast<uint8_t*>(q);
                    void* qr = nullptr;
                    const mcError_t er = mcIpcOpenMemHandle(
                        &qr, res[d].rf_handle, mcIpcMemLazyEnablePeerAccess);
                    if (er != mcSuccess) {
                        std::fprintf(stderr, "rank%d open peer%d rf failed: %d\n",
                                     rank, d, (int)er);
                        std::exit(1);
                    }
                    r.peer_rf[d] = static_cast<ranks_flags*>(qr);
                }
                for (int d = 0; d < R; ++d) {
                    if (d == rank) continue;
                    int can = 0;
                    if (mcDeviceCanAccessPeer(&can, rank, d) == mcSuccess && can)
                        mcDeviceEnablePeerAccess(d, 0);
                }
                std::vector<uint8_t*> base_ptrs(R);
                std::vector<ranks_flags*> rf_ptrs(R);
                for (int d = 0; d < R; ++d) {
                    base_ptrs[d] = (d == rank) ? r.sym : r.peer_sym[d];
                    rf_ptrs[d] = (d == rank) ? r.rf : r.peer_rf[d];
                }
                CHECK(mcMalloc((void**)&r.d_base_ptrs, R * sizeof(uint8_t*)));
                CHECK(mcMemcpy(r.d_base_ptrs, base_ptrs.data(),
                               R * sizeof(uint8_t*), mcMemcpyHostToDevice));
                CHECK(mcMalloc((void**)&r.d_rf_ptrs, R * sizeof(ranks_flags*)));
                CHECK(mcMemcpy(r.d_rf_ptrs, rf_ptrs.data(),
                               R * sizeof(ranks_flags*),
                               mcMemcpyHostToDevice));
                ready.fetch_add(1);
            });
        while (ready.load() < R) {}
        for (auto& t : pool) t.join();
    }

    // ---- 逐测例：graph 链全链计时 ----
    for (const TkCase& c : cases) {
        const int T = c.tokens;
        o.scene.tokens = T;
        const std::vector<std::vector<int64_t>>& routes = c.routes;

        // 清零 + 上传路由（一次；每档测量前后均已同步，无在途写）
        {
            std::vector<std::thread> th(R);
            for (int rank = 0; rank < R; ++rank)
                th[rank] = std::thread([&, rank]() {
                    RankRes& r = res[rank];
                    CHECK(mcSetDevice(rank));
                    CHECK(mcMemcpy(r.sym + L.topk_idx_off, routes[rank].data(),
                                   routes[rank].size() * sizeof(int64_t),
                                   mcMemcpyHostToDevice));
                    CHECK(mcMemset(r.sym + L.src_index_table_off, 0,
                                   (size_t)L.src_index_table_bytes));
                    CHECK(mcMemset(r.sym + L.barrier_slots_off, 0,
                                   (size_t)R * R * 4));
                    CHECK(mcMemset(r.lf, 0, sizeof(local_flags)));
                    CHECK(mcMemset(r.rf, 0, sizeof(ranks_flags)));
                    rank_barrier();
                });
            for (auto& t : th) t.join();
        }

        std::vector<double> per(R, -1.0);
        std::vector<std::string> cal(R, "-");
        std::vector<int> caps(R, 0);
        {
            std::atomic<int> go{0};
            std::vector<std::thread> th(R);
            for (int rank = 0; rank < R; ++rank)
                th[rank] = std::thread([&, rank]() {
                    RankRes& r = res[rank];
                    CHECK(mcSetDevice(rank));
                    mcStream_t st = nullptr;
                    CHECK(mcStreamCreate(&st));
                    // [P50-D] 逐例 win_cap（≥kM，且 ≤ 全局分配窗口容量）
                    const int wc_case = std::min(win_cap, c.win_cap);
                    fp8mega::MegaLaunchArgs a = make_mega_args(
                        o, L, r, wc_case, combine_f32_off);
                    // [BUGFIX 261007] 不粗化(merge=1)会跨波执行，多波下
                    //   wait-all/分片旗存在陈旧读 → 结果错误（harness case 1
                    //   rel_max_diff=0.229）。暂回退缺省单波粗化以恢复正确性。
                    // if (c.imbalanced) a.merge_grid = 0x3fffffff;
                    while (go.load(std::memory_order_acquire) == 0) {}
                    bool cap = false;
                    // [R38] 跨 graph 节点前序复用：同 graph 内路由不变，仅首节点
                    //   跑 ROUTE/PUBLISH/XBAR，其余节点跳过（flag 由 TASK_SETUP 重臂）。
                    //   MX_NO_REUSE=1 可关闭做 A/B。
                    const bool reuse_prologue =
                        std::getenv("MX_NO_REUSE") == nullptr;
                    int reuse_mode = 2;
                    if (const char* rm = std::getenv("MX_REUSE_MODE"))
                        reuse_mode = std::atoi(rm);
                    cal[rank] = chain_measure(
                        st, nodes, replays,
                        [&](mcStream_t s, int k) {
                            fp8mega::MegaLaunchArgs b = a;
                            b.skip_prologue =
                                (reuse_prologue && k != 0) ? reuse_mode : 0;
                            fp8mega::launch_full(b, s);
                        },
                        &per[rank], &cap);
                    caps[rank] = cap ? 1 : 0;
                    mcStreamDestroy(st);
                });
            go.store(1, std::memory_order_release);
            for (auto& t : th) t.join();
        }
        double best = 1e30;
        for (int rank = 0; rank < R; ++rank)
            if (per[rank] > 0.0) best = std::min(best, per[rank]);
        if (best >= 1e30) best = -1.0;
        const bool all_cap = std::count(caps.begin(), caps.end(), 1) == R;
        std::string note = all_cap ? cal[0] : (cal[0] + "(部分降级)");
        if (o.tk_all) {
            std::printf("%-9d %-5d %-6d %-7d %-6d %12.2f %s\n", c.launch,
                        c.ranks, c.tokens, c.max_expert, c.nz_expert, best,
                        note.c_str());
        } else {
            if (tk_mode) note += " max_expert=" + std::to_string(c.max_expert);
            std::printf("%-4d %-6d %12.2f %14s\n", R, T, best, note.c_str());
        }
        std::fflush(stdout);
        if (std::getenv("PHASE_DUMP") != nullptr) {
            const double clk = 2000.0;
            for (int rank = 0; rank < R; ++rank) {
                CHECK(mcSetDevice(res[rank].dev));
                unsigned long long tp[16] = {0};
                CHECK(mcMemcpy(tp, res[rank].lf->pre_ticks,
                               16 * sizeof(unsigned long long),
                               mcMemcpyDeviceToHost));
                if (tp[7] > tp[3]) {
                    int proc[kGrid] = {0};
                    CHECK(mcMemcpy(proc, res[rank].lf->out_processed,
                                   kGrid * sizeof(int), mcMemcpyDeviceToHost));
                    int sm = 0, mx = 0;
                    for (int b = 0; b < kGrid; ++b) {
                        sm += proc[b];
                        mx = std::max(mx, proc[b]);
                    }
                    float sm1 = 0.0f, sm2 = 0.0f;
                    {
                        std::vector<float> c1(kGrid, 0.0f), c2(kGrid, 0.0f);
                        CHECK(mcMemcpy(c1.data(), res[rank].lf->l1_chk,
                                       kGrid * sizeof(float),
                                       mcMemcpyDeviceToHost));
                        CHECK(mcMemcpy(c2.data(), res[rank].lf->l2_chk,
                                       kGrid * sizeof(float),
                                       mcMemcpyDeviceToHost));
                        for (int b = 0; b < kGrid; ++b) {
                            sm1 += c1[b];
                            sm2 += c2[b];
                        }
                    }
                    std::vector<unsigned long long> tr((size_t)kGrid * 8, 0);
                    CHECK(mcMemcpy(tr.data(),
                                   res[rank].lf->pre_ticks + (size_t)kGrid * 8,
                                   tr.size() * sizeof(unsigned long long),
                                   mcMemcpyDeviceToHost));
                    unsigned long long maxl1 = 0, maxl2 = 0;
                    for (int b = 0; b < kGrid; ++b) {
                        maxl1 = std::max(maxl1, tr[(size_t)b * 8 + 0]);
                        maxl2 = std::max(maxl2, tr[(size_t)b * 8 + 1]);
                    }
                    std::printf(
                        "[phase] rank=%d us=%.0f mega=%.1f reduce=%.1f "
                        "dispatch=%.1f L1=%.1f "
                        "L2tail=%.1f chain=%.1f tasks=%d maxblk=%d "
                        "maxL1=%.1f maxL2=%.1f ck1=%.4e ck2=%.4e\n",
                        rank, per[rank],
                        (double)(tp[15] - tp[6]) / clk,
                        per[rank] - (double)(tp[15] - tp[6]) / clk,
                        (double)(tp[4] - tp[3]) / clk,
                        (double)(tp[5] - tp[4]) / clk,
                        (double)(tp[7] - tp[5]) / clk,
                        (double)(tp[7] - tp[3]) / clk, sm, mx,
                        (double)maxl1 / 40.0 / clk,
                        (double)maxl2 / 40.0 / clk, (double)sm1, (double)sm2);
                }
            }
            std::fflush(stdout);
        }
    }
    return 0;
}

// ---------------------------------------------------------------- main --
// ======================================================== MPI 模式 --
// 每进程一张卡：rank = MPI rank，R = MPI size。用 MPI_Allgather 交换 IPC handle、
// MPI_Barrier 做栅栏；每进程只建本 rank 的资源。与 run_aligned 同口径（均衡/tk）。
static int run_aligned_mpi(RunOpts o) {
    const int R = g_mpi_size;
    const int rank = g_mpi_rank;
    o.scene.ranks = R;
    CHECK(mcSetDevice(rank));

    const bool tk_mode = !o.tk_info.empty();
    std::vector<TkCase> cases;
    if (o.tk_all) {
        std::string e;
        if (!load_tk_all(o.tk_info, R, kTopk, kEPR, &cases, &e)) {
            std::fprintf(stderr, "[bench] %s\n", e.c_str());
            return 2;
        }
        if (o.tk_first > 0 && (int)cases.size() > o.tk_first) cases.resize(o.tk_first);
    } else if (tk_mode) {
        TkCase c;
        std::string e;
        if (!load_tk_case(o.tk_info, o.tk_launch, R, kTopk, kEPR, &c, &e)) {
            std::fprintf(stderr, "[bench] %s\n", e.c_str());
            return 2;
        }
        cases.push_back(c);
    } else {
        const std::vector<int> tprs =
            o.tpr_list.empty() ? aligned_tpr_sweep() : o.tpr_list;
        for (int t : tprs) {
            TkCase c;
            c.tokens = t;
            c.max_expert = t * kTopk / kEPR;
            c.routes.resize(R);
            for (int rr = 0; rr < R; ++rr)
                c.routes[rr] = make_routes(RoutePattern::Balanced, rr, R, t);
            cases.push_back(c);
        }
    }
    int tmax = 0, cap_need = 0;
    for (const TkCase& c : cases) {
        tmax = std::max(tmax, c.tokens);
        cap_need = std::max(cap_need, c.max_expert);
    }
    const int max_tokens =
        o.max_tokens_set
            ? std::max(std::max(o.scene.max_tokens, tmax), cap_need)
            : std::max(std::max(512, tmax), cap_need);
    bool imbalanced = false;
    for (TkCase& c : cases) {
        std::map<int, int> cnt;
        for (const auto& rr : c.routes)
            for (int64_t e : rr)
                if (e >= 0 && e < (int64_t)R * kEPR) ++cnt[(int)e];
        int med = 1;
        if (!cnt.empty()) {
            std::vector<int> nz;
            nz.reserve(cnt.size());
            for (const auto& kv : cnt) nz.push_back(kv.second);
            std::sort(nz.begin(), nz.end());
            med = nz[nz.size() / 2];
        }
        c.imbalanced = (c.max_expert >= 3 * std::max(1, med));
        if (c.imbalanced) imbalanced = true;
        int wc;
        if (o.scene.win_cap > 0) {
            wc = o.scene.win_cap;
        } else if (c.imbalanced) {
            int tgt = (med >= 32) ? (c.max_expert + kRing - 1) / kRing
                                  : std::min((int)48, c.max_expert);
            tgt = ((tgt + kM - 1) / kM) * kM;
            wc = std::min(max_tokens, std::max((int)kM, tgt));
        } else {
            wc = std::max(128, max_tokens);
        }
        c.win_cap = std::max((int)kM, wc);
    }
    int win_cap = kM;
    for (const TkCase& c : cases) win_cap = std::max(win_cap, c.win_cap);
    if (o.scene.win_cap > 0) win_cap = o.scene.win_cap;
    const int nodes = o.nodes_set ? o.scene.nodes : 20;
    const int replays = o.replays_set ? o.measure_replays : 2;
    o.scene.tokens = tmax;
    o.scene.max_tokens = max_tokens;
    o.scene.win_cap = win_cap;

    ProdLayout L = compute_layout(R, max_tokens, kTopk, kHidden, kIntermediate,
                                  win_cap + o.scene.win_pad);
    L.win_capacity = win_cap;
    const int64_t barrier_pad_off = ((L.sym_bytes + 255) / 256) * 256;
    const int64_t recv_pad_off =
        ((barrier_pad_off + (int64_t)R * 64 * 4 + 255) / 256) * 256;
    const int64_t combine_f32_off =
        ((recv_pad_off + (int64_t)R * 64 * 4 + 255) / 256) * 256;
    const int64_t combine_rows_end =
        ((combine_f32_off + (int64_t)kTopk * max_tokens * kHidden * 2 + 255) /
         256) * 256;
    const int64_t sym_alloc = combine_rows_end;

    if (rank == 0) {
        std::printf("设备数=%d(MPI) topk=%d 路由=%s graph: nodes=%d replays=%d\n",
                    R, kTopk,
                    o.tk_all ? "TK_ALL" : (tk_mode ? "不均衡(TK_INFO)" : "均衡"),
                    nodes, replays);
        if (tk_mode && !o.tk_all) {
            const TkCase& c = cases[0];
            std::printf("[TK_INFO] launch=%d tokens=%d 单专家最大收行=%d 非零专家=%d/%d\n",
                        c.launch, c.tokens, c.max_expert, c.nz_expert, R * kEPR);
        }
        if (o.tk_all)
            std::printf("%-9s %-5s %-6s %-7s %-6s %10s %10s %10s %10s\n", "launch",
                        "ranks", "tokens", "maxexp", "nz", "us_sel", "us_min",
                        "us_max", "us_mean");
        else
            std::printf("%-4s %-6s %10s %10s %10s %14s\n", "R", "TPR", "us_sel",
                        "us_min", "us_max", "口径");
    }

    // ---- 本 rank 资源 ----
    RankRes r;
    r.rank = rank;
    r.dev = rank;
    void* p = nullptr;
    if (mcExtMallocWithFlags(&p, (size_t)sym_alloc, mcDeviceMallocFinegrained) !=
            mcSuccess || p == nullptr)
        CHECK(mcMalloc(&p, (size_t)sym_alloc));
    r.sym = static_cast<uint8_t*>(p);
    CHECK(mcMemset(r.sym, 0, (size_t)sym_alloc));
    CHECK(mcMalloc((void**)&r.lf, sizeof(local_flags)));
    CHECK(mcMemset(r.lf, 0, sizeof(local_flags)));
    {
        void* rfp = nullptr;
        if (mcExtMallocWithFlags(&rfp, sizeof(ranks_flags), mcDeviceMallocFinegrained) !=
                mcSuccess || rfp == nullptr)
            CHECK(mcMalloc(&rfp, sizeof(ranks_flags)));
        r.rf = static_cast<ranks_flags*>(rfp);
        CHECK(mcMemset(r.rf, 0, sizeof(ranks_flags)));
    }
    CHECK(mcMalloc((void**)&r.w1, (size_t)kEPR * 2 * kIntermediate * kHidden));
    CHECK(mcMalloc((void**)&r.w2, (size_t)kEPR * kHidden * kIntermediate));
    fp8mega::fill_weights_lane(r.w1, kEPR * 2 * kIntermediate, kHidden, 17u, 0);
    fp8mega::fill_weights_lane(r.w2, kEPR * kHidden, kIntermediate, 91u, 0);
    CHECK(mcMalloc((void**)&r.ffn_hidden,
                   (size_t)kEPR * kRing * win_cap * kIntermediate));
    CHECK(mcMemset(r.ffn_hidden, 0, (size_t)kEPR * kRing * win_cap * kIntermediate));
    CHECK(mcMalloc((void**)&r.l2_out, (size_t)kGrid * win_cap * kHidden * 2));
    CHECK(mcMemset(r.l2_out, 0, (size_t)kGrid * win_cap * kHidden * 2));
    CHECK(mcMalloc((void**)&r.y, (size_t)max_tokens * kHidden * sizeof(float)));
    CHECK(mcMemset(r.y, 0, (size_t)max_tokens * kHidden * sizeof(float)));
    r.frag_a = nullptr;
    r.frag2 = nullptr;
    if (imbalanced) {
        const int mt_total = (win_cap + 15) >> 4;
        const size_t asz = (size_t)(kHidden / 32) * mt_total * 512;
        const size_t f2sz = (size_t)(kIntermediate / 32) * mt_total * 512;
        if (mt_total > 0) {
            CHECK(mcMalloc((void**)&r.frag_a, (size_t)kEPR * kRing * asz));
            CHECK(mcMalloc((void**)&r.frag2, (size_t)kEPR * kRing * f2sz));
        }
    }
    {
        std::vector<uint8_t> xb((size_t)max_tokens * kHidden);
        for (int t = 0; t < max_tokens; ++t)
            for (int cc = 0; cc < kHidden; ++cc)
                xb[(size_t)t * kHidden + cc] =
                    (uint8_t)((rank * 37 + t * 7 + cc * 3) & 0x3f);
        CHECK(mcMemcpy(r.sym + L.x_off, xb.data(), xb.size(), mcMemcpyHostToDevice));
        std::vector<float> wv((size_t)max_tokens * kTopk);
        for (int i = 0; i < max_tokens * kTopk; ++i)
            wv[i] = 1.0f + rank * 0.25f + (i % 7) * 0.01f;
        CHECK(mcMemcpy(r.sym + L.topk_weights_off, wv.data(),
                       wv.size() * sizeof(float), mcMemcpyHostToDevice));
    }
    r.peer_sym.assign(R, nullptr);
    r.peer_rf.assign(R, nullptr);
    CHECK(mcIpcGetMemHandle(&r.handle, r.sym));
    CHECK(mcIpcGetMemHandle(&r.rf_handle, r.rf));

    // ---- MPI 交换 IPC handle ----
    std::vector<mcIpcMemHandle_t> all_h(R), all_rfh(R);
    MPI_Allgather(&r.handle, (int)sizeof(mcIpcMemHandle_t), MPI_BYTE, all_h.data(),
                  (int)sizeof(mcIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Allgather(&r.rf_handle, (int)sizeof(mcIpcMemHandle_t), MPI_BYTE,
                  all_rfh.data(), (int)sizeof(mcIpcMemHandle_t), MPI_BYTE,
                  MPI_COMM_WORLD);
    for (int d = 0; d < R; ++d) {
        if (d == rank) continue;
        void* q = nullptr;
        if (mcIpcOpenMemHandle(&q, all_h[d], mcIpcMemLazyEnablePeerAccess) !=
            mcSuccess) {
            std::fprintf(stderr, "rank%d open peer%d sym failed\n", rank, d);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        r.peer_sym[d] = static_cast<uint8_t*>(q);
        void* qr = nullptr;
        if (mcIpcOpenMemHandle(&qr, all_rfh[d], mcIpcMemLazyEnablePeerAccess) !=
            mcSuccess) {
            std::fprintf(stderr, "rank%d open peer%d rf failed\n", rank, d);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        r.peer_rf[d] = static_cast<ranks_flags*>(qr);
    }
    for (int d = 0; d < R; ++d) {
        if (d == rank) continue;
        int can = 0;
        if (mcDeviceCanAccessPeer(&can, rank, d) == mcSuccess && can)
            mcDeviceEnablePeerAccess(d, 0);
    }
    {
        std::vector<uint8_t*> bp(R);
        std::vector<ranks_flags*> rp(R);
        for (int d = 0; d < R; ++d) {
            bp[d] = (d == rank) ? r.sym : r.peer_sym[d];
            rp[d] = (d == rank) ? r.rf : r.peer_rf[d];
        }
        CHECK(mcMalloc((void**)&r.d_base_ptrs, R * sizeof(uint8_t*)));
        CHECK(mcMemcpy(r.d_base_ptrs, bp.data(), R * sizeof(uint8_t*),
                       mcMemcpyHostToDevice));
        CHECK(mcMalloc((void**)&r.d_rf_ptrs, R * sizeof(ranks_flags*)));
        CHECK(mcMemcpy(r.d_rf_ptrs, rp.data(), R * sizeof(ranks_flags*),
                       mcMemcpyHostToDevice));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ---- 逐测例 ----
    for (const TkCase& c : cases) {
        o.scene.tokens = c.tokens;
        CHECK(mcMemcpy(r.sym + L.topk_idx_off, c.routes[rank].data(),
                       c.routes[rank].size() * sizeof(int64_t),
                       mcMemcpyHostToDevice));
        CHECK(mcMemset(r.sym + L.src_index_table_off, 0,
                       (size_t)L.src_index_table_bytes));
        CHECK(mcMemset(r.sym + L.barrier_slots_off, 0, (size_t)R * R * 4));
        CHECK(mcMemset(r.lf, 0, sizeof(local_flags)));
        CHECK(mcMemset(r.rf, 0, sizeof(ranks_flags)));
        MPI_Barrier(MPI_COMM_WORLD);

        mcStream_t st = nullptr;
        CHECK(mcStreamCreate(&st));
        const int wc_case = std::min(win_cap, c.win_cap);
        fp8mega::MegaLaunchArgs a = make_mega_args(o, L, r, wc_case, combine_f32_off);
        const bool reuse_prologue = std::getenv("MX_NO_REUSE") == nullptr;
        int reuse_mode = 2;
        if (const char* rm = std::getenv("MX_REUSE_MODE")) reuse_mode = std::atoi(rm);
        bool cap = false;
        double per_r = -1.0;
        std::string cal = chain_measure(
            st, nodes, replays,
            [&](mcStream_t s, int k) {
                fp8mega::MegaLaunchArgs b = a;
                b.skip_prologue = (reuse_prologue && k != 0) ? reuse_mode : 0;
                fp8mega::launch_full(b, s);
            },
            &per_r, &cap);
        mcStreamDestroy(st);

        std::vector<double> allper(R, -1.0);
        MPI_Allgather(&per_r, 1, MPI_DOUBLE, allper.data(), 1, MPI_DOUBLE,
                      MPI_COMM_WORLD);
        double vmin = 1e30, vmax = -1.0, vsum = 0.0;
        int vcnt = 0;
        for (int d = 0; d < R; ++d)
            if (allper[d] > 0.0) {
                vmin = std::min(vmin, allper[d]);
                vmax = std::max(vmax, allper[d]);
                vsum += allper[d];
                ++vcnt;
            }
        double vmean = vcnt ? vsum / vcnt : -1.0;
        double best = vmin;   // 与 run_aligned 同：us_sel = 跨 rank 取 min
        if (rank == 0) {
            if (o.tk_all)
                std::printf("%-9d %-5d %-6d %-7d %-6d %10.2f %10.2f %10.2f %10.2f %s\n",
                            c.launch, c.ranks, c.tokens, c.max_expert, c.nz_expert,
                            best, vmin, vmax, vmean, cal.c_str());
            else
                std::printf("%-4d %-6d %10.2f %10.2f %10.2f %s\n", R, c.tokens, best,
                            vmin, vmax, cal.c_str());
        }
        std::fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return 0;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_mpi_size);
    RunOpts o;
    std::string err;
    if (!parse_args(argc, argv, &o, &err)) {
        std::fprintf(stderr, "[bench] %s\n", err.c_str());
        usage(argv[0]);
        MPI_Finalize();
        return 2;
    }
    int rc = 0;
    if (g_mpi_size > 1) {
        // MPI 模式：每进程一 rank/一卡（mpirun -np R）
        if (o.aligned) {
            rc = run_aligned_mpi(o);
        } else {
            if (g_mpi_rank == 0)
                std::fprintf(stderr, "[bench] MPI 模式仅支持 --aligned 1\n");
            rc = 2;
        }
    } else if (o.aligned) {
        rc = run_aligned(o);
    } else if (o.sweep) {
        rc = run_sweep(o);
    } else {
        // 单档（旧行为）：--aligned 0 且不带 --sweep 时与改造前完全一致
        SceneResult r;
        rc = run_scene(o, &r);
    }
    MPI_Finalize();
    return rc;
}
