// ============================================================================
// mybench.cu — 每进程一张卡的 MoE mega kernel 基准（MPI）
//
// 启动：mpirun -np R ./mybench --s 0 [--tprs 1,4,16,32,64,128] [--nodes 20]
//        [--replays 2] [--max-tokens 128] [--quiet]
//
// 模型：R 个 MPI 进程 = R 个 rank，各占一张卡（mcSetDevice(rank)）。
//   用 MPI_Allgather 交换 mcIpcMemHandle（80B），MPI_Barrier 做跨 rank 栅栏。
//
// 流程（--s 0 均衡模式）：
//   每 rank 资源初始化(sym/lf/rf/权重/staging) -> IPC 交换/开 peer ->
//   foreach tpr: data_init -> eager -> graph capture(nodes) -> warmup ->
//                measure(replays) -> MPI_Allgather -> 打印 min/max/mean
//
// 口径：us/kernel = 图计时(ms) * 1000 / (nodes * replays)。
// 依赖：链接 megakernel.so（fp8mega::launch_full 等）+ MACA ompi(-lmpi)。
// ============================================================================
#include <mcr/mc_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "common.cuh"
#include "flags_defs.cuh"
#include "host_layout.cuh"
#include "bench_check.cuh"
#include "megakernel.h"

#define MYCHK(call)                                                          \
    do {                                                                     \
        mcError_t e_ = (call);                                               \
        if (e_ != mcSuccess) {                                               \
            std::fprintf(stderr, "[mybench] MACA error %d at %s:%d: %s\n",   \
                         (int)e_, __FILE__, __LINE__, #call);                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
    } while (0)

static int g_rank = 0;   // MPI rank = 本进程负责的 rank
static int g_size = 1;   // MPI size = R

// bar flag 索引（与生产布局常量一致）
static constexpr int64_t kXreleaseIdx = deep_gemm::layout::kBarXreleaseIdx;
static constexpr int64_t kSliceFlagsIdx = deep_gemm::layout::kBarSliceFlagsIdx;
static constexpr int64_t kL1ProgIdx = deep_gemm::layout::kBarL1ProgressIdx;
static constexpr int64_t kL2DoneIdx = deep_gemm::layout::kBarL2DoneIdx;
static constexpr int64_t kTaskReadyIdx = deep_gemm::layout::kBarTaskReadyIdx;
static constexpr int64_t kTaskCounterIdx = deep_gemm::layout::kBarTaskCounterIdx;

struct MyOpts {
    int s = 0;                              // 0=均衡（首期）
    std::vector<int> tprs;                  // --tprs
    int nodes = 20;                         // --nodes
    int replays = 2;                        // --replays
    int max_tokens = 128;                   // --max-tokens（固定容量，全 TPR 复用）
    int block_n = 256;                      // --bn
    int window = 8;                         // 生产 smem 计算用窗口
    bool quiet = false;                     // --quiet
    std::string tk_info;                    // --tk-info（--s 1 预留）
    int tk_launch = 0;                      // --tk-launch（--s 1 预留）
};

struct RankRes1 {
    int rank = 0;
    int dev = 0;
    uint8_t* sym = nullptr;
    local_flags* lf = nullptr;
    ranks_flags* rf = nullptr;
    uint8_t* w1 = nullptr;
    uint8_t* w2 = nullptr;
    uint8_t* ffn_hidden = nullptr;
    uint16_t* l2_out = nullptr;
    float* y = nullptr;
    uint8_t** d_base_ptrs = nullptr;
    ranks_flags** d_rf_ptrs = nullptr;
    mcIpcMemHandle_t handle;
    mcIpcMemHandle_t rf_handle;
    std::vector<uint8_t*> peer_sym;
    std::vector<ranks_flags*> peer_rf;
};

static void usage(const char* p) {
    std::fprintf(
        stderr,
        "用法: %s [选项]（须用 mpirun -np R 启动，每进程一张卡）\n"
        "  --s 0|1            场景：0=均衡（默认），1=TK_INFO（预留）\n"
        "  --tprs L           逗号分隔 TPR 列表（默认 1,4,16,32,64,128）\n"
        "  --nodes N          每张图 kernel 节点数（默认 20）\n"
        "  --replays N        计时 replay 次数（默认 2）\n"
        "  --max-tokens M     槽容量（默认 128，全 TPR 复用）\n"
        "  --bn 512|256|128   N tile（默认 256）\n"
        "  --window N         生产 smem 计算用窗口（默认 8）\n"
        "  --tk-info F        --s 1：TK_INFO 日志\n"
        "  --tk-launch N      --s 1：指定 launch\n"
        "  --quiet            只打印结果行\n"
        "环境: MACA_VISIBLE_DEVICES（应含全部 R 张卡，各进程按 rank 选卡）\n",
        p);
}

static std::vector<int> parse_ints(const std::string& s) {
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) out.push_back(std::atoi(s.substr(i, j - i).c_str()));
        i = j + 1;
    }
    return out;
}

static bool parse_args(int argc, char** argv, MyOpts* o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto val = [&]() -> const char* {
            return (i + 1 < argc) ? argv[++i] : "";
        };
        if (a == "--s") o->s = std::atoi(val());
        else if (a == "--tprs") o->tprs = parse_ints(val());
        else if (a == "--nodes") o->nodes = std::atoi(val());
        else if (a == "--replays") o->replays = std::atoi(val());
        else if (a == "--max-tokens") o->max_tokens = std::atoi(val());
        else if (a == "--bn") o->block_n = std::atoi(val());
        else if (a == "--window") o->window = std::atoi(val());
        else if (a == "--tk-info") o->tk_info = val();
        else if (a == "--tk-launch") o->tk_launch = std::atoi(val());
        else if (a == "--quiet") o->quiet = true;
        else if (a == "-h" || a == "--help") { usage(argv[0]); return false; }
        else { std::fprintf(stderr, "[mybench] 未知参数: %s\n", a.c_str()); return false; }
    }
    if (o->tprs.empty()) o->tprs = {1, 4, 16, 32, 64, 128};
    if (o->nodes < 1) o->nodes = 20;
    if (o->replays < 1) o->replays = 2;
    return true;
}

static fp8mega::MegaLaunchArgs make_args(const MyOpts& o, const ProdLayout& L,
                                         const RankRes1& r, int R, int tokens,
                                         int win_cap, int64_t combine_rows_off) {
    fp8mega::MegaLaunchArgs a{};
    a.topk_idx = reinterpret_cast<const int64_t*>(r.sym + L.topk_idx_off);
    a.topk_idx_off = L.topk_idx_off;
    a.self = r.sym;
    a.peer_bases = r.d_base_ptrs;
    a.R = R;
    a.rank = r.rank;
    a.tokens = tokens;
    a.slot_capacity = o.max_tokens;
    a.win_cap = win_cap;
    a.pool_stride = win_cap;
    a.tbl_cap = o.max_tokens;
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
    a.w_off = L.token_route_weights_off;
    a.meta_off = L.token_src_meta_off;
    a.max_spins = (int)kSpinLimit;
    a.publish_epoch = kPublishEpoch;
    a.recv_stride = (int)L.recv_stride;
    a.w1 = r.w1;
    a.l1_n_total = kIntermediate * 2;
    a.l1_block_n = 256;
    a.l1_k_out = kIntermediate;
    a.ffn_hidden = r.ffn_hidden;
    a.w2 = r.w2;
    a.l2_out = r.l2_out;
    a.l2_k_out = kHidden;
    a.combine_rows_off = combine_rows_off;
    a.combine_cap = o.max_tokens;
    a.sums_stride = deep_gemm::layout::kMegaMoeMaxCombineSlices;
    a.y = r.y;
    a.grid = kGrid;
    a.merge_grid = kGrid;
    const int bn = prod_block_n(o.block_n);
    a.threads = bn == 512 ? 1024 : 512;
    a.smem = prod_smem(R, o.window, bn);
    return a;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);
    const int R = g_size;
    const int rank = g_rank;

    MyOpts o;
    if (!parse_args(argc, argv, &o)) {
        MPI_Finalize();
        return 2;
    }
    if (o.s != 0) {
        if (rank == 0) std::fprintf(stderr, "[mybench] --s %d 暂未实现（仅 0=均衡）\n", o.s);
        MPI_Finalize();
        return 2;
    }
    MYCHK(mcSetDevice(rank));

    const int max_tokens = o.max_tokens;
    const int win_cap = std::max(128, max_tokens);
    const int nodes = o.nodes;
    const int replays = o.replays;

    ProdLayout L = compute_layout(R, max_tokens, kTopk, kHidden, kIntermediate,
                                  win_cap);
    const int64_t barrier_pad_off = ((L.sym_bytes + 255) / 256) * 256;
    const int64_t recv_pad_off =
        ((barrier_pad_off + (int64_t)R * 64 * 4 + 255) / 256) * 256;
    const int64_t combine_f32_off =
        ((recv_pad_off + (int64_t)R * 64 * 4 + 255) / 256) * 256;
    const int64_t combine_rows_end =
        ((combine_f32_off + (int64_t)kTopk * max_tokens * kHidden * 2 + 255) / 256) *
        256;
    const int64_t sym_alloc = combine_rows_end;

    if (rank == 0 && !o.quiet) {
        std::printf("mybench(MPI) R=%d topk=%d s=%d nodes=%d replays=%d "
                    "max_tokens=%d win_cap=%d\n",
                    R, kTopk, o.s, nodes, replays, max_tokens, win_cap);
        std::printf("%-4s %-6s %10s %10s %10s\n", "R", "TPR", "us_min", "us_max",
                    "us_mean");
    }

    // ---- 每 rank 资源 ----
    RankRes1 r;
    r.rank = rank;
    r.dev = rank;
    void* p = nullptr;
    if (mcExtMallocWithFlags(&p, (size_t)sym_alloc, mcDeviceMallocFinegrained) !=
            mcSuccess || p == nullptr)
        MYCHK(mcMalloc(&p, (size_t)sym_alloc));
    r.sym = static_cast<uint8_t*>(p);
    MYCHK(mcMemset(r.sym, 0, (size_t)sym_alloc));
    MYCHK(mcMalloc((void**)&r.lf, sizeof(local_flags)));
    MYCHK(mcMemset(r.lf, 0, sizeof(local_flags)));
    {
        void* rfp = nullptr;
        if (mcExtMallocWithFlags(&rfp, sizeof(ranks_flags),
                                 mcDeviceMallocFinegrained) != mcSuccess ||
            rfp == nullptr)
            MYCHK(mcMalloc(&rfp, sizeof(ranks_flags)));
        r.rf = static_cast<ranks_flags*>(rfp);
        MYCHK(mcMemset(r.rf, 0, sizeof(ranks_flags)));
    }
    MYCHK(mcMalloc((void**)&r.w1, (size_t)kEPR * 2 * kIntermediate * kHidden));
    MYCHK(mcMalloc((void**)&r.w2, (size_t)kEPR * kHidden * kIntermediate));
    fp8mega::fill_weights_lane(r.w1, kEPR * 2 * kIntermediate, kHidden, 17u, 0);
    fp8mega::fill_weights_lane(r.w2, kEPR * kHidden, kIntermediate, 91u, 0);
    MYCHK(mcMalloc((void**)&r.ffn_hidden,
                   (size_t)kEPR * kRing * win_cap * kIntermediate));
    MYCHK(mcMemset(r.ffn_hidden, 0, (size_t)kEPR * kRing * win_cap * kIntermediate));
    MYCHK(mcMalloc((void**)&r.l2_out, (size_t)kGrid * win_cap * kHidden * 2));
    MYCHK(mcMemset(r.l2_out, 0, (size_t)kGrid * win_cap * kHidden * 2));
    MYCHK(mcMalloc((void**)&r.y, (size_t)max_tokens * kHidden * sizeof(float)));
    MYCHK(mcMemset(r.y, 0, (size_t)max_tokens * kHidden * sizeof(float)));

    // x / topk_weights（确定性，全 TPR 复用；kernel 只读 [0,tokens) 行）
    {
        std::vector<uint8_t> xb((size_t)max_tokens * kHidden);
        for (int t = 0; t < max_tokens; ++t)
            for (int c = 0; c < kHidden; ++c)
                xb[(size_t)t * kHidden + c] =
                    (uint8_t)((rank * 37 + t * 7 + c * 3) & 0x3f);
        MYCHK(mcMemcpy(r.sym + L.x_off, xb.data(), xb.size(), mcMemcpyHostToDevice));
        std::vector<float> wv((size_t)max_tokens * kTopk);
        for (int i = 0; i < max_tokens * kTopk; ++i)
            wv[i] = 1.0f + rank * 0.25f + (i % 7) * 0.01f;
        MYCHK(mcMemcpy(r.sym + L.topk_weights_off, wv.data(),
                       wv.size() * sizeof(float), mcMemcpyHostToDevice));
    }

    // ---- IPC：取 handle -> MPI 交换 -> 开 peer ----
    r.peer_sym.assign(R, nullptr);
    r.peer_rf.assign(R, nullptr);
    MYCHK(mcIpcGetMemHandle(&r.handle, r.sym));
    MYCHK(mcIpcGetMemHandle(&r.rf_handle, r.rf));
    std::vector<mcIpcMemHandle_t> all_h(R), all_rfh(R);
    MPI_Allgather(&r.handle, (int)sizeof(mcIpcMemHandle_t), MPI_BYTE, all_h.data(),
                  (int)sizeof(mcIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Allgather(&r.rf_handle, (int)sizeof(mcIpcMemHandle_t), MPI_BYTE,
                  all_rfh.data(), (int)sizeof(mcIpcMemHandle_t), MPI_BYTE,
                  MPI_COMM_WORLD);
    for (int d = 0; d < R; ++d) {
        if (d == rank) continue;
        void* q = nullptr;
        if (mcIpcOpenMemHandle(&q, all_h[d], mcIpcMemLazyEnablePeerAccess) != mcSuccess) {
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
        MYCHK(mcMalloc((void**)&r.d_base_ptrs, R * sizeof(uint8_t*)));
        MYCHK(mcMemcpy(r.d_base_ptrs, bp.data(), R * sizeof(uint8_t*),
                       mcMemcpyHostToDevice));
        MYCHK(mcMalloc((void**)&r.d_rf_ptrs, R * sizeof(ranks_flags*)));
        MYCHK(mcMemcpy(r.d_rf_ptrs, rp.data(), R * sizeof(ranks_flags*),
                       mcMemcpyHostToDevice));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // ---- 逐 TPR ----
    mcStream_t st = nullptr;
    MYCHK(mcStreamCreate(&st));
    for (int tpr : o.tprs) {
        // data_init：路由（均衡）+ 清零旗区
        std::vector<int64_t> route =
            make_routes(RoutePattern::Balanced, rank, R, tpr);
        MYCHK(mcMemcpy(r.sym + L.topk_idx_off, route.data(),
                       route.size() * sizeof(int64_t), mcMemcpyHostToDevice));
        MYCHK(mcMemset(r.sym + L.src_index_table_off, 0,
                       (size_t)L.src_index_table_bytes));
        MYCHK(mcMemset(r.sym + L.barrier_slots_off, 0, (size_t)R * R * 4));
        MYCHK(mcMemset(r.lf, 0, sizeof(local_flags)));
        MYCHK(mcMemset(r.rf, 0, sizeof(ranks_flags)));
        MPI_Barrier(MPI_COMM_WORLD);

        fp8mega::MegaLaunchArgs a =
            make_args(o, L, r, R, tpr, win_cap, combine_f32_off);

        // eager（预热）
        fp8mega::launch_full(a, st);
        MYCHK(mcStreamSynchronize(st));
        MPI_Barrier(MPI_COMM_WORLD);

        // graph capture：nodes 个同构节点
        mcGraph_t g = nullptr;
        mcGraphExec_t ex = nullptr;
        if (mcStreamBeginCapture(st, mcStreamCaptureModeThreadLocal) != mcSuccess) {
            std::fprintf(stderr, "rank%d BeginCapture 失败\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int k = 0; k < nodes; ++k) fp8mega::launch_full(a, st);
        MYCHK(mcStreamEndCapture(st, &g));
        MYCHK(mcGraphInstantiate(&ex, g, nullptr, nullptr, 0));
        mcGraphDestroy(g);

        // warmup
        for (int i = 0; i < 2; ++i) MYCHK(mcGraphLaunch(ex, st));
        MYCHK(mcStreamSynchronize(st));
        MPI_Barrier(MPI_COMM_WORLD);

        // measure：replays 次整图
        mcEvent_t e0 = nullptr, e1 = nullptr;
        MYCHK(mcEventCreate(&e0));
        MYCHK(mcEventCreate(&e1));
        MYCHK(mcEventRecord(e0, st));
        for (int i = 0; i < replays; ++i) MYCHK(mcGraphLaunch(ex, st));
        MYCHK(mcEventRecord(e1, st));
        MYCHK(mcStreamSynchronize(st));
        float ms = 0.f;
        MYCHK(mcEventElapsedTime(&ms, e0, e1));
        double us = (double)ms * 1000.0 / (double)(nodes * replays);
        mcEventDestroy(e0);
        mcEventDestroy(e1);
        mcGraphExecDestroy(ex);

        // gather 各 rank 耗时
        std::vector<double> all(R, -1.0);
        MPI_Allgather(&us, 1, MPI_DOUBLE, all.data(), 1, MPI_DOUBLE, MPI_COMM_WORLD);
        if (rank == 0) {
            double vmin = 1e30, vmax = -1.0, vsum = 0.0;
            int n = 0;
            for (double v : all)
                if (v > 0.0) {
                    vmin = std::min(vmin, v);
                    vmax = std::max(vmax, v);
                    vsum += v;
                    ++n;
                }
            double vmean = n ? vsum / n : -1.0;
            std::printf("%-4d %-6d %10.2f %10.2f %10.2f\n", R, tpr, vmin, vmax,
                        vmean);
            std::fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    mcStreamDestroy(st);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
