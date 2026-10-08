// ============================================================================
// test/mega_common.cuh — mybench / check 共用：每进程一卡的运行态、布局、路由
//
// 模型：mpirun -np R，每进程 = 1 个 rank = 1 张卡（mcSetDevice(rank)）。
//   对称区 / ranks_flags 经 mcIpcGetMemHandle + MPI_Allgather 交换后 peer 映射。
//
// 路由（TK 日志）：pair j（按专家序排列，同专家连续）-> src = j % R，
//   q = j / R，token = q % tokens，slot k = q / tokens。
//   要求 max_expert <= R*tokens，此时同一 token 不会出现重复专家（构造期检查）。
//   注：bench.cu 的 build_tk_case 用 idx/topk 顺序填充，会产生重复专家，已弃用。
// ============================================================================
#pragma once

#include <mcr/mc_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "common.cuh"
#include "flags_defs.cuh"
#include "host_layout.cuh"
#include "bench_check.cuh"
#include "megakernel.h"

namespace mt {

using Routes = std::vector<std::vector<int64_t>>;   // [R][tokens*topk] 全局专家号

constexpr int64_t kXreleaseIdx = deep_gemm::layout::kBarXreleaseIdx;
constexpr int64_t kSliceFlagsIdx = deep_gemm::layout::kBarSliceFlagsIdx;
constexpr int64_t kL1ProgIdx = deep_gemm::layout::kBarL1ProgressIdx;
constexpr int64_t kL2DoneIdx = deep_gemm::layout::kBarL2DoneIdx;
constexpr int64_t kTaskReadyIdx = deep_gemm::layout::kBarTaskReadyIdx;
constexpr int64_t kTaskCounterIdx = deep_gemm::layout::kBarTaskCounterIdx;

// 默认 TK 日志（相对仓库根目录；scripts/run.sh 已切到仓库根）
inline const char* kDefaultTkLog = "test/70.r32_tpr_perf/0928_tk_info.log";

// ------------------------------------------------------------ 几何/布局 --
struct Geo {
    int R = 0;
    int max_tokens = 0;     // 槽容量（sym 布局）
    int win = 0;            // 布局用窗口容量（各 launch 的 win_cap 不得超过）
    ProdLayout L{};
    int64_t combine_off = 0;   // combine_rows（BF16 [topk][max_tokens][H]）偏移
    int64_t sym_alloc = 0;
    int block_n = 0;
    int threads = 0;
    int64_t smem = 0;
};

inline Geo make_geo(int R, int max_tokens, int win) {
    Geo g;
    g.R = R;
    g.max_tokens = max_tokens;
    g.win = win;
    g.L = compute_layout(R, max_tokens, kTopk, kHidden, kIntermediate, win);
    g.L.win_capacity = win;
    g.L.x_off = (int64_t)offsetof(local_data, x);
    g.L.topk_idx_off = (int64_t)offsetof(local_data, topk_idx);
    g.L.topk_weights_off = (int64_t)offsetof(local_data, topk_weights);
    g.L.route_slot_counts_off = (int64_t)offsetof(local_data, route_slot_counts);
    g.L.prefix_tbl_off = (int64_t)offsetof(local_data, prefix_tbl);
    g.L.expert_tokens_off = (int64_t)offsetof(local_data, expert_tokens);
    g.L.token_route_weights_off =
        (int64_t)offsetof(local_data, token_route_weights);
    g.L.token_src_meta_off = (int64_t)offsetof(local_data, token_src_meta);
    g.L.barrier_slots_off = 0;
    g.L.src_index_table_off = 0;
    g.L.expert_recv_counts_off = 0;
    g.L.sym_bytes = (int64_t)sizeof(local_data);
    g.combine_off = (int64_t)offsetof(ranks_data, combine_rows);
    g.sym_alloc = (int64_t)sizeof(local_data);
    g.block_n = prod_block_n(256);
    g.threads = g.block_n == 512 ? 1024 : 512;
    g.smem = prod_smem(R, 8, g.block_n);
    return g;
}

// 每 launch 的窗口容量。已验证正确：win == slot == max_tokens。
//   自适应窗（win<slot）在当前内核会令部分 launch 的 REDUCE/graph 失败（见 test/mega_common.cuh 注释与
//   test/77.correctness/测试流程.md），其精确位置在 COMBINE/L2 窗槽 stride 关系，尚未根治。
//   现行策略在此保持等式约束；adaptive 已停用不构成行为变化。
inline int choose_win_cap_counts(const std::vector<int>& per_expert, int max_tokens) {
    (void)per_expert;
    return max_tokens;
}

// 逐专家计数（展平为 [E]，E = R*EPR）
inline std::vector<int> counts_of(const Routes& routes, int E) {
    std::vector<int> cnt(E, 0);
    for (const auto& rr : routes)
        for (int64_t v : rr)
            if (v >= 0 && v < E) ++cnt[(size_t)v];
    return cnt;
}

// ------------------------------------------------------------ 设备资源 --
struct Dev {
    int rank = 0;
    uint8_t* sym = nullptr;          // 对称区（FG）
    local_data* ld = nullptr;        // 本 rank 本地数据（FG）
    local_flags* lf = nullptr;       // 私有旗区
    ranks_flags* rf = nullptr;       // 跨 rank 旗区（FG）
    ranks_data* rd = nullptr;        // 远端字段（FG）
    ranks_data** d_rd = nullptr;     // 各 rank ranks_data 基址表（设备数组）
    uint8_t* w1 = nullptr;           // [EPR][2I][H] lane 序（设备生成）
    uint8_t* w2 = nullptr;           // [EPR][H][I] lane 序
    uint8_t* ffn = nullptr;          // [EPR][ring][win][I] FP8
    uint16_t* l2 = nullptr;          // [grid][win][H] BF16
    float* y = nullptr;              // REDUCE 输出 [rows][H]
    std::vector<float*> ynode;       // graph 逐节点输出（同形状）
    uint8_t** d_base = nullptr;
    ranks_flags** d_rf = nullptr;
    mcIpcMemHandle_t h{}, rh{}, rdh{};
    mcStream_t st = nullptr;
    mcGraphExec_t exec = nullptr;
};

inline size_t ffn_bytes(const Geo& g) {
    return (size_t)kEPR * kRing * g.win * kIntermediate;
}
inline size_t l2_bytes(const Geo& g) {
    return (size_t)kGrid * g.win * kHidden * 2;
}

// 确定性输入（与 bench.cu DISPATCH 视图一致）：x 与 topk_weights 的 max_tokens 行
inline void fill_inputs(const Geo& g, Dev& d) {
    std::vector<uint8_t> xb((size_t)g.max_tokens * kHidden);
    for (int t = 0; t < g.max_tokens; ++t)
        for (int c = 0; c < kHidden; ++c)
            xb[(size_t)t * kHidden + c] =
                (uint8_t)((d.rank * 37 + t * 7 + c * 3) & 0x3f);
    CHECK(mcMemcpy(d.sym + g.L.x_off, xb.data(), xb.size(), mcMemcpyHostToDevice));
    std::vector<float> wv((size_t)g.max_tokens * kTopk);
    for (int i = 0; i < g.max_tokens * kTopk; ++i)
        wv[i] = 1.0f + d.rank * 0.25f + (i % 7) * 0.01f;
    CHECK(mcMemcpy(d.sym + g.L.topk_weights_off, wv.data(), wv.size() * sizeof(float),
                   mcMemcpyHostToDevice));
}

// 分配本 rank 的全部设备资源（不含 IPC）。ynode_rows = 任一 launch 的最大 token 数。
inline void alloc_dev(Dev& d, const Geo& g, int rank, int ynode_rows, int nodes) {
    CHECK(mcSetDevice(rank));
    d.rank = rank;
    void* p = nullptr;
    if (mcExtMallocWithFlags(&p, sizeof(local_data), mcDeviceMallocFinegrained) !=
            mcSuccess || !p)
        CHECK(mcMalloc(&p, sizeof(local_data)));
    d.ld = static_cast<local_data*>(p);
    d.sym = reinterpret_cast<uint8_t*>(d.ld);
    CHECK(mcMemset(d.sym, 0, sizeof(local_data)));
    CHECK(mcMalloc((void**)&d.lf, sizeof(local_flags)));
    CHECK(mcMemset(d.lf, 0, sizeof(local_flags)));
    void* rfp = nullptr;
    if (mcExtMallocWithFlags(&rfp, sizeof(ranks_flags), mcDeviceMallocFinegrained) !=
            mcSuccess || !rfp)
        CHECK(mcMalloc(&rfp, sizeof(ranks_flags)));
    d.rf = static_cast<ranks_flags*>(rfp);
    CHECK(mcMemset(d.rf, 0, sizeof(ranks_flags)));

    void* rdp = nullptr;
    if (mcExtMallocWithFlags(&rdp, sizeof(ranks_data),
                             mcDeviceMallocFinegrained) != mcSuccess ||
        !rdp)
        CHECK(mcMalloc(&rdp, sizeof(ranks_data)));
    d.rd = static_cast<ranks_data*>(rdp);
    CHECK(mcMemset(d.rd, 0, sizeof(ranks_data)));

    d.w1 = &d.ld->w1[0][0][0];
    d.w2 = &d.ld->w2[0][0][0];
    fp8mega::fill_weights_lane(d.w1, kEPR * 2 * kIntermediate, kHidden, 17u, 0);
    fp8mega::fill_weights_lane(d.w2, kEPR * kHidden, kIntermediate, 91u, 0);
    CHECK(mcDeviceSynchronize());

    d.ffn = &d.ld->ffn_hidden[0][0][0];
    d.l2 = &d.ld->l2_out[0][0][0];
    const size_t y_bytes = (size_t)ynode_rows * kHidden * sizeof(float);
    d.y = &d.ld->y[0][0];
    d.ynode.assign(nodes, nullptr);
    for (int n = 0; n < nodes; ++n) {
        CHECK(mcMalloc((void**)&d.ynode[n], y_bytes));
        CHECK(mcMemset(d.ynode[n], 0, y_bytes));
    }
    fill_inputs(g, d);
    CHECK(mcStreamCreate(&d.st));
    CHECK(mcIpcGetMemHandle(&d.h, d.sym));
    CHECK(mcIpcGetMemHandle(&d.rh, d.rf));
    CHECK(mcIpcGetMemHandle(&d.rdh, d.rd));
}

// MPI 交换 IPC 句柄，打开 peer，构造 d_base / d_rf 设备指针表。
// 必须在所有 rank 的 alloc_dev 之后调用（内部含 MPI 集合通信）。
inline void exchange_ipc(Dev& d, int R) {
    std::vector<mcIpcMemHandle_t> all_h(R), all_rh(R), all_rdh(R);
    MPI_Allgather(&d.h, (int)sizeof(mcIpcMemHandle_t), MPI_BYTE, all_h.data(),
                  (int)sizeof(mcIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Allgather(&d.rh, (int)sizeof(mcIpcMemHandle_t), MPI_BYTE, all_rh.data(),
                  (int)sizeof(mcIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    MPI_Allgather(&d.rdh, (int)sizeof(mcIpcMemHandle_t), MPI_BYTE, all_rdh.data(),
                  (int)sizeof(mcIpcMemHandle_t), MPI_BYTE, MPI_COMM_WORLD);
    CHECK(mcSetDevice(d.rank));
    std::vector<uint8_t*> bp(R);
    std::vector<ranks_flags*> rp(R);
    std::vector<ranks_data*> rdp(R);
    for (int s = 0; s < R; ++s) {
        if (s == d.rank) {
            bp[s] = d.sym;
            rp[s] = d.rf;
            rdp[s] = d.rd;
            continue;
        }
        void* q = nullptr;
        CHECK(mcIpcOpenMemHandle(&q, all_h[s], mcIpcMemLazyEnablePeerAccess));
        bp[s] = static_cast<uint8_t*>(q);
        void* qr = nullptr;
        CHECK(mcIpcOpenMemHandle(&qr, all_rh[s], mcIpcMemLazyEnablePeerAccess));
        rp[s] = static_cast<ranks_flags*>(qr);
        void* qd = nullptr;
        CHECK(mcIpcOpenMemHandle(&qd, all_rdh[s], mcIpcMemLazyEnablePeerAccess));
        rdp[s] = static_cast<ranks_data*>(qd);
    }
    for (int s = 0; s < R; ++s) {
        if (s == d.rank) continue;
        int can = 0;
        if (mcDeviceCanAccessPeer(&can, d.rank, s) == mcSuccess && can)
            mcDeviceEnablePeerAccess(s, 0);   // 已开启时返回 AlreadyEnabled，忽略
    }
    CHECK(mcMalloc((void**)&d.d_base, R * sizeof(uint8_t*)));
    CHECK(mcMemcpy(d.d_base, bp.data(), R * sizeof(uint8_t*), mcMemcpyHostToDevice));
    CHECK(mcMalloc((void**)&d.d_rf, R * sizeof(ranks_flags*)));
    CHECK(mcMemcpy(d.d_rf, rp.data(), R * sizeof(ranks_flags*), mcMemcpyHostToDevice));
    CHECK(mcMalloc((void**)&d.d_rd, R * sizeof(ranks_data*)));
    CHECK(mcMemcpy(d.d_rd, rdp.data(), R * sizeof(ranks_data*), mcMemcpyHostToDevice));
}

// 上传本 rank 的路由并清零一次 launch 所需的全部状态（不清 ynode/y）。
inline void reset_dev(const Geo& g, const Dev& d, const Routes& routes, int tokens) {
    const ProdLayout& L = g.L;
    const auto& route = routes[d.rank];
    CHECK(mcSetDevice(d.rank));
    CHECK(mcMemcpy(d.sym + L.topk_idx_off, route.data(), route.size() * sizeof(int64_t),
                   mcMemcpyHostToDevice));
    CHECK(mcMemset(d.sym + L.route_slot_counts_off, 0, (size_t)g.R * kEPR * sizeof(int)));
    CHECK(mcMemset(d.sym + L.prefix_tbl_off, 0, (size_t)L.prefix_tbl_bytes));
    CHECK(mcMemset(d.sym + L.token_route_weights_off, 0,
                   (size_t)kGrid * g.win * sizeof(float)));
    CHECK(mcMemset(d.sym + L.token_src_meta_off, 0,
                   (size_t)kGrid * g.win * 3 * sizeof(int)));
    CHECK(mcMemset(d.ld->expert_tokens, 0, sizeof(d.ld->expert_tokens)));
    CHECK(mcMemset(d.lf, 0, sizeof(local_flags)));
    CHECK(mcMemset(d.rf, 0, sizeof(ranks_flags)));
    CHECK(mcMemset(d.rd, 0, sizeof(ranks_data)));
    CHECK(mcMemset(d.ffn, 0, ffn_bytes(g)));
    CHECK(mcMemset(d.l2, 0, l2_bytes(g)));
    (void)tokens;
}

// 构造 launch 入参。win = 本 launch 的窗口容量（<= g.win）。
inline fp8mega::MegaLaunchArgs make_args(const Geo& g, const Dev& d, int tokens, int win,
                                         float* y) {
    const ProdLayout& L = g.L;
    fp8mega::MegaLaunchArgs a{};
    a.topk_idx = reinterpret_cast<const int64_t*>(d.sym + L.topk_idx_off);
    a.topk_idx_off = L.topk_idx_off;
    a.self = d.sym;
    a.peer_bases = d.d_base;
    a.R = g.R;
    a.rank = d.rank;
    a.tokens = tokens;
    a.slot_capacity = g.max_tokens;
    a.win_cap = win;
    a.pool_stride = win;
    a.tbl_cap = g.max_tokens;
    a.k_in = kHidden;
    a.topk = kTopk;
    a.ring = kRing;
    a.src_off = L.src_index_table_off;
    a.rsc_off = L.route_slot_counts_off;
    a.recv_off = L.expert_recv_counts_off;
    a.prefix_off = L.prefix_tbl_off;
    a.barrier_off = L.barrier_slots_off;
    a.lf = d.lf;
    a.rf = d.rf;
    a.rf_peers = d.d_rf;
    a.remote = d.rd;
    a.remote_peers = d.d_rd;
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
    a.prefix_ptr = nullptr;
    a.w1 = d.w1;
    a.l1_n_total = kIntermediate * 2;
    a.l1_block_n = 256;
    a.l1_k_out = kIntermediate;
    a.ffn_hidden = d.ffn;
    a.w2 = d.w2;
    a.l2_out = d.l2;
    a.l2_k_out = kHidden;
    a.combine_rows_off = g.combine_off;
    a.combine_cap = g.max_tokens;
    a.sums_stride = deep_gemm::layout::kMegaMoeMaxCombineSlices;
    a.y = y;
    a.grid = kGrid;
    a.merge_grid = kGrid;
    a.threads = g.threads;
    a.smem = g.smem;
    return a;
}

// ------------------------------------------------------------ 路由 --
inline Routes balanced_routes(int R, int tokens) {
    Routes out(R);
    for (int r = 0; r < R; ++r) out[r] = make_routes(RoutePattern::Balanced, r, R, tokens);
    return out;
}

// 同一 token 不出现重复专家（R*tokens 之内各专家 pair 连续，见文件头注释）
inline bool routes_duplicate_free(const Routes& routes, int tokens, int topk) {
    for (const auto& rr : routes)
        for (int t = 0; t < tokens; ++t)
            for (int a = 0; a < topk; ++a)
                for (int b = a + 1; b < topk; ++b)
                    if (rr[(size_t)t * topk + a] == rr[(size_t)t * topk + b]) return false;
    return true;
}

// 一条 TK 实录 launch：complete（全部 R 个 rank 行齐全、总 pair 整除 R*topk）
struct TkLaunch {
    int id = 0;
    int tokens = 0;                 // 每 rank token 数 = total / (R*topk)
    int max_expert = 0;
    std::vector<int> per_expert;    // [E]，rank-major（专家 e = r*EPR + i）
    std::vector<std::vector<int>> cnt;  // [R][EPR]
};

// 解析日志，返回全部合法 complete launch（按 launch 号升序）
inline bool load_tk_launches(const std::string& path, int R, int topk,
                             std::vector<TkLaunch>* out, std::string* err) {
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
        if (std::sscanf(line.c_str() + p, "[TK_INFO] rank=%d launch=%d n=%d :", &rank,
                        &launch, &n) != 3)
            continue;
        if (n != kEPR || rank < 0 || rank >= R) continue;
        const size_t colon = line.find(':', p);
        if (colon == std::string::npos) continue;
        std::istringstream is(line.substr(colon + 1));
        std::vector<int> vals;
        int v = 0;
        while (is >> v) vals.push_back(v);
        if ((int)vals.size() != kEPR) continue;
        data[launch][rank] = vals;
    }
    out->clear();
    for (const auto& kv : data) {
        if ((int)kv.second.size() != R) continue;           // 不完整
        TkLaunch t;
        t.id = kv.first;
        t.cnt.assign(R, std::vector<int>(kEPR, 0));
        long long total = 0;
        for (const auto& rk : kv.second) {
            t.cnt[rk.first] = rk.second;
            for (int c : rk.second) {
                total += c;
                t.max_expert = std::max(t.max_expert, c);
            }
        }
        if (total <= 0 || total % ((long long)R * topk) != 0) continue;
        t.tokens = (int)(total / ((long long)R * topk));
        t.per_expert.reserve((size_t)R * kEPR);
        for (int r = 0; r < R; ++r)
            for (int e = 0; e < kEPR; ++e) t.per_expert.push_back(t.cnt[r][e]);
        out->push_back(std::move(t));
    }
    if (out->empty()) {
        *err = "TK_INFO 中无合法 complete launch: " + path;
        return false;
    }
    return true;
}

// 由展平计数 per_expert[R*EPR]（rank-major）构造每 rank 的路由（分配规则见文件头）
inline bool routes_from_counts(const std::vector<int>& per_expert, int R, int tokens,
                               int topk, int id, Routes* out, std::string* err) {
    out->assign(R, std::vector<int64_t>((size_t)tokens * topk, 0));
    long long j = 0;
    for (int r = 0; r < R; ++r)
        for (int e = 0; e < kEPR; ++e) {
            const int64_t ge = (int64_t)r * kEPR + e;
            const int c_re = per_expert[(size_t)r * kEPR + e];
            for (int c = 0; c < c_re; ++c, ++j) {
                const int src = (int)(j % R);
                const int q = (int)(j / R);
                const int token = q % tokens;
                const int k = q / tokens;
                (*out)[src][(size_t)token * topk + k] = ge;
            }
        }
    if (!routes_duplicate_free(*out, tokens, topk)) {
        *err = "launch=" + std::to_string(id) + " 构造后仍有重复专家（R*tokens=" +
               std::to_string(R * tokens) + "）";
        return false;
    }
    return true;
}

// 回放一致性：路由逐专家计数与实录逐字一致。返回 max|实际-实录|。
inline long long route_count_deviation(const Routes& routes, const std::vector<int>& want,
                                       int E) {
    std::vector<long long> got(E, 0);
    for (const auto& rr : routes)
        for (int64_t v : rr)
            if (v >= 0 && v < E) ++got[(size_t)v];
    long long dev = 0;
    for (int e = 0; e < E; ++e)
        dev = std::max(dev, std::llabs(got[e] - (long long)want[e]));
    return dev;
}

// 逗号分隔整数列表
inline std::vector<int> parse_csv_ints(const std::string& s) {
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

// 一个用例（只存计数；路由按需构造，避免全量日志常驻内存）
struct Spec {
    std::string label;
    int kind = 0;                 // 0 = 均衡（按 tokens），1 = TK 实录
    int tokens = 0;               // 每 rank token 数
    int launch_id = 0;            // TK 实录号（kind=1）
    int max_expert = 0;
    int win = 0;                  // 本用例窗口容量（choose_win_cap_counts）
    std::vector<int> per_expert;  // [R*EPR]，rank-major
};

inline Spec balanced_spec(int R, int tokens) {
    Spec s;
    s.label = "TPR=" + std::to_string(tokens);
    s.kind = 0;
    s.tokens = tokens;
    s.per_expert = counts_of(balanced_routes(R, tokens), R * kEPR);
    for (int c : s.per_expert) s.max_expert = std::max(s.max_expert, c);
    return s;
}

inline Spec tk_spec(const TkLaunch& t) {
    Spec s;
    s.label = "launch=" + std::to_string(t.id);
    s.kind = 1;
    s.tokens = t.tokens;
    s.launch_id = t.id;
    s.max_expert = t.max_expert;
    s.per_expert = t.per_expert;
    return s;
}

// 按用例重建路由（kind=0 均衡；kind=1 由计数 + 固定分配规则）
inline bool spec_routes(const Spec& s, int R, int topk, Routes* out, std::string* err) {
    if (s.kind == 0) {
        *out = balanced_routes(R, s.tokens);
        return true;
    }
    return routes_from_counts(s.per_expert, R, s.tokens, topk, s.launch_id, out, err);
}

}  // namespace mt
