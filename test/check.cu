// ============================================================================
// test/check.cu — fp8xfp8 mega kernel 正确性测试（mpirun -np 32，每进程一卡）
//
// 启动（经 scripts/run.sh）：
//   scripts/run.sh check --smoke     # 冒烟：单次 <10s 目标
//   scripts/run.sh check --all       # 全量：均衡 TPR 表 + 全部 TK 实录 launch，不限时
//
// 每个用例（均衡 TPR 或 TK 实录 launch）可执行的检查：
//   L0      前序：route / publish / task_setup 与 host 模型逐项一致（launch_prelude）
//   L1/L2   expert0 窗口0 的 ffn_hidden（FP8）与 l2_out（BF16）vs CPU 参考（rank0）
//   REDUCE  每 rank 的 y（FP32）vs CPU 参考（ref_reduce，全部 token）
//   graph   图内 N 节点（N=--graph-nodes，默认 50）每个节点的 y 与 eager 的 y 逐元素比较：
//           rel_max = max|a-b| / max|ref| <= 1e-3 且 finite 位置一致（抓陈旧旗子）
//
// 冷启动：首个 launch 的跨 rank 同步在冷状态下不可靠（见 测试流程.md），
//   因此所有检查前先做一次不计入结果的 eager 预热。
// 退出码：0 = 全部 PASS；1 = 有 FAIL；2 = 参数/环境错误
// ============================================================================

#include <mcr/mc_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <thread>
#include <vector>

#include "mega_common.cuh"

namespace {

using mt::Spec;

constexpr int kCase0 = 804900;               // 实录 launch（= MXMOEMega FIXED_CASES['804900']）
const std::vector<int> kBalancedTprs = {1, 4, 16, 32, 33, 64, 128, 256, 512};

struct Flags {
    bool l0 = false;      // 前序检查
    bool eager = false;   // eager 全链 + L1/L2 + REDUCE
    int graph = 0;        // >0：graph 图内节点数（与 eager 逐节点比较）
};

// 与设备 fill_weights_lane 同字节布局的 host 参考：生成「整张量中第 row_base 行起的
// n_rows 行」，即某个专家的切片（设备按整张量的全局行号取值，专家 e 的行基址 = e*n_rows）。
void fill_lane_host_w(std::vector<uint8_t>& buf, int n_rows, int k_cols, uint32_t seed,
                      int row_base) {
    buf.assign((size_t)n_rows * k_cols, 0);
    const int k_iters = k_cols / 32;
    const int npairs = n_rows / 32;
    for (int np = 0; np < npairs; ++np)
        for (int kt = 0; kt < k_iters; ++kt) {
            uint8_t* tile = buf.data() + ((size_t)np * k_iters + kt) * 1024;
            for (int L = 0; L < 64; ++L) {
                const int row0 = row_base + np * 32 + (L % 16);
                const int row1 = row0 + 16;
                const int k0 = kt * 32 + (L / 16) * 8;
                for (int b = 0; b < 8; ++b) {
                    tile[L * 16 + b] = WPROBE_VAL(seed, row0, k0 + b);
                    tile[L * 16 + 8 + b] = WPROBE_VAL(seed, row1, k0 + b);
                }
            }
        }
}

struct Env {
    int R = 0;
    int rank = 0;
    int check_rows = 16;
    mt::Geo geo;
    mt::Dev d;
    std::vector<uint8_t> w1h, w2h;   // host 参考权重：当前 w_expert 的切片（仅 rank0 使用）
    int w_expert = -1;
    std::vector<std::vector<uint8_t>> w1_cache, w2_cache;   // 按本地专家缓存（rank0）
};

struct Result {
    std::string label;
    bool l0 = true, l12 = true, red = true, graph = true;
    double red_max_abs = 0.0;
    double graph_rel = 0.0;
    long long graph_pattern = 0;
    double ms = 0.0;
};

// ----------------------------------------------------------- 汇总 --
bool agree(bool local) {
    int v = local ? 1 : 0, g = 0;
    MPI_Allreduce(&v, &g, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    return g == 1;
}
double agree_max(double v) {
    double g = 0.0;
    MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    return g;
}
long long agree_sum(long long v) {
    long long g = 0;
    MPI_Allreduce(&v, &g, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return g;
}

// 每 rank 打印前 2 条失败细节，避免 32 份输出刷屏
int g_detail_budget = 2;
void detail(const Env& e, const char* what, const std::string& msg) {
    if (g_detail_budget <= 0) return;
    --g_detail_budget;
    std::printf("    [detail] rank%d %s: %s\n", e.rank, what, msg.c_str());
    std::fflush(stdout);
}

// 行级并行执行 f(i)，i in [0, n)（host 参考计算用；n 很小时开销可忽略）
template <class F>
void parallel_rows(int n, F&& f) {
    const int nt = std::max(1, std::min(16, n));
    std::vector<std::thread> th;
    th.reserve(nt);
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t]() {
            for (int i = t; i < n; i += nt) f(i);
        });
    for (auto& x : th) x.join();
}

// ------------------------------------------------- L0 前序（launch_prelude） --
bool run_l0(Env& e, const Spec& s, const mt::Routes& routes) {
    mt::reset_dev(e.geo, e.d, routes, s.tokens);
    MPI_Barrier(MPI_COMM_WORLD);
    fp8mega::MegaLaunchArgs a = mt::make_args(e.geo, e.d, s.tokens, s.win, e.d.y);
    fp8mega::launch_prelude(a, e.d.st);
    CHECK(mcStreamSynchronize(e.d.st));
    MPI_Barrier(MPI_COMM_WORLD);

    // validate_task_setup 的 cap = L.win_capacity。内核的窗口切分与路由截断均用
    // slot_capacity（= max_tokens，见 prologue.cuh 的 task_setup_body / publish），
    // 而非按 launch 的 win_cap；故模型也按 slot_capacity 建模。
    ProdLayout Lc = e.geo.L;
    Lc.win_capacity = e.geo.max_tokens;
    std::string msg;
    const uint8_t* ld_base = reinterpret_cast<const uint8_t*>(e.d.ld);
    bool ok = validate_route(Lc, e.R, s.tokens, e.rank, routes, ld_base, &msg,
                             e.d.rd);
    if (ok)
        ok = validate_publish(Lc, e.R, s.tokens, e.rank, routes, ld_base, kPublishEpoch,
                              &msg, (int)Lc.recv_stride, Lc.expert_recv_counts_off,
                              e.d.rd);
    if (ok) {
        std::vector<int> ts(kGrid, 0);
        CHECK(mcSetDevice(e.rank));
        CHECK(mcMemcpy(ts.data(), e.d.lf->out_total, kGrid * sizeof(int),
                       mcMemcpyDeviceToHost));
        ok = validate_task_setup(Lc, e.R, e.rank, routes, ld_base, nullptr, ts.data(), &msg);
    }
    if (!ok) detail(e, "L0", msg);
    return ok;
}

// ------------------------------------------------------- eager 全链 --
void run_eager(Env& e, const Spec& s, const mt::Routes& routes, std::vector<float>* y) {
    mt::reset_dev(e.geo, e.d, routes, s.tokens);
    MPI_Barrier(MPI_COMM_WORLD);
    fp8mega::MegaLaunchArgs a = mt::make_args(e.geo, e.d, s.tokens, s.win, e.d.y);
    fp8mega::launch_full(a, e.d.st);
    CHECK(mcStreamSynchronize(e.d.st));
    MPI_Barrier(MPI_COMM_WORLD);
    y->assign((size_t)s.tokens * kHidden, 0.0f);
    CHECK(mcMemcpy(y->data(), e.d.y, y->size() * sizeof(float), mcMemcpyDeviceToHost));
}

// L1 诊断（MT_L1_DIAG=1）：把「设备 pool 行」「本次用例真实输入行」「设备 ffn」三者分开核对。
// 结论用途：判断 FAIL 来自 pool 内容（peer 写可见性/残留）还是 kernel 的 L1 计算。
void l1_diag(Env& e, const Spec& s, const mt::Routes& routes, int ex, int rows,
             const std::vector<uint8_t>& xd, const std::vector<uint8_t>& ffn_d) {
    std::vector<std::pair<int, int>> pairs;   // 路由到专家 ex 的 (src, token)
    for (int r = 0; r < e.R; ++r)
        for (int t = 0; t < s.tokens; ++t)
            for (int k = 0; k < kTopk; ++k)
                if (routes[r][(size_t)t * kTopk + k] == ex) pairs.push_back({r, t});
    auto hash_of = [](const uint8_t* p, size_t n) {
        uint64_t h = 1469598103934665603ull;
        for (size_t i = 0; i < n; ++i) {
            h ^= p[i];
            h *= 1099511628211ull;
        }
        return h;
    };
    auto xrow = [](int src, int t, uint8_t* out) {
        for (int c = 0; c < kHidden; ++c)
            out[c] = (uint8_t)((src * 37 + t * 7 + c * 3) & 0x3f);
    };
    std::unordered_map<uint64_t, std::vector<std::pair<int, int>>> exp_map;
    for (const auto& pr : pairs) {
        std::vector<uint8_t> row(kHidden);
        xrow(pr.first, pr.second, row.data());
        exp_map[hash_of(row.data(), row.size())].push_back(pr);
    }
    int hit = 0;
    std::vector<std::pair<int, int>> pool_of((size_t)rows, {-1, -1});
    for (int i = 0; i < rows; ++i) {
        auto it = exp_map.find(hash_of(xd.data() + (size_t)i * kHidden, kHidden));
        if (it != exp_map.end() && !it->second.empty()) {
            ++hit;
            pool_of[i] = it->second.back();
            it->second.pop_back();
        }
    }
    std::printf("    [L1diag] launch=%s expert=%d pairs=%d 核验=%d | pool 行命中本次真实输入=%d\n",
                s.label.c_str(), ex, (int)pairs.size(), rows, hit);
    // 用「真实输入行」重算 L1 参考，与设备 ffn 比较：若通过则 kernel 正确、问题在 pool 读出
    std::vector<uint8_t> xd_true((size_t)rows * kHidden);
    for (int i = 0; i < rows; ++i)
        if (pool_of[i].first >= 0)
            xrow(pool_of[i].first, pool_of[i].second, xd_true.data() + (size_t)i * kHidden);
    std::vector<float> ref((size_t)rows * kIntermediate);
    parallel_rows(rows, [&](int i) {
        ref_l1(xd_true.data() + (size_t)i * kHidden, e.w1h.data(), 0, 1, kHidden,
               kIntermediate, ref.data() + (size_t)i * kIntermediate);
    });
    std::vector<uint8_t> rq(ref.size());
    ref_quantize_fp8(ref.data(), (long long)ref.size(), rq.data());
    std::vector<float> rf(ref.size());
    for (size_t i = 0; i < ref.size(); ++i) rf[i] = fp8_e4m3_decode(rq[i]);
    print_check_stats("L1diag ffn(对真实输入行)", compare_fp8(ffn_d.data(), rf.data(),
                                                            (int)rf.size(), 7e-2, 2e-3));
    // 扫描 ffn 缓冲的全部 (expert, ring) 槽：找出哪一槽真的装着该参考输出
    const int nslot = kEPR * kRing;
    std::vector<uint8_t> all((size_t)nslot * s.win * kIntermediate);
    CHECK(mcMemcpy(all.data(), e.d.ffn, all.size(), mcMemcpyDeviceToHost));
    const size_t slot_b = (size_t)s.win * kIntermediate;
    for (int sl = 0; sl < nslot; ++sl) {
        const uint8_t* p = all.data() + (size_t)sl * slot_b;
        const CheckStats st = compare_fp8(p, rf.data(), (int)rf.size(), 7e-2, 2e-3);
        if (st.bad == 0)
            std::printf("    [L1diag] ffn 匹配槽: slot=%d (expert=%d, ring=%d) "
                        "我读的槽=%d (expert=%d, ring=%d)\n",
                        sl, sl / kRing, sl % kRing, ex * kRing, ex, 0);
    }
}

// --------------------------------------------- L1 / L2（expert0，rank0） --
bool check_l1l2(Env& e, const Spec& s, const mt::Routes& routes) {
    if (e.rank != 0) return true;
    const int win = s.win;
    // 选择 rank0 本地专家中收行最多者（专家 e 的 pool 槽 = e*kRing）
    int ex = -1, best = 0;
    for (int i = 0; i < kEPR; ++i)
        if (s.per_expert[i] > best) {
            best = s.per_expert[i];
            ex = i;
        }
    if (ex < 0) {
        std::printf("    [L1/L2] rank0 本地专家均无 pair，跳过\n");
        return true;
    }
    if (e.w_expert != ex) {
        if (e.w1_cache.empty()) {
            e.w1_cache.resize(kEPR);
            e.w2_cache.resize(kEPR);
        }
        if (e.w1_cache[ex].empty()) {
            fill_lane_host_w(e.w1_cache[ex], 2 * kIntermediate, kHidden, 17u,
                             ex * 2 * kIntermediate);
            fill_lane_host_w(e.w2_cache[ex], kHidden, kIntermediate, 91u, ex * kHidden);
        }
        e.w1h = e.w1_cache[ex];   // 拷贝（每切片 25MB，rank0 一次性）
        e.w2h = e.w2_cache[ex];
        e.w_expert = ex;
    }
    const int rows = std::max(1, std::min(std::min(best, win), e.check_rows));
    const ProdLayout& L = e.geo.L;
    const int pool_slot = ex * kRing;   // 专家 ex，窗口 w0
    std::printf("    [L1/L2] expert=%d (rank0 本地) pairs=%d 核验行数=%d\n", ex, best, rows);

    CHECK(mcSetDevice(0));
    std::vector<uint8_t> xd((size_t)rows * kHidden);
    CHECK(mcMemcpy(xd.data(),
                   (e.d.ld ? &e.d.ld->expert_tokens[0][0][0]
                           : e.d.sym + L.expert_tokens_off) +
                       (size_t)pool_slot * win * kHidden,
                   xd.size(), mcMemcpyDeviceToHost));
    std::vector<uint8_t> ffn_d((size_t)rows * kIntermediate);
    CHECK(mcMemcpy(ffn_d.data(), e.d.ffn + (size_t)pool_slot * win * kIntermediate,
                   ffn_d.size(), mcMemcpyDeviceToHost));
    // L1 输出布局：行主序（ring 0）或 A-fragment 路径的 f2 布局（compute.cuh 的 f_slots /
    // f2_slots 公式；compute_big.cuh 的 cm_store_ffn_frag 与 f2p 寻址）。
    //   f_slots = ceil((k_in/32)*mt*512 / (win*I))，f2_slot = 1 + f_slots，
    //   f2 槽 = e*ring + f2_slot，字节(row,col) = ((col>>5)*mt + (row>>4))*512 + (row&15)*32 + (col&31)。
    {
        const int mt = (win + 15) >> 4;
        const int64_t slot_bytes = (int64_t)win * kIntermediate;
        const int64_t f_bytes = (int64_t)(kHidden / 32) * mt * 512;
        const int f_slots = (int)((f_bytes + slot_bytes - 1) / slot_bytes);
        const int f2_slot = 1 + f_slots;
        long long nz0 = 0;
        for (size_t i = 0; i < (size_t)rows * kIntermediate; ++i)
            if (ffn_d[i] != 0) ++nz0;
        if (nz0 < (long long)(rows * kIntermediate) / 2) {
            std::vector<uint8_t> f2((size_t)win * kIntermediate);
            CHECK(mcMemcpy(f2.data(),
                           e.d.ffn + ((size_t)ex * kRing + f2_slot) * win * kIntermediate,
                           f2.size(), mcMemcpyDeviceToHost));
            for (int r = 0; r < rows; ++r)
                for (int c = 0; c < kIntermediate; ++c)
                    ffn_d[(size_t)r * kIntermediate + c] =
                        f2[((size_t)(c >> 5) * mt + (r >> 4)) * 512 + (r & 15) * 32 + (c & 31)];
            std::printf("    [L1/L2] 布局=A-fragment（f2 槽 e*ring+%d），已解码为行主序\n", f2_slot);
        } else {
            std::printf("    [L1/L2] 布局=行主序（ring 0）\n");
        }
    }

    // L1：CPU 参考 -> E4M3 量化 -> 解码后与设备字节比较
    std::vector<float> ffn_exp((size_t)rows * kIntermediate);
    parallel_rows(rows, [&](int i) {
        ref_l1(xd.data() + (size_t)i * kHidden, e.w1h.data(), 0, 1, kHidden, kIntermediate,
               ffn_exp.data() + (size_t)i * kIntermediate);
    });
    std::vector<uint8_t> ffn_q(ffn_exp.size());
    ref_quantize_fp8(ffn_exp.data(), (long long)ffn_exp.size(), ffn_q.data());
    std::vector<float> ffn_exp_f(ffn_exp.size());
    for (size_t i = 0; i < ffn_exp.size(); ++i) ffn_exp_f[i] = fp8_e4m3_decode(ffn_q[i]);
    const CheckStats s1 = compare_fp8(ffn_d.data(), ffn_exp_f.size() > 0 ? ffn_exp_f.data()
                                                                        : nullptr,
                                      (int)ffn_exp_f.size(), 7e-2, 2e-3);
    print_check_stats("L1 ffn_hidden(expert0)", s1);
    if (std::getenv("MT_L1_DIAG") != nullptr) l1_diag(e, s, routes, ex, rows, xd, ffn_d);
    // L2：CPU 参考（以设备 ffn 字节为输入）vs 各 block 的 l2_out 槽。
    // 每个 block 只写自己负责的 N-slice 列；对每个 slice 查找「存在一个 block 与参考一致」。
    std::vector<float> out_exp((size_t)rows * kHidden);
    parallel_rows(rows, [&](int i) {
        ref_l2(ffn_d.data() + (size_t)i * kIntermediate, e.w2h.data(), 0, 1,
               kIntermediate, kHidden, out_exp.data() + (size_t)i * kHidden);
    });
    std::vector<uint16_t> l2_all((size_t)kGrid * rows * kHidden);
    for (int b = 0; b < kGrid; ++b)
        CHECK(mcMemcpy(l2_all.data() + (size_t)b * rows * kHidden,
                       e.d.l2 + (size_t)b * win * kHidden,
                       (size_t)rows * kHidden * 2, mcMemcpyDeviceToHost));
    const int cs = kHidden / kNSlice;
    bool l2_ok = true;
    for (int sl = 0; sl < kNSlice; ++sl) {
        CheckStats best;
        int best_block = -1;
        for (int b = 0; b < kGrid; ++b) {
            const CheckStats st = compare_bf16_rows(
                l2_all.data() + (size_t)b * rows * kHidden, out_exp.data(), rows, kHidden,
                kHidden, sl * cs, (sl + 1) * cs, 1e-2, 1e-3);
            if (best_block < 0 || st.bad < best.bad) {
                best = st;
                best_block = b;
            }
            if (st.pass()) break;
        }
        char tag[64];
        std::snprintf(tag, sizeof(tag), "L2 out slice%d(block%d)", sl, best_block);
        print_check_stats(tag, best);
        l2_ok = l2_ok && best.pass();
    }
    return s1.pass() && l2_ok;
}

// ----------------------------------------------------- REDUCE（每 rank） --
CheckStats check_reduce(Env& e, const Spec& s, const mt::Routes& routes,
                        const std::vector<float>& y) {
    const ProdLayout& L = e.geo.L;
    const int T = s.tokens;
    const int E = e.R * kEPR;
    CHECK(mcSetDevice(e.rank));
    // combine_rows 为 [topk][kMaxTPR][H]；紧凑为 [topk][T][H] 供 ref_reduce。
    // eager 单 launch：reset 后 current_time[0][1]=1，REDUCE 片 = 0。
    const int cyc_rows = 0;
    std::vector<uint8_t> rows_h((size_t)kTopk * T * kHidden * 2);
    const uint8_t* rows_dev =
        reinterpret_cast<const uint8_t*>(e.d.rd->combine_rows[cyc_rows]);
    for (int k = 0; k < kTopk; ++k)
        CHECK(mcMemcpy(rows_h.data() + (size_t)k * T * kHidden * 2,
                       rows_dev + (size_t)k * kMaxTPR * kHidden * 2,
                       (size_t)T * kHidden * 2, mcMemcpyDeviceToHost));
    std::vector<float> w_h((size_t)T * kTopk);
    CHECK(mcMemcpy(w_h.data(), e.d.sym + L.topk_weights_off, w_h.size() * sizeof(float),
                   mcMemcpyDeviceToHost));
    std::vector<float> y_ref((size_t)T * kHidden);
    for (int t = 0; t < T; ++t)
        ref_reduce(rows_h.data(), routes[e.rank].data(), w_h.data(), t, kTopk, E, T,
                   kHidden, y_ref.data() + (size_t)t * kHidden);
    const CheckStats st = compare_f32(y.data(), y_ref.data(), (int)y_ref.size(), 1e-5, 1e-6);
    if (const char* dm = std::getenv("MT_DIAG_REDUCE")) {
        // 逐 k 统计 combine_rows 非零行 vs 参考 y 非零行，判断缺的是哪个贡献槽
        const size_t H = kHidden;
        const int E = e.R * kEPR;
        std::vector<long long> exp_nz(kTopk, 0), got_nz(kTopk, 0);
        for (int k = 0; k < kTopk; ++k)
            for (int t = 0; t < T; ++t) {
                bool any_ref = false, any_got = false;
                for (size_t c = 0; c < H; ++c) {
                    if (std::fabs(y_ref[(size_t)t * H + c]) > 1e-6f) any_ref = true;
                    bool ra = false; unsigned short bf; const uint16_t* row =
                        reinterpret_cast<const uint16_t*>(rows_h.data() +
                            ((size_t)k * T + t) * H * 2);
                    for (size_t cc = 0; cc < H; ++cc)
                        if (row[cc] != 0) { ra = true; break; }
                    if (ra) any_got = true;
                }
                if (any_ref) ++exp_nz[k];
                if (any_got) ++got_nz[k];
            }
        std::printf("  [REDdiag] launch=%s rank=%d 期望非零行/槽 = [", s.label.c_str(), e.rank);
        for (int k = 0; k < kTopk; ++k) std::printf("%lld,", exp_nz[k]);
        std::printf("]  实际 = [");
        for (int k = 0; k < kTopk; ++k) std::printf("%lld,", got_nz[k]);
        std::printf("]\n");
    }
    if (!st.pass()) {
        char msg[160];
        std::snprintf(msg, sizeof(msg), "bad=%lld max_abs=%.3e worst_got=%.6g want=%.6g",
                      st.bad, st.max_abs, (double)st.worst_got, (double)st.worst_want);
        detail(e, "REDUCE", msg);
    }
    return st;
}

// ------------------------------------------------------ graph vs eager --
struct RelStat {
    double rel_max = 0.0;
    long long pattern = 0;
};
RelStat rel_compare(const std::vector<float>& got, const std::vector<float>& want) {
    RelStat s;
    double scale = 0.0, maxd = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const bool gf = std::isfinite(got[i]), wf = std::isfinite(want[i]);
        if (gf != wf) {
            ++s.pattern;
            continue;
        }
        if (!wf) continue;
        scale = std::max(scale, (double)std::fabs(want[i]));
        maxd = std::max(maxd, (double)std::fabs(got[i] - want[i]));
    }
    s.rel_max = maxd / std::max(scale, 1e-30);
    return s;
}

// 图内 nodes 个 launch_full 节点，各写独立 ynode[n]；整图只 replay 一次，
// 节点之间不清零（测试旗子跨节点复用）。
bool run_graph(Env& e, const Spec& s, const mt::Routes& routes,
               const std::vector<float>& y_ref, int nodes, Result* out) {
    const int T = s.tokens;
    CHECK(mcSetDevice(e.rank));
    int graph_diag_rank = 0;
    if (const char* diag_rank = std::getenv("MT_GRAPH_DIAG_RANK"))
        graph_diag_rank = std::atoi(diag_rank);
    const bool dump_graph_rows =
        std::getenv("MT_GRAPH_ROWS_DIFF") != nullptr && e.rank == graph_diag_rank;
    std::vector<uint8_t> eager_rows;
    if (dump_graph_rows) {
        long long eager_ct = 0;
        CHECK(mcMemcpy(&eager_ct, &(e.d.lf->current_time[0][1]),
                       sizeof(eager_ct), mcMemcpyDeviceToHost));
        const int eager_cyc = (int)((eager_ct - 1) % M_CYCLE);
        eager_rows.assign((size_t)kTopk * kMaxTPR * kHidden * 2, 0);
        CHECK(mcMemcpy(eager_rows.data(),
                       e.d.rd->combine_rows[eager_cyc],
                       eager_rows.size(), mcMemcpyDeviceToHost));
    }
    CHECK(mcStreamBeginCapture(e.d.st, mcStreamCaptureModeThreadLocal));
    for (int n = 0; n < nodes; ++n) {
        fp8mega::MegaLaunchArgs a = mt::make_args(e.geo, e.d, T, s.win, e.d.ynode[n]);
        fp8mega::launch_full(a, e.d.st);
    }
    mcGraph_t g = nullptr;
    CHECK(mcStreamEndCapture(e.d.st, &g));
    CHECK(mcGraphInstantiate(&e.d.exec, g, nullptr, nullptr, 0));
    mcGraphDestroy(g);

    mt::reset_dev(e.geo, e.d, routes, T);
    for (int n = 0; n < nodes; ++n)
        CHECK(mcMemset(e.d.ynode[n], 0, (size_t)T * kHidden * sizeof(float)));
    MPI_Barrier(MPI_COMM_WORLD);
    CHECK(mcGraphLaunch(e.d.exec, e.d.st));
    CHECK(mcStreamSynchronize(e.d.st));
    MPI_Barrier(MPI_COMM_WORLD);

    bool pass = true;
    std::vector<float> got((size_t)T * kHidden);
    for (int n = 0; n < nodes; ++n) {
        CHECK(mcMemcpy(got.data(), e.d.ynode[n], got.size() * sizeof(float),
                       mcMemcpyDeviceToHost));
        const RelStat rs = rel_compare(got, y_ref);
        out->graph_rel = std::max(out->graph_rel, rs.rel_max);
        out->graph_pattern += rs.pattern;
        if (rs.rel_max > 1e-3 || rs.pattern > 0) {
            if (pass) {
                char msg[160];
                std::snprintf(msg, sizeof(msg), "node%d rel_max=%.3e pattern=%lld", n,
                              rs.rel_max, rs.pattern);
                detail(e, "graph", msg);
                if (dump_graph_rows && n == 0) {
                    const uint8_t* cur_dev =
                        reinterpret_cast<const uint8_t*>(e.d.rd->combine_rows[0]);
                    std::vector<uint8_t> cur_rows(eager_rows.size());
                    CHECK(mcMemcpy(cur_rows.data(), cur_dev, cur_rows.size(),
                                   mcMemcpyDeviceToHost));
                    const uint16_t* a16 = reinterpret_cast<const uint16_t*>(eager_rows.data());
                    const uint16_t* b16 = reinterpret_cast<const uint16_t*>(cur_rows.data());
                    const size_t n16 = eager_rows.size() / 2;
                    size_t first = n16;
                    long long regions = 0;
                    bool in_run = false;
                    for (size_t i = 0; i < n16; ++i) {
                        if (a16[i] != b16[i]) {
                            if (first == n16) first = i;
                            if (!in_run) { ++regions; in_run = true; }
                        } else {
                            in_run = false;
                        }
                    }
                    if (first < n16) {
                        const size_t kk = first / ((size_t)kMaxTPR * kHidden);
                        const size_t rem = first % ((size_t)kMaxTPR * kHidden);
                        const size_t stok = rem / kHidden;
                        const size_t col = rem % kHidden;
                        const int64_t dst_expert =
                            routes[e.rank][stok * (size_t)kTopk + kk];
                        std::printf("    [graph-raw] rank=%d node=%d raw字节差=%s 区域数=%lld "
                                    "首差:k=%zu,token=%zu,col=%zu,期望=%u,实际=%u,目标专家=%lld\n",
                                    e.rank, n,
                                    (cur_rows == eager_rows ? "无" : "有"), regions,
                                    kk, stok, col, (unsigned)a16[first],
                                    (unsigned)b16[first], (long long)dst_expert);
                    } else {
                        std::printf("    [graph-raw] rank=%d node=%d raw字节差=无\n",
                                    e.rank, n);
                    }
                    std::fflush(stdout);
                }
                if (std::getenv("MT_GRAPH_ROWS") != nullptr && e.rank == graph_diag_rank) {
                    long long ct01 = 0;
                    CHECK(mcSetDevice(e.rank));
                    CHECK(mcMemcpy(&ct01, &(e.d.lf->current_time[0][1]),
                                   sizeof(ct01), mcMemcpyDeviceToHost));
                    const int cyc_rows = n % M_CYCLE;
                    std::vector<uint8_t> g_rows((size_t)kTopk * T * kHidden * 2);
                    const uint8_t* g_rows_dev =
                        reinterpret_cast<const uint8_t*>(e.d.rd->combine_rows[cyc_rows]);
                    for (int k = 0; k < kTopk; ++k)
                        CHECK(mcMemcpy(g_rows.data() + (size_t)k * T * kHidden * 2,
                                       g_rows_dev + (size_t)k * kMaxTPR * kHidden * 2,
                                       (size_t)T * kHidden * 2, mcMemcpyDeviceToHost));
                    std::vector<float> g_w((size_t)T * kTopk);
                    CHECK(mcMemcpy(g_w.data(), e.d.sym + e.geo.L.topk_weights_off,
                                   g_w.size() * sizeof(float), mcMemcpyDeviceToHost));
                    std::vector<float> y_data((size_t)T * kHidden);
                    for (int t = 0; t < T; ++t)
                        ref_reduce(g_rows.data(), routes[e.rank].data(), g_w.data(), t,
                                   kTopk, e.R * kEPR, T, kHidden,
                                   y_data.data() + (size_t)t * kHidden);
                    const CheckStats data_ref = compare_f32(y_data.data(), y_ref.data(),
                                                            (int)y_data.size(), 1e-5, 1e-6);
                    const CheckStats got_data = compare_f32(got.data(), y_data.data(),
                                                            (int)y_data.size(), 1e-5, 1e-6);
                    const uint64_t epoch_expect = (uint64_t)ct01;
                    std::vector<uint64_t> sum_flags((size_t)kTopk * kMaxCombineRows *
                                                    kCombineSlicesMax, 0);
                    CHECK(mcMemcpy(sum_flags.data(),
                                   &e.d.rf->combine_sums[cyc_rows][0][0][0],
                                   sum_flags.size() * sizeof(uint64_t),
                                   mcMemcpyDeviceToHost));
                    std::printf("    [graph-rows] rank=%d node=%d ct01=%lld cyc=%d "
                                "数据对eager bad=%lld max=%.3e 最差idx=%lld "
                                "输出对数据 bad=%lld max=%.3e 最差idx=%lld\n",
                                e.rank, n, ct01, cyc_rows,
                                data_ref.bad, data_ref.max_abs, data_ref.worst,
                                got_data.bad, got_data.max_abs, got_data.worst);
                    std::fflush(stdout);
                    if (data_ref.worst >= 0) {
                        const long long dtok = data_ref.worst / kHidden;
                        std::printf("    [graph-k] rank=%d node=%d token=%lld 标志期望=%llu 逻辑槽=%d",
                                    e.rank, n, dtok, (unsigned long long)epoch_expect,
                                    kNSlice);
                        for (int k = 0; k < kTopk; ++k) {
                            const int64_t ex = routes[e.rank][(size_t)dtok * kTopk + k];
                            const float wv = g_w[(size_t)dtok * kTopk + k];
                            const uint16_t* row = reinterpret_cast<const uint16_t*>(
                                g_rows.data() + ((size_t)k * T + dtok) * kHidden * 2);
                            long long nz = 0;
                            double ssum = 0.0, asum = 0.0;
                            for (int c = 0; c < kHidden; ++c) {
                                const float v = bf16_to_f32(row[c]);
                                if (v != 0.0f) ++nz;
                                ssum += v;
                                asum += std::fabs(v);
                            }
                            std::printf(" [k%d:专家=%lld,权重=%.6g,非零=%lld,和=%.6g,绝和=%.6g,标志=",
                                        k, (long long)ex, (double)wv, nz, ssum, asum);
                            for (int s2 = 0; s2 < kNSlice && s2 < kCombineSlicesMax; ++s2)
                                std::printf("%llu,", (unsigned long long)sum_flags
                                            [((size_t)k * kMaxCombineRows + dtok) *
                                             kCombineSlicesMax + s2]);
                            std::printf("]");
                        }
                        std::printf("\n");
                        std::fflush(stdout);
                    }
                }
                if (std::getenv("MT_GRAPH_DIAG") != nullptr) {
                    double scale = 0.0;
                    for (size_t i = 0; i < y_ref.size(); ++i)
                        if (std::isfinite(y_ref[i]))
                            scale = std::max(scale, (double)std::fabs(y_ref[i]));
                    const double tol = 1e-3 * std::max(scale, 1e-30);
                    size_t first_bad = got.size(), worst = 0;
                    double maxd = 0.0;
                    long long bad = 0;
                    for (size_t i = 0; i < got.size(); ++i) {
                        if (!std::isfinite(got[i]) || !std::isfinite(y_ref[i]))
                            continue;
                        const double d = std::fabs((double)got[i] - (double)y_ref[i]);
                        if (d > tol) {
                            ++bad;
                            if (first_bad == got.size()) first_bad = i;
                            if (d > maxd) { maxd = d; worst = i; }
                        }
                    }
                    long long seg[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    size_t last_bad = 0;
                    if (first_bad < got.size()) {
                        const size_t ftok = first_bad / kHidden;
                        const size_t segw = kHidden / 8;
                        for (size_t i = ftok * kHidden; i < (ftok + 1) * kHidden; ++i) {
                            if (!std::isfinite(got[i]) || !std::isfinite(y_ref[i]))
                                continue;
                            if (std::fabs((double)got[i] - (double)y_ref[i]) > tol) {
                                ++seg[(i % kHidden) / segw];
                                last_bad = i;
                            }
                        }
                    }
                    std::printf("    [graph-diag] rank=%d node=%d n=%zu bad=%lld tol=%.3e "
                                "first=%zu(token=%zu,col=%zu) worst=%zu(token=%zu,col=%zu,"
                                "got=%.6g,want=%.6g) last=%zu seg=[%lld,%lld,%lld,%lld,"
                                "%lld,%lld,%lld,%lld]\n",
                                e.rank, n, got.size(), bad, tol,
                                first_bad, first_bad / kHidden, first_bad % kHidden,
                                worst, worst / kHidden, worst % kHidden,
                                (double)got[worst], (double)y_ref[worst], last_bad,
                                seg[0], seg[1], seg[2], seg[3],
                                seg[4], seg[5], seg[6], seg[7]);
                    std::fflush(stdout);
                }
            }
            pass = false;
        }
    }
    mcGraphExecDestroy(e.d.exec);
    e.d.exec = nullptr;
    return pass;
}

// --------------------------------------------------------- 单用例 --
void print_row(const Result& r, const Flags& f) {
    auto tag = [](bool ok) { return ok ? "PASS" : "FAIL"; };
    std::printf("[%s] L0=%s", r.label.c_str(), f.l0 ? tag(r.l0) : "-");
    std::printf(" L1L2=%s",
                f.eager ? tag(r.l12) : "-");
    std::printf(" REDUCE=%s", f.eager ? tag(r.red) : "-");
    if (f.eager) std::printf("(max_abs=%.2e)", r.red_max_abs);
    if (f.graph > 0)
        std::printf(" graph×%d=%s(rel=%.2e,pat=%lld)", f.graph, tag(r.graph), r.graph_rel,
                    r.graph_pattern);
    std::printf("  %.0f ms\n", r.ms);
    std::fflush(stdout);
}

bool run_spec(Env& e, const Spec& s, const Flags& f) {
    Result r;
    r.label = s.label;
    const auto t0 = std::chrono::steady_clock::now();
    mt::Routes routes;
    std::string err;
    if (!mt::spec_routes(s, e.R, kTopk, &routes, &err)) {
        if (e.rank == 0) std::fprintf(stderr, "[check] %s\n", err.c_str());
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    if (f.l0) {
        const bool ok = run_l0(e, s, routes);
        r.l0 = agree(ok);
    }
    std::vector<float> y;
    if (f.eager || f.graph > 0) {
        run_eager(e, s, routes, &y);
        // [DUMP] MT_DUMP_Y=<dir>：把本 rank 的 eager 输出 y（tokens x H float）写入
        //   <dir>/y_<label>_r<rank>.bin，供 torch 对比脚本逐元素比较。只读输出，不改数值路径。
        if (const char* dd = std::getenv("MT_DUMP_Y")) {
            char path[512];
            std::snprintf(path, sizeof(path), "%s/y_%s_r%d.bin", dd,
                          s.label.c_str(), e.d.rank);
            if (FILE* fp = std::fopen(path, "wb")) {
                std::fwrite(y.data(), sizeof(float), y.size(), fp);
                std::fclose(fp);
            }
            // 同时导出本 rank 的路由（int64，tokens*topk，行主序 [token][k]），供 torch 对拍复现输入
            std::snprintf(path, sizeof(path), "%s/route_%s_r%d.bin", dd,
                          s.label.c_str(), e.d.rank);
            if (FILE* fr = std::fopen(path, "wb")) {
                const auto& rr = routes[(size_t)e.d.rank];
                std::fwrite(rr.data(), sizeof(int64_t), rr.size(), fr);
                std::fclose(fr);
            }
        }
        if (f.eager) {
            r.l12 = agree(check_l1l2(e, s, routes));
            const CheckStats red = check_reduce(e, s, routes, y);
            r.red = agree(red.pass());
            r.red_max_abs = agree_max(red.max_abs);
        }
    }
    if (f.graph > 0) {
        Result gr;
        const bool ok = run_graph(e, s, routes, y, f.graph, &gr);
        r.graph = agree(ok);
        r.graph_rel = agree_max(gr.graph_rel);
        r.graph_pattern = agree_sum(gr.graph_pattern);
    }
    r.ms = std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0).count();
    if (e.rank == 0) print_row(r, f);
    const bool pass = (!f.l0 || r.l0) && (!f.eager || (r.l12 && r.red)) &&
                      (f.graph == 0 || r.graph);
    return pass;
}

void usage(const char* p) {
    std::printf(
        "用法: %s --smoke | --all  [选项]（经 scripts/run.sh check 启动，R=32）\n"
        "  --smoke           冒烟：均衡 16 + TK launch %d（前序/eager/graph×1/graph×N）\n"
        "  --all             全量：均衡 TPR 表 + 全部 TK 合法 launch（每个含前序/eager/graph×N）\n"
        "  --tk-info F       TK_INFO 日志（缺省 %s）\n"
        "  --graph-nodes N   graph 图内节点数（缺省 50）\n"
        "  --check-rows N    L1/L2 核验行数（缺省 16）\n",
        p, kCase0, mt::kDefaultTkLog);
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, R = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &R);
    std::setvbuf(stdout, nullptr, _IOLBF, 0);   // 超时被杀时保留已输出的阶段日志

    bool smoke = false, all = false;
    std::vector<int> only;             // --only：只跑指定 TK launch（调试用）
    std::vector<int> tprs;             // --tprs：覆盖 --all 的均衡 TPR 列表（调试用）
    std::string tk_path = mt::kDefaultTkLog;
    int graph_nodes = 50, check_rows = 16;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--smoke") smoke = true;
        else if (a == "--all") all = true;
        else if (a == "--tk-info") tk_path = val();
        else if (a == "--only") only = mt::parse_csv_ints(val());
        else if (a == "--tprs") tprs = mt::parse_csv_ints(val());
        else if (a == "--graph-nodes") graph_nodes = std::atoi(val().c_str());
        else if (a == "--check-rows") check_rows = std::atoi(val().c_str());
        else if (a == "-h" || a == "--help") {
            if (rank == 0) usage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) std::fprintf(stderr, "未知参数: %s\n", a.c_str());
            MPI_Finalize();
            return 2;
        }
    }
    if (smoke == all) {
        if (rank == 0) {
            std::fprintf(stderr, "请指定且只指定一个: --smoke 或 --all\n");
            usage(argv[0]);
        }
        MPI_Finalize();
        return 2;
    }
    if (graph_nodes < 1 || check_rows < 1) {
        if (rank == 0) std::fprintf(stderr, "--graph-nodes / --check-rows 须 >= 1\n");
        MPI_Finalize();
        return 2;
    }

    // ---- 用例列表（每进程独立解析，一致） ----
    std::string err;
    std::vector<std::pair<Spec, Flags>> plan;
    std::vector<mt::TkLaunch> tk;
    if (smoke) {
        plan.push_back({mt::balanced_spec(R, 16), Flags{true, true, 0}});
        plan.push_back({mt::balanced_spec(R, 16), Flags{false, false, 1}});
    }
    if (smoke || all) {
        if (!mt::load_tk_launches(tk_path, R, kTopk, &tk, &err)) {
            if (rank == 0) std::fprintf(stderr, "[check] %s\n", err.c_str());
            MPI_Finalize();
            return 2;
        }
    }
    if (smoke) {
        const mt::TkLaunch* c0 = nullptr;
        for (const auto& t : tk)
            if (t.id == kCase0) c0 = &t;
        if (!c0) {
            if (rank == 0)
                std::fprintf(stderr, "[check] TK_INFO 中无 launch=%d（R=%d）\n", kCase0, R);
            MPI_Finalize();
            return 2;
        }
        plan.push_back({mt::tk_spec(*c0), Flags{true, true, 1}});
        plan.push_back({mt::tk_spec(*c0), Flags{false, true, graph_nodes}});
        plan.push_back({mt::balanced_spec(R, 16), Flags{false, true, graph_nodes}});
    }
    if (all) {
        // --tprs 子集：仍自动包含 TPR=512，保证 max_tokens=512 的全局布局与全量一致
        std::vector<int> bal = tprs.empty()
            ? std::vector<int>(std::begin(kBalancedTprs), std::end(kBalancedTprs))
            : tprs;
        if (!tprs.empty() && std::find(bal.begin(), bal.end(), 512) == bal.end())
            bal.push_back(512);
        for (int t : bal)
            plan.push_back({mt::balanced_spec(R, t), Flags{true, true, graph_nodes}});
        for (const auto& t : tk)
            plan.push_back({mt::tk_spec(t), Flags{true, true, graph_nodes}});
    }
    if (!only.empty()) {
        std::vector<std::pair<Spec, Flags>> keep;
        for (const auto& p : plan) {
            const bool pick = p.first.kind == 1 &&
                              std::find(only.begin(), only.end(), p.first.launch_id) != only.end();
            // 保留 TPR=512 均衡用例：它决定 max_tokens=512 的全局布局（复现 --all 的布局）
            const bool pin = p.first.kind == 0 && p.first.tokens == 512;
            // --tprs 指定的均衡用例同样保留（子集验证用）
            const bool bal = p.first.kind == 0 && !tprs.empty() &&
                             std::find(tprs.begin(), tprs.end(), p.first.tokens) != tprs.end();
            if (pick || pin || bal) keep.push_back(p);
        }
        plan.swap(keep);
    }
    if (plan.empty()) {
        if (rank == 0) std::fprintf(stderr, "[check] 无用例\n");
        MPI_Finalize();
        return 2;
    }

    // ---- 全局布局 ----
    int max_tokens = 16;
    for (const auto& p : plan) max_tokens = std::max(max_tokens, p.first.tokens);
    int win_global = 0;
    const char* fwin = std::getenv("MT_FORCE_WIN");   // 调试：强制某个 win，用于最小复现
    for (auto& p : plan) {
        p.first.win = mt::choose_win_cap_counts(p.first.per_expert, max_tokens);
        // [MT_FORCE_WIN] 扩展：原仅 TK 用例；对拍需要均衡用例指定同一 win，故两类都生效
        if (fwin && p.first.kind == 1) p.first.win = std::atoi(fwin);
        win_global = std::max(win_global, p.first.win);
    }
    Env e;
    e.R = R;
    e.rank = rank;
    e.check_rows = check_rows;
    e.geo = mt::make_geo(R, max_tokens, win_global);
    if (e.geo.smem > 227 * 1024) {
        if (rank == 0) std::fprintf(stderr, "[check] smem=%lld > 227KB\n", (long long)e.geo.smem);
        MPI_Finalize();
        return 2;
    }

    const auto t_start = std::chrono::steady_clock::now();
    CHECK(mcSetDevice(rank));
    mt::alloc_dev(e.d, e.geo, rank, max_tokens, graph_nodes);
    const auto t_alloc = std::chrono::steady_clock::now();
    mt::exchange_ipc(e.d, R);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto t_ipc = std::chrono::steady_clock::now();
    if (rank == 0)
        std::printf("check R=%d mode=%s cases=%zu max_tokens=%d win_global=%d graph_nodes=%d\n",
                    R, smoke ? "smoke" : "all", plan.size(), max_tokens, win_global,
                    graph_nodes);

    // 冷启动预热（不计入结果）
    const auto t_warm0 = std::chrono::steady_clock::now();
    {
        Spec warm = mt::balanced_spec(R, 16);
        warm.win = mt::choose_win_cap_counts(warm.per_expert, e.geo.max_tokens);
        mt::Routes wr = mt::balanced_routes(R, 16);
        std::vector<float> y;
        run_eager(e, warm, wr, &y);
    }
    if (rank == 0) {
        auto ms = [](std::chrono::steady_clock::time_point a,
                     std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        std::printf("[timing] alloc=%.0f ms ipc=%.0f ms warm=%.0f ms\n",
                    ms(t_start, t_alloc), ms(t_alloc, t_ipc), ms(t_warm0,
                    std::chrono::steady_clock::now()));
    }

    bool all_pass = true;
    int failed = 0;
    for (const auto& p : plan) {
        const bool ok = run_spec(e, p.first, p.second);
        if (!ok) {
            ++failed;
            all_pass = false;
        }
        if (smoke && !all_pass) break;   // 冒烟：首个失败即停止（便于定位）
        g_detail_budget = 2;
    }
    const double wall = std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - t_start).count();
    if (rank == 0)
        std::printf("[总体] %s  用例=%zu 失败=%d  耗时=%.2f s（含 setup+预热）\n",
                    all_pass ? "PASS" : "FAIL", plan.size(), failed, wall);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return all_pass ? 0 : 1;
}
