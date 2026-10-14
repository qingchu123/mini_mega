// ============================================================================
// test/mybench.cu — 每进程一张卡的 MoE mega kernel 性能基准（MPI）
//
// 启动（经 scripts/run.sh，R=32 个进程）：
//   scripts/run.sh mybench -R 32 -s 1 -tprs 1,2,4,6,8,12,18,24,30,36,42,48,54,60,66,72,78,84,90,128,256   # 均衡（默认）
//   scripts/run.sh mybench -R 32 -s 2 -top 50                              # TK 前 50 个
//   scripts/run.sh mybench -R 32 -s 2 -cases 804900,807500                 # 指定 TK launch
//
// 选项（单横杠或双横杠均可）：
//   -s 1|2          场景：1=均衡（默认）；2=TK 实录（不均衡路由）
//   -tprs L         [s1] TPR 列表（默认 1,2,4,6,8,12,18,24,30,36,42,48,54,60,66,72,78,84,90,128,256）
//   -top N          [s2] 取日志中前 N 个合法 complete launch（默认 50）
//   -cases L        [s2] 指定 launch 号列表（与 -top 互斥，优先 -cases）
//   -tk-info F      [s2] TK_INFO 日志（默认 test/70.r32_tpr_perf/0928_tk_info.log）
//   -nodes N        每张图的 kernel 节点数（默认 20）
//   -replays N      计时 replay 次数（默认 2）
//   -max-tokens M   槽容量下限（默认 512；s2 取 max(M, 用例最大 token)）
//   -quiet          只打印结果行
//
// 口径：us/kernel = 图计时(ms) * 1000 / (nodes * replays)，跨 rank 取 min/max/mean。
// ============================================================================
#include <mcr/mc_runtime.h>
#include <mpi.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "mega_common.cuh"

namespace {

#define MYCHK(call)                                                          \
    do {                                                                     \
        mcError_t e_ = (call);                                               \
        if (e_ != mcSuccess) {                                               \
            std::fprintf(stderr, "[mybench] MACA error %d at %s:%d: %s\n",   \
                         (int)e_, __FILE__, __LINE__, #call);                \
            MPI_Abort(MPI_COMM_WORLD, 1);                                    \
        }                                                                    \
    } while (0)

struct Opts {
    int scene = 1;
    std::vector<int> tprs = {1, 2, 4, 6, 8, 12, 18, 24, 30, 36,
                             42, 48, 54, 60, 66, 72, 78, 84, 90, 128, 256};
    int top = 50;
    std::vector<int> cases;
    std::string tk_info = mt::kDefaultTkLog;
    int nodes = 20;
    int replays = 2;
    int max_tokens = 512;
    bool quiet = false;
};

void usage(const char* p) {
    std::fprintf(stderr,
                 "用法: %s [选项]（须用 scripts/run.sh mybench -R R 启动）\n"
                 "  -s 1|2        场景：1=均衡（默认）；2=TK 实录\n"
                 "  -tprs L       [s1] TPR 列表（默认 1,2,4,6,8,12,18,24,30,36,42,48,54,60,66,72,78,84,90,128,256）\n"
                 "  -top N        [s2] 前 N 个合法 launch（默认 50）\n"
                 "  -cases L      [s2] 指定 launch 号列表\n"
                 "  -tk-info F    [s2] TK_INFO 日志\n"
                 "  -nodes N      图节点数（默认 20）\n"
                 "  -replays N    replay 次数（默认 2）\n"
                 "  -max-tokens M 槽容量下限（默认 512）\n"
                 "  -quiet        只打印结果行\n",
                 p);
}

std::vector<int> parse_ints(const std::string& s) {
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

bool parse_args(int argc, char** argv, Opts* o) {
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        while (!a.empty() && a[0] == '-') a.erase(0, 1);   // -x / --x 等价
        auto val = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "s") o->scene = std::atoi(val().c_str());
        else if (a == "tprs") o->tprs = parse_ints(val());
        else if (a == "top") o->top = std::atoi(val().c_str());
        else if (a == "cases") o->cases = parse_ints(val());
        else if (a == "tk-info") o->tk_info = val();
        else if (a == "nodes") o->nodes = std::atoi(val().c_str());
        else if (a == "replays") o->replays = std::atoi(val().c_str());
        else if (a == "max-tokens") o->max_tokens = std::atoi(val().c_str());
        else if (a == "quiet") o->quiet = true;
        else if (a == "h" || a == "help") { usage(argv[0]); return false; }
        else {
            std::fprintf(stderr, "[mybench] 未知参数: %s\n", argv[i]);
            return false;
        }
    }
    if (o->scene != 1 && o->scene != 2) {
        std::fprintf(stderr, "[mybench] -s 只能是 1 或 2\n");
        return false;
    }
    if (o->nodes < 1) o->nodes = 20;
    if (o->replays < 1) o->replays = 2;
    if (o->top < 1) o->top = 1;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank = 0, R = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &R);
    if (rank == 0 && std::getenv("MT_DUMP_RES")) fp8mega::dump_kernel_resources();

    Opts o;
    if (!parse_args(argc, argv, &o)) {
        MPI_Finalize();
        return 2;
    }

    // ---- 构造用例（每个进程独立解析，结果确定、一致；只存计数） ----
    std::vector<mt::Spec> specs;
    std::string err;
    if (o.scene == 1) {
        for (int t : o.tprs)
            if (t >= 1) specs.push_back(mt::balanced_spec(R, t));
    } else {
        std::vector<mt::TkLaunch> all;
        if (!mt::load_tk_launches(o.tk_info, R, kTopk, &all, &err)) {
            if (rank == 0) std::fprintf(stderr, "[mybench] %s\n", err.c_str());
            MPI_Finalize();
            return 2;
        }
        std::vector<const mt::TkLaunch*> pick;
        if (!o.cases.empty()) {
            for (int id : o.cases) {
                const mt::TkLaunch* hit = nullptr;
                for (const auto& t : all)
                    if (t.id == id) hit = &t;
                if (!hit) {
                    if (rank == 0)
                        std::fprintf(stderr, "[mybench] TK_INFO 中无合法 launch=%d\n", id);
                    MPI_Finalize();
                    return 2;
                }
                pick.push_back(hit);
            }
        } else {
            for (size_t i = 0; i < all.size() && (int)i < o.top; ++i)
                pick.push_back(&all[i]);
        }
        for (const mt::TkLaunch* t : pick) specs.push_back(mt::tk_spec(*t));
    }
    if (specs.empty()) {
        if (rank == 0) std::fprintf(stderr, "[mybench] 无可运行用例\n");
        MPI_Finalize();
        return 2;
    }

    // ---- 全局布局：最大 token 与最大窗口 ----
    int max_tokens = o.max_tokens;
    for (const auto& s : specs) max_tokens = std::max(max_tokens, s.tokens);
    int win_global = 0;
    // [实验] MT_FORCE_WIN=N：强制所有用例的窗口容量为 N（扫固定 win 用）。
    const char* fwin = std::getenv("MT_FORCE_WIN");
    const int fwin_val = fwin ? std::atoi(fwin) : 0;
    for (auto& s : specs) {
        s.win = (fwin_val > 0)
                    ? fwin_val
                    : mt::choose_win_cap_counts(s.per_expert, max_tokens);
        win_global = std::max(win_global, s.win);
    }
    const mt::Geo geo = mt::make_geo(R, max_tokens, win_global);
    if (geo.smem > 227 * 1024) {
        if (rank == 0) std::fprintf(stderr, "[mybench] smem=%lld > 227KB\n", (long long)geo.smem);
        MPI_Finalize();
        return 2;
    }
    if (rank == 0 && !o.quiet) {
        std::printf("mybench(MPI) R=%d topk=%d scene=%d cases=%zu nodes=%d replays=%d "
                    "max_tokens=%d win_global=%d\n",
                    R, kTopk, o.scene, specs.size(), o.nodes, o.replays, max_tokens,
                    win_global);
        if (o.scene == 1)
            std::printf("%-4s %-12s %10s %10s %10s\n", "R", "case", "us_min", "us_max",
                        "us_mean");
        else
            std::printf("%-4s %-16s %7s %6s %9s %10s %10s %10s\n", "R", "case", "tokens",
                        "win", "max_exp", "us_min", "us_max", "us_mean");
    }

    // ---- 每 rank 资源 + IPC（一次） ----
    MYCHK(mcSetDevice(rank));
    mt::Dev d;
    mt::alloc_dev(d, geo, rank, max_tokens, 0);
    mt::exchange_ipc(d, R);
    MPI_Barrier(MPI_COMM_WORLD);

    // ---- 逐用例 ----
    for (size_t i = 0; i < specs.size(); ++i) {
        const mt::Spec& c = specs[i];
        mt::Routes routes;
        if (!mt::spec_routes(c, R, kTopk, &routes, &err)) {
            if (rank == 0) std::fprintf(stderr, "[mybench] %s\n", err.c_str());
            MPI_Abort(MPI_COMM_WORLD, 2);
        }
        const int tokens = c.tokens;
        const int win = c.win;

        mt::reset_dev(geo, d, routes, tokens);
        fp8mega::MegaLaunchArgs a = mt::make_args(geo, d, tokens, win, d.y);

        // eager 预热（冷启动后首次发射，不计时）
        MPI_Barrier(MPI_COMM_WORLD);
        fp8mega::launch_full(a, d.st);
        MYCHK(mcStreamSynchronize(d.st));
        MPI_Barrier(MPI_COMM_WORLD);

        // graph capture：nodes 个同构节点
        mcGraph_t g = nullptr;
        mcGraphExec_t ex = nullptr;
        if (mcStreamBeginCapture(d.st, mcStreamCaptureModeThreadLocal) != mcSuccess) {
            std::fprintf(stderr, "rank%d BeginCapture 失败\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int k = 0; k < o.nodes; ++k) fp8mega::launch_full(a, d.st);
        MYCHK(mcStreamEndCapture(d.st, &g));
        MYCHK(mcGraphInstantiate(&ex, g, nullptr, nullptr, 0));
        mcGraphDestroy(g);

        // warmup
        for (int k = 0; k < 2; ++k) MYCHK(mcGraphLaunch(ex, d.st));
        MYCHK(mcStreamSynchronize(d.st));
        MPI_Barrier(MPI_COMM_WORLD);

        // measure
        mcEvent_t e0 = nullptr, e1 = nullptr;
        MYCHK(mcEventCreate(&e0));
        MYCHK(mcEventCreate(&e1));
        MYCHK(mcEventRecord(e0, d.st));
        for (int k = 0; k < o.replays; ++k) MYCHK(mcGraphLaunch(ex, d.st));
        MYCHK(mcEventRecord(e1, d.st));
        MYCHK(mcStreamSynchronize(d.st));
        float ms = 0.f;
        MYCHK(mcEventElapsedTime(&ms, e0, e1));
        const double us = (double)ms * 1000.0 / (double)(o.nodes * o.replays);
        mcEventDestroy(e0);
        mcEventDestroy(e1);
        mcGraphExecDestroy(ex);

        // [诊断锚点] 采样对齐用：本用例 graph 计时结束的 epoch 秒（立即 flush）
        if (rank == 0) {
            const double te = std::chrono::duration<double>(
                                  std::chrono::system_clock::now().time_since_epoch())
                                  .count();
            std::printf("[anchor] TPR=%d t_meas_end=%.3f us=%.2f\n", tokens, te, us);
            std::fflush(stdout);
        }

        // [诊断] MYBENCH_PHASE=1：计时后做一次 reset + 单次 eager launch，读取 kernel 内置相位 tick。
        // 注意：route..reduce 与 chain 只来自 block0 的时间线（kernel 结束要等所有 block），
        // 真正的关键路径需看逐 block 数据（MYBENCH_PHASE_RANK=<rank>，需 -DR24_DIAG_PHASE=1）。
        if (std::getenv("MYBENCH_PHASE") != nullptr) {
            // [诊断·第三 bank] 内核首尾/循环入口出口时间戳（需 R24_DIAG_PHASE=1 构建）。
            // 在 reset 之前调用 = 取「最后一次 graph replay」的时间线；eager 后调用 = eager 的。
#if R24_DIAG_PHASE
            auto dump_span = [&](const char* tag, double wall_us) {
                unsigned long long gg[4] = {0, 0, 0, 0};
                std::vector<unsigned long long> tg((size_t)kGrid * 8, 0ull);
                if (mcMemcpy(gg, d.lf->pre_ticks + 8, sizeof(gg),
                             mcMemcpyDeviceToHost) != mcSuccess)
                    return;
                if (mcMemcpy(tg.data(), d.lf->pre_ticks + (size_t)kGrid * 16,
                             tg.size() * sizeof(unsigned long long),
                             mcMemcpyDeviceToHost) != mcSuccess)
                    return;
                if (rank != 0) return;
                const unsigned long long period =
                    (gg[1] != 0 && gg[0] > gg[1]) ? gg[0] - gg[1] : 0;
                unsigned long long span_max = 0, smin = ~0ull, emax = 0;
                int bi = -1;
                for (int b = 0; b < kGrid; ++b) {
                    const unsigned long long* q = &tg[(size_t)b * 8];
                    if (q[0] == 0 || q[1] < q[0] || q[1] > (1ull << 60)) continue;
                    if (q[1] - q[0] > span_max) { span_max = q[1] - q[0]; bi = b; }
                    smin = std::min(smin, q[0]);
                    emax = std::max(emax, q[1]);
                }
                if (bi < 0) {
                    std::printf("[span-%s] TPR=%d 无第三 bank 数据（需 -DR24_DIAG_PHASE=1 构建）\n",
                                tag, tokens);
                    std::fflush(stdout);
                    return;
                }
                const unsigned long long* qb = &tg[(size_t)bi * 8];
                const unsigned long long* q0 = &tg[0];
                auto dd = [](unsigned long long hi, unsigned long long lo) {
                    return hi >= lo ? hi - lo : 0ull;
                };
                std::printf("[span-%s] TPR=%d wall=%.2fus clk=%.0fc/us period=%llu | "
                            "total=%llu kspan=%llu(b%d s+%llu in=%llu loop=%llu post=%llu) "
                            "| b0span=%llu(in=%llu loop=%llu post=%llu)\n",
                            tag, tokens, wall_us,
                            (period && wall_us > 0) ? (double)period / wall_us : 0.0,
                            period, emax - smin, span_max, bi, dd(qb[0], smin),
                            dd(qb[2], qb[0]), dd(qb[3], qb[2]), dd(qb[1], qb[3]),
                            dd(q0[1], q0[0]), dd(q0[2], q0[0]), dd(q0[3], q0[2]),
                            dd(q0[1], q0[3]));
                // [诊断·headroom 分解] reduce 时间戳（第三 bank block0 区 +4..+6；
                // 无 reduce 发射（MX_SKIP_REDUCE）或未编译时恒 0）
                if (q0[4] != 0) {
                    const unsigned long long r_end = q0[6] != 0 ? q0[6] : q0[5];
                    const unsigned long long next_start = period ? gg[0] + period : 0;
                    std::printf("[spanR-%s] TPR=%d gap1=%llu rblk0=%llu rspan=%llu gap2=%llu"
                                " wend=%lld\n",
                                tag, tokens, dd(q0[4], emax), dd(q0[5], q0[4]),
                                dd(r_end, q0[4]), next_start ? dd(next_start, r_end) : 0,
                                gg[3] ? (long long)(gg[3] - emax) : 0LL);
                } else {
                    std::printf("[spanR-%s] TPR=%d reduce 未发射（时间戳为 0）\n", tag,
                                tokens);
                }
                // [诊断·任务粒度] 第四 bank：单任务 span 分布（跨全部发射累计，
                // 需 -DR24_DIAG_PHASE 构建；n = 该 block 累计任务数）
                {
                    std::vector<unsigned long long> t4((size_t)kGrid * 8, 0ull);
                    if (mcMemcpy(t4.data(), d.lf->pre_ticks + (size_t)kGrid * 24,
                                 t4.size() * sizeof(unsigned long long),
                                 mcMemcpyDeviceToHost) == mcSuccess) {
                        auto pr4 = [&](int b) {
                            const unsigned long long* q = &t4[(size_t)b * 8];
                            const unsigned long long n = q[3];
                            std::printf(
                                "[spanT-%s] TPR=%d b=%02d n=%llu tmin=%llu tmax=%llu tavg=%llu\n",
                                tag, tokens, b, n, q[0], q[1],
                                n ? q[2] / n : 0ull);
                        };
                        pr4(0);
                        if (bi != 0) pr4(bi);
                    }
                }
                // [诊断·reduce 逐块] 第五 bank：最晚 reduce 块的 token/自旋分解 +
                // 逐切片全局自旋（累计值；waiters=80 发射 × 每块 ~32 等待线程）
                {
                    std::vector<unsigned long long> t5(256 * 8 + 64, 0ull);
                    if (mcMemcpy(t5.data(), d.lf->pre_ticks + (size_t)kGrid * 32,
                                 t5.size() * sizeof(unsigned long long),
                                 mcMemcpyDeviceToHost) == mcSuccess) {
                        unsigned long long smin5 = ~0ull, emax5 = 0;
                        int bmax = -1, nblk = 0;
                        for (int b = 0; b < 256; ++b) {
                            const unsigned long long* q = &t5[(size_t)b * 8];
                            if (q[0] == 0 || q[1] < q[0] || q[1] > (1ull << 60))
                                continue;
                            ++nblk;
                            smin5 = std::min(smin5, q[0]);
                            if (q[1] > emax5) { emax5 = q[1]; bmax = b; }
                        }
                        if (bmax >= 0) {
                            int c50 = 0, c100 = 0, c200 = 0;
                            for (int b = 0; b < 256; ++b) {
                                const unsigned long long* q = &t5[(size_t)b * 8];
                                if (q[0] == 0 || q[1] < q[0] || q[1] > (1ull << 60))
                                    continue;
                                if (q[1] - smin5 > 50000) ++c50;
                                if (q[1] - smin5 > 100000) ++c100;
                                if (q[1] - smin5 > 200000) ++c200;
                            }
                            const unsigned long long* q = &t5[(size_t)bmax * 8];
                            std::printf(
                                "[spanR2-%s] TPR=%d nblk=%d lastblk=%d token=%llu "
                                "start+=%llu end+=%llu spin_sum=%llu waiters=%llu "
                                "long_w=%llu top=(%llu,slot=%llu) | "
                                "late50=%d late100=%d late200=%d\n",
                                tag, tokens, nblk, bmax, q[4], q[0] - smin5,
                                q[1] - smin5, q[2], q[3], q[6], q[5] >> 8,
                                q[5] & 255ull, c50, c100, c200);
                            const unsigned long long* g5 = &t5[256 * 8];
                            std::printf(
                                "[spanR3-%s] TPR=%d 逐切片自旋/等待: s0=%llu/%llu "
                                "s1=%llu/%llu s2=%llu/%llu s3=%llu/%llu\n",
                                tag, tokens, g5[0], g5[4], g5[1], g5[5], g5[2],
                                g5[6], g5[3], g5[7]);
                            // [spanR4] 逐 k 标记到达时刻（相对 reduce 起点，
                            //   取该 k 下 4 个 slice 的最晚）
                            const unsigned long long* g6 = &t5[256 * 8 + 16];
                            std::printf("[spanR4-%s] TPR=%d 标记到达(k=topk槽):", tag,
                                        tokens);
                            for (int k = 0; k < 8; ++k) {
                                unsigned long long arr = 0;
                                for (int s = 0; s < 4; ++s)
                                    arr = std::max(arr, g6[k * 4 + s]);
                                if (arr) std::printf(" k%d=+%llu", k, dd(arr, q0[4]));
                            }
                            std::printf("\n");
                        }
                    }
                }
                std::fflush(stdout);
            };
#endif  // R24_DIAG_PHASE（dump_span：读第三/四/五 bank，仅诊断构建有数据）
            // [诊断] 逐 rank 相位表：graph 稳态（reset 前读取 = 最后一次 replay）与
            // eager（单发后读取）分开打印。行：route publish xbar setup dispatch
            //   L1 L2tail compute reduce chain | trL1max trL2max maxtasks
            auto phase_tables = [&](const char* tag, double wall_us) {
                unsigned long long tp[16] = {0};
                MYCHK(mcMemcpy(tp, d.lf->pre_ticks, sizeof(tp), mcMemcpyDeviceToHost));
                std::vector<unsigned long long> tr((size_t)kGrid * 8, 0ull);
                MYCHK(mcMemcpy(tr.data(), d.lf->pre_ticks + (size_t)kGrid * 8,
                               tr.size() * sizeof(unsigned long long),
                               mcMemcpyDeviceToHost));
                int proc[kGrid] = {0};
                MYCHK(mcMemcpy(proc, d.lf->out_processed, sizeof(proc),
                               mcMemcpyDeviceToHost));
                auto cyc = [&](int hi, int lo) {
                    return (double)((long long)tp[hi] - (long long)tp[lo]);
                };
                double rowv[12];
                rowv[0] = cyc(0, 6);
                rowv[1] = cyc(1, 0);
                rowv[2] = cyc(2, 1);
                rowv[3] = cyc(3, 2);
                rowv[4] = cyc(4, 3);
                rowv[5] = cyc(5, 4);
                rowv[6] = cyc(7, 5);
                rowv[7] = cyc(7, 6);
                rowv[8] = cyc(15, 7);
                rowv[9] = cyc(15, 6);
                unsigned long long l1max = 0, l2max = 0;
                for (int b = 0; b < kGrid; ++b) {
                    l1max = std::max(l1max, tr[(size_t)b * 8 + 0]);
                    l2max = std::max(l2max, tr[(size_t)b * 8 + 1]);
                }
                rowv[10] = (double)l1max;
                rowv[11] = (double)l2max;
                int maxtasks = 0;
                for (int b = 0; b < kGrid; ++b) maxtasks = std::max(maxtasks, proc[b]);
                // 指定 rank 的逐 block 明细（需 R24_DIAG_PHASE=1；L1/L2 为 atomicAdd 累计周期）
                if (const char* pr = std::getenv("MYBENCH_PHASE_RANK")) {
                    if (std::atoi(pr) == rank) {
                        std::printf("[phase-blk-%s] rank=%d wall=%.1fus block: L1 L2 pull move wait pw loop tail tasks\n",
                                    tag, rank, wall_us);
                        for (int b = 0; b < kGrid; ++b)
                            std::printf("[phase-blk-%s] b=%02d %10llu %10llu %10llu %10llu %10llu %10llu %10llu %10llu %d\n",
                                        tag, b, tr[(size_t)b * 8 + 0], tr[(size_t)b * 8 + 1],
                                        tr[(size_t)b * 8 + 2], tr[(size_t)b * 8 + 5],
                                        tr[(size_t)b * 8 + 6], tr[(size_t)b * 8 + 3],
                                        tr[(size_t)b * 8 + 4], tr[(size_t)b * 8 + 7], proc[b]);
                        unsigned long long m0 = 0, m1 = 0, m2 = 0, m3 = 0, m4 = 0, m5 = 0,
                                           m6 = 0, m7 = 0;
                        unsigned long long s2 = 0, s3 = 0, s5 = 0, s6 = 0, s7 = 0;
                        for (int b = 0; b < kGrid; ++b) {
                            const unsigned long long* q = &tr[(size_t)b * 8];
                            m0 = std::max(m0, q[0]); m1 = std::max(m1, q[1]);
                            m2 = std::max(m2, q[2]); m3 = std::max(m3, q[3]);
                            m4 = std::max(m4, q[4]); m5 = std::max(m5, q[5]);
                            m6 = std::max(m6, q[6]); m7 = std::max(m7, q[7]);
                            s2 += q[2]; s3 += q[3]; s5 += q[5]; s6 += q[6]; s7 += q[7];
                        }
                        std::printf("[phase-sum-%s] max: L1=%llu L2=%llu pull=%llu move=%llu wait=%llu "
                                    "pw=%llu loop=%llu tail=%llu | sum pull=%llu move=%llu wait=%llu "
                                    "pw=%llu tail=%llu\n", tag, m0, m1, m2, m5, m6, m3, m4, m7,
                                    s2, s5, s6, s3, s7);
                        std::fflush(stdout);
                    }
                }
                std::vector<double> allrow((size_t)R * 13, 0.0);
                double sendrow[13];
                for (int i = 0; i < 12; ++i) sendrow[i] = rowv[i];
                sendrow[12] = (double)maxtasks;
                MPI_Gather(sendrow, 13, MPI_DOUBLE, allrow.data(), 13, MPI_DOUBLE, 0,
                           MPI_COMM_WORLD);
                if (rank == 0) {
                    std::printf("[phase-%s] 单位=GPU 周期（clock64，block0 时间线）；trL1max/trL2max 需 R24_DIAG_PHASE=1\n",
                                tag);
                    std::printf("%-4s %9s %8s %8s %8s %8s %8s %8s %9s %8s %9s %9s %9s\n", "rank",
                                "route", "publish", "xbar", "setup", "dispatch", "L1", "L2tail",
                                "compute", "reduce", "chain", "trL1max", "trL2max");
                    for (int r = 0; r < R; ++r) {
                        const double* q = allrow.data() + (size_t)r * 13;
                        std::printf("%-4d %9.0f %8.0f %8.0f %8.0f %8.0f %8.0f %8.0f %9.0f %8.0f %9.0f %9.0f %9.0f  tasks=%d\n",
                                    r, q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7], q[8],
                                    q[9], q[10], q[11], (int)q[12]);
                    }
                    std::fflush(stdout);
                }
            };
            phase_tables("graph", us);
#if R24_DIAG_PHASE
            dump_span("graph", us);
#endif
            mt::reset_dev(geo, d, routes, tokens);
            MPI_Barrier(MPI_COMM_WORLD);
            auto tw0 = std::chrono::steady_clock::now();
            fp8mega::launch_full(a, d.st);
            MYCHK(mcStreamSynchronize(d.st));
            auto tw1 = std::chrono::steady_clock::now();
            MPI_Barrier(MPI_COMM_WORLD);
            const double eager_us =
                std::chrono::duration<double, std::micro>(tw1 - tw0).count();
            phase_tables("eager", eager_us);
#if R24_DIAG_PHASE
            dump_span("eager", eager_us);
#endif
        }

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
            const double vmean = n ? vsum / n : -1.0;
            if (o.scene == 1)
                std::printf("%-4d %-12s %10.2f %10.2f %10.2f\n", R, c.label.c_str(), vmin,
                            vmax, vmean);
            else
                std::printf("%-4d %-16s %7d %6d %9d %10.2f %10.2f %10.2f\n", R,
                            c.label.c_str(), c.tokens, win, c.max_expert, vmin, vmax, vmean);
            std::fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
