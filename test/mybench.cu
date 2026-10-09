// ============================================================================
// test/mybench.cu — 每进程一张卡的 MoE mega kernel 性能基准（MPI）
//
// 启动（经 scripts/run.sh，R=32 个进程）：
//   scripts/run.sh mybench -R 32 -s 1 -tprs 1,4,16,32,33,64,128,256,512   # 均衡（默认）
//   scripts/run.sh mybench -R 32 -s 2 -top 50                              # TK 前 50 个
//   scripts/run.sh mybench -R 32 -s 2 -cases 804900,807500                 # 指定 TK launch
//
// 选项（单横杠或双横杠均可）：
//   -s 1|2          场景：1=均衡（默认）；2=TK 实录（不均衡路由）
//   -tprs L         [s1] TPR 列表（默认 1,4,16,32,33,64,128,256,512）
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
    std::vector<int> tprs = {1, 4, 16, 32, 33, 64, 128, 256, 512};
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
                 "  -tprs L       [s1] TPR 列表（默认 1,4,16,32,33,64,128,256,512）\n"
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
    for (auto& s : specs) {
        s.win = mt::choose_win_cap_counts(s.per_expert, max_tokens);
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

        // [诊断] MYBENCH_PHASE=1：计时后做一次 reset + 单次 eager launch，读取 kernel 内置相位 tick。
        // 注意：route..reduce 与 chain 只来自 block0 的时间线（kernel 结束要等所有 block），
        // 真正的关键路径需看逐 block 数据（MYBENCH_PHASE_RANK=<rank>，需 -DR24_DIAG_PHASE=1）。
        if (std::getenv("MYBENCH_PHASE") != nullptr) {
            mt::reset_dev(geo, d, routes, tokens);
            MPI_Barrier(MPI_COMM_WORLD);
            fp8mega::launch_full(a, d.st);
            MYCHK(mcStreamSynchronize(d.st));
            MPI_Barrier(MPI_COMM_WORLD);
            unsigned long long tp[16] = {0};
            MYCHK(mcMemcpy(tp, d.lf->pre_ticks, sizeof(tp), mcMemcpyDeviceToHost));
            std::vector<unsigned long long> tr((size_t)kGrid * 8, 0ull);
            MYCHK(mcMemcpy(tr.data(), d.lf->pre_ticks + (size_t)kGrid * 8,
                           tr.size() * sizeof(unsigned long long), mcMemcpyDeviceToHost));
            int proc[kGrid] = {0};
            MYCHK(mcMemcpy(proc, d.lf->out_processed, sizeof(proc), mcMemcpyDeviceToHost));
            // 行：route publish xbar setup dispatch L1 L2tail compute reduce chain | trL1max trL2max maxtasks
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
                    std::printf("[phase-blk] rank=%d block: L1cyc L2cyc tasks\n", rank);
                    for (int b = 0; b < kGrid; ++b)
                        std::printf("[phase-blk] b=%02d %10llu %10llu %d\n", b,
                                    tr[(size_t)b * 8 + 0], tr[(size_t)b * 8 + 1], proc[b]);
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
                std::printf("[phase] 单位=GPU 周期（clock64，block0 时间线）；trL1max/trL2max 需 R24_DIAG_PHASE=1\n");
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
