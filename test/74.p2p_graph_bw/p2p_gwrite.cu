// ============================================================================
// p2p_gwrite.cu — GPU 间 P2P「远写」单 block、graph 口径带宽实测（MACA/MetaX）
//
// 口径：
//   * 每个可见 rank 用 FG 显存（mcExtMallocWithFlags + mcDeviceMallocFinegrained）
//     分配缓冲，导出 IPC handle，其余 rank open peer（mcDeviceCanAccessPeer 检查）。
//   * 对每个 (from->to, size) 建一张 CUDA graph，内含 N=10 个同构节点；
//     每节点 = 1 个 block（blockDim=1024）把 size 字节远写到 to 的 peer 缓冲。
//   * 变体：__stwt 与普通 store，各测一遍，取更快。
//   * 预热 warmup 次；replay R 次取 min；event 夹住；per-node = 总时间/N；
//     带宽 = size / per-node。
//   * 回读校验（在 to 本地读自己的缓冲），SIGALRM 超时保护。
//
// 编译：见 build.sh
// 运行：MACA_VISIBLE_DEVICES=0,1,...,15 ./p2p_gwrite [--pairs 0:1,0:8,0:15]
// ============================================================================

#include <mcr/mc_runtime.h>

#include <algorithm>
#include <atomic>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <unistd.h>

#define CK(x)                                                                  \
    do {                                                                       \
        mcError_t _e = (x);                                                    \
        if (_e != mcSuccess) {                                                 \
            std::fprintf(stderr, "ERR %s = %d @%d\n", #x, (int)_e, __LINE__);  \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// ---- 运行参数 ----
static const size_t kMaxBytes = 768 * 1024;   // 最大粒度
static const int    kNodes    = 10;           // 每张 graph 的节点数
static const int    kBlock    = 1024;         // 单节点 block 线程数
static const int    kWarmup   = 2;
static const int    kReplays  = 3;

static const size_t kSizes[] = {6 * 1024,   12 * 1024,  24 * 1024,  48 * 1024,
                                96 * 1024,  192 * 1024, 384 * 1024, 768 * 1024};
static const int kNumSizes = (int)(sizeof(kSizes) / sizeof(kSizes[0]));

// ---- 全局资源（每 rank 一份）----
static int g_ndev = 0;
static uint8_t* g_buf[64] = {nullptr};
static mcIpcMemHandle_t g_handle[64];
static uint8_t* g_peer[64][64];
static bool g_open_ok[64][64];
static bool g_alloc_ok[64];
static mcStream_t g_stream[64] = {nullptr};
static int g_enabled[64], g_unavail[64];

// 校验用：writer 为该 (from,size) 选中的写方式与 seed
static int g_vmode[64][16];
static unsigned g_vseed[64][16];

// ---- 跨线程屏障（与 bench.cu rank_barrier 同款）----
static std::atomic<int> g_bcnt{0};
static std::atomic<int> g_bgen{0};
static int g_bn = 0;
static void rbarrier() {
    if (g_bn <= 1) return;
    const int gen = g_bgen.load(std::memory_order_acquire);
    if (g_bcnt.fetch_add(1, std::memory_order_acq_rel) + 1 == g_bn) {
        g_bcnt.store(0, std::memory_order_relaxed);
        g_bgen.fetch_add(1, std::memory_order_release);
    } else {
        while (g_bgen.load(std::memory_order_acquire) == gen) {}
    }
}

// ---- 单 block 远写 kernel：MODE 1 = __stwt；0 = 普通 store ----
template <int MODE>
__global__ void gwrite_kernel(uint4* __restrict__ dst, size_t nvec, unsigned seed) {
    const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = tid; i < nvec; i += (size_t)blockDim.x) {
        const uint4 v = make_uint4(seed, (unsigned)i, seed ^ (unsigned)i, 0xA5A5A5A5u);
        if (MODE == 1)
            __stwt(dst + i, v);
        else
            dst[i] = v;
    }
}

static void launch_node(int mode, mcStream_t st, uint8_t* dst, size_t nvec, unsigned seed) {
    if (mode == 1)
        gwrite_kernel<1><<<1, kBlock, 0, st>>>((uint4*)dst, nvec, seed);
    else
        gwrite_kernel<0><<<1, kBlock, 0, st>>>((uint4*)dst, nvec, seed);
}

// 测一种写方式：建 graph 捕获 N 节点，预热 + R 次 replay 取 min，返回 per-node µs。
static double measure_mode(int mode, mcStream_t st, uint8_t* dst, size_t nvec,
                           unsigned seed, const char** caliber, bool* captured) {
    mcGraph_t g = nullptr;
    mcGraphExec_t ex = nullptr;
    bool cap = (mcStreamBeginCapture(st, mcStreamCaptureModeThreadLocal) == mcSuccess);
    if (cap) {
        for (int k = 0; k < kNodes; ++k) launch_node(mode, st, dst, nvec, seed);
        cap = (mcStreamEndCapture(st, &g) == mcSuccess);
    }
    if (cap) {
        mcError_t e = mcGraphInstantiate(&ex, g, nullptr, nullptr, 0);
        mcGraphDestroy(g);
        if (e != mcSuccess || ex == nullptr) cap = false;
    }
    if (captured) *captured = cap;

    std::vector<double> per(kReplays, 0.0);

    if (!cap) {  // eager 降级口径
        if (caliber) *caliber = "eager(降级)";
        for (int i = 0; i < kWarmup; ++i) {
            for (int k = 0; k < kNodes; ++k) launch_node(mode, st, dst, nvec, seed);
            CK(mcStreamSynchronize(st));
        }
        std::vector<mcEvent_t> c0(kReplays), c1(kReplays);
        for (int r = 0; r < kReplays; ++r) {
            CK(mcEventCreate(&c0[r]));
            CK(mcEventCreate(&c1[r]));
            CK(mcEventRecord(c0[r], st));
            for (int k = 0; k < kNodes; ++k) launch_node(mode, st, dst, nvec, seed);
            CK(mcEventRecord(c1[r], st));
            CK(mcStreamSynchronize(st));
        }
        for (int r = 0; r < kReplays; ++r) {
            float ms = 0;
            CK(mcEventElapsedTime(&ms, c0[r], c1[r]));
            per[r] = (double)ms * 1000.0 / kNodes;
            mcEventDestroy(c0[r]);
            mcEventDestroy(c1[r]);
        }
    } else {
        if (caliber) *caliber = "graph";
        for (int i = 0; i < kWarmup; ++i) CK(mcGraphLaunch(ex, st));
        CK(mcStreamSynchronize(st));
        std::vector<mcEvent_t> c0(kReplays), c1(kReplays);
        for (int r = 0; r < kReplays; ++r) {
            CK(mcEventCreate(&c0[r]));
            CK(mcEventCreate(&c1[r]));
            CK(mcEventRecord(c0[r], st));
            CK(mcGraphLaunch(ex, st));
            CK(mcEventRecord(c1[r], st));
        }
        CK(mcStreamSynchronize(st));
        for (int r = 0; r < kReplays; ++r) {
            float ms = 0;
            CK(mcEventElapsedTime(&ms, c0[r], c1[r]));
            per[r] = (double)ms * 1000.0 / kNodes;
            mcEventDestroy(c0[r]);
            mcEventDestroy(c1[r]);
        }
        mcGraphExecDestroy(ex);
    }
    std::sort(per.begin(), per.end());
    return per.front();
}

static void on_alarm(int) {
    const char m[] = "\n[TIMEOUT] SIGALRM 触发，强制退出\n";
    ssize_t r = write(2, m, sizeof(m) - 1);
    (void)r;
    _exit(124);
}

struct Pair { int from, to; };

int main(int argc, char** argv) {
    std::signal(SIGALRM, on_alarm);
    alarm(60);

    // 解析 --pairs a:b,c:d
    std::vector<Pair> pairs;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--pairs") == 0 && i + 1 < argc) {
            std::string s = argv[++i];
            size_t pos = 0;
            while (pos < s.size()) {
                size_t c = s.find(',', pos);
                std::string tok = s.substr(pos, c == std::string::npos ? std::string::npos : c - pos);
                size_t colon = tok.find(':');
                if (colon != std::string::npos)
                    pairs.push_back({std::atoi(tok.substr(0, colon).c_str()),
                                     std::atoi(tok.substr(colon + 1).c_str())});
                if (c == std::string::npos) break;
                pos = c + 1;
            }
        }
    }
    if (pairs.empty()) {
        pairs.push_back({0, 1});
        pairs.push_back({0, 8});
        pairs.push_back({0, 15});
    }

    CK(mcGetDeviceCount(&g_ndev));
    if (g_ndev < 2) {
        std::fprintf(stderr, "可见设备数=%d，无法测 peer 远写\n", g_ndev);
        return 2;
    }
    g_bn = g_ndev;
    std::printf("可见设备=%d，每张 graph N=%d 节点，block=%d，warmup=%d，replays=%d\n",
                g_ndev, kNodes, kBlock, kWarmup, kReplays);

    // ---- 阶段1：FG 分配 + IPC handle 导出 ----
    {
        std::vector<std::thread> pool;
        for (int r = 0; r < g_ndev; ++r) {
            pool.emplace_back([r]() {
                CK(mcSetDevice(r));
                void* p = nullptr;
                mcError_t e = mcExtMallocWithFlags(&p, kMaxBytes, mcDeviceMallocFinegrained);
                if (e != mcSuccess || p == nullptr) {
                    std::fprintf(stderr, "rank%d FG 分配失败 err=%d\n", r, (int)e);
                    g_alloc_ok[r] = false;
                    return;
                }
                g_buf[r] = (uint8_t*)p;
                CK(mcMemset(g_buf[r], 0, kMaxBytes));
                CK(mcIpcGetMemHandle(&g_handle[r], g_buf[r]));
                g_alloc_ok[r] = true;
            });
        }
        for (auto& t : pool) t.join();
        for (int r = 0; r < g_ndev; ++r)
            if (!g_alloc_ok[r]) {
                std::fprintf(stderr, "rank%d 分配失败，退出\n", r);
                return 3;
            }
    }

    // ---- 阶段2：open peer + peer access + stream ----
    {
        std::vector<std::thread> pool;
        for (int r = 0; r < g_ndev; ++r) {
            pool.emplace_back([r]() {
                CK(mcSetDevice(r));
                for (int d = 0; d < g_ndev; ++d) {
                    if (d == r) continue;
                    void* q = nullptr;
                    mcError_t e = mcIpcOpenMemHandle(&q, g_handle[d],
                                                     mcIpcMemLazyEnablePeerAccess);
                    if (e != mcSuccess) {
                        std::fprintf(stderr, "rank%d open peer%d 失败 err=%d\n", r, d, (int)e);
                        g_open_ok[r][d] = false;
                        continue;
                    }
                    g_peer[r][d] = (uint8_t*)q;
                    g_open_ok[r][d] = true;
                    int can = 0;
                    if (mcDeviceCanAccessPeer(&can, r, d) == mcSuccess && can) {
                        mcError_t ee = mcDeviceEnablePeerAccess(d, 0);
                        if (ee == mcSuccess || ee == mcErrorPeerAccessAlreadyEnabled)
                            g_enabled[r]++;
                        else
                            g_unavail[r]++;
                    } else {
                        g_unavail[r]++;
                    }
                }
                CK(mcStreamCreate(&g_stream[r]));
            });
        }
        for (auto& t : pool) t.join();
    }

    // 打印 peer 矩阵（canAccessPeer / open 状态）
    std::printf("\n---- peer 可达性 (行=from, 列=to) ----\n");
    for (int r = 0; r < g_ndev; ++r) {
        std::printf("rank%2d  open=%d enabled=%d unavail=%d : ", r, g_enabled[r],
                    g_enabled[r], g_unavail[r]);
        for (int d = 0; d < g_ndev; ++d) {
            if (d == r) std::printf("  . ");
            else {
                int can = 0;
                mcDeviceCanAccessPeer(&can, r, d);
                std::printf("%d/%d ", can, g_open_ok[r][d] ? 1 : 0);
            }
        }
        std::printf("\n");
    }

    // ---- 阶段3：测量 + 校验 ----
    std::vector<std::thread> pool;
    for (int r = 0; r < g_ndev; ++r) {
        pool.emplace_back([r, &pairs]() {
            CK(mcSetDevice(r));
            // 全局预热：吸收首次发射/DVFS 一次性开销
            {
                const char* c = "";
                bool cp = false;
                for (int w = 0; w < 3; ++w) {
                    for (int k = 0; k < kNodes * 4; ++k)
                        launch_node(0, g_stream[r], g_peer[r][(r + 1) % g_ndev],
                                    kMaxBytes / 16, 0x5A5A0000u);
                    CK(mcStreamSynchronize(g_stream[r]));
                }
                measure_mode(0, g_stream[r], g_peer[r][(r + 1) % g_ndev], 4096, 0x5A5A0000u,
                             &c, &cp);
                usleep(200 * 1000);
            }
            // 完整跑两遍：第一遍预热各 (size,mode) 的 graph/kernel 缓存，
            // 只报告第二遍（否则第一个 pair 会因冷启动系统性偏慢 ~10%）。
            for (int pass = 0; pass < 2; ++pass) {
            const bool report = (pass == 1);
            for (const Pair& pr : pairs) {
                // 测量段：仅 from 线程工作，其余等屏障
                rbarrier();
                if (r == pr.from) {
                    const char* cal0 = "";
                    bool cap0 = false;
                    const double us0 =
                        measure_mode(0, g_stream[r], g_peer[r][pr.to], 0, 0u, &cal0, &cap0);
                    if (report) {
                        std::printf("\n==== 远写 from=%d -> to=%d ====\n", pr.from, pr.to);
                        std::printf("  空 block(0B) per-node=%.3f us (%s, cap=%d)\n", us0, cal0,
                                    (int)cap0);
                        std::printf("  %-8s  %-9s  %-10s  %-9s  %-8s  %s\n", "size", "stwt GB/s",
                                    "plain GB/s", "best", "caliber", "per-node us");
                    }
                    for (int si = 0; si < kNumSizes; ++si) {
                        const size_t bytes = kSizes[si];
                        const size_t nvec = bytes / 16;
                        const uint32_t seed = 0xC0DE0000u ^ ((unsigned)pr.from << 8) ^
                                              ((unsigned)pr.to << 4) ^ (unsigned)si;
                        const char* cal[2] = {"", ""};
                        bool cap[2] = {false, false};
                        double us[2];
                        us[0] = measure_mode(0, g_stream[r], g_peer[r][pr.to], nvec, seed,
                                             &cal[0], &cap[0]);
                        us[1] = measure_mode(1, g_stream[r], g_peer[r][pr.to], nvec, seed,
                                             &cal[1], &cap[1]);
                        double bw[2];
                        bw[0] = (double)bytes / (us[0] * 1000.0);  // bytes/us -> GB/s
                        bw[1] = (double)bytes / (us[1] * 1000.0);
                        const int best = (bw[1] > bw[0]) ? 1 : 0;
                        g_vmode[r][si] = best;
                        g_vseed[r][si] = seed;
                        if (report)
                            std::printf("  %-8zu  %-9.2f  %-10.2f  %-9s  %-8s  %.3f\n", bytes,
                                        bw[0], bw[1], best ? "stwt" : "plain",
                                        cap[best] ? "graph" : "eager", us[best]);
                        (void)cal;
                    }
                }

                // 校验段：writer 单发一次，owner 本地回读（仅报告遍执行）
                if (!report) continue;
                for (int si = 0; si < kNumSizes; ++si) {
                    const size_t bytes = kSizes[si];
                    const size_t nvec = bytes / 16;
                    if (r == pr.from) {
                        launch_node(g_vmode[r][si], g_stream[r], g_peer[r][pr.to], nvec,
                                    g_vseed[r][si]);
                        CK(mcStreamSynchronize(g_stream[r]));
                        mcError_t le = mcGetLastError();
                        if (le != mcSuccess)
                            std::fprintf(stderr, "  [校验] launch err=%d size=%zu\n", (int)le,
                                         bytes);
                    }
                    rbarrier();
                    if (r == pr.to) {
                        std::vector<uint8_t> h(bytes);
                        mcError_t ce = mcMemcpy(h.data(), g_buf[r], bytes, mcMemcpyDeviceToHost);
                        if (ce != mcSuccess) {
                            std::printf("  [校验] pair %d->%d size=%zu memcpy err=%d\n",
                                        pr.from, pr.to, bytes, (int)ce);
                        } else {
                            const uint4* hv = (const uint4*)h.data();
                            size_t bad = 0;
                            for (size_t i = 0; i < nvec; ++i) {
                                uint4 exp = make_uint4(g_vseed[pr.from][si], (unsigned)i,
                                                       g_vseed[pr.from][si] ^ (unsigned)i,
                                                       0xA5A5A5A5u);
                                if (hv[i].x != exp.x || hv[i].y != exp.y || hv[i].z != exp.z ||
                                    hv[i].w != exp.w) {
                                    if (bad < 3)
                                        std::printf("  [校验] MISMATCH pair %d->%d size=%zu idx=%zu "
                                                    "got=(%u,%u,%u,%u) exp=(%u,%u,%u,%u)\n",
                                                    pr.from, pr.to, bytes, i, hv[i].x, hv[i].y,
                                                    hv[i].z, hv[i].w, exp.x, exp.y, exp.z, exp.w);
                                    ++bad;
                                }
                            }
                            std::printf("  [校验] pair %d->%d size=%zu mode=%s : %s (bad=%zu/%zu)\n",
                                        pr.from, pr.to, bytes,
                                        g_vmode[pr.from][si] ? "stwt" : "plain",
                                        bad == 0 ? "PASS" : "FAIL", bad, nvec);
                        }
                    }
                    rbarrier();
                }
                rbarrier();
            }
            }  // pass
        });
    }
    for (auto& t : pool) t.join();

    alarm(0);
    std::printf("\n完成。\n");
    return 0;
}
