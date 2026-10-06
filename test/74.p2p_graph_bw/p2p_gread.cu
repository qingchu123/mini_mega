// ============================================================================
// p2p_gread.cu — GPU 间 P2P「远读」(remote read) 单 block / graph 口径带宽
//
//   每个可见 rank：FG 显存缓冲 + IPC handle + open peer（canAccessPeer 检查）。
//   每个 (reader, target, size, variant) 建一张 CUDA graph：
//     10 个同构节点，每节点 = 1 个 block(blockDim=1024) 从 peer 缓冲远读 size 字节。
//   variant：普通 32-bit load / 128-bit 向量化 load。
//   预热 2；replay 3 取 min；event 夹住；per-node 时间 = 总时间 / 10；
//   带宽 = size / per-node 时间。回读校验（reduce 求和 + 超时标志）。
//
// 只用卡 16..31：运行前设置 MACA_VISIBLE_DEVICES=16,17,...,31（逻辑 0..15）。
// 编译：bash build.sh    运行：bash run.sh
// ============================================================================

#include <mcr/mc_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// ------------------------------------------------------------------ 常量 --
static constexpr int kNodes = 10;    // 单图节点数
static constexpr int kBlock = 1024;  // 每节点 blockDim
static constexpr int kWarmup = 2;
static constexpr int kReplays = 3;

// size 粒度：6KB .. 768KB
static const size_t kSizes[] = {
    6 * 1024,   12 * 1024,  24 * 1024,  48 * 1024,
    96 * 1024,  192 * 1024, 384 * 1024, 768 * 1024,
};
static constexpr int kNumSizes = (int)(sizeof(kSizes) / sizeof(kSizes[0]));
static constexpr size_t kMaxSize = 768 * 1024;

#define CHECK(x)                                                          \
    do {                                                                  \
        mcError_t _e = (x);                                               \
        if (_e != mcSuccess) {                                            \
            std::fprintf(stderr, "FAIL %s:%d  err=%d\n", __FILE__,        \
                         __LINE__, (int)_e);                              \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

// ------------------------------------------------------------------ kernel --
// VEC  : false=普通 32-bit load；true=128-bit 向量化 load
// REDUCE: false=计时（DCE 阻止）；true=回读校验（block 归约求和）
template <bool VEC, bool REDUCE>
__global__ void p2p_read_kernel(const unsigned int* __restrict__ src,
                                unsigned int nwords,
                                unsigned int* __restrict__ sink,
                                unsigned long long* __restrict__ terr) {
    const unsigned int tid = threadIdx.x;
    unsigned int acc = 0u;
    const unsigned long long t0 = clock64();
    const unsigned long long kLimit = 4000000000ull;  // 超时保护（周期数）

    if (VEC) {
        const uint4* s4 = reinterpret_cast<const uint4*>(src);
        const unsigned int n4 = nwords >> 2;
        for (unsigned int i = tid; i < n4; i += blockDim.x) {
            const uint4 v = s4[i];
            acc += v.x + v.y + v.z + v.w;
            if ((i & 0xFFu) == 0u && (clock64() - t0) > kLimit) {
                if (tid == 0u) terr[0] = 1ull;
                break;
            }
        }
    } else {
        for (unsigned int i = tid; i < nwords; i += blockDim.x) {
            acc += src[i];
            if ((i & 0xFFu) == 0u && (clock64() - t0) > kLimit) {
                if (tid == 0u) terr[0] = 1ull;
                break;
            }
        }
    }

    if (REDUCE) {
        __shared__ unsigned int sm[kBlock];
        sm[tid] = acc;
        __syncthreads();
        for (unsigned int s = blockDim.x >> 1; s > 0u; s >>= 1) {
            if (tid < s) sm[tid] += sm[tid + s];
            __syncthreads();
        }
        if (tid == 0u) sink[0] = sm[0];
    } else {
        // 仅当 acc 恰好命中魔数才写回：迫使编译器保留全部 load，且不引入额外流量
        if (acc == 0xDEADBEEFu && tid == 0u) sink[0] = acc;
    }
}

__global__ void fill_one_kernel(unsigned int* __restrict__ p, size_t n,
                                unsigned int val) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        p[i] = val;
}

// ------------------------------------------------------------------ 运行态 --
struct RankRes {
    int rank = -1;
    unsigned int* buf = nullptr;         // FG 显存，kMaxSize
    mcIpcMemHandle_t handle{};
    std::vector<unsigned int*> peer;     // peer[d]（d != rank）
    std::vector<int> can;                // canAccessPeer 结果
    std::vector<int> enable_err;         // EnablePeerAccess 返回码
    std::vector<int> open_err;           // OpenMemHandle 返回码
};

static std::atomic<int> g_alloc_ready{0};
static std::atomic<int> g_open_ready{0};

// ---------------------------------------------------- graph 构建 + 测量 --
template <bool VEC>
struct Meas {
    double gbps = 0.0;
    double per_us = 0.0;
    bool captured = false;
    mcError_t err = mcSuccess;
    const char* note = "graph";
    bool valid = false;
    unsigned int got = 0u;
    unsigned int want = 0u;
};

template <bool VEC>
static Meas<VEC> measure_one(mcStream_t st, unsigned int* src, size_t size,
                             unsigned int pat, unsigned int* d_sink,
                             unsigned long long* d_terr, unsigned int* h_sink,
                             unsigned long long* h_terr) {
    Meas<VEC> m;
    const unsigned int nwords = (unsigned int)(size / 4);
    m.want = pat * nwords;   // 目标缓冲每 32-bit 字填 pat，故期望和 = pat*nwords
    mcGraph_t g = nullptr;
    mcGraphExec_t ex = nullptr;

    mcError_t e = mcStreamBeginCapture(st, mcStreamCaptureModeThreadLocal);
    if (e != mcSuccess) {
        m.err = e;
        m.note = "BeginCapture 失败";
        return m;
    }
    for (int k = 0; k < kNodes; ++k) {
        if (VEC)
            p2p_read_kernel<true, false><<<1, kBlock, 0, st>>>(src, nwords,
                                                               d_sink, d_terr);
        else
            p2p_read_kernel<false, false><<<1, kBlock, 0, st>>>(src, nwords,
                                                                d_sink, d_terr);
    }
    e = mcStreamEndCapture(st, &g);
    if (e != mcSuccess || g == nullptr) {
        m.err = e;
        m.note = "EndCapture 失败";
        return m;
    }
    e = mcGraphInstantiate(&ex, g, nullptr, nullptr, 0);
    mcGraphDestroy(g);
    if (e != mcSuccess || ex == nullptr) {
        m.err = e;
        m.note = "GraphInstantiate 失败";
        return m;
    }
    m.captured = true;

    for (int i = 0; i < kWarmup; ++i) CHECK(mcGraphLaunch(ex, st));
    CHECK(mcStreamSynchronize(st));

    std::vector<mcEvent_t> c0(kReplays, nullptr), c1(kReplays, nullptr);
    for (int i = 0; i < kReplays; ++i) {
        CHECK(mcEventCreate(&c0[i]));
        CHECK(mcEventCreate(&c1[i]));
        CHECK(mcEventRecord(c0[i], st));
        CHECK(mcGraphLaunch(ex, st));
        CHECK(mcEventRecord(c1[i], st));
        CHECK(mcStreamSynchronize(st));
    }
    double best = 1e30;
    for (int i = 0; i < kReplays; ++i) {
        float ms = 0.f;
        CHECK(mcEventElapsedTime(&ms, c0[i], c1[i]));
        if ((double)ms < best) best = (double)ms;
        mcEventDestroy(c0[i]);
        mcEventDestroy(c1[i]);
    }
    mcGraphExecDestroy(ex);

    m.per_us = best * 1000.0 / (double)kNodes;   // ms->us, /nodes
    m.gbps = (double)size / (m.per_us * 1e-6) / 1e9;

    // ---- 回读校验（eager，仅一次）----
    CHECK(mcMemset(d_terr, 0, sizeof(unsigned long long)));
    if (VEC)
        p2p_read_kernel<true, true><<<1, kBlock, 0, st>>>(src, nwords, d_sink,
                                                          d_terr);
    else
        p2p_read_kernel<false, true><<<1, kBlock, 0, st>>>(src, nwords, d_sink,
                                                           d_terr);
    CHECK(mcStreamSynchronize(st));
    CHECK(mcMemcpy(h_sink, d_sink, sizeof(unsigned int), mcMemcpyDeviceToHost));
    CHECK(mcMemcpy(h_terr, d_terr, sizeof(unsigned long long),
                   mcMemcpyDeviceToHost));
    m.got = *h_sink;
    m.valid = (*h_terr == 0ull) && (m.got == m.want);
    if (*h_terr != 0ull) m.note = "timeout";
    return m;
}

// ------------------------------------------------------------------ main --
static std::vector<int> parse_targets(const char* s, int n) {
    std::vector<int> out;
    if (!s) return out;
    const char* p = s;
    while (*p) {
        int v = std::atoi(p);
        if (v >= 0 && v < n) out.push_back(v);
        const char* c = std::strchr(p, ',');
        if (!c) break;
        p = c + 1;
    }
    return out;
}

int main(int argc, char** argv) {
    int n = 0;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--ranks") == 0 && i + 1 < argc)
            n = std::atoi(argv[++i]);
    }
    if (n <= 0) {
        const char* vd = std::getenv("MACA_VISIBLE_DEVICES");
        if (vd && *vd) {
            for (const char* p = vd; *p;) {
                ++n;
                const char* c = std::strchr(p, ',');
                if (!c) break;
                p = c + 1;
            }
        } else {
            n = 16;
        }
    }
    int reader = 0;
    if (const char* rd = std::getenv("GREAD_READER")) reader = std::atoi(rd);
    std::vector<int> targets = parse_targets(std::getenv("GREAD_TARGETS"), n);
    if (targets.empty()) {
        targets.push_back(1);
        if (n / 2 != 1) targets.push_back(n / 2);
        if (n - 1 != 1) targets.push_back(n - 1);
    }

    std::printf("=== P2P 远读 (FG) 单block graph口径 ===\n");
    std::printf("可见 ranks=%d  reader=%d  targets=", n, reader);
    for (size_t i = 0; i < targets.size(); ++i)
        std::printf("%s%d", i ? "," : "", targets[i]);
    std::printf("  nodes=%d block=%d warmup=%d replays=%d\n", kNodes, kBlock,
                kWarmup, kReplays);

    std::vector<RankRes> res(n);
    std::vector<std::thread> pool;

    // 1) FG 分配 + 填值 + IPC handle
    for (int rank = 0; rank < n; ++rank) {
        pool.emplace_back([&, rank]() {
            RankRes& r = res[rank];
            r.rank = rank;
            CHECK(mcSetDevice(rank));
            void* p = nullptr;
            const mcError_t me = mcExtMallocWithFlags(
                &p, kMaxSize, mcDeviceMallocFinegrained);
            if (me != mcSuccess || p == nullptr) {
                std::fprintf(stderr, "rank%d FG 分配失败 err=%d\n", rank,
                             (int)me);
                std::exit(1);
            }
            r.buf = static_cast<unsigned int*>(p);
            CHECK(mcMemset(r.buf, 0, kMaxSize));
            fill_one_kernel<<<(kMaxSize / 4 + 255) / 256, 256>>>(
                r.buf, kMaxSize / 4, (unsigned int)(rank + 1));
            CHECK(mcDeviceSynchronize());
            CHECK(mcIpcGetMemHandle(&r.handle, r.buf));
            g_alloc_ready.fetch_add(1);
        });
    }
    while (g_alloc_ready.load() < n) {
    }
    for (auto& t : pool) t.join();
    pool.clear();

    // 2) open peer + canAccessPeer + enable
    for (int rank = 0; rank < n; ++rank) {
        pool.emplace_back([&, rank]() {
            RankRes& r = res[rank];
            CHECK(mcSetDevice(rank));
            r.peer.assign(n, nullptr);
            r.can.assign(n, 0);
            r.enable_err.assign(n, 0);
            r.open_err.assign(n, 0);
            for (int d = 0; d < n; ++d) {
                if (d == rank) continue;
                void* q = nullptr;
                const mcError_t e = mcIpcOpenMemHandle(
                    &q, res[d].handle, mcIpcMemLazyEnablePeerAccess);
                r.open_err[d] = (int)e;
                if (e == mcSuccess)
                    r.peer[d] = static_cast<unsigned int*>(q);
                int can = 0;
                if (mcDeviceCanAccessPeer(&can, rank, d) == mcSuccess && can) {
                    r.can[d] = 1;
                    const mcError_t ee = mcDeviceEnablePeerAccess(d, 0);
                    r.enable_err[d] = (int)ee;
                }
            }
            g_open_ready.fetch_add(1);
        });
    }
    while (g_open_ready.load() < n) {
    }
    for (auto& t : pool) t.join();
    pool.clear();

    // 打印拓扑可访问性
    {
        RankRes& r = res[reader];
        std::printf("reader=%d  peer 状态: ", reader);
        for (int d = 0; d < n; ++d) {
            if (d == reader) continue;
            std::printf("[%d can=%d open=%d en=%d] ", d, r.can[d],
                        r.open_err[d], r.enable_err[d]);
        }
        std::printf("\n");
    }

    // 3) reader 线程测量
    int rc = 0;
    std::thread meas_thread([&]() {
        RankRes& r = res[reader];
        CHECK(mcSetDevice(reader));
        mcStream_t st = nullptr;
        CHECK(mcStreamCreate(&st));

        unsigned int* d_sink = nullptr;
        unsigned long long* d_terr = nullptr;
        CHECK(mcMalloc((void**)&d_sink, sizeof(unsigned int)));
        CHECK(mcMalloc((void**)&d_terr, sizeof(unsigned long long)));
        unsigned int h_sink = 0u;
        unsigned long long h_terr = 0ull;

        // 表：size x target 的更快带宽
        for (size_t ti = 0; ti < targets.size(); ++ti) {
            const int tgt = targets[ti];
            if (!r.can[tgt] || r.peer[tgt] == nullptr) {
                std::printf("\n[skip] pair %d->%d 不可用 (can=%d open_err=%d "
                            "enable_err=%d)\n",
                            reader, tgt, r.can[tgt], r.open_err[tgt],
                            r.enable_err[tgt]);
                rc = 1;
                continue;
            }
            std::printf("\n---- pair %d -> %d ----\n", reader, tgt);
            std::printf("%-9s %14s %14s %14s %10s\n", "size", "load32 GB/s",
                        "vec128 GB/s", "best GB/s", "per-node us");
            double peak = 0;
            size_t peak_size = 0;
            const char* peak_var = "";
            for (int si = 0; si < kNumSizes; ++si) {
                const size_t size = kSizes[si];
                const unsigned int pat = (unsigned int)(tgt + 1);
                Meas<false> ms = measure_one<false>(st, r.peer[tgt], size, pat,
                                                    d_sink, d_terr, &h_sink,
                                                    &h_terr);
                Meas<true> mv = measure_one<true>(st, r.peer[tgt], size, pat,
                                                  d_sink, d_terr, &h_sink,
                                                  &h_terr);
                const bool vec_faster = mv.gbps >= ms.gbps;
                const double best = vec_faster ? mv.gbps : ms.gbps;
                const double best_us = vec_faster ? mv.per_us : ms.per_us;
                std::printf("%-9zu %14.2f %14.2f %14.2f %10.3f  [%s%s]\n", size,
                            ms.gbps, mv.gbps, best, best_us,
                            ms.valid ? "" : "L32-BAD ",
                            mv.valid ? "" : "V128-BAD");
                std::fflush(stdout);
                if (best > peak) {
                    peak = best;
                    peak_size = size;
                    peak_var = vec_faster ? "vec128" : "load32";
                }
            }
            std::printf("  >> pair %d->%d 峰值 %.2f GB/s @ %zu B (%s)\n", reader,
                        tgt, peak, peak_size, peak_var);
        }

        mcFree(d_sink);
        mcFree(d_terr);
        mcStreamDestroy(st);
    });
    meas_thread.join();

    // 清理（仅释放自身分配；peer 句柄随进程退出回收）
    for (int rank = 0; rank < n; ++rank) {
        if (res[rank].buf) {
            mcSetDevice(rank);
            mcFree(res[rank].buf);
        }
    }
    std::printf("\n[done] rc=%d\n", rc);
    return rc;
}
