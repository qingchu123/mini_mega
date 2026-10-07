// ============================================================================
// fg_rwk_read.cu — 同布局下 FG(fine-grained) vs RWK(普通 mcMalloc) 本地读带宽
//
//   动机：L1 的 A(pool) 当前放在对称区(FG)。本基准用【完全相同的访问布局】分别
//   读 FG / RWK 两个缓冲，量化「本地读」是否因显存类型而异，作为是否把 L1 A 单独
//   放进普通显存(RWK)的依据。
//
//   两个变体：
//     contig16 —— uint4 连续读（raw 本地读带宽上限，同类型对照）
//     afrag    —— 复刻 cm_load_a 的 native-fragment 8B/lane 装载：
//                 row = m_tile*16 + (lane&15)，addr = row*k_stride + kk*32
//                 + ((lane>>4)&3)*8；每个 64 线程组一次读 16 行×32B=512B。
//
//   口径：每个 size 单独一次 kernel 读完整缓冲；预热 2、5 次 event 计时取 min；
//         带宽 = size / min 时间。校验：两缓冲 reduce16 求和必须相等。
//   运行：bash run.sh   （单卡，默认 dev0；FG_BENCH_DEV 覆盖）
// ============================================================================

#include <mcr/mc_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x)                                                          \
    do {                                                                  \
        mcError_t _e = (x);                                               \
        if (_e != mcSuccess) {                                            \
            std::fprintf(stderr, "FAIL %s:%d err=%d %s\n", __FILE__,      \
                         __LINE__, (int)_e, mcGetErrorString(_e));        \
            std::exit(1);                                                 \
        }                                                                 \
    } while (0)

static constexpr int kBlock = 1024;
static constexpr int kGrid = 64;
static constexpr int kWarmup = 3;
static constexpr int kTrials = 15;
static constexpr int kKStride = 2048;  // 复刻 pool 行 stride（= hidden 2048B）
static mcStream_t g_st = nullptr;

// ---------------------------------------------------------------- kernels --
__global__ void fill_kernel(unsigned int* __restrict__ p, size_t n,
                            unsigned int seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        p[i] = seed + (unsigned int)i * 2654435761u;
}

// 连续 uint4 读（raw 上限）
__global__ void contig16_kernel(const uint4* __restrict__ p, size_t n4,
                                unsigned int* __restrict__ sink) {
    unsigned int acc = 0u;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 0xDEADBEEFu && threadIdx.x == 0u) sink[0] = acc;
}

// 复刻 cm_load_a：8B/lane，16 行×32B tile
__global__ void afrag_kernel(const uint8_t* __restrict__ pool, long long nbytes,
                             int k_stride, unsigned int* __restrict__ sink) {
    const int tid = threadIdx.x;
    const int lane = tid & 63;
    const int group = tid >> 6;  // 64 线程组
    const int ngroups = (int)(blockDim.x >> 6);
    const int tiles_per_row = k_stride >> 5;          // k/32
    const long long rows = nbytes / k_stride;
    const long long total_tiles = (rows >> 4) * (long long)tiles_per_row;
    const long long stride = (long long)gridDim.x * ngroups;
    unsigned int acc = 0u;
    long long t = (long long)blockIdx.x * ngroups + group;
    // 4 路展开：暴露 MLP，避免串行 load→sum
    for (; t + 3 * stride < total_tiles; t += 4 * stride) {
        long long tt[4] = {t, t + stride, t + 2 * stride, t + 3 * stride};
        uint2 v[4];
#pragma unroll
        for (int u = 0; u < 4; ++u) {
            const int mi = (int)(tt[u] / tiles_per_row);
            const int kk = (int)(tt[u] % tiles_per_row);
            const long long row = (long long)mi * 16 + (lane & 15);
            const uint8_t* addr =
                pool + row * k_stride + kk * 32 + ((lane >> 4) & 3) * 8;
            v[u] = *reinterpret_cast<const uint2*>(addr);
        }
#pragma unroll
        for (int u = 0; u < 4; ++u) acc += v[u].x + v[u].y;
    }
    for (; t < total_tiles; t += stride) {
        const int mi = (int)(t / tiles_per_row);
        const int kk = (int)(t % tiles_per_row);
        const long long row = (long long)mi * 16 + (lane & 15);
        const uint8_t* addr =
            pool + row * k_stride + kk * 32 + ((lane >> 4) & 3) * 8;
        const uint2 v = *reinterpret_cast<const uint2*>(addr);
        acc += v.x + v.y;
    }
    if (acc == 0xDEADBEEFu && tid == 0u) sink[0] = acc;
}

// 校验：uint4 逐元素求和（block 内归约 + atomicAdd）
__global__ void reduce16_kernel(const uint4* __restrict__ p, size_t n4,
                                unsigned long long* __restrict__ out) {
    unsigned long long acc = 0ull;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4;
         i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc += (unsigned long long)v.x + v.y + v.z + v.w;
    }
    __shared__ unsigned long long sm[kBlock];
    sm[threadIdx.x] = acc;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (threadIdx.x < s) sm[threadIdx.x] += sm[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0u) atomicAdd(out, sm[0]);
}

// ------------------------------------------------------------------ 工具 --
static bool is_finegrained(void* p) {
    mcPointerAttribute_t attr;
    std::memset(&attr, 0, sizeof(attr));
    if (mcPointerGetAttributes(&attr, p) != mcSuccess) return false;
    return (attr.allocationFlags & mcDeviceMallocFinegrained) != 0u ||
           (attr.type == mcMemoryTypeUnified && attr.isManaged);
}

struct LaunchArg {
    const void* p;
    size_t bytes;
    unsigned int* sink;
};

// A/B 交错计时：同一 (size,variant) 下 FG/RWK 轮流各测一轮取 min，抑制漂移。
// args[0]=FG args[1]=RWK；out_best[mi] 为 us，out_gbps[mi] 为带宽。
static void time_pair(void (*launch)(void*), LaunchArg* args, size_t bytes,
                      double* out_best, double* out_gbps) {
    mcEvent_t e0, e1;
    CHECK(mcEventCreate(&e0));
    CHECK(mcEventCreate(&e1));
    for (int i = 0; i < kWarmup; ++i)
        for (int mi = 0; mi < 2; ++mi) launch(&args[mi]);
    CHECK(mcDeviceSynchronize());
    double best[2] = {1e30, 1e30};
    for (int i = 0; i < kTrials; ++i) {
        for (int mi = 0; mi < 2; ++mi) {
            CHECK(mcEventRecord(e0, g_st));
            launch(&args[mi]);
            CHECK(mcEventRecord(e1, g_st));
            CHECK(mcDeviceSynchronize());
            float ms = 0.f;
            CHECK(mcEventElapsedTime(&ms, e0, e1));
            const double us = ms * 1000.0;
            if (us < best[mi]) best[mi] = us;
        }
    }
    mcEventDestroy(e0);
    mcEventDestroy(e1);
    for (int mi = 0; mi < 2; ++mi) {
        out_best[mi] = best[mi];
        out_gbps[mi] = (double)bytes / (best[mi] * 1e-6) / 1e9;
    }
}

static void launch_contig16(void* a) {
    LaunchArg* x = (LaunchArg*)a;
    contig16_kernel<<<kGrid, kBlock, 0, g_st>>>((const uint4*)x->p,
                                                x->bytes / 16, x->sink);
}

static void launch_afrag(void* a) {
    LaunchArg* x = (LaunchArg*)a;
    afrag_kernel<<<kGrid, kBlock, 0, g_st>>>((const uint8_t*)x->p,
                                             (long long)x->bytes, kKStride,
                                             x->sink);
}

// ------------------------------------------------------------------ main --
int main() {
    int dev = 0;
    if (const char* s = std::getenv("FG_BENCH_DEV")) dev = std::atoi(s);
    CHECK(mcSetDevice(dev));
    CHECK(mcStreamCreate(&g_st));

    static const size_t kSizes[] = {8u << 20, 32u << 20, 128u << 20,
                                    512u << 20};
    static const int kNumSizes = (int)(sizeof(kSizes) / sizeof(kSizes[0]));
    const size_t kMaxSize = kSizes[kNumSizes - 1];

    void* fg = nullptr;
    void* rwk = nullptr;
    CHECK(mcExtMallocWithFlags(&fg, kMaxSize, mcDeviceMallocFinegrained));
    CHECK(mcMalloc(&rwk, kMaxSize));
    std::printf("dev=%d  FG=%p(fg=%d)  RWK=%p(fg=%d)\n", dev, fg,
                (int)is_finegrained(fg), rwk, (int)is_finegrained(rwk));
    if (!is_finegrained(fg)) std::fprintf(stderr, "WARN: FG 缓冲被判定为非 fine-grained\n");

    unsigned int* sink = nullptr;
    CHECK(mcMalloc((void**)&sink, sizeof(unsigned int)));

    // 相同填充
    {
        size_t nwords = kMaxSize / 4;
        fill_kernel<<<kGrid, kBlock>>>((unsigned int*)fg, nwords, 0x12345678u);
        fill_kernel<<<kGrid, kBlock>>>((unsigned int*)rwk, nwords, 0x12345678u);
        CHECK(mcDeviceSynchronize());
    }
    // 校验两缓冲一致
    {
        unsigned long long *sf = nullptr, *sr = nullptr;
        CHECK(mcMalloc((void**)&sf, sizeof(unsigned long long)));
        CHECK(mcMalloc((void**)&sr, sizeof(unsigned long long)));
        CHECK(mcMemset(sf, 0, sizeof(unsigned long long)));
        CHECK(mcMemset(sr, 0, sizeof(unsigned long long)));
        reduce16_kernel<<<kGrid, kBlock>>>((const uint4*)fg, kMaxSize / 16, sf);
        reduce16_kernel<<<kGrid, kBlock>>>((const uint4*)rwk, kMaxSize / 16, sr);
        CHECK(mcDeviceSynchronize());
        unsigned long long f = 0, r = 0;
        CHECK(mcMemcpy(&f, sf, sizeof(f), mcMemcpyDeviceToHost));
        CHECK(mcMemcpy(&r, sr, sizeof(r), mcMemcpyDeviceToHost));
        std::printf("校验 reduce: FG=%llu RWK=%llu %s\n", f, r,
                    (f == r) ? "PASS" : "FAIL");
        if (f != r) return 2;
        mcFree(sf);
        mcFree(sr);
    }

    std::printf("\n%-8s %-10s %-4s %10s %12s %12s\n", "size", "variant", "mem",
                "us", "GB/s", "RWK vs FG");
    for (int si = 0; si < kNumSizes; ++si) {
        const size_t sz = kSizes[si];
        void* ptrs[2] = {fg, rwk};
        for (int vi = 0; vi < 2; ++vi) {
            const char* vn = vi == 0 ? "contig16" : "afrag";
            LaunchArg args[2] = {{ptrs[0], sz, sink}, {ptrs[1], sz, sink}};
            double best[2] = {0.0, 0.0}, gbps[2] = {0.0, 0.0};
            time_pair(vi == 0 ? launch_contig16 : launch_afrag, args, sz, best,
                      gbps);
            for (int mi = 0; mi < 2; ++mi) {
                char rel[32];
                if (mi == 0)
                    std::snprintf(rel, sizeof(rel), "-");
                else
                    std::snprintf(rel, sizeof(rel), "%+.1f%%",
                                  (gbps[1] / gbps[0] - 1.0) * 100.0);
                std::printf("%-8zu %-10s %-4s %10.2f %12.1f %12s\n", sz, vn,
                            mi == 0 ? "FG" : "RWK", best[mi], gbps[mi], rel);
            }
        }
    }

    mcFree(fg);
    mcFree(rwk);
    mcFree(sink);
    std::printf("\ndone\n");
    return 0;
}
