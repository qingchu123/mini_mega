// ============================================================================
// test/92.r19_l1_bench — r19_l1_big 单 block 隔离扫描
//
// 目的：把生产 L1 计算函数 r19_l1_big（M≤48 各档 + 回退）单独拿出来，在
//   单 block（1×512、8 warp）下扫描 M 与模板参数 (kMT,kPF,kBatch)，得到纯
//   L1 计算延迟，作为独立优化的基线。
//
// 口径：K=H=6144、N=2I=4096、block_n=256、nwarps=8、单 block、grid=1；
//   A 走 fragment（fbase 非空）；每个 M/变体采样 REP 次取中位，采样前冲刷
//   缓存（memset 64MB），逼近冷读。us/launch。
//
// 生产档（balanced，g_l1=4）：
//   M<=16 : r19<1,2,4>        17..32: r19<4,2,4>（g_l1=2 时 <4,2,2>）
//   33..48: r19<3,2,4>（或 r24，见 compute_big.cuh 分派）
// ============================================================================
#include <mcr/mc_runtime.h>
#include <cstdio>
#include <cstdint>
#include <algorithm>

#include "common.cuh"
#include "flags_defs.cuh"
#include "compute.cuh"
#include "compute_big.cuh"

namespace {
constexpr int kI = 2048, kH = 6144, kN = 4096, kBN = 256;
constexpr int kGL1 = (kN / kBN) / kNSlice;   // 每档 N 组数 = 4
constexpr int kNwarp = 8;
constexpr int kThreads = 512;

uint8_t* g_flush = nullptr;
size_t g_flushB = 0;

template <int MT, int PF, int BATCH>
__global__ void __launch_bounds__(kThreads) k_r19(
        const uint8_t* pool, const uint8_t* frag, const uint8_t* w1,
        uint8_t* ffn, int mt_total, int count) {
    const int lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
    r19_l1_big<MT, PF, BATCH>(pool, frag, mt_total, w1, 0, count, kH, kN, kBN,
                              lane, warp, kNwarp, nullptr, 0, kGL1, ffn,
                              nullptr, 0, kI);
}

template <class F>
double timeit(F launch, int rep) {
    mcStream_t st = nullptr; mcStreamCreate(&st);
    for (int w = 0; w < 5; ++w) launch(st);       // 预热
    mcStreamSynchronize(st);
    if (g_flush) mcMemset(g_flush, 0xA5, g_flushB);  // 冷缓存
    mcEvent_t e0, e1; mcEventCreate(&e0); mcEventCreate(&e1);
    mcEventRecord(e0, st);
    for (int r = 0; r < rep; ++r) launch(st);
    mcEventRecord(e1, st); mcStreamSynchronize(st);
    float ms = 0; mcEventElapsedTime(&ms, e0, e1);
    mcEventDestroy(e0); mcEventDestroy(e1); mcStreamDestroy(st);
    return (double)ms * 1000.0 / rep;
}

#define MYCHK(c) do { mcError_t e_=(c); if(e_!=mcSuccess){ \
    std::fprintf(stderr,"[r19] MACA err %d at %d\n",(int)e_,__LINE__); return 1; } } while(0)
}  // namespace

int main() {
    MYCHK(mcSetDevice(0));
    const int Mmax = 512, mt = (Mmax + 15) / 16;
    const size_t poolB = (size_t)Mmax * kH;
    const size_t w1B = (size_t)kEPR * kN * kH;
    const size_t ffnB = (size_t)Mmax * kI;
    const size_t fragB = (size_t)(kH / 32) * mt * 512;
    uint8_t *pool = nullptr, *w1 = nullptr, *ffn = nullptr, *frag = nullptr;
    MYCHK(mcMalloc((void**)&pool, poolB));
    MYCHK(mcMalloc((void**)&w1, w1B));
    MYCHK(mcMalloc((void**)&ffn, ffnB));
    MYCHK(mcMalloc((void**)&frag, fragB));
    MYCHK(mcMemset(pool, 0x3c, poolB));
    MYCHK(mcMemset(w1, 0x3c, w1B));
    MYCHK(mcMemset(frag, 0x3c, fragB));
    g_flushB = 64ull << 20;
    MYCHK(mcMalloc((void**)&g_flush, g_flushB));

    const int Ms[] = {16, 17, 24, 32, 33, 40, 48, 64};
    const int REP = 100;

    auto bench = [&](auto kfn, int M) {
        return (int)timeit([&](mcStream_t st) {
            kfn<<<1, kThreads, 0, st>>>(pool, frag, w1, ffn, mt, M);
        }, REP);
    };

    // 全局预热（避免首行冷启动；抖动主要来自首个 kernel 的模块/TLB/时钟）
    {
        mcStream_t st = nullptr; mcStreamCreate(&st);
        for (int i = 0; i < 8; ++i) {
            k_r19<2,2,4><<<1,kThreads,0,st>>>(pool,frag,w1,ffn,mt,64);
            k_r19<4,4,4><<<1,kThreads,0,st>>>(pool,frag,w1,ffn,mt,64);
        }
        mcStreamSynchronize(st); mcStreamDestroy(st);
    }

    std::printf("r19_l1_big 单block扫描  K=H=6144 N=4096 block_n=256 8warp grid=1  "
                "REP=%d  (us/launch)\n", REP);
    std::printf("%-5s %9s %9s %9s %9s %9s %9s %9s\n", "M",
                "2,2,4", "2,4,4", "2,8,4", "3,2,4", "3,4,4", "4,2,4", "4,4,4");

    for (int M : Ms) {
        int t[7];
        t[0] = bench(k_r19<2,2,4>, M);
        t[1] = bench(k_r19<2,4,4>, M);
        t[2] = bench(k_r19<2,8,4>, M);
        t[3] = bench(k_r19<3,2,4>, M);
        t[4] = bench(k_r19<3,4,4>, M);
        t[5] = bench(k_r19<4,2,4>, M);
        t[6] = bench(k_r19<4,4,4>, M);
        std::printf("%-5d %9d %9d %9d %9d %9d %9d %9d\n",
                    M, t[0], t[1], t[2], t[3], t[4], t[5], t[6]);
    }

    MYCHK(mcFree(g_flush));
    MYCHK(mcFree(pool)); MYCHK(mcFree(w1)); MYCHK(mcFree(ffn)); MYCHK(mcFree(frag));
    return 0;
}
