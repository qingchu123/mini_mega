// ============================================================================
// test/93.r19_l2_bench — r19_l2_big / l2_gemm_body 单 block 隔离扫描（L2 场景）
//
// L2 = 下投影 GEMM：C[M,H] = A[M,I] · W2[I,H]
//   M=count、K=k_in=I=2048（64 个 k-step）、N=H=6144（=24 组 × 256）；
//   输出 BF16；无 SwiGLU（只 pack）。
//   生产每 block 处理 g_l2=(H/256)/nslice_eff 组（NS=4 → 6 组 → N=1536）。
//
// 口径：单 block（1×512、8 warp）、grid=1；A 走 fragment（f2）；
//   5 次预热 + 冲刷 64MB 缓存 × REP=100 取均值，us/launch。
// ============================================================================
#include <mcr/mc_runtime.h>
#include <cstdio>
#include <cstdint>

#include "common.cuh"
#include "flags_defs.cuh"
#include "compute.cuh"
#include "compute_big.cuh"

namespace {
constexpr int kI = 2048, kH = 6144, kBN = 256;
constexpr int kGL2 = (kH / kBN) / kNSlice;   // 每 block N 组数 = 6
constexpr int kNwarp = 8;
constexpr int kThreads = 512;

uint8_t* g_flush = nullptr;
size_t g_flushB = 0;

template <int MT, int PF, int BATCH>
__global__ void __launch_bounds__(kThreads) k_l2(
        const uint8_t* ffn, const uint8_t* f2, const uint8_t* w2,
        uint16_t* out, int mt_total, int count, int ngroups) {
    const int lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
    r19_l2_big<MT, PF, BATCH>(ffn, f2, mt_total, w2, count, kI, kH, kBN, lane,
                              warp, kNwarp, nullptr, 0, ngroups, out, kH);
}

// l2_gemm_body（小 count 用）：<PF, NT, MT, BATCH, frag>
template <int PF, int MT, int BATCH>
__global__ void __launch_bounds__(kThreads) k_l2small(
        const uint8_t* ffn, const uint8_t* f2, const uint8_t* w2,
        uint16_t* out, int mt_total, int count, int ngroups) {
    const int lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
    l2_gemm_body<PF, kNTilesPerWarp, MT, BATCH, true>(
        ffn, w2, count, kI, kH, kBN, lane, warp, kNwarp, nullptr, 0, ngroups,
        out, kH, f2, mt_total);
}

template <class F>
double timeit(F launch, int rep) {
    mcStream_t st = nullptr; mcStreamCreate(&st);
    for (int w = 0; w < 5; ++w) launch(st);
    mcStreamSynchronize(st);
    if (g_flush) mcMemset(g_flush, 0xA5, g_flushB);
    mcEvent_t e0, e1; mcEventCreate(&e0); mcEventCreate(&e1);
    mcEventRecord(e0, st);
    for (int r = 0; r < rep; ++r) launch(st);
    mcEventRecord(e1, st); mcStreamSynchronize(st);
    float ms = 0; mcEventElapsedTime(&ms, e0, e1);
    mcEventDestroy(e0); mcEventDestroy(e1); mcStreamDestroy(st);
    return (double)ms * 1000.0 / rep;
}

#define MYCHK(c) do { mcError_t e_=(c); if(e_!=mcSuccess){ \
    std::fprintf(stderr,"[l2] MACA err %d at %d\n",(int)e_,__LINE__); return 1; } } while(0)
}  // namespace

int main() {
    MYCHK(mcSetDevice(0));
    const int Mmax = 512, mt = (Mmax + 15) / 16;
    const size_t ffnB = (size_t)Mmax * kI;         // FP8 A [M][I]
    const size_t w2B = (size_t)kEPR * kH * kI;     // [EPR][H][I]
    const size_t f2B = (size_t)(kI / 32) * mt * 512;
    const size_t outB = (size_t)Mmax * kH * 2;     // BF16 [M][H]
    uint8_t *ffn = nullptr, *w2 = nullptr, *f2 = nullptr;
    uint16_t* out = nullptr;
    MYCHK(mcMalloc((void**)&ffn, ffnB));
    MYCHK(mcMalloc((void**)&w2, w2B));
    MYCHK(mcMalloc((void**)&f2, f2B));
    MYCHK(mcMalloc((void**)&out, outB));
    MYCHK(mcMemset(ffn, 0x3c, ffnB));
    MYCHK(mcMemset(w2, 0x3c, w2B));
    MYCHK(mcMemset(f2, 0x3c, f2B));
    g_flushB = 64ull << 20;
    MYCHK(mcMalloc((void**)&g_flush, g_flushB));

    const int Ms[] = {16, 24, 32, 33, 48, 64, 96, 128};
    const int REP = 100;

    auto bench = [&](auto kfn, int M, int ng) {
        return (int)timeit([&](mcStream_t st) {
            kfn<<<1, kThreads, 0, st>>>(ffn, f2, w2, out, mt, M, ng);
        }, REP);
    };

    {   // 预热
        mcStream_t st = nullptr; mcStreamCreate(&st);
        for (int i = 0; i < 8; ++i) {
            k_l2<4,1,3><<<1,kThreads,0,st>>>(ffn,f2,w2,out,mt,64,kGL2);
            k_l2<6,1,3><<<1,kThreads,0,st>>>(ffn,f2,w2,out,mt,64,kGL2);
        }
        mcStreamSynchronize(st); mcStreamDestroy(st);
    }

    std::printf("r19_l2_big 单block扫描 K=I=2048 N=6组x256 block_n=256 8warp grid=1 "
                "REP=%d (us/launch)\n", REP);
    std::printf("%-5s %9s %9s %9s %9s %9s %9s %9s\n", "M",
                "2,1,3", "3,1,3", "4,1,3", "4,2,3", "6,1,3", "3,2,3", "6,1,6");

    for (int M : Ms) {
        int t[7];
        t[0] = bench(k_l2<2,1,3>, M, kGL2);
        t[1] = bench(k_l2<3,1,3>, M, kGL2);
        t[2] = bench(k_l2<4,1,3>, M, kGL2);
        t[3] = bench(k_l2<4,2,3>, M, kGL2);
        t[4] = bench(k_l2<6,1,3>, M, kGL2);
        t[5] = bench(k_l2<3,2,3>, M, kGL2);
        t[6] = bench(k_l2<6,1,6>, M, kGL2);
        std::printf("%-5d %9d %9d %9d %9d %9d %9d %9d\n",
                    M, t[0], t[1], t[2], t[3], t[4], t[5], t[6]);
    }

    // l2_gemm_body（小档）与 r19_l2_big 对照，M=16/32
    std::printf("\nl2_gemm_body vs r19_l2_big (K=2048,N=1536)\n");
    std::printf("%-5s %-16s %9s\n", "M", "variant", "us");
    for (int M : {16, 32}) {
        std::printf("%-5d %-16s %9d\n", M, "l2_body<2,1,6>", bench(k_l2small<2,1,6>, M, kGL2));
        std::printf("%-5d %-16s %9d\n", M, "l2_body<2,2,3>", bench(k_l2small<2,2,3>, M, kGL2));
        std::printf("%-5d %-16s %9d\n", M, "r19_l2<2,1,3>",  bench(k_l2<2,1,3>, M, kGL2));
        std::printf("%-5d %-16s %9d\n", M, "r19_l2<2,2,3>",  bench(k_l2<2,2,3>, M, kGL2));
    }

    // N 缩放（M=64）
    std::printf("\nN 缩放 (M=64, <4,1,3>)\n");
    std::printf("%-6s %-10s %9s\n", "N", "ngroups", "us");
    for (int ng : {1, 2, 3, 6, 12, 24}) {
        int batch = (ng >= 3) ? 3 : ((ng >= 2) ? 2 : 1);
        int us = (batch == 3) ? bench(k_l2<4,1,3>, 64, ng)
                 : (batch == 2) ? bench(k_l2<4,1,2>, 64, ng)
                                : bench(k_l2<4,1,1>, 64, ng);
        std::printf("%-6d %-10d %9d\n", ng * 256, ng, us);
    }

    MYCHK(mcFree(g_flush));
    MYCHK(mcFree(ffn)); MYCHK(mcFree(w2)); MYCHK(mcFree(f2)); MYCHK(mcFree(out));
    return 0;
}
