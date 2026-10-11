// ============================================================================
// test/94.r19_ksplit_bench — L1 场景 K 维切分（split-K）性能曲线
//
// L1 = gate/up 投影：C[M,2I] = A[M,H] · W1[H,2I]，K=H=6144、N=2I=4096。
// 本测例把 N 切 NSLICE=4 片（每片 1024=4 组×256，与生产 nslice=4 同），
// 再把 K 切 NK∈{1,2,4,8} 片；grid = (NSLICE, NK)，每 block 算一个 (nslice,kchunk)
// 的 FP32 部分和写入 partial[NK][M][2I]，再由 k_reduce 求和 + SwiGLU + 量化。
//
// 口径：单窗口 M 行、单机 32 SM；测 (GEMM 部分和 + 归约) 总时间，比较 NK 的
//   加速比与归约开销。us/launch。
// ============================================================================
#include <mcr/mc_runtime.h>
#include <cstdio>
#include <cstdint>

#include "common.cuh"
#include "flags_defs.cuh"
#include "compute.cuh"
#include "compute_big.cuh"

namespace {
constexpr int kH = 6144, kNtot = 4096, kI = 2048;
constexpr int kBN = 256;
constexpr int kNSLICE = 4;          // N 切 4 片
constexpr int kNslice = kNtot / kNSLICE;   // 1024 = 4 组
constexpr int kGslice = kNslice / kBN;     // 4
constexpr int kNwarp = 8;
constexpr int kThreads = 512;
constexpr int kKStep = 32;

// ---- GEMM 部分和：一个 (nslice, kchunk) block ----
template <int kMT, int kPF, int kBATCH>
__global__ void __launch_bounds__(kThreads) k_l1_partial(
        const uint8_t* A, const uint8_t* W, float* part, int M, int NK) {
    const int lane = threadIdx.x & 63, warp = threadIdx.x >> 6;
    const int ns = blockIdx.x;                    // 0..NSLICE-1
    const int kc = blockIdx.y;                    // 0..NK-1
    const int n0 = ns * kNslice;
    const int kchunk = kH / NK;                   // 每块 K 片字节数
    const int k0 = kc * kchunk;
    const int k_iters = kchunk / kKStep;
    const int n_per_warp = kBN / kNwarp;          // 32
    const uint8_t* A0 = A + (size_t)k0;           // A 的 K 起始偏移
    for (int c0 = 0; c0 < M; c0 += kMT * kM) {
        const int cc = (M - c0 < kMT * kM) ? (M - c0) : kMT * kM;
        const uint8_t* a_c = A0 + (size_t)c0 * kH;
        for (int g0 = 0; g0 < kGslice; g0 += kBATCH) {
            int nbase[kBATCH];
            #pragma unroll
            for (int q = 0; q < kBATCH; ++q)
                nbase[q] = n0 + (g0 + q) * kBN + warp * n_per_warp;
            R19Frag a_pipe[kPF][kMT];
            R19Frag2 b_pair[kPF][kBATCH];
            #pragma unroll
            for (int sp = 0; sp < kPF; ++sp) {
                if (sp >= k_iters) break;
                #pragma unroll
                for (int m = 0; m < kMT; ++m)
                    a_pipe[sp][m] = r19_load_a(a_c, kH, sp * kKStep, cc, m);
                #pragma unroll
                for (int q = 0; q < kBATCH; ++q)
                    b_pair[sp][q] =
                        r19_load_b16(W, nbase[q], k0 + sp * kKStep, kH);
            }
            R19Acc acc[kBATCH][kMT][2];
            #pragma unroll
            for (int q = 0; q < kBATCH; ++q)
                #pragma unroll
                for (int m = 0; m < kMT; ++m)
                    #pragma unroll
                    for (int t = 0; t < 2; ++t)
                        acc[q][m][t] = R19Acc{0.f, 0.f, 0.f, 0.f};
            #pragma unroll kPF
            for (int kk = 0; kk < k_iters; ++kk) {
                const int sp = kk & (kPF - 1);
                #pragma unroll
                for (int m = 0; m < kMT; ++m) {
                    #pragma unroll
                    for (int q = 0; q < kBATCH; ++q) {
                        const R19Frag2 bp = b_pair[sp][q];
                        const R19Frag b0 = {bp[0], bp[1]};
                        const R19Frag b1 = {bp[2], bp[3]};
                        acc[q][m][0] = __builtin_mxc_mma_16x16x32f8(
                            a_pipe[sp][m], b0, acc[q][m][0]);
                        acc[q][m][1] = __builtin_mxc_mma_16x16x32f8(
                            a_pipe[sp][m], b1, acc[q][m][1]);
                    }
                }
                const int nk = kk + kPF;
                if (nk < k_iters) {
                    #pragma unroll
                    for (int m = 0; m < kMT; ++m)
                        a_pipe[sp][m] =
                            r19_load_a(a_c, kH, nk * kKStep, cc, m);
                    #pragma unroll
                    for (int q = 0; q < kBATCH; ++q)
                        b_pair[sp][q] =
                            r19_load_b16(W, nbase[q], k0 + nk * kKStep, kH);
                }
            }
            float* p = part + (size_t)kc * M * kNtot;
            #pragma unroll
            for (int q = 0; q < kBATCH; ++q)
                #pragma unroll
                for (int m = 0; m < kMT; ++m)
                    #pragma unroll
                    for (int t = 0; t < 2; ++t)
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int row = c0 + m * kM + (lane >> 4) * 4 + i;
                            const int col = (g0 + q) * kBN + warp * n_per_warp +
                                            t * 16 + (lane & 15);
                            if (row < M)
                                p[(size_t)row * kNtot + n0 + col] =
                                    acc[q][m][t][i];
                        }
        }
    }
}

// ---- 归约 + SwiGLU + 量化：每线程一个输出 (m, hidden) ----
__global__ void __launch_bounds__(256) k_reduce(
        const float* part, uint8_t* ffn, int M, int NK) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * kI) return;
    const int m = idx / kI, p = idx % kI;
    float g = 0.f, u = 0.f;
    for (int kc = 0; kc < NK; ++kc) {
        const float* pr = part + (size_t)kc * M * kNtot + (size_t)m * kNtot;
        g += pr[2 * p];
        u += pr[2 * p + 1];
    }
    const float s = g / (1.0f + __expf(-g)) * u;
    ffn[(size_t)m * kI + p] = (uint8_t)__float2int_rn(s * 0.25f);
}

template <class F>
double timeit(F launch, int rep) {
    mcStream_t st = nullptr; mcStreamCreate(&st);
    for (int w = 0; w < 3; ++w) launch(st);
    mcStreamSynchronize(st);
    mcEvent_t e0, e1; mcEventCreate(&e0); mcEventCreate(&e1);
    mcEventRecord(e0, st);
    for (int r = 0; r < rep; ++r) launch(st);
    mcEventRecord(e1, st); mcStreamSynchronize(st);
    float ms = 0; mcEventElapsedTime(&ms, e0, e1);
    mcEventDestroy(e0); mcEventDestroy(e1); mcStreamDestroy(st);
    return (double)ms * 1000.0 / rep;
}

#define MYCHK(c) do { mcError_t e_=(c); if(e_!=mcSuccess){ \
    std::fprintf(stderr,"[ksplit] MACA err %d at %d\n",(int)e_,__LINE__); return 1; } } while(0)
}  // namespace

int main() {
    MYCHK(mcSetDevice(0));
    const int Mmax = 512;
    const size_t AB = (size_t)Mmax * kH;
    const size_t WB = (size_t)kNtot * kH;
    const size_t ffnB = (size_t)Mmax * kI;
    const size_t partB = (size_t)8 * Mmax * kNtot * sizeof(float);  // NK<=8
    uint8_t *A = nullptr, *W = nullptr, *ffn = nullptr;
    float* part = nullptr;
    MYCHK(mcMalloc((void**)&A, AB));
    MYCHK(mcMalloc((void**)&W, WB));
    MYCHK(mcMalloc((void**)&ffn, ffnB));
    MYCHK(mcMalloc((void**)&part, partB));
    MYCHK(mcMemset(A, 0x3c, AB));
    MYCHK(mcMemset(W, 0x3c, WB));

    const int Ms[] = {64, 128, 256};
    const int NKs[] = {1, 2, 4, 8};
    const int REP = 50;

    std::printf("L1 split-K  形状 N=4096(NSLICE=4 x 1024) K=6144  单窗口 (us/launch, 含归约)\n");
    std::printf("%-6s", "M");
    for (int nk : NKs) std::printf("  NK=%d", nk);
    std::printf("\n");

    for (int M : Ms) {
        std::printf("%-6d", M);
        for (int NK : NKs) {
            auto l = [&](mcStream_t st) {
                dim3 g(kNSLICE, NK);
                k_l1_partial<4, 2, 4><<<g, kThreads, 0, st>>>(A, W, part, M, NK);
                k_reduce<<<(M * kI + 255) / 256, 256, 0, st>>>(part, ffn, M, NK);
            };
            std::printf(" %7.1f", timeit(l, REP));
        }
        std::printf("\n");
    }

    std::printf("\n仅 GEMM 部分和（不含归约）(us/launch)\n");
    std::printf("%-6s", "M");
    for (int nk : NKs) std::printf("  NK=%d", nk);
    std::printf("\n");
    for (int M : Ms) {
        std::printf("%-6d", M);
        for (int NK : NKs) {
            auto l = [&](mcStream_t st) {
                dim3 g(kNSLICE, NK);
                k_l1_partial<4, 2, 4><<<g, kThreads, 0, st>>>(A, W, part, M, NK);
            };
            std::printf(" %7.1f", timeit(l, REP));
        }
        std::printf("\n");
    }

    MYCHK(mcFree(A)); MYCHK(mcFree(W)); MYCHK(mcFree(ffn)); MYCHK(mcFree(part));
    return 0;
}
