// ============================================================================
// compute_big.cuh — 大 count（>kM）任务独立编译单元
//
// 任务隔离：大 count 整体（pull + 分片旗 + L1 M 档位 + l1_prog + L2 M 档位 +
//   tail）放进 __noinline__ 的 r19_big_task；count<=kM 走 compute.cuh 的
//   r19_small_task（同 __noinline__），两者寄存器/栈互不污染。
// M 档位：外层 c0 按 kMT*16 行分组，组内 B 在 kMT 个 M-tile 间共享（权重遍数
//   = ceil(count/(kMT*16))）；acc[kBatch][kMT][kNT] 常驻，a_pipe/b_pair 预取环。
// A 布局：[R21] pool 行主序 + fragment 副本（local_data.frag_a/frag2），4-line
//   连续装载（消除 16-line 散射与 L2 straggler）。
// A-smem：[R24/R34] r24_*_big 走 A-smem 双缓冲（bsm 异步）+ 行区间 [c0_begin,
//   c0_end) + kMask 尾块掩码；MT8/B2（acc 128 顶格），KS=8 保 smem 64KB。
// 融合推送：[R29] L2 r24 每算完一个 (c0,g) 列块即 __stwt 推送，任务尾只写标记。
// 协议语义与基线逐字一致（wait-all、分片旗、l1_prog、l2_done、checksum、epoch）。
// 各优化专项（R21/R24/R27/R29/R34…）的实验与标定见 docs/ 与 perf.md。
// ============================================================================
#pragma once

#include "common.cuh"
#include "flags.cuh"
#include "combine.cuh"   // [R29] CombineFuseCtx/combine_push_chunk（融合推送）

// [OPT] 相位诊断（默认 0；写入 pre_ticks 第二 bank 的每 block L1/L2 累计周期）
#ifndef R24_DIAG_PHASE
#define R24_DIAG_PHASE 0
#endif

// [R29] COMBINE×L2 融合总开关（0 = 完全回退旧行为：L2 不推送、任务尾整段推送）
#ifndef R29_COMBINE_FUSE
#define R29_COMBINE_FUSE 1
#endif
// [R30] 列块计算顺序按 blockIdx 交错（打散各 block 同时推送同一批目标 rank 的
//   收端热点；只影响独立列块的处理次序，不改协议）
//   0=自然序；1=c0+g 都交错；2=只交错 c0；3=只交错 g
#ifndef R30_STAGGER
#define R30_STAGGER 1
#endif
// [R30] 一次延迟推送：上一列块的推送挪到下一列块 stage0 的 __syncthreads() 之后
//   （A 预取已发出、且省掉列块尾额外一次 __syncthreads；数据可见性由该 barrier
//   保证）。默认 0。
#ifndef R30_DEFER1
#define R30_DEFER1 0
#endif

// [CC·261006] 中等 count（33..48）改走 r24 A-smem 异步路径（A 只搬一遍/块，
//   消除每 warp 重复的 8× global A 散射装载，并借 bsm 异步隐藏 A 延迟）。
//   kBatch 用 CM_L1_BATCH（=4，偶数，gate/up 配对）。默认开。
#ifndef CM_MED_R24
#define CM_MED_R24 1
#endif

// [BALANCE·docs/20 §4.2] L1 split-K：把每个 (专家,窗口) 的 L1 沿 K=H 切成 kNK 段，
//   各段把 gate/up 的 FP32 部分和写/加到 l1_partial[slot][s]，最后一段读回→
//   SwiGLU→FP8→ffn_hidden/frag2，然后再走现有 l1_prog 发布。默认 0 = 现行为
//   （逐字不变：所有新路径经 if (L1_SPLITK&&...) 编译期消除）。1 = 开启验证。
#ifndef L1_SPLITK
#define L1_SPLITK 0
#endif


// [L1L2-SPLIT] 任务体阶段：拆成两个 kernel 时，L1 kernel 只跑 DISPATCH+L1，
//   L2 kernel 只跑 L2+tail。kPhaseBoth = 原合并路径（行为逐字不变）。
//   阶段判断全部用 if constexpr，编译期消除另一段 → 两 kernel 寄存器/栈独立。
enum class MegaPhase : int { Both = 0, L1 = 1, L2 = 2 };

// 大路径函数属性：默认 __noinline__（独立寄存器分配）；可用宏切换实验。
#ifndef R19_BIG_ATTR
#define R19_BIG_ATTR __noinline__
#endif
// 小 count 热路径任务体属性：默认独立 __noinline__（隔离）；可 -D 切 forceinline
#ifndef R19_SMALL_ATTR
#define R19_SMALL_ATTR __noinline__
#endif

// M 档位 / 预取槽 / N 组批（可用 -D 覆盖扫描）
// [R27·delivered] 大 count 档位由 MT4/B4 升为 MT8/B2：B（权重）的 M 遍数
//   由 count/(4*16)=4 降到 count/(8*16)=2（TPR=256），权重流量减半；kBatch
//   减半使 A（smem 搬运）流量 ×2（1.6→3.1MB/block，占 B 的 1/4，净收益大）。
//   回退档（非整除 count）固定 MT4，避免同一任务体内叠两份大 acc 实例
//   （MT8 全档位回退实测栈 1880B→2688B、71 机 TPR=128 +4.7%；固定 MT4 回退后
//   栈 2072B、TPR=128 回归收窄到 +2.8%）。
//   71 机同会话（R=8 TPR=256）：full 1921→1839µs（−4.3%），L1 909→713µs。
#ifndef R19_L1_MT
#define R19_L1_MT 8
#endif
#ifndef R19_L1_PF
#define R19_L1_PF 2
#endif
#ifndef R19_L1_BATCH
#define R19_L1_BATCH 2
#endif
// [BALANCE·r19balance] 49..64（满窗）档的 L1 N 组批宽：该档 kMT=4、g_l1=4，
//   用 4 组一次覆盖整片 N（A 只装一遍；默认 R19_L1_BATCH=2 会装两遍）。
//   仅当 g_l1 % R19_L1_BATCH64 == 0 时进入该档，否则回落原路径。
#ifndef R19_L1_BATCH64
#define R19_L1_BATCH64 4
#endif
#ifndef R19_L2_MT
#define R19_L2_MT 8
#endif
#ifndef R19_L2_PF
#define R19_L2_PF 1
#endif
#ifndef R19_L2_BATCH
#define R19_L2_BATCH 2
#endif

using R19Frag = __NATIVE_VECTOR__(2, uint32_t);
using R19Frag2 = __NATIVE_VECTOR__(4, uint32_t);
using R19Acc = __NATIVE_VECTOR__(4, float);
using R19F32x2 = __NATIVE_VECTOR__(2, float);

// BF16 打包（L2 输出；与 compute.cuh cm_pack_f2bf16 同构）
__device__ __forceinline__ uint32_t r19_pack_f2bf16(float a, float b) {
    const R19F32x2 v = {a, b};
    return __builtin_bit_cast(uint32_t, __builtin_mxc_cvt_pk_f32tobf16(v));
}

// A（激活）native-map 直装（与 compute.cuh cm_load_a 逐字同构）
__device__ __forceinline__ R19Frag r19_load_a(
        const uint8_t* __restrict__ pool, int k_stride, int k0, int count,
        int m_tile) {
    const int lane = threadIdx.x & 63;
    const int row = m_tile * kM + (lane & 15);
    R19Frag f = {0u, 0u};
    if (row < count)
        f = *reinterpret_cast<const R19Frag*>(
            pool + (size_t)row * k_stride + k0 + ((lane >> 4) & 3) * 8);
    return f;
}

// B（权重）16B/lane 配对 tile（与 compute.cuh cm_load_b_lane16 逐字同构）
__device__ __forceinline__ R19Frag2 r19_load_b16(
        const uint8_t* __restrict__ w, int n, int k, int k_hidden) {
    const int lane = threadIdx.x & 63;
    const size_t n_tiles = (size_t)(k_hidden >> 5);
    const uint8_t* tile =
        w + ((size_t)(n >> 5) * n_tiles + (size_t)(k >> 5)) * 1024;
    return *reinterpret_cast<const R19Frag2*>(tile + (size_t)lane * 16);
}

#ifdef R24_DIAG_NOB
__device__ __forceinline__ R19Frag2 r19_load_b16z(
        const uint8_t* __restrict__ w, int n, int k, int k_hidden) {
    return R19Frag2{0u, 0u, 0u, 0u};
}
#define R19_LDB(w, n, k, kh) r19_load_b16z(w, n, k, kh)
#else
#define R19_LDB(w, n, k, kh) r19_load_b16(w, n, k, kh)
#endif

// ============================================================================
// [R21] A fragment 布局：pool 仍行主序，另写一份 A 的 K-step×M-tile 分块副本到
//   local_data.frag_a/frag2（F[step][m][row][k]，step=k/32、m=row/16），使 L1/L2
//   的 A 装载落连续 512B（4-line），消除 16-line 散射与 L2 warp 末尾 straggler。
// ============================================================================
// F 装载：一步的 A fragment（lane 布局与 r19_load_a 完全一致）
__device__ __forceinline__ R19Frag r19_load_a_frag(
        const uint8_t* __restrict__ fbase, int kk, int m_abs, int mt_total,
        int row_limit) {
    const int lane = threadIdx.x & 63;
    const int row = m_abs * kM + (lane & 15);
    R19Frag f = {0u, 0u};
    if (row < row_limit)
        f = *reinterpret_cast<const R19Frag*>(
            fbase + ((size_t)kk * mt_total + m_abs) * 512 + (lane & 15) * 32 +
            (lane >> 4) * 8);
    return f;
}

// F 写：把 16B 源块（行 s2、字节偏移 o）写入 F
__device__ __forceinline__ void r19_store_a_frag(
        uint8_t* __restrict__ fbase, int s2, int o, const uint4 v,
        int mt_total) {
#ifdef CM_DIAG_NOFWRITE
    (void)fbase; (void)s2; (void)o; (void)v; (void)mt_total;
    return;
#endif
    const int c = o >> 4;   // 16B chunk 号
    uint8_t* p = fbase + (size_t)(s2 >> 4) * 512 + (s2 & 15) * 32 +
                 (size_t)(c >> 1) * mt_total * 512 + (c & 1) * 16;
    *reinterpret_cast<uint4*>(p) = v;
}

// [R21] L2 的 A（ffn_hidden）fragment 副本布局：同一 [step][m][16][32B] 分块，
// step = k/32（I=2048 → 64 步），m = row/16（win_cap/16 个）。L1 epilogue 按
// 字节写入；L2 一次 warp 装载落连续 512B（4 line）。
__device__ __forceinline__ R19Frag r19_load_a2_frag(
        const uint8_t* __restrict__ f2, int kk, int m_abs, int mt_total,
        int row_limit) {
    const int lane = threadIdx.x & 63;
    const int row = m_abs * kM + (lane & 15);
    R19Frag f = {0u, 0u};
    if (row < row_limit)
        f = *reinterpret_cast<const R19Frag*>(
            f2 + ((size_t)kk * mt_total + m_abs) * 512 + (lane & 15) * 32 +
            ((lane >> 4) & 3) * 8);
    return f;
}

// ------------------------------------------------------- 调度参数（循环不变量）--
// dispatch_pull_body 的循环不变量集中打包，按 const& 传给两个 noinline 任务体，
// 避免大参数表经 ABI 展开（也便于两个任务体共享）。
struct R19Sched {
    // [S-SHRINK] 只缓存派生量/间接指针；纯 a-> 字段直接经 a（落在 smem s_a）读。
    const MegaArgs* a;          // 输入视图（smem）
    int cycle;                  // [M_CYCLE] 本轮远端片
    uint8_t* pool_base;         // [A-RWK] pool 基址（nullptr → self+pool_off）
    uint32_t* bar_flags;        // = lf->bar[cycle]
    int* dbg;                   // = lf->dbg
    float* l1_chk;              // = lf->l1_chk
    float* l2_chk;              // = lf->l2_chk
    unsigned long long* pt;     // = lf->pre_ticks
    uint8_t* frag_a;            // = ld->frag_a[0]
    uint8_t* frag2;             // = ld->frag2[0]
    // [BALANCE·docs/20 §4.2] L1 split-K 部分和缓冲 / 段完成计数（kernel 内部派生）
    float* l1_partial;          // = ld->l1_partial[0][0][0]（FP32 gate/up 部分和）
    uint32_t* l1_part_cnt;      // = lf->l1_part_cnt[0][0]（每 (slot,s) 段计数）
    bool frag_all;
    uint64_t epoch;             // = current_time[0][1]
    int merge;
    int nslice_eff;
    int mt_total;
    unsigned long long* ctr = nullptr;
};

// [R21] F 槽基址（已弃用：fragment 改用 local_data.frag_a 专用缓冲）。
// ---------------------------------------------------------------------- L1 --
// 一组 M 块（kMT 个 16 行 tile）共享 B 的 K 循环：SwiGLU + FP8 量化回写。
// M 循环全静态：调用者按 kMT*kM 行分组并传 cc；尾组不足时 r19_load_a 对
// row>=cc 零填充、写回按 row<cc 跳过，数值等价。
// [R21] A 装载选择：fbase 非空走 fragment 布局（4 line/指令），否则行主序池
__device__ __forceinline__ R19Frag r19_zero_a() { return R19Frag{0u, 0u}; }
#ifdef R24_DIAG_NOALOAD
#define R21_LDA(step, m) r19_zero_a()
#else
#define R21_LDA(step, m)                                                    \
    ((fbase != nullptr)                                                     \
         ? r19_load_a_frag(fbase, k_step_begin + (step), m_abs0 + (m),      \
                           mt_total, m_abs0 * kM + count)                   \
         : r19_load_a(pool, k_hidden, (k_step_begin + (step)) * kKStep,     \
                      count, (m)))
#endif

// [R21] L2 的 A 装载选择：f2 非空走 fragment 布局
#ifdef R24_DIAG_NOALOAD
#define R21_L2A(step, m) r19_zero_a()
#else
#define R21_L2A(step, m)                                                    \
    ((f2 != nullptr)                                                        \
         ? r19_load_a2_frag(f2, (step), m_abs0 + (m), mt_total,             \
                            m_abs0 * kM + count)                            \
         : r19_load_a(a_pool, k_in, (step) * kKStep, count, (m)))
#endif

// 大 count L1：外层 c0 按 kMT*16 行分组，组内 B 只读一遍。
// [R21] fbase = A fragment 副本基址（nullptr 走池行主序）；m_abs0 = 本 c0 的
// 起始 M tile 号；mt_total = win_cap/16（F 的 m 维长度）。
template <int kMT, int kPF, int kBatch>
__device__ __forceinline__ void r19_l1_big(
        const uint8_t* pool, const uint8_t* fbase, int mt_total,
        const uint8_t* w1, int expert, int count,
        int k_hidden, int n_total, int block_n, int lane, int warp, int nwarps,
        float* checksum, int n_group_begin, int n_group_count,
        uint8_t* ffn_out, uint8_t* ffn2, int out_col_base, int k_out,
        int m_tile_base = 0,
        // [BALANCE·docs/20 §4.2] K 范围（默认全 K，既有调用点不变）：
        //   k_step_begin = 起始 K step（= 全局 k/32），k_steps = 本调用的 step 数
        //   （<0 → 全 K）。l1_part 非空进入「部分和 epilogue」：跳过 SwiGLU/FP8，
        //   gate/up 原值 FP32：seg==0 直写、否则 atomicAdd 到 l1_partial。
        int k_step_begin = 0, int k_steps = -1,
        float* l1_part = nullptr, int l1_seg = 0, int l1_row0 = 0) {
    const uint8_t* w_e = w1 + (size_t)expert * n_total * k_hidden;
    float acc_chk = 0.0f;
    for (int c0 = 0; c0 < count; c0 += kMT * kM) {
        const int cc = (count - c0 < kMT * kM) ? (count - c0) : kMT * kM;
        const uint8_t* a_c = pool + (size_t)c0 * k_hidden;
        uint8_t* f_c = (ffn_out != nullptr) ? ffn_out + (size_t)c0 * k_out
                                            : nullptr;
        const int m_abs0 = (c0 >> 4) + m_tile_base;
        int g0 = n_group_begin;
        for (; g0 + kBatch <= n_group_begin + n_group_count; g0 += kBatch) {
            constexpr int kKStep = 32;
            constexpr int kNT = 2;
            const int k_iters = (k_steps >= 0) ? k_steps : (k_hidden / kKStep);
            const int n_per_warp = block_n / nwarps;
            int nbase[kBatch];
            #pragma unroll
            for (int q = 0; q < kBatch; ++q)
                nbase[q] = (g0 + q) * block_n + warp * n_per_warp;
            R19Frag a_pipe[kPF][kMT];
            R19Frag2 b_pair[kPF][kBatch];
            #pragma unroll
            for (int sp = 0; sp < kPF; ++sp) {
                if (sp >= k_iters) break;
                #pragma unroll
                for (int m = 0; m < kMT; ++m)
                    a_pipe[sp][m] = R21_LDA(sp, m);
                #pragma unroll
                for (int q = 0; q < kBatch; ++q)
                    b_pair[sp][kBatch - 1 - q] =
                        R19_LDB(w_e, nbase[q],
                                (k_step_begin + sp) * kKStep, k_hidden);
            }
            R19Acc acc[kBatch][kMT][kNT];
            #pragma unroll
            for (int q = 0; q < kBatch; ++q)
                #pragma unroll
                for (int m = 0; m < kMT; ++m)
                    #pragma unroll
                    for (int t = 0; t < kNT; ++t)
                        acc[q][m][t] = R19Acc{0.f, 0.f, 0.f, 0.f};
            #pragma unroll kPF
            for (int kk = 0; kk < k_iters; ++kk) {
                const int sp = kk & (kPF - 1);
                #pragma unroll
                for (int m = 0; m < kMT; ++m) {
                    #pragma unroll
                    for (int q = 0; q < kBatch; ++q) {
                        const R19Frag2 bp = b_pair[sp][kBatch - 1 - q];
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
                        a_pipe[sp][m] = R21_LDA(nk, m);
                    #pragma unroll
                    for (int q = 0; q < kBatch; ++q)
                        b_pair[sp][kBatch - 1 - q] =
                            R19_LDB(w_e, nbase[q],
                                    (k_step_begin + nk) * kKStep, k_hidden);
                }
            }
            #pragma unroll
            for (int pi = 0; pi < kBatch / 2; ++pi) {
                const int qa = 2 * pi, qb = 2 * pi + 1;
                #pragma unroll
                for (int m = 0; m < kMT; ++m) {
                    #pragma unroll
                    for (int t = 0; t < kNT; ++t) {
                        // [BALANCE·docs/20 §4.2] 部分和 epilogue：不做 SwiGLU/FP8，
                        //   把 gate/up 原 acc 以 FP32 写入 l1_partial[..][2*hc(+1)]。
                        if (l1_part != nullptr) {
                            const int hc =
                                (((g0 - n_group_begin) >> 1) + pi) * block_n +
                                warp * n_per_warp + t * 16 + (lane & 15);
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int row = m * kM + (lane >> 4) * 4 + i;
                                if (row < cc) {
                                    float* pp = l1_part +
                                        (size_t)(l1_row0 + c0 + row) *
                                            kL1SliceCols + 2 * hc;
                                    const float gv = acc[qa][m][t][i];
                                    const float uv = acc[qb][m][t][i];
                                    if (l1_seg == 0) {
                                        pp[0] = gv;
                                        pp[1] = uv;
                                    } else {
                                        atomicAdd(pp, gv);
                                        atomicAdd(pp + 1, uv);
                                    }
                                }
                            }
                            continue;
                        }
                        R19Acc sw;
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const float gv = acc[qa][m][t][i];
                            const float uv = acc[qb][m][t][i];
                            sw[i] = gv / (1.0f + __expf(-gv)) * uv;
                            acc_chk += sw[i];
                        }
                        if (f_c != nullptr || ffn2 != nullptr) {
                            const uint32_t packed =
                                __builtin_mxc_cvt_pk4_f32tof8_cfg(
                                    sw, _MACA_FROUND_TO_NEAREST_EVEN,
                                    _MACA_DENORMAL_FLUSH_NONE,
                                    _MACA_SRC_NO_MODIFIER, false, true);
                            const int col = out_col_base +
                                (((g0 - n_group_begin) >> 1) + pi) * block_n +
                                warp * n_per_warp + t * 16 + (lane & 15);
                            uint8_t* f2p = nullptr;
                            if (ffn2 != nullptr)
                                f2p = ffn2 +
                                    ((size_t)(col >> 5) * mt_total +
                                     (m_abs0 + m)) * 512 +
                                    (lane >> 4) * 4 * 32 + (col & 31);
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                const int row = m * kM + (lane >> 4) * 4 + i;
                                if (row < cc) {
                                    const uint8_t byte = (uint8_t)(packed >> (8 * i));
                                    if (f_c != nullptr)
                                        f_c[(size_t)row * k_out + col] = byte;
#ifndef R24_DIAG_NOMAINT
                                    if (f2p != nullptr) f2p[i * 32] = byte;
#endif
                                }
                            }
                        }
                    }
                }
            }
        }
        // 余数组（组对语义下必为偶数）：kBatch=2 收尾，兼容非生产几何。
        if (g0 + 2 <= n_group_begin + n_group_count) {
            // 复用同一回退入口处理余下 gate/up 对；把当前 M tile 的
            // fragment 基址折到局部 0，保持 R21_LDA/R21_L2A 的布局契约。
            r19_l1_big<kMT, kPF, 2>(
                a_c, fbase,
                mt_total, w1, expert, cc, k_hidden, n_total, block_n, lane,
                warp, nwarps, checksum, g0, n_group_count - (g0 - n_group_begin),
                f_c, ffn2, out_col_base, k_out, m_abs0,
                k_step_begin, k_steps, l1_part, l1_seg, l1_row0 + c0);
        }
    }
    if (checksum != nullptr && lane == 0)
        atomicAdd(checksum, acc_chk + (float)count * 0.0f);
}

// [regspill] 大档 L1（kMT>=2）隔离 shim：__noinline__ 独立寄存器/栈，避免其
//   acc 峰值（最多 128 floats）传染 r19_big_task。forceinline 的 r19_l1_big 会
//   内联进本 shim，故 shim 函数体即该档的独立帧。small task 的 kMT=1 不走此路。
template <int kMT, int kPF, int kBatch>
__device__ R19_L1BIG_ISO void r19_l1_big_iso(
        const uint8_t* pool, const uint8_t* fbase, int mt_total,
        const uint8_t* w1, int expert, int count, int k_hidden, int n_total,
        int block_n, int lane, int warp, int nwarps, float* checksum,
        int n_group_begin, int n_group_count, uint8_t* ffn_out, uint8_t* ffn2,
        int out_col_base, int k_out, int m_tile_base = 0) {
    r19_l1_big<kMT, kPF, kBatch>(
        pool, fbase, mt_total, w1, expert, count, k_hidden, n_total, block_n,
        lane, warp, nwarps, checksum, n_group_begin, n_group_count, ffn_out,
        ffn2, out_col_base, k_out, m_tile_base);
}

// ============================================================================
// [BALANCE·docs/20 §4.2] L1 split-K 归约：读 l1_partial 的 NK 段累加结果
//   （gate/up 原值，布局 [row][2*hc(+1)]）→ SwiGLU → FP8 → ffn_hidden/frag2。
//   布局公式与 r19_l1_big 的 epilogue 逐字同构（:336-344/:579-589）。
// ============================================================================
template <int kMT>
__device__ __forceinline__ void r19_l1_reduce(
        const float* part, int count, int block_n, int lane, int warp,
        int nwarps, int out_col_base, int k_out, uint8_t* ffn_out,
        uint8_t* ffn2, int mt_total, int m_tile_base = 0) {
    constexpr int kNT = 2;
    const int n_per_warp = block_n / nwarps;
    // 本 slice 的输出 tile 组数 = (I/nslice_eff)/block_n = (kL1SliceCols/2)/block_n
    const int g_pairs = (kL1SliceCols / 2) / block_n;
    for (int c0 = 0; c0 < count; c0 += kMT * kM) {
        const int cc = (count - c0 < kMT * kM) ? (count - c0) : kMT * kM;
        uint8_t* f_c = (ffn_out != nullptr) ? ffn_out + (size_t)c0 * k_out
                                            : nullptr;
        const int m_abs0 = (c0 >> 4) + m_tile_base;
        for (int gi = 0; gi < g_pairs; ++gi) {
            #pragma unroll
            for (int m = 0; m < kMT; ++m) {
                #pragma unroll
                for (int t = 0; t < kNT; ++t) {
                    const int hc = gi * block_n + warp * n_per_warp + t * 16 +
                                   (lane & 15);
                    R19Acc sw = {0.f, 0.f, 0.f, 0.f};
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int row = m * kM + (lane >> 4) * 4 + i;
                        if (row < cc) {
                            const float* pr =
                                part + (size_t)(c0 + row) * kL1SliceCols +
                                2 * hc;
                            const float gv = pr[0], uv = pr[1];
                            sw[i] = gv / (1.0f + __expf(-gv)) * uv;
                        }
                    }
                    if (f_c != nullptr || ffn2 != nullptr) {
                        const uint32_t packed =
                            __builtin_mxc_cvt_pk4_f32tof8_cfg(
                                sw, _MACA_FROUND_TO_NEAREST_EVEN,
                                _MACA_DENORMAL_FLUSH_NONE,
                                _MACA_SRC_NO_MODIFIER, false, true);
                        const int col = out_col_base + hc;
                        uint8_t* f2p = nullptr;
                        if (ffn2 != nullptr)
                            f2p = ffn2 +
                                  ((size_t)(col >> 5) * mt_total + (m_abs0 + m)) *
                                      512 +
                                  (lane >> 4) * 4 * 32 + (col & 31);
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int row = m * kM + (lane >> 4) * 4 + i;
                            if (row < cc) {
                                const uint8_t byte = (uint8_t)(packed >> (8 * i));
                                if (f_c != nullptr)
                                    f_c[(size_t)row * k_out + col] = byte;
#ifndef R24_DIAG_NOMAINT
                                if (f2p != nullptr) f2p[i * 32] = byte;
#endif
                            }
                        }
                    }
                }
            }
        }
    }
}

// [BALANCE·docs/20 §4.2] 块内 split-K：同一任务的 block 自己跑 kNK 个 K 段，
//   各段（seg）用 r19_l1_big 的部分和 epilogue 写/加到 part[slot][s]；
//   tid0 用 l1_part_cnt 的 atomicAdd 判「最后一段」触发归约（写 ffn/frag2）。
//   计数在本窗第一段（seg==0）开始前清零（窗口复用处，不依赖残留值）。
//   归约完成于调用者发布 l1_prog 之前。part/cnt 已由调用者定位到本 (slot,s)；
template <int kMT, int kPF, int kBatch>
__device__ __noinline__ void r19_l1_splitk_iso(
        const uint8_t* pool_w, const uint8_t* fbase, int mt_total,
        const uint8_t* w1, int expert, int count, int k_hidden, int n_total,
        int block_n, int lane, int warp, int nwarps, int n_group_begin,
        int n_group_count, int out_col_base, int k_out, float* part,
        uint32_t* cnt, uint8_t* ffn_out, uint8_t* ffn2) {
    const int chunk = (k_hidden / kNK) / 32;   // 每 K 段 step 数（kKStep=32）
    __shared__ int s_last;
    #pragma unroll 1
    for (int seg = 0; seg < kNK; ++seg) {
        // [BALANCE·docs/20 §4.2] 窗口复用处清零：本窗第一段开始前把计数归零
        //   （不依赖初始/上一代的残留值；ring 门保证同槽前代已结束）。
        if (seg == 0 && lane == 0 && warp == 0)
            *reinterpret_cast<volatile uint32_t*>(cnt) = 0u;
        r19_l1_big<kMT, kPF, kBatch>(
            pool_w, fbase, mt_total, w1, expert, count, k_hidden, n_total,
            block_n, lane, warp, nwarps, nullptr, n_group_begin, n_group_count,
            nullptr, nullptr, out_col_base, k_out, /*m_tile_base=*/0,
            /*k_step_begin=*/seg * chunk, /*k_steps=*/chunk,
            /*l1_part=*/part, /*l1_seg=*/seg, /*l1_row0=*/0);
        // 全 block 各 warp 的部分和写必须完成后再计段，否则 tid0 可能先于其它
        //   warp 的 epilogue 完成而触发归约 → 读到未写完的部分和。
        __syncthreads();
        if (lane == 0 && warp == 0) {
            __threadfence();
            const unsigned old = atomicAdd(cnt, 1u);
            s_last = (old == (unsigned)(kNK - 1));
        }
        __syncthreads();
    }
    if (s_last) {
        __threadfence();
        r19_l1_reduce<kMT>(part, count, block_n, lane, warp, nwarps,
                           out_col_base, k_out, ffn_out, ffn2, mt_total);
        __syncthreads();
    }
}

// ============================================================================
// [BALANCE·docs/20 §4.2b] 跨 block K 切：一段（kseg）的 L1 部分和 + 最后段归约。
//   与 r19_l1_splitk_iso 的区别：段不再由同一 block 顺序执行，而是不同 block
//   各跑一段（part 已由 owner 清零），全部段对 part 做 FP32 atomicAdd；
//   l1_part_cnt atomic 判最后一段 → r19_l1_reduce（SwiGLU→FP8→ffn/frag2）→
//   由该段发布 l1_prog（owner 在任务体下方只等不发布）。
// ============================================================================
#if R19_BLOCK_KSPLIT
template <int kMT>
__device__ __noinline__ void r19_l1_ksplit_iso(
        const uint8_t* pool_w, const uint8_t* fbase, int mt_total,
        const uint8_t* w1, int expert, int count, int k_hidden, int n_total,
        int block_n, int lane, int warp, int nwarps, int n_group_begin,
        int n_group_count, int out_col_base, int k_out, float* part,
        uint32_t* cnt, uint8_t* ffn_out, uint8_t* ffn2, int kseg,
        uint32_t* bar_flags, int l1_prog_idx, int pool_slot, int slice,
        unsigned gen, int nslice_eff, long long max_spins, int* dbg) {
    (void)nslice_eff; (void)max_spins; (void)dbg;
    const int chunk = (k_hidden / 32) / kNK;   // 每 K 段 step 数
    __shared__ int s_last;
    // 所有段一律 atomicAdd（part 已由 owner 清零）；l1_seg 传 1 → 走 atomicAdd 分支。
    r19_l1_big<kMT, R19_L1_PF, 4>(
        pool_w, fbase, mt_total, w1, expert, count, k_hidden, n_total,
        block_n, lane, warp, nwarps, nullptr, n_group_begin, n_group_count,
        nullptr, nullptr, out_col_base, k_out, /*m_tile_base=*/0,
        /*k_step_begin=*/kseg * chunk, /*k_steps=*/chunk,
        /*l1_part=*/part, /*l1_seg=*/1, /*l1_row0=*/0);
    __syncthreads();
    if (lane == 0 && warp == 0) {
        __threadfence();
        const unsigned old = atomicAdd(cnt, 1u);
        s_last = (old == (unsigned)(kNK - 1));
    }
    __syncthreads();
    if (s_last) {
        __threadfence();
        r19_l1_reduce<kMT>(part, count, block_n, lane, warp, nwarps,
                           out_col_base, k_out, ffn_out, ffn2, mt_total);
        __syncthreads();
        if (threadIdx.x == 0)
            l1_progress_publish(bar_flags, l1_prog_idx, pool_slot, slice, gen,
                                nslice_eff);
    }
}

// [BALANCE·docs/20 §4.2b] owner 的 L1 入口等待：等齐本窗全部 nslice_eff 个
//   slice 的 l1_prog 旗（由各 slice 的最后 K 段发布），不自行发布。
__device__ __forceinline__ void r19_l1_wait_all(
        const R19Sched& S, int pool_slot, int w) {
    const int tid = threadIdx.x;
#ifdef NO_GEN_FIX
    const unsigned gen = (unsigned)S.epoch;
#else
    const unsigned gen =
        (unsigned)S.epoch + (unsigned)(w / (kRing > 0 ? kRing : 1));
#endif
    __syncthreads();
    if (tid == 0) {
        for (int s2 = 0; s2 < S.nslice_eff; ++s2) {
            long long spin = 0;
            while (*(volatile const uint32_t*)(S.bar_flags + S.a->l1_prog_idx +
                                               (size_t)pool_slot *
                                                   S.nslice_eff +
                                               s2) != gen) {
                if (++spin > S.a->max_spins) {
                    if (S.dbg != nullptr) atomicAdd(S.dbg + 4, 1);
                    break;
                }
            }
        }
        __threadfence();
    }
    __syncthreads();
}
#endif

// ============================================================================
// [R24] L1 的 A 走 smem：warp 协作 bsm 异步搬运（__builtin_mxc_ldg_b128_bsm）。
//   每 stage = kKS 个 K step × kMT 个 M-tile；smem 布局同 F_A [j][m][row][32B]，
//   双缓冲（stage st 消费时 st+1 在飞）；B 直装 + 预取环 kBPF。[R34] 支持任意
//   count（行区间 [c0_begin,c0_end)：整块无掩码 + 尾块 kMask 掩码）；k_iters 须为
//   kKS 整数倍，否则回退 r19_l1_big。[R27] KS=8 使 MT8 双缓冲保持 64KB。
//   [OPT-KCAP-REMOVE] MT8 大档寄存器/栈由 R19_GEMM_ATTR=__noinline__ 隔离。
// ============================================================================
#ifndef R24_KS
#define R24_KS 8
#endif
// B 预取环深（2 的幂；R24 路径专用）
#ifndef R24_BPF
#define R24_BPF 2
#endif
// 阶段发布等待模式：默认仅等 bsm 搬运（不拖住 B 直装的 gvmcnt 流）
#ifndef R24_BSMCNT_WAIT
#define R24_BSMCNT_WAIT 1
#endif
#if R24_BSMCNT_WAIT
#define R24_WAIT_BSM() __builtin_mxc_arrive(4096)
#else
#define R24_WAIT_BSM() __builtin_mxc_arrive(64)
#endif
// smem 缓冲的 M 档位（L1/L2 共享同一块 smem，取两路较大者）
#ifndef R24_MT
#define R24_MT ((R19_L1_MT > R19_L2_MT) ? R19_L1_MT : R19_L2_MT)
#endif

// ============================================================================
// [D1·B-smem] B（权重）经 smem 异步双缓冲（默认关，R24_BSMEM_B=0 → 逐字等价）。
//   开启时 r24_*_big 的 B 也用 __builtin_mxc_ldg_b128_bsm 搬入 smem 环，与 A 的
//   bsm 异步流水对齐；B fragment 由 smem LDS.128 读，从而把 B 的装载从寄存器
//   scoreboard 摘除（诊断：B 裸露 ≈211µs，见 perf.md）。
//   环深 R24_BRING（2 的幂，默认 2）；每 kstep 每线程发 kBatch 个 16B bsm。
//   **等待队列**：该 builtin 的拷贝计入 gvmcnt（不是 bsmcnt！实测 arrive(4096)
//   =bsmcnt(0) 等不到它，必须 arrive(64+N)=gvmcnt(N)），标准段允许「环深-1」个
//   kstep 的 B 在飞、尾部 gvmcnt(0) 全等；A 的可见性仍由 stage 顶 bsmcnt(0)+
//   __syncthreads() 保证。
//   smem 预算：A 双缓冲后紧接 B 环（实例偏移 = 2*kMT*kKS*512），全局数组取各
//   实例最大值（med L1 <3,16,4> = 48KB A + 64KB B = 112KB，见 kABSharedBytes）。
//   [结论] 实测 s2 top50 中性偏负（+1.8%，763 vs 749µs）：B 为带宽受限而非可
//   隐藏延迟，故默认关（见 perf.md）。
// ============================================================================
#ifndef R24_BSMEM_B
#define R24_BSMEM_B 0
#endif
#ifndef R24_BRING
#define R24_BRING 2
#endif
// B 环 smem 字节数 = 环深 × kBatch 组 × kComputeWarps warp × 1024B/kstep/warp
#define R24_BSM_BYTES(kBatch) ((R24_BRING) * (kBatch) * kComputeWarps * 1024)
// kernel 作用域 A+B 共享数组字节数。默认（B 关）保持原 A 双缓冲 64KB 不变，避免
//   缩小 L1 carveout 拖慢 B 直装；仅 R24_BSMEM_B=1 时放大到 med L1 <3,16,4> 需求。
#if R24_BSMEM_B
#define kABSharedBytes (2 * 3 * 16 * 512 + R24_BSM_BYTES(4))
#else
#define kABSharedBytes (2 * R24_MT * R24_KS * 512)
#endif
// B 的 bsm 拷贝计入 gvmcnt 队列（实测 bsmcnt(0) 不能等它；gvmcnt(N) 才行）。
//   N = 允许在飞的每线程 16B 拷贝笔数（0 = 全等）。
#define R24_WAIT_BSM_N(N) __builtin_mxc_arrive(64 + (N))
// 主循环 B 等待阈值：标准段允许「环深-1」个 kstep 的 B 在飞（不含 A）；可用
//   -DR24_BSM_WAITSTD=0 强制每 kstep 全等（调试/校验收敛）。
#ifndef R24_BSM_WAITSTD
#define R24_BSM_WAITSTD ((kBRing - 1) * kBatch)
#endif

// 把一个 stage 的 A 协作搬入 smem（每线程 kMT*kKS*32/512 片 16B）
template <int kMT, int kKS, bool kMask = false, int kMTAct = kMT>
__device__ __forceinline__ void r24_issue_a(
        uint8_t* sbuf, const uint8_t* fbase, int step0, int mt_total,
        int m_abs0, int cc = 0) {
    (void)cc;
    constexpr int kPieces = kMT * kKS * 32;
    static_assert(kPieces % 512 == 0, "stage pieces must divide 512 threads");
    const int tid = (int)threadIdx.x;
    // [R36] kMTAct = 尾块活跃 M tile 数（编译期）：越界 tile 不搬运（A 流量按需）
    constexpr int n_m = kMTAct;
    #pragma unroll
    for (int i = 0; i < kPieces / 512; ++i) {
        const int p = tid + i * 512;
        const int f = p >> 5;            // fragment = j*kMT + m
        const int off = (p & 31) << 4;
        const int j = f / kMT, m = f % kMT;
        if constexpr (kMask) {
            if (m >= n_m) continue;      // 越界 tile 不搬
        }
        // [R34] 尾块（kMask）m_abs0+kMT 可越出 ceil(win_cap/16) 的 F 范围：
        //   越界 tile 折叠到最后一 tile（数据无效但内存安全，掩码丢弃）
        int mb = m_abs0 + m;
        if constexpr (kMask) {
            if (mb >= mt_total) mb = mt_total - 1;
        }
        const uint8_t* g = fbase +
            ((size_t)(step0 + j) * mt_total + mb) * 512 + off;
        __builtin_mxc_ldg_b128_bsm(sbuf + (size_t)p * 16,
                                   const_cast<uint8_t*>(g), 0, -1, true, true,
                                   false, true);
    }
}

// 从 stage smem 读 native fragment（lane 布局与 r19_load_a_frag 一致）
template <int kMT>
__device__ __forceinline__ R19Frag r24_lds_a(const uint8_t* sbuf, int j, int m) {
    const int lane = (int)(threadIdx.x & 63);
    return *reinterpret_cast<const R19Frag*>(
        sbuf + (size_t)(j * kMT + m) * 512 + (lane & 15) * 32 +
        ((lane >> 4) & 3) * 8);
}

// [D1·B-smem] 把一个 kstep 的 B（kBatch 组 × warp 私有 32N×32K tile）bsm 异步搬
//   入 smem 环槽 s。源地址与 r19_load_b16 逐字一致（lane*16 偏移），数值不变。
template <int kBatch>
__device__ __forceinline__ void r24_issue_b(
        uint8_t* bmem, int s, const uint8_t* w, const int* nbase, int kstep,
        int k_hidden) {
    const int lane = (int)(threadIdx.x & 63);
    const int warp = (int)(threadIdx.x >> 6);
    const size_t n_tiles = (size_t)(k_hidden >> 5);
    #pragma unroll
    for (int q = 0; q < kBatch; ++q) {
        uint8_t* d = bmem +
            (((size_t)s * kBatch + q) * kComputeWarps + warp) * 1024 +
            (size_t)lane * 16;
#ifdef R24_DIAG_NOB
        *reinterpret_cast<R19Frag2*>(d) = R19Frag2{0u, 0u, 0u, 0u};
#else
        const uint8_t* g = w +
            ((size_t)(nbase[q] >> 5) * n_tiles + (size_t)kstep) * 1024 +
            (size_t)lane * 16;
        __builtin_mxc_ldg_b128_bsm(d, const_cast<uint8_t*>(g), 0, -1, true,
                                   true, false, true);
#endif
    }
}

// [D1·B-smem] 从环槽 s 读第 q 组的 B fragment（lane 私有 16B）
template <int kBatch>
__device__ __forceinline__ R19Frag2 r24_lds_b(const uint8_t* bmem, int s,
                                              int q) {
    const int lane = (int)(threadIdx.x & 63);
    const int warp = (int)(threadIdx.x >> 6);
    return *reinterpret_cast<const R19Frag2*>(
        bmem + (((size_t)s * kBatch + q) * kComputeWarps + warp) * 1024 +
        (size_t)lane * 16);
}

template <int kMT, int kKS, int kBatch, bool kMask = false, int kMTAct = kMT,
          int kBPF = R24_BPF, int kBsmB = 0>
__device__ R19_GEMM_ATTR void r24_l1_big(
        uint8_t* s_mem, const uint8_t* fbase, int mt_total, const uint8_t* w1,
        int expert, int c0_begin, int c0_end, int k_hidden, int n_total,
        int block_n, int lane, int warp, int nwarps, float* checksum,
        int n_group_begin, int n_group_count, uint8_t* ffn_out, uint8_t* ffn2,
        int out_col_base, int k_out,
        // [BALANCE·docs/20 §4.2] K 范围（默认全 K，既有调用点不变）
        int k_step_begin = 0, int k_steps = -1) {
    constexpr int kKStep = 32;
    constexpr int kNT = 2;
    constexpr int kBufBytes = kMT * kKS * 512;
    const int k_iters = (k_steps >= 0) ? k_steps : (k_hidden / kKStep);
    const int nstage = k_iters / kKS;
    const int n_per_warp = block_n / nwarps;
    const uint8_t* w_e = w1 + (size_t)expert * n_total * k_hidden;
    float acc_chk = 0.0f;
    // [D1·B-smem] B 环 smem 紧接 A 双缓冲之后（实例偏移）；仅 kBsmB>0 时使用。
    constexpr int kBRing = (kBsmB > 0) ? kBsmB : 1;
    uint8_t* bmem = s_mem + 2 * kBufBytes;
    static_assert(kBsmB == 0 ||
                  (2 * kBufBytes + kBRing * kBatch * kComputeWarps * 1024 <=
                   kABSharedBytes),
                  "A+B smem 超预算");
    // [R34] 任意行区间 [c0_begin,c0_end)：整块 kMT*16 行 + 尾块 cc（cc 可 <
    //   kMT*16，A 越界行为垃圾数据，消费端按 row<cc 掩码丢弃 → 数值等价零填充）
    for (int c0 = c0_begin; c0 < c0_end; c0 += kMT * kM) {
        const int cc = (c0_end - c0 < kMT * kM) ? (c0_end - c0) : kMT * kM;
        const int m_abs0 = c0 >> 4;
        uint8_t* f_c = (ffn_out != nullptr) ? ffn_out + (size_t)c0 * k_out
                                            : nullptr;
      for (int g0 = n_group_begin; g0 + kBatch <= n_group_begin + n_group_count;
           g0 += kBatch) {
        int nbase[kBatch];
        #pragma unroll
        for (int q = 0; q < kBatch; ++q)
            nbase[q] = (g0 + q) * block_n + warp * n_per_warp;
        r24_issue_a<kMT, kKS, kMask, kMTAct>(s_mem, fbase, k_step_begin,
                                        mt_total, m_abs0);
        R19Acc acc[kBatch][kMT][kNT];
        #pragma unroll
        for (int q = 0; q < kBatch; ++q)
            #pragma unroll
            for (int m = 0; m < kMT; ++m)
                #pragma unroll
                for (int t = 0; t < kNT; ++t)
                    acc[q][m][t] = R19Acc{0.f, 0.f, 0.f, 0.f};
        if constexpr (kBsmB > 0) {
            // B 走 smem 环：每个 stage 顶部预取本 stage 前 kBRing 个 kstep，
            //   循环内按 bsmcnt 精确等待「仅 B」的在飞数（A 由 stage 顶 wait-all
            //   保证），尾部改 bsmcnt(0) 全等以覆盖预取不足段。
            for (int st = 0; st < nstage; ++st) {
                R24_WAIT_BSM();          // 排空 A(st) 与上一 stage 残留 B
                __syncthreads();
                if (st + 1 < nstage)
                    r24_issue_a<kMT, kKS, kMask, kMTAct>(
                        s_mem + ((st + 1) & 1) * kBufBytes, fbase,
                        k_step_begin + (st + 1) * kKS, mt_total, m_abs0);
                const uint8_t* sb = s_mem + (st & 1) * kBufBytes;
                #pragma unroll
                for (int p = 0; p < kBRing; ++p) {
                    const int gp = st * kKS + p;
                    if (gp < k_iters)
                        r24_issue_b<kBatch>(bmem, gp & (kBRing - 1), w_e, nbase,
                                            k_step_begin + gp, k_hidden);
                }
                #pragma unroll
                for (int j = 0; j < kKS; ++j) {
                    const int g = st * kKS + j;
                    // 精确等待 B(g)：标准段允许 B(g+1..g+kBRing-1) 在飞，
                    //   尾部无后续预取 → 全等（bsmcnt 0）。
                    if (g + kBRing < k_iters)
                        R24_WAIT_BSM_N(R24_BSM_WAITSTD);
                    else
                        R24_WAIT_BSM_N(0);
                    // [D1] gvmcnt 等待已保证本线程拷贝完成
                    R19Frag a_pipe[kMT];
                    #pragma unroll
                    for (int m = 0; m < kMT; ++m) {
                        if (kMask && (m >= kMTAct)) continue;   // [R36] 越界 tile
                        a_pipe[m] = r24_lds_a<kMT>(sb, j, m);
                    }
                    #pragma unroll
                    for (int m = 0; m < kMT; ++m) {
                        if (kMask && (m >= kMTAct)) continue;   // [R36] 越界 tile
                        #pragma unroll
                        for (int q = 0; q < kBatch; ++q) {
                            const R19Frag2 bp =
                                r24_lds_b<kBatch>(bmem, g & (kBRing - 1), q);
                            const R19Frag b0 = {bp[0], bp[1]};
                            const R19Frag b1 = {bp[2], bp[3]};
                            acc[q][m][0] = __builtin_mxc_mma_16x16x32f8(
                                a_pipe[m], b0, acc[q][m][0]);
                            acc[q][m][1] = __builtin_mxc_mma_16x16x32f8(
                                a_pipe[m], b1, acc[q][m][1]);
                        }
                    }
                    const int nk = g + kBRing;
                    if (nk < k_iters)
                        r24_issue_b<kBatch>(bmem, nk & (kBRing - 1), w_e, nbase,
                                            k_step_begin + nk, k_hidden);
                }
            }
        } else {
        R19Frag2 b_pair[kBPF][kBatch];
        #pragma unroll
        for (int sp = 0; sp < kBPF; ++sp) {
            if (sp >= k_iters) break;
            #pragma unroll
            for (int q = 0; q < kBatch; ++q)
                b_pair[sp][q] = R19_LDB(w_e, nbase[q],
                                        (k_step_begin + sp) * kKStep, k_hidden);
        }
        for (int st = 0; st < nstage; ++st) {
            // 等 stage st 的 bsm 完成 + 发布
            R24_WAIT_BSM();
            __syncthreads();
            if (st + 1 < nstage)
                r24_issue_a<kMT, kKS, kMask, kMTAct>(
                    s_mem + ((st + 1) & 1) * kBufBytes, fbase,
                    k_step_begin + (st + 1) * kKS, mt_total, m_abs0);
            const uint8_t* sb = s_mem + (st & 1) * kBufBytes;
            #pragma unroll
            for (int j = 0; j < kKS; ++j) {
                const int g = st * kKS + j;
                const int sp = g & (kBPF - 1);
                R19Frag a_pipe[kMT];
                #pragma unroll
                for (int m = 0; m < kMT; ++m) {
                    if (kMask && (m >= kMTAct)) continue;   // [R36] 越界 tile
                    a_pipe[m] = r24_lds_a<kMT>(sb, j, m);
                }
                #pragma unroll
                for (int m = 0; m < kMT; ++m) {
                    if (kMask && (m >= kMTAct)) continue;   // [R36] 越界 tile
                    #pragma unroll
                    for (int q = 0; q < kBatch; ++q) {
                        const R19Frag2 bp = b_pair[sp][q];
                        const R19Frag b0 = {bp[0], bp[1]};
                        const R19Frag b1 = {bp[2], bp[3]};
                        acc[q][m][0] = __builtin_mxc_mma_16x16x32f8(
                            a_pipe[m], b0, acc[q][m][0]);
                        acc[q][m][1] = __builtin_mxc_mma_16x16x32f8(
                            a_pipe[m], b1, acc[q][m][1]);
                    }
                }
                const int nk = g + kBPF;
                if (nk < k_iters) {
                    #pragma unroll
                    for (int q = 0; q < kBatch; ++q)
                        b_pair[sp][q] = R19_LDB(w_e, nbase[q],
                                                (k_step_begin + nk) * kKStep,
                                                k_hidden);
                }
            }
        }
        }   // end kBsmB
        // SwiGLU + FP8 量化回写（与 r19_l1_tile 同构）
        #pragma unroll
        for (int pi = 0; pi < kBatch / 2; ++pi) {
            const int qa = 2 * pi, qb = 2 * pi + 1;
            #pragma unroll
            for (int m = 0; m < kMT; ++m) {
                if (kMask && (m >= kMTAct)) continue;   // [R36] 越界 tile
                #pragma unroll
                for (int t = 0; t < kNT; ++t) {
                    R19Acc sw;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gv = acc[qa][m][t][i];
                        const float uv = acc[qb][m][t][i];
                        sw[i] = gv / (1.0f + __expf(-gv)) * uv;
                        // [R34] 尾块越界行不进 checksum（与零填充路径一致）
                        const int row = m * kM + (lane >> 4) * 4 + i;
                        if (!kMask || row < cc) acc_chk += sw[i];
                    }
                    if (f_c != nullptr || ffn2 != nullptr) {
                        const uint32_t packed =
                            __builtin_mxc_cvt_pk4_f32tof8_cfg(
                                sw, _MACA_FROUND_TO_NEAREST_EVEN,
                                _MACA_DENORMAL_FLUSH_NONE,
                                _MACA_SRC_NO_MODIFIER, false, true);
                        const int col = out_col_base +
                                        (((g0 - n_group_begin) >> 1) + pi) *
                                            block_n +
                                        warp * n_per_warp + t * 16 +
                                        (lane & 15);
                        uint8_t* f2p = nullptr;
                        if (ffn2 != nullptr)
                            f2p = ffn2 +
                                  ((size_t)(col >> 5) * mt_total + (m_abs0 + m)) *
                                      512 +
                                  (lane >> 4) * 4 * 32 + (col & 31);
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int row = m * kM + (lane >> 4) * 4 + i;
                            if (!kMask || row < cc) {
                                const uint8_t byte = (uint8_t)(packed >> (8 * i));
                                if (f_c != nullptr)
                                    f_c[(size_t)row * k_out + col] = byte;
#ifndef R24_DIAG_NOMAINT
                                if (f2p != nullptr) f2p[i * 32] = byte;
#endif
                            }
                        }
                    }
                }
            }
        }
      }
    }
    if (checksum != nullptr && lane == 0)
        atomicAdd(checksum, acc_chk + (float)(c0_end - c0_begin) * 0.0f);
}

// ---------------------------------------------------------------------- L2 --
template <int kMT, int kPF, int kBatch>
__device__ __forceinline__ void r19_l2_tile(
        const uint8_t* a_pool, const uint8_t* f2, int m_abs0, int mt_total,
        const uint8_t* w2, int count, int k_in,
        int block_n, int lane, int warp, int nwarps, int g0, uint16_t* out,
        int k_out, float& acc_chk) {
    constexpr int kKStep = 32;
    constexpr int kNT = 2;
    const int k_iters = k_in / kKStep;
    const int n_per_warp = block_n / nwarps;
    (void)f2; (void)m_abs0; (void)mt_total;
    int nbase[kBatch];
    #pragma unroll
    for (int q = 0; q < kBatch; ++q)
        nbase[q] = (g0 + q) * block_n + warp * n_per_warp;
    R19Frag a_pipe[kPF][kMT];
    R19Frag2 b_pair[kPF][kBatch];
    #pragma unroll
    for (int sp = 0; sp < kPF; ++sp) {
        if (sp >= k_iters) break;
        #pragma unroll
        for (int m = 0; m < kMT; ++m)
            a_pipe[sp][m] = R21_L2A(sp, m);
        #pragma unroll
        for (int q = 0; q < kBatch; ++q)
            b_pair[sp][q] = R19_LDB(w2, nbase[q], sp * kKStep, k_in);
    }
    R19Acc acc[kBatch][kMT][kNT];
    #pragma unroll
    for (int q = 0; q < kBatch; ++q)
        #pragma unroll
        for (int m = 0; m < kMT; ++m)
            #pragma unroll
            for (int t = 0; t < kNT; ++t)
                acc[q][m][t] = R19Acc{0.f, 0.f, 0.f, 0.f};
    #pragma unroll kPF
    for (int kk = 0; kk < k_iters; ++kk) {
        const int sp = kk & (kPF - 1);
        #pragma unroll
        for (int m = 0; m < kMT; ++m) {
            #pragma unroll
            for (int q = 0; q < kBatch; ++q) {
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
                a_pipe[sp][m] = R21_L2A(nk, m);
            #pragma unroll
            for (int q = 0; q < kBatch; ++q)
                b_pair[sp][q] = R19_LDB(w2, nbase[q], nk * kKStep, k_in);
        }
    }
    // BF16 输出（每组写自己的列块）
    #pragma unroll
    for (int q = 0; q < kBatch; ++q) {
        #pragma unroll
        for (int m = 0; m < kMT; ++m) {
            #pragma unroll
            for (int t = 0; t < kNT; ++t) {
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int row = m * kM + (lane >> 4) * 4 + i;
                    const int col = (g0 + q) * block_n + warp * n_per_warp +
                                    t * 16 + (lane & 15);
                    acc_chk += acc[q][m][t][i];
                    if (out != nullptr && row < count) {
                        out[(size_t)row * k_out + col] =
                            (uint16_t)(r19_pack_f2bf16(acc[q][m][t][i], 0.0f) &
                                       0xFFFFu);
                    }
                }
            }
        }
    }
}

// 大 count L2。
template <int kMT, int kPF, int kBatch>
__device__ __forceinline__ void r19_l2_big(
        const uint8_t* a_pool, const uint8_t* f2, int mt_total,
        const uint8_t* w2, int count, int k_in,
        int n_total, int block_n, int lane, int warp, int nwarps,
        float* checksum, int n_group_begin, int n_group_count, uint16_t* out,
        int k_out) {
    (void)n_total;
    float acc_chk = 0.0f;
    for (int c0 = 0; c0 < count; c0 += kMT * kM) {
        const int cc = (count - c0 < kMT * kM) ? (count - c0) : kMT * kM;
        const uint8_t* a_c = a_pool + (size_t)c0 * k_in;
        uint16_t* o_c =
            (out != nullptr) ? out + (size_t)c0 * k_out : nullptr;
        int g0 = n_group_begin;
        for (; g0 + kBatch <= n_group_begin + n_group_count; g0 += kBatch) {
            r19_l2_tile<kMT, kPF, kBatch>(a_c, f2, c0 >> 4, mt_total, w2, cc,
                                          k_in, block_n, lane, warp, nwarps,
                                          g0, o_c, k_out, acc_chk);
        }
        // 余数组：用 kBatch=2 收尾（不足 2 组时逐组 kBatch=1）
        if (g0 + 2 <= n_group_begin + n_group_count) {
            r19_l2_tile<kMT, kPF, 2>(a_c, f2, c0 >> 4, mt_total, w2, cc, k_in,
                                     block_n, lane, warp, nwarps, g0, o_c,
                                     k_out, acc_chk);
        } else if (g0 < n_group_begin + n_group_count) {
            r19_l2_tile<kMT, kPF, 1>(a_c, f2, c0 >> 4, mt_total, w2, cc, k_in,
                                     block_n, lane, warp, nwarps, g0, o_c,
                                     k_out, acc_chk);
        }
    }
    if (checksum != nullptr && lane == 0) atomicAdd(checksum, acc_chk);
}

// [R24] L2 的 A 走 smem：f2 已是 fragment 布局，复用 r24_issue_a / r24_lds_a。
// g0 批次在外层：每批一次完整 K 流水（与 r19_l2_big 的 A 读次数相同）。
template <int kMT, int kKS, int kBatch, bool kMask = false, int kMTAct = kMT,
          int kBPF = R24_BPF, int kBsmB = 0>
__device__ R19_GEMM_ATTR void r24_l2_big(
        uint8_t* s_mem, const uint8_t* f2, int mt_total, const uint8_t* w2,
        int c0_begin, int c0_end, int k_in, int n_total, int block_n, int lane,
        int warp, int nwarps, float* checksum, int n_group_begin,
        int n_group_count, uint16_t* out, int k_out,
        const CombineFuseCtx* fuse = nullptr) {
    constexpr int kKStep = 32;
    constexpr int kNT = 2;
    constexpr int kBufBytes = kMT * kKS * 512;
    const int k_iters = k_in / kKStep;
    const int nstage = k_iters / kKS;
    const int n_per_warp = block_n / nwarps;
    (void)n_total;
    float acc_chk = 0.0f;
    // [D1·B-smem] B 环 smem 紧接 A 双缓冲之后（实例偏移）；仅 kBsmB>0 时使用。
    constexpr int kBRing = (kBsmB > 0) ? kBsmB : 1;
    uint8_t* bmem = s_mem + 2 * kBufBytes;
    static_assert(kBsmB == 0 ||
                  (2 * kBufBytes + kBRing * kBatch * kComputeWarps * 1024 <=
                   kABSharedBytes),
                  "A+B smem 超预算");
#if R30_PUSH_DEFER || R30_DEFER1
    // [R30] 推送流水：上一 (c0,g) 列块的行/列区间，延迟到本列块 stage 循环
    //   首 barrier 之后发出（fabric 排空与后续 MMA 重叠；PUSH_DEFER 分片、
    //   DEFER1 整段）。数据由上一列块 epilogue 写完、本列块 stage 循环首个
    //   __syncthreads 保证块内可见；标记仍由任务尾统一写（序不变）。
    int pend_lo = 0, pend_hi = 0, pend_c0 = 0, pend_c1 = 0;
#endif
#if R30_STAGGER == 1 || R30_STAGGER == 2
    const int c0n_s = (c0_end - c0_begin + kMT * kM - 1) / (kMT * kM);
    for (int ci_s = 0; ci_s < c0n_s; ++ci_s) {
        const int c0 = c0_begin + ((ci_s + (int)blockIdx.x) % c0n_s) * kMT * kM;
#else
    for (int c0 = c0_begin; c0 < c0_end; c0 += kMT * kM) {
#endif
        // [R34] 行区间 [c0_begin,c0_end)：尾块 cc<kMT*16 按 row<cc 掩码
        const int cc = (c0_end - c0 < kMT * kM) ? (c0_end - c0) : kMT * kM;
        const int m_abs0 = c0 >> 4;
        uint16_t* o_c = (out != nullptr) ? out + (size_t)c0 * k_out : nullptr;
#if R30_STAGGER == 1 || R30_STAGGER == 3
        const int gn_s = n_group_count / kBatch;
        for (int gi_s = 0; gi_s < gn_s; ++gi_s) {
            const int g0 = n_group_begin +
                           ((gi_s + (int)blockIdx.x) % gn_s) * kBatch;
#else
        for (int g0 = n_group_begin; g0 + kBatch <= n_group_begin + n_group_count;
             g0 += kBatch) {
#endif
            int nbase[kBatch];
            #pragma unroll
            for (int q = 0; q < kBatch; ++q)
                nbase[q] = (g0 + q) * block_n + warp * n_per_warp;
            r24_issue_a<kMT, kKS, kMask, kMTAct>(s_mem, f2, 0, mt_total,
                                            m_abs0);
            R19Acc acc[kBatch][kMT][kNT];
            #pragma unroll
            for (int q = 0; q < kBatch; ++q)
                #pragma unroll
                for (int m = 0; m < kMT; ++m)
                    #pragma unroll
                    for (int t = 0; t < kNT; ++t)
                        acc[q][m][t] = R19Acc{0.f, 0.f, 0.f, 0.f};
            if constexpr (kBsmB > 0) {
            for (int st = 0; st < nstage; ++st) {
                R24_WAIT_BSM();
                __syncthreads();
                if (st + 1 < nstage)
                    r24_issue_a<kMT, kKS, kMask, kMTAct>(s_mem + ((st + 1) & 1) * kBufBytes,
                                          f2, (st + 1) * kKS, mt_total,
                                          m_abs0);
#if R30_PUSH_DEFER
                if (fuse != nullptr && pend_hi > pend_lo) {
                    const int nrp = pend_hi - pend_lo;
                    const int per = (nrp + nstage - 1) / nstage;
                    const int lo = pend_lo + st * per;
                    const int hi = (lo + per < pend_hi) ? (lo + per) : pend_hi;
                    if (hi > lo)
                        combine_push_chunk_rows(*fuse, lo, hi, pend_c0,
                                                pend_c1, lane, warp, nwarps);
                }
#elif R30_DEFER1
                if (st == 0 && fuse != nullptr && pend_hi > pend_lo) {
                    combine_push_chunk_rows(*fuse, pend_lo, pend_hi, pend_c0,
                                            pend_c1, lane, warp, nwarps);
                    pend_hi = 0;
                }
#endif
                const uint8_t* sb = s_mem + (st & 1) * kBufBytes;
                #pragma unroll
                for (int p = 0; p < kBRing; ++p) {
                    const int gp = st * kKS + p;
                    if (gp < k_iters)
                        r24_issue_b<kBatch>(bmem, gp & (kBRing - 1), w2, nbase,
                                            gp, k_in);
                }
                #pragma unroll
                for (int j = 0; j < kKS; ++j) {
                    const int g = st * kKS + j;
                    if (g + kBRing < k_iters)
                        R24_WAIT_BSM_N(R24_BSM_WAITSTD);
                    else
                        R24_WAIT_BSM_N(0);
                    // [D1] gvmcnt 等待已保证本线程拷贝完成
                    R19Frag a_pipe[kMT];
                    #pragma unroll
                    for (int m = 0; m < kMT; ++m) {
                        if (kMask && (m >= kMTAct)) continue;   // [R36] 越界 tile
                        a_pipe[m] = r24_lds_a<kMT>(sb, j, m);
                    }
                    #pragma unroll
                    for (int m = 0; m < kMT; ++m) {
                        if (kMask && (m >= kMTAct)) continue;   // [R36] 越界 tile
                        #pragma unroll
                        for (int q = 0; q < kBatch; ++q) {
                            const R19Frag2 bp =
                                r24_lds_b<kBatch>(bmem, g & (kBRing - 1), q);
                            const R19Frag b0 = {bp[0], bp[1]};
                            const R19Frag b1 = {bp[2], bp[3]};
                            acc[q][m][0] = __builtin_mxc_mma_16x16x32f8(
                                a_pipe[m], b0, acc[q][m][0]);
                            acc[q][m][1] = __builtin_mxc_mma_16x16x32f8(
                                a_pipe[m], b1, acc[q][m][1]);
                        }
                    }
                    const int nk = g + kBRing;
                    if (nk < k_iters)
                        r24_issue_b<kBatch>(bmem, nk & (kBRing - 1), w2, nbase,
                                            nk, k_in);
                }
            }
            } else {
            R19Frag2 b_pair[kBPF][kBatch];
            #pragma unroll
            for (int sp = 0; sp < kBPF; ++sp) {
                if (sp >= k_iters) break;
                #pragma unroll
                for (int q = 0; q < kBatch; ++q)
                    b_pair[sp][q] = R19_LDB(w2, nbase[q], sp * kKStep, k_in);
            }
            for (int st = 0; st < nstage; ++st) {
                R24_WAIT_BSM();
                __syncthreads();
                if (st + 1 < nstage)
                    r24_issue_a<kMT, kKS, kMask, kMTAct>(s_mem + ((st + 1) & 1) * kBufBytes,
                                          f2, (st + 1) * kKS, mt_total,
                                          m_abs0);
#if R30_PUSH_DEFER
                // 本 stage 分摊上一列块的推送行片（A 预取已先发出）
                if (fuse != nullptr && pend_hi > pend_lo) {
                    const int nrp = pend_hi - pend_lo;
                    const int per = (nrp + nstage - 1) / nstage;
                    const int lo = pend_lo + st * per;
                    const int hi = (lo + per < pend_hi) ? (lo + per) : pend_hi;
                    if (hi > lo)
                        combine_push_chunk_rows(*fuse, lo, hi, pend_c0,
                                                pend_c1, lane, warp, nwarps);
                }
#elif R30_DEFER1
                // 上一列块整段推送（首 barrier 已保证 epilogue 写可见；A 已发出）
                if (st == 0 && fuse != nullptr && pend_hi > pend_lo) {
                    combine_push_chunk_rows(*fuse, pend_lo, pend_hi, pend_c0,
                                            pend_c1, lane, warp, nwarps);
                    pend_hi = 0;
                }
#endif
                const uint8_t* sb = s_mem + (st & 1) * kBufBytes;
                #pragma unroll
                for (int j = 0; j < kKS; ++j) {
                    const int g = st * kKS + j;
                    const int sp = g & (kBPF - 1);
                    R19Frag a_pipe[kMT];
                    #pragma unroll
                    for (int m = 0; m < kMT; ++m) {
                        if (kMask && (m >= kMTAct)) continue;   // [R36] 越界 tile
                        a_pipe[m] = r24_lds_a<kMT>(sb, j, m);
                    }
                    #pragma unroll
                    for (int m = 0; m < kMT; ++m) {
                        if (kMask && (m >= kMTAct)) continue;   // [R36] 越界 tile
                        #pragma unroll
                        for (int q = 0; q < kBatch; ++q) {
                            const R19Frag2 bp = b_pair[sp][q];
                            const R19Frag b0 = {bp[0], bp[1]};
                            const R19Frag b1 = {bp[2], bp[3]};
                            acc[q][m][0] = __builtin_mxc_mma_16x16x32f8(
                                a_pipe[m], b0, acc[q][m][0]);
                            acc[q][m][1] = __builtin_mxc_mma_16x16x32f8(
                                a_pipe[m], b1, acc[q][m][1]);
                        }
                    }
                    const int nk = g + kBPF;
                    if (nk < k_iters) {
                        #pragma unroll
                        for (int q = 0; q < kBatch; ++q)
                            b_pair[sp][q] = R19_LDB(w2, nbase[q], nk * kKStep,
                                                    k_in);
                    }
                }
            }
            }   // end kBsmB
            #pragma unroll
            for (int q = 0; q < kBatch; ++q) {
                #pragma unroll
                for (int m = 0; m < kMT; ++m) {
                    if (kMask && (m >= kMTAct)) continue;   // [R36] 越界 tile
                    #pragma unroll
                    for (int t = 0; t < kNT; ++t) {
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int row = m * kM + (lane >> 4) * 4 + i;
                            const int col = (g0 + q) * block_n +
                                            warp * n_per_warp + t * 16 +
                                            (lane & 15);
                            if (!kMask || row < cc)
                                acc_chk += acc[q][m][t][i];
                            if (o_c != nullptr &&
                                (!kMask || row < cc)) {
                                o_c[(size_t)row * k_out + col] =
                                    (uint16_t)(r19_pack_f2bf16(
                                                   acc[q][m][t][i], 0.0f) &
                                                0xFFFFu);
                            }
                        }
                    }
                }
            }
            // [R29] 融合推送：本 (c0,g) 列块 staging 已写完，块同步后立即推送，
            //   与后续 L2 计算重叠（fabric 排空不再串在任务尾）；完成标记留待
            //   任务尾统一写。行索引与任务尾一致（block_idx*staging_cap+s2）。
            if (fuse != nullptr && fuse->on && out != nullptr) {
#if R30_DIAG_NOPUSH
                (void)0;   // 诊断：跳过整段推送（标记仍由任务尾写）
#elif R30_PUSH_DEFER || R30_DEFER1
                // 登记本列块待推区间（PUSH_DEFER 下一列块分片 / DEFER1 整段）
                pend_lo = c0;
                pend_hi = c0 + cc;
                pend_c0 = g0 * block_n;
                pend_c1 = (g0 + kBatch) * block_n;
#else
#if R33_WARPPUSH
                // [R33] 每 warp 只推自己写的列段：免块同步（末列块推送与自身
                //   epilogue/计算重叠；各 warp 时刻错开）。协议发布序不变。
                __syncwarp();
                combine_push_warp_cols(*fuse, c0, c0 + cc, g0, kBatch,
                                       block_n, n_per_warp, lane, warp);
#else
                __syncthreads();
                combine_push_chunk(*fuse, c0, c0 + cc, g0 * block_n,
                                   (g0 + kBatch) * block_n, lane, warp,
                                   nwarps);
#endif
#endif
            }
        }
    }
    // [R35] 奇数 N 组余量：kBatch=2 主循环后若本 slice 还剩 1 组（如 NSLICE=8 时
    //   g_l2=24/8=3），用 kBatch=1 实例补齐（kBatch=1 无进一步余量 → 不递归）。
    //   每个 c0 行块都跳过同一余量组，故此处对全部 c0 再走一遍 kBatch=1 补算；
    //   checksum 由各实例分别 atomicAdd，总和不变。
    if constexpr (kBatch > 1) {
        const int g_rem = n_group_begin + (n_group_count / kBatch) * kBatch;
        if (g_rem < n_group_begin + n_group_count)
            r24_l2_big<kMT, kKS, 1, kMask, kMTAct, kBPF>(
                s_mem, f2, mt_total, w2, c0_begin, c0_end, k_in, n_total,
                block_n, lane, warp, nwarps, checksum, g_rem,
                n_group_begin + n_group_count - g_rem, out, k_out, fuse);
    }
#if R30_PUSH_DEFER || R30_DEFER1
    // 收尾：最后一个列块无后继可分摊，整段发出（仍在任务尾标记之前）
    if (fuse != nullptr && pend_hi > pend_lo) {
        __syncthreads();
        combine_push_chunk_rows(*fuse, pend_lo, pend_hi, pend_c0, pend_c1, lane,
                                warp, nwarps);
    }
#endif
    if (checksum != nullptr && lane == 0) atomicAdd(checksum, acc_chk);
}

// ============================================================================
// 任务体公共段（两个 noinline 任务体共享，forceinline 进各自函数体）：
//   pull → 分片旗 → wait-all → l1_prog 发布/等待 → tail
//   代码自 compute.cuh 基线逐字搬移，仅变量改经 R19Sched 传入。
// ============================================================================
// [L1L2-SPLIT] kPhase 拆分语义（pool_slot 参数 = 数据槽；旗槽在函数内按 ring 派生）：
//   Both —— 与基线逐字一致（数据+staging+分片旗+wait-all 全做）；
//   L1   —— 只做数据搬移（pool/frag）+ 分片旗 + wait-all，不写 wt/meta staging；
//   L2   —— 只重算 wt/meta staging（COMBINE tail 消费；本 kernel 领号可能与 L1
//           不同，必须按自己的任务重算），不碰 pool/frag/分片旗。
template <MegaPhase kPhase = MegaPhase::Both>
__device__ __forceinline__ void r19_pull_flags(
        const R19Sched& S, int task, int e, int w, int count, int slice,
        unsigned epoch, int pool_slot, const int* pfx, int kseg = 0) {
    const int tid = threadIdx.x;
    // [BALANCE·docs/20 §4.2b] 跨 block K 切：仅 owner（kseg==0）执行 pull/staging/
    //   分片旗；其余 K 段只等待分片旗（数据由 owner 搬齐）。未启用时 own≡true，
    //   行为与原实现逐字一致。
    const bool own = R19_BLOCK_KSPLIT ? (kseg == 0) : true;
    (void)kseg;
    const int warp = tid / 64;
    const int lane = tid % 64;
    const int nwarps = (int)(blockDim.x / 64);
    const int s_row_off = w * S.a->win_cap;
    // 旗槽：保持 ring 索引（旗区容量 EPR*ring*nslice=256 不变）；数据槽 pool_slot
    //   在拆分下为 e*slot_win+w（每窗口独占，见 dispatch_pull_body）。
    const int flag_slot = e * kRing + (w % kRing);
    // [GEN] 槽携带代际：pool_slot=e*ring+w%ring 会绕回复用；旗值改用单调代际
    //   epoch + w/ring（= 该槽第几次被用），wait 端用本窗算出的同一值等待，
    //   避免上一代窗口遗留的旗被误判就绪（多波 merge=1 陈旧读修复）。
#ifdef NO_GEN_FIX
    const unsigned gen = epoch;
#else
    const unsigned gen = epoch + (unsigned)(w / (kRing > 0 ? kRing : 1));
#endif
    (void)flag_slot;
    // [R35] 物理任务覆盖 merge 个连续逻辑 slice：K 片为连续 merge 段
    const int u4_slice = (kHidden / 16) / S.nslice_eff;   // 本任务 uint4 数
    const int u4_off = slice * u4_slice;
    uint8_t* pool = S.pool_base;
    float* wt = reinterpret_cast<float*>(const_cast<uint8_t*>(S.a->self) + S.a->w_off);
    int* meta = reinterpret_cast<int*>(const_cast<uint8_t*>(S.a->self) + S.a->meta_off);
    const uint32_t* src_tbl =
        &(&S.a->unk->rd)->src_index_table[S.cycle][0][0][0];
    const bool kTick = (S.pt != nullptr) && (blockIdx.x == 0) && (task == 0) &&
                       (tid == 0);
    // ---- 行均分 K 片 pull：半 warp 一行（8 warp × 2 行并行），staging 写 weights/meta
    // [H1] 每个 warp 的两个半 warp 各负责一行，且相邻行配对（同源 rank）
    const int half = lane >> 5;           // 半 warp 号（0/1）
    const int ll = lane & 31;             // 半 warp 内 lane
    // [R21] A fragment 副本（拉取时顺带写 F_A：pool 仍行主序，供 bench 校验）
    const bool kAfrag = S.frag_all && count > 0;
    const bool kSkipPool = false;   // 实测跳过 pool 写更慢，保留
    uint8_t* frag_base = nullptr;
    if (kAfrag) {
        // 按 (expert, window) 数据槽定位专用 A fragment（local_data.frag_a）
        const size_t fsz = (size_t)(kHidden / 32) * (size_t)S.mt_total * 512;
        frag_base = S.frag_a + (size_t)pool_slot * fsz;
    }
    const int mt_total = S.mt_total;   // [R34] ceil(win_cap/16)
    // [R26] warp 独占一段连续行（= 同源 rank 段）：每个 warp 的 8 次迭代
    // 始终读同一 peer，消除跨 peer 轮转的 fabric 排队惩罚（71 机 TPR=128
    // 同会话 −60µs；dispatch 慢 rank 160~210→110~125µs）。数据放置不变。
    const int wchunk = (count + nwarps - 1) / nwarps;
    const int wlo = warp * wchunk;
    const int whi = (wlo + wchunk < count) ? (wlo + wchunk) : count;
    if (own)
    for (int s2 = wlo + half; s2 < whi; s2 += 2) {
        const int global_row = s_row_off + s2;
        int src = 0;
        while (src + 1 < S.a->R && pfx[src + 1] <= global_row) ++src;
        const int slot_in_rank = global_row - pfx[src];
        const uint32_t encoded =
            src_tbl[(size_t)(e * S.a->R + src) * kMaxTPR + slot_in_rank];
        const int src_token = (int)(encoded / (uint32_t)kTopk);
        const bool valid = src_token >= 0 && src_token < S.a->slot_capacity;
        uint4* dstp = reinterpret_cast<uint4*>(
            pool + ((size_t)pool_slot * S.a->win_cap + s2) * kHidden);
        if (valid) {
            // [O2] 权重远端读先于整行搬运发出：与搬运 store 并行，
            // 避免原顺序（先搬运后读权重）把远端读延迟暴露在任务尾。
            // [L1L2-SPLIT] wt/meta staging 仅 Both/L2 需要（L1 阶段跳过权重读）。
            float wv = 0.0f;
            if constexpr (kPhase != MegaPhase::L1) {
                if (ll == 0)
                    wv = *(reinterpret_cast<const float*>(
                               S.a->peer_bases[src] + S.a->weights_off) + encoded);
            }
            // [L1L2-SPLIT] pool/frag 数据搬移仅 Both/L1（L2 阶段不碰输入缓冲）。
            if constexpr (kPhase != MegaPhase::L2) {
                // [BF16-IN] bf16 输入时读本地 fold 后的 fp8 区，否则读 fp8 x。
                const int64_t xoff =
                    S.a->in_bf16 ? S.a->prefold_off : S.a->x_off;
#ifdef PULL_LOCAL_DIAG
                // [诊断] 远端读换成本 rank 本地读（同 volume/pattern），量化远端代价
                const uint4* srcp = reinterpret_cast<const uint4*>(
                                        S.a->self + xoff) +
                                    (size_t)(src_token % S.a->slot_capacity) *
                                        (kHidden / 16) + u4_off;
#else
                const uint4* srcp = reinterpret_cast<const uint4*>(
                                        S.a->peer_bases[src] + xoff) +
                                    (size_t)src_token * (kHidden / 16) + u4_off;
#endif
                // [H2] 本场景 u4_slice=96（6144/16/4）：3 个远端装载先全量发出，
                // 再连续 store，消除 load→store→load 的串行依赖（−1.7µs）。
                if (u4_slice == 96) {
                    const uint4 v0 = srcp[ll];
                    const uint4 v1 = srcp[ll + 32];
                    const uint4 v2 = srcp[ll + 64];
                    if (!kSkipPool) {
                        dstp[u4_off + ll] = v0;
                        dstp[u4_off + ll + 32] = v1;
                        dstp[u4_off + ll + 64] = v2;
                    }
                    if (kAfrag) {
                        r19_store_a_frag(frag_base, s2, (u4_off + ll) * 16, v0,
                                         mt_total);
                        r19_store_a_frag(frag_base, s2, (u4_off + ll + 32) * 16, v1,
                                         mt_total);
                        r19_store_a_frag(frag_base, s2, (u4_off + ll + 64) * 16, v2,
                                         mt_total);
                    }
                } else {
                    for (int i = ll; i < u4_slice; i += 32) {
                        const uint4 v = srcp[i];
                        if (!kSkipPool) dstp[u4_off + i] = v;
                        if (kAfrag)
                            r19_store_a_frag(frag_base, s2, (u4_off + i) * 16, v,
                                             mt_total);
                    }
                }
            }
            if constexpr (kPhase != MegaPhase::L1) {
                if (ll == 0) {
                    wt[(size_t)blockIdx.x * S.a->win_cap + s2] = wv;
                    meta[((size_t)blockIdx.x * S.a->win_cap + s2) * 3 + 0] = src;
                    meta[((size_t)blockIdx.x * S.a->win_cap + s2) * 3 + 1] = src_token;
                    meta[((size_t)blockIdx.x * S.a->win_cap + s2) * 3 + 2] =
                        (int)(encoded % (uint32_t)kTopk);
                }
            }
        } else {
            // 非法行零填充（本 K 片 + staging），保证确定性与 COMBINE 跳过
            if constexpr (kPhase != MegaPhase::L2) {
                const uint4 z4{0u, 0u, 0u, 0u};
                for (int i = ll; i < u4_slice; i += 32) {
                    if (!kSkipPool) dstp[u4_off + i] = z4;
                    if (kAfrag)
                        r19_store_a_frag(frag_base, s2, (u4_off + i) * 16, z4,
                                         mt_total);
                }
            }
            if constexpr (kPhase != MegaPhase::L1) {
                if (ll == 0) {
                    wt[(size_t)blockIdx.x * S.a->win_cap + s2] = 0.0f;
                    meta[((size_t)blockIdx.x * S.a->win_cap + s2) * 3 + 0] = 0;
                    meta[((size_t)blockIdx.x * S.a->win_cap + s2) * 3 + 1] = 0;
                    meta[((size_t)blockIdx.x * S.a->win_cap + s2) * 3 + 2] = 0;
                }
            }
        }
    }
#if R19_BLOCK_KSPLIT
    // [BALANCE·docs/20 §4.2b] owner 在置分片旗前把本 (slot,slice) 的部分和缓冲清零：
    //   之后各 K 段用 FP32 atomicAdd 累加，无需区分「首写」。清零在分片旗之前、
    //   经同一 fence 序列发布，wait-all 后各段可见。仅 K 切开启且为真窗口时做。
    if (own && kseg == 0 && S.a->win_cap <= R19_BLOCK_KSPLIT_WINMAX &&
        (kNK > 1) && S.frag_all) {
        float* part = S.l1_partial +
            ((size_t)pool_slot * kNSlice + slice) * kMaxTPR * kL1SliceCols;
        const int n4 = (count * kL1SliceCols) / 4;
        uint4* p4 = reinterpret_cast<uint4*>(part);
        const uint4 z4{0u, 0u, 0u, 0u};
        for (int i = tid; i < n4; i += (int)blockDim.x) p4[i] = z4;
        // 段完成计数也必须每代清零（槽跨 ring 复用；否则 reduce 触发条件
        //   old==kNK-1 只在第一代成立）。由 owner 在置分片旗前清零，可见性同 part。
        if (tid == 0)
            *reinterpret_cast<volatile uint32_t*>(
                S.l1_part_cnt + (size_t)pool_slot * kNSlice + slice) = 0u;
    }
#endif
    // [L1L2-SPLIT] 以下 flush/分片旗/wait-all 仅 Both/L1（L2 的输入齐备性由
    //   kernel 边界保证：L1 kernel 全部完成后 L2 kernel 才启动）。
    if constexpr (kPhase != MegaPhase::L2) {
#ifndef PULL_FENCE_NONE
        __threadfence();
#endif
        __syncthreads();
        // ---- 分片就绪旗（本 slice 的 K 片已落 pool；L1 消费）----
        // [BALANCE·docs/20 §4.2b] 非 owner（kseg>0）不置旗；旗由各 slice 的 owner 置。
        if (own && tid == 0) {
#ifndef PULL_FENCE_MIN
            __threadfence();
#endif
            // 同设备分片旗：普通写 + fence（__stwt 在普通显存上不更新 L2）
            // [R36b] flag 粒度=物理 slice（nslice_eff）：粗化后 stride≤4，区容量
            //   EPR×ring×nslice_eff ≤ 256 恒成立（不再随 kNSlice 放大）；
            //   kNSlice=16 因此无需改布局常量即可用。
            // [L1L2-SPLIT] 旗槽恒用 ring 索引（容量 256 不随数据槽扩容）。
            *reinterpret_cast<volatile uint32_t*>(
                S.bar_flags + S.a->slice_flags_idx +
                (size_t)flag_slot * S.nslice_eff + slice) = gen;
#ifndef PULL_FENCE_MIN
            __threadfence();
#endif
        }
        __syncthreads();
        // 诊断：block0 首任务 DISPATCH 完成（分片旗已写）
        if (kTick) S.pt[4] = clock64();
        // ---- L1 入口 wait-all：一次等齐本窗全部分片就绪旗 ----
        // 骨架结论：packed+配对下 wait-all 优于 K 循环内每步分片门；任务按窗口连续
        // 编号，同窗全部 slice 旗由本窗 nslice_eff 个物理任务写满。
        if (count > 0 && tid == 0) {
            for (int s2 = 0; s2 < S.nslice_eff; ++s2) {
                // owner 跳过自己已置的 slice；非 owner 必须等自己 slice 的 owner。
                if (s2 == slice && own) continue;
                long long spin = 0;
                // [集71] wait-all（分片就绪旗）等待轮次
                MX_CTR_ADD(S.ctr, kCtrWaitAllSlice, 1);
                while (*(volatile const uint32_t*)(S.bar_flags +
                                                   S.a->slice_flags_idx +
                                                   (size_t)flag_slot *
                                                       S.nslice_eff +
                                                   s2) != gen) {
                    if (++spin > S.a->max_spins) {
                        if (S.dbg != nullptr) atomicAdd(S.dbg + 4, 1);
                        break;
                    }
                }
            }
            __threadfence();
        }
    }
    __syncthreads();
}

__device__ __forceinline__ void r19_l1_pub_wait(
        const R19Sched& S, int task, int count, int slice, unsigned epoch,
        int pool_slot, int w) {
    const int tid = threadIdx.x;
    const bool kTick = (S.pt != nullptr) && (blockIdx.x == 0) && (task == 0) &&
                       (tid == 0);
    // [GEN] 与 slice_flags 同代际，避免 l1_prog 跨窗口复用陈旧读
#ifdef NO_GEN_FIX
    const unsigned gen = epoch;
#else
    const unsigned gen = epoch + (unsigned)(w / (kRing > 0 ? kRing : 1));
#endif
    __syncthreads();
    // ---- l1_prog 发布 + L1→L2 依赖（本窗全部 NSLICE 个旗）----
    if (tid == 0) {
        __threadfence();
        // [R36b] 同 slice 旗：flag 粒度=物理 slice（stride=nslice_eff）
        l1_progress_publish(S.bar_flags, (int)S.a->l1_prog_idx, pool_slot, slice,
                            gen, (int)S.nslice_eff);
        for (int s2 = 0; s2 < S.nslice_eff; ++s2) {
            if (s2 == slice) continue;
            long long spin = 0;
            // [集71] wait-all（l1_prog 旗）等待轮次
            MX_CTR_ADD(S.ctr, kCtrWaitAllL1Prog, 1);
            while (*(volatile const uint32_t*)(S.bar_flags + S.a->l1_prog_idx +
                                               (size_t)pool_slot *
                                                   S.nslice_eff +
                                               s2) != gen) {
                if (++spin > S.a->max_spins) {
                    if (S.dbg != nullptr) atomicAdd(S.dbg + 4, 1);
                    break;
                }
            }
        }
        __threadfence();
    }
    __syncthreads();
    // 诊断：block0 首任务 L1 完成（已等齐本窗 slice 旗）
    if (kTick) S.pt[5] = clock64();
}

__device__ __forceinline__ CombineCtx cm_build_ctx(const R19Sched& S) {
    CombineCtx c;
    c.meta = reinterpret_cast<const int*>(S.a->self + S.a->meta_off);
    c.weights = reinterpret_cast<const float*>(S.a->self + S.a->w_off);
    c.peer_bases = S.a->peer_bases;
    c.unk_peers = S.a->unk_peers;
    c.rows_off = S.a->combine_rows_off;
    c.epoch = S.epoch;
    c.block_idx = (int)blockIdx.x;
    c.staging_cap = S.a->win_cap;
    c.cycle = S.cycle;
    return c;
}
__device__ __forceinline__ void r19_tail_common(
        const R19Sched& S, const CombineCtx& ctx, bool fused, int task, int e,
        int w, int count, int slice, unsigned epoch, int pool_slot) {
    (void)S; (void)task; (void)e; (void)w; (void)pool_slot; (void)epoch;
    if (ctx.rows_off >= 0) {
        const int lane = (int)(threadIdx.x % 64);
        const int warp = (int)(threadIdx.x / 64);
        CombineCtx c = ctx;
        c.slice = slice;
        c.nslice_eff = S.nslice_eff;
        c.merge = S.merge;
        if (fused)
            combine_mark_rows(c, count, lane, warp, 8);
        else
            combine_push_pair(c, reinterpret_cast<const uint8_t*>(S.a->l2_out),
                              count, lane, warp, 8);
    }
    __syncthreads();
}

// ============================================================================
// 大 count 任务体（__noinline__ 独立寄存器分配）
//   自包含：pull → 分片旗 → wait-all → L1（M 档位）→ l1_prog 发布/等待
//   → L2（M 档位）→ tail。调用者（dispatch_pull_body 循环）只做领号/解码与
//   任务尾 l2_done/processed 计数（与基线逐字一致）。
// ============================================================================
// [FRAG-ALWAYS] 去掉 kFragCap：fragment 分支恒编译（运行期按 frag_a/frag_all 决定）
// [L1L2-SPLIT] kPhase：Both=原合并任务体；L1=只跑 DISPATCH+L1（L2/tail 编译期
//   消除）；L2=只跑 L2+tail（DISPATCH/pull/L1 编译期消除）。
template <MegaPhase kPhase = MegaPhase::Both>
__device__ R19_BIG_ATTR void r19_big_task(
        const R19Sched& S, int task, int e, int w, int count, int slice,
        int kseg, unsigned epoch, int pool_slot, const int* pfx,
        uint8_t* a_smem, uint8_t* b_smem) {
    const int tid = threadIdx.x;
    const int warp = tid / 64;
    const int lane = tid % 64;
    // 档位运行期选（无编译期 kCap）；重型 r24_* 用 R19_GEMM_ATTR=__noinline__ 隔离。
    // [R38] win_cap<=64（tk/长尾）r24 用更深 B 预取环(4)，均衡大档保持 R24_BPF(2)。
    constexpr int kR24BPF = R24_BPF;
    // [D1·B-smem] 仅 tk/长尾（win_cap<=64）B 实例开启 smem 环；均衡大档保持直装。
    constexpr int kBsmB = (R24_BSMEM_B) ? R24_BRING : 0;
    (void)a_smem;  // [R24] kernel 作用域 A-smem（由调用方声明；小任务不用）
    // [BALANCE·docs/20 §4.2b] 本窗是否走跨 block K 切（小窗 + frag 可用 + 开关）。
    const bool ksplit = R19_BLOCK_KSPLIT && (S.a->win_cap <= R19_BLOCK_KSPLIT_WINMAX) &&
                        (kNK > 1) && S.frag_all;
    // [tail 内联] ctx/fuse 由 S 现构造（CombineTail 已去除）
    const CombineCtx cctx = cm_build_ctx(S);
    CombineFuseCtx fuse;
    fuse.capable = (cctx.rows_off >= 0);
    fuse.c = &cctx;
    fuse.out_base = reinterpret_cast<const uint8_t*>(S.a->l2_out);
    // [L1L2-SPLIT] pull 恒调用：其内部按 kPhase 决定搬数据（L1/Both）还是
    //   只重算 wt/meta staging（L2）；L2 的 COMBINE tail 依赖此 staging。
    r19_pull_flags<kPhase>(S, task, e, w, count, slice, epoch, pool_slot, pfx, kseg);
#if R24_DIAG_PHASE
    const unsigned long long t_l1s = (tid == 0) ? clock64() : 0ull;
#endif
    // ---- L1：M 档位（kMT 个 16 行 tile 共享 B 一遍）----
    if constexpr (kPhase != MegaPhase::L2) {
    const uint8_t* pool = S.pool_base;
    const uint8_t* pool_w = pool + (size_t)pool_slot * S.a->win_cap * kHidden;
    const int g_l1 = ((2 * S.a->l1_k_out) / kL1BlockN) / S.nslice_eff;
    const int g0_l1 = slice * g_l1;
    // [集71] L1 阶段执行次数（count>0 的任务，tid0 计一次）
    if (tid == 0 && count > 0) MX_CTR_ADD(S.ctr, kCtrL1Exec, 1);
    if (CM_L1_GATE) {
        uint8_t* ffn_slot = (S.a->ffn_hidden != nullptr)
            ? S.a->ffn_hidden + (size_t)pool_slot * S.a->win_cap * S.a->l1_k_out
            : nullptr;
        const int out_col_base = slice * (S.a->l1_k_out / S.nslice_eff);
        // A fragment 副本：一律走 local_data.frag_a/frag2 专用缓冲
        const bool kFragNew = S.frag_all && count > 0;
        const uint8_t* fbase = nullptr;
        uint8_t* ffn2 = nullptr;
        if (kFragNew) {
            const size_t asz =
                (size_t)(kHidden / 32) * (size_t)S.mt_total * 512;
            const size_t f2sz =
                (size_t)(S.a->l1_k_out / 32) * (size_t)S.mt_total * 512;
            const size_t soff = (size_t)pool_slot;   // [L1L2-SPLIT] 数据槽
            if (CM_FRAG_A) fbase = S.frag_a + soff * asz;
            if (CM_FRAG_L2 && CM_FRAG_F2) ffn2 = S.frag2 + soff * f2sz;
        }
        // 行主序 ffn_hidden 仅 bench 数值校验读 slot0；fragment 时其余槽只写 F2
        if ((kFragNew && CM_FRAG_L2) && pool_slot != 0)
            ffn_slot = nullptr;
#if R19_BLOCK_KSPLIT
        // [BALANCE·docs/20 §4.2b] 跨 block K 切：本任务只算一个 K 段的部分和，
        //   最后一段归约 + 发布 l1_prog；owner（kseg==0）在下方等 l1_prog。
        //   要求 A 走 fragment（fbase）以便按全局 K step 定位；否则退回原链。
        if (ksplit && fbase != nullptr) {
            const int part_slot = e * kRing + (w % kRing);
            float* part = S.l1_partial +
                ((size_t)part_slot * kNSlice + slice) * kMaxTPR * kL1SliceCols;
            uint32_t* cnt = S.l1_part_cnt + (size_t)part_slot * kNSlice + slice;
            const unsigned gen =
                epoch + (unsigned)(w / (kRing > 0 ? kRing : 1));
            #define R19_KSPLIT_CALL(KMT)                                          \
                r19_l1_ksplit_iso<KMT>(                                           \
                    pool_w, fbase, S.mt_total, S.a->w1, e, count, kHidden,        \
                    (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps,    \
                    g0_l1, g_l1, out_col_base, S.a->l1_k_out, part, cnt,          \
                    ffn_slot, ffn2, kseg, S.bar_flags, (int)S.a->l1_prog_idx,     \
                    pool_slot, slice, gen, (int)S.nslice_eff, S.a->max_spins,     \
                    S.dbg)
            if (count <= 16) R19_KSPLIT_CALL(1);
            else if (count <= 32) R19_KSPLIT_CALL(2);
            else if (count <= 48) R19_KSPLIT_CALL(3);
            else R19_KSPLIT_CALL(4);
            #undef R19_KSPLIT_CALL
        } else
#endif
#if L1_SPLITK
        // [BALANCE·docs/20 §4.2] split-K 路径：块内跑 kNK 个 K 段 + 归约。
        //   要求 A 走 fragment（fbase）以便按全局 K step 定位；否则退回原链。
        if (fbase != nullptr) {
            const int part_slot = e * kRing + (w % kRing);
            float* part = S.l1_partial +
                ((size_t)part_slot * kNSlice + slice) * kMaxTPR * kL1SliceCols;
            uint32_t* cnt =
                S.l1_part_cnt + (size_t)part_slot * kNSlice + slice;
            r19_l1_splitk_iso<4, R19_L1_PF, 4>(
                pool_w, fbase, S.mt_total, S.a->w1, e, count, kHidden,
                (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps,
                g0_l1, g_l1, out_col_base, S.a->l1_k_out, part, cnt,
                ffn_slot, ffn2);
        } else
#endif
        {
        // [R24/R34] A-smem 路径：F 可用 + kKS 整除 + 单批（g_l1==kBatch）。
        // [R34] 去掉 count%(MT*16)==0 门控：count>64 时走"全块 + 尾块"两段式
        // （尾块按 row<cc 掩码），权重遍数 = ceil(count/(MT*16))（旧整段 MT4
        // 回退为 ceil(count/64)）。count<=64 时权重同为 1 遍而 async 多一遍
        // stage 同步且尾块掩码有开销，实测更慢 → 保持非 async 回退。
        const bool kAsync = (fbase != nullptr) && (count > 64) &&
                            ((kHidden / 32) % R24_KS == 0) &&
                            (g_l1 % R19_L1_BATCH == 0);
        if (count <= 32 && g_l1 % 4 == 0) {
            // [L1TUNE-B] 17..32 档：MT2（同 1 组权重、acc 减半）替代 MT4。
            //   实测 TPR24 630→483（−23%，docs/15/16）。count≤16 走 small 档。
            // [261012_方向2] PF1→PF2：本档 B 为延迟受限，PF1 每步暴露全局装载
            //   延迟；regspill 隔离（R19_L1BIG_ISO noinline）落地后 PF2 的额外
            //   寄存器预算可承受。实测 -s2 top50 750.4→734.4µs（−2.1%）、
            //   均衡 TPR32 489.8→470.1µs（−4.0%），check --all 762/762 PASS。
            r19_l1_big_iso<2, 2, 4>(
                pool_w, fbase, S.mt_total, S.a->w1, e, count, kHidden,
                (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps, S.l1_chk,
                g0_l1, g_l1, ffn_slot, ffn2, out_col_base, S.a->l1_k_out);
        } else if (count <= 32 && g_l1 % 2 == 0 && g_l1 >= 2) {
            // [R35] NSLICE=8 等 g_l1=2 几何：单批 2 组小档（A 装载一遍、MT2
            //   runtime 裁剪），与 NSLICE=4 的 <2,kNT,2,4> 同构按比例缩小；
            //   避免落入 r19_l1_big<4,...> 的 MT4 档（A 读 2× + M 空转）。
            r19_l1_big_iso<4, R19_L1_PF, 2>(
                pool_w, fbase, S.mt_total, S.a->w1, e, count, kHidden,
                (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps, S.l1_chk,
                g0_l1, g_l1, ffn_slot, ffn2, out_col_base, S.a->l1_k_out);
        } else if (count <= 48 && g_l1 % 4 == 0) {
            // [OPT-33] 33..48 档：MT3（恰好 ceil(count/16)=3 个 tile），acc 比
            //   MT4 小（[4][3][2] → [4][3][2] 不变，但省 1 个 M tile 的寄存器/
            //   发射槽），减少尾块 M 空转（原 MT4 对 33..48 仍按 4 tile 分配）。
            bool med_r24 = false;
#if CM_MED_R24
            if (fbase != nullptr && ((kHidden / 32) % 16 == 0)) {
                // [OPT-KCAP-REMOVE] win_cap<=64（tk/长尾）用更深 B 预取环(4)；
                //   均衡大档保持 R24_BPF(2)。运行期等价原 kCap>0 编译期选择。
                if (S.a->win_cap <= 64) {
                    r24_l1_big<3, 16, CM_L1_BATCH, true, 3, 4, kBsmB>(
                        a_smem, fbase, S.mt_total, S.a->w1, e, 0, count, kHidden,
                        (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps,
                        S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2, out_col_base,
                        S.a->l1_k_out);
                } else {
                    r24_l1_big<3, 16, CM_L1_BATCH, true, 3, R24_BPF>(
                        a_smem, fbase, S.mt_total, S.a->w1, e, 0, count, kHidden,
                        (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps,
                        S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2, out_col_base,
                        S.a->l1_k_out);
                }
                med_r24 = true;
            }
#endif
            if (!med_r24) {
                r19_l1_big_iso<3, R19_L1_PF, 4>(
                pool_w, fbase, S.mt_total, S.a->w1, e, count, kHidden,
                (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps, S.l1_chk,
                g0_l1, g_l1, ffn_slot, ffn2, out_col_base, S.a->l1_k_out);
            }
        } else if (count <= 48 && g_l1 % 2 == 0 && g_l1 >= 2) {
            // [OPT-NS8] NSLICE=8 时 g_l1=2：33..48 档用 MT3/B2（A 装一遍、
            //   runtime M 裁剪），避免落入 r19_l1_big<4,...> 的 MT4 慢档。
            r19_l1_big_iso<3, R19_L1_PF, 2>(
                pool_w, fbase, S.mt_total, S.a->w1, e, count, kHidden,
                (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps, S.l1_chk,
                g0_l1, g_l1, ffn_slot, ffn2, out_col_base, S.a->l1_k_out);
        } else if (count > 48 && count <= 64 && fbase != nullptr &&
                   ((kHidden / 32) % R24_KS == 0)) {
            // [L1TUNE-D] 49..64 档走 r24 A-smem：整块(64)无掩码、49..63 尾块掩码。
            //   原条件仅 count%64==0 进入，49..63 掉进 r19 慢回退（TPR54 826→653，docs/16）。
            // [OPT-KCAP-REMOVE] win_cap<=64 用 BPF=4，均衡大档 BPF=2。
            // [BALANCE·r19balance] 满窗档 N 组批宽用 R19_L1_BATCH64（默认 4）：本档
            //   kMT=4、g_l1=4，4 组一次覆盖整片 N → A 只装一遍（batch=2 时装两遍）。
            //   g_l1 不是其倍数（如 NS=8 → g_l1=2）时回落原 R19_L1_BATCH 路径。
            const bool kB64 = (g_l1 % R19_L1_BATCH64 == 0);
            if (count % (4 * kM) == 0) {
                if (S.a->win_cap <= 64) {
                    if (kB64)
                        r24_l1_big<4, R24_KS, R19_L1_BATCH64, false, 4, 4, kBsmB>(
                            a_smem, fbase, S.mt_total, S.a->w1, e, 0, count, kHidden,
                            (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps,
                            S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2, out_col_base,
                            S.a->l1_k_out);
                    else
                        r24_l1_big<4, R24_KS, R19_L1_BATCH, false, 4, 4, kBsmB>(
                            a_smem, fbase, S.mt_total, S.a->w1, e, 0, count, kHidden,
                            (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps,
                            S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2, out_col_base,
                            S.a->l1_k_out);
                } else {
                    r24_l1_big<4, R24_KS, R19_L1_BATCH, false, 4, R24_BPF>(
                        a_smem, fbase, S.mt_total, S.a->w1, e, 0, count, kHidden,
                        (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps,
                        S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2, out_col_base,
                        S.a->l1_k_out);
                }
            } else {
                if (S.a->win_cap <= 64) {
                    if (kB64)
                        r24_l1_big<4, R24_KS, R19_L1_BATCH64, true, 4, 4, kBsmB>(
                            a_smem, fbase, S.mt_total, S.a->w1, e, 0, count, kHidden,
                            (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps,
                            S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2, out_col_base,
                            S.a->l1_k_out);
                    else
                        r24_l1_big<4, R24_KS, R19_L1_BATCH, true, 4, 4, kBsmB>(
                            a_smem, fbase, S.mt_total, S.a->w1, e, 0, count, kHidden,
                            (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps,
                            S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2, out_col_base,
                            S.a->l1_k_out);
                } else {
                    r24_l1_big<4, R24_KS, R19_L1_BATCH, true, 4, R24_BPF>(
                        a_smem, fbase, S.mt_total, S.a->w1, e, 0, count, kHidden,
                        (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps,
                        S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2, out_col_base,
                        S.a->l1_k_out);
                }
            }
        } else if (kAsync) {
            // [R34] 全块（无掩码，原 async 快路径）+ 尾块（kMask=true）两段式：
            //   尾块 A 读取折叠到 F 最后一 tile、写/checksum 按 row<cc 掩码，
            //   权重遍数 = ceil(count/(MT*16))（旧整段 MT4 回退为 ceil(count/64)）
            const int full = count - count % (R19_L1_MT * kM);
            if (full > 0)
                r24_l1_big<R19_L1_MT, R24_KS, R19_L1_BATCH, false, R19_L1_MT, kR24BPF>(
                    a_smem, fbase, S.mt_total, S.a->w1, e, 0, full, kHidden,
                    (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps,
                    S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2, out_col_base,
                    S.a->l1_k_out);
            if (full < count) {
                // [R34] 尾块只占 1 tile（r<=16，TPR=129/257/513 皆 r=1）时用
                //   MT1/KS16 小档：A/MMA 降到 1/8（B 遍数不变），并省一半 stage
                //   同步；几何不匹配时回退 MT8 掩码档
                const bool kTiny = (count - full <= kM) &&
                                   ((kHidden / 32) % 16 == 0);
                // [R36] 尾块活跃 tile 数（编译期裁剪 A/MMA/epilogue）
                const int l1_act = (count - full + kM - 1) / kM;
                if (kTiny)
                    r24_l1_big<1, 16, R19_L1_BATCH, true, 1, kR24BPF>(
                        a_smem, fbase, S.mt_total, S.a->w1, e, full, count,
                        kHidden, (2 * S.a->l1_k_out), kL1BlockN, lane, warp,
                        kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2,
                        out_col_base, S.a->l1_k_out);
                // [TAIL-MT] 尾块按精确活跃 tile 数选档（原仅 5/8）：count∈65..127
                //   原先一律用 MTAct=8（按 128 行算）→ 白算（TPR 84/90 台阶）。
                else if (l1_act == 2)
                    r24_l1_big<R19_L1_MT, R24_KS, R19_L1_BATCH, true, 2, kR24BPF>(
                        a_smem, fbase, S.mt_total, S.a->w1, e, full, count,
                        kHidden, (2 * S.a->l1_k_out), kL1BlockN, lane, warp,
                        kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2,
                        out_col_base, S.a->l1_k_out);
                else if (l1_act == 3)
                    r24_l1_big<R19_L1_MT, R24_KS, R19_L1_BATCH, true, 3, kR24BPF>(
                        a_smem, fbase, S.mt_total, S.a->w1, e, full, count,
                        kHidden, (2 * S.a->l1_k_out), kL1BlockN, lane, warp,
                        kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2,
                        out_col_base, S.a->l1_k_out);
                else if (l1_act == 4)
                    r24_l1_big<R19_L1_MT, R24_KS, R19_L1_BATCH, true, 4, kR24BPF>(
                        a_smem, fbase, S.mt_total, S.a->w1, e, full, count,
                        kHidden, (2 * S.a->l1_k_out), kL1BlockN, lane, warp,
                        kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2,
                        out_col_base, S.a->l1_k_out);
                else if (l1_act == 5)
                    r24_l1_big<R19_L1_MT, R24_KS, R19_L1_BATCH, true, 5, kR24BPF>(
                        a_smem, fbase, S.mt_total, S.a->w1, e, full, count,
                        kHidden, (2 * S.a->l1_k_out), kL1BlockN, lane, warp,
                        kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2,
                        out_col_base, S.a->l1_k_out);
                else if (l1_act == 6)
                    r24_l1_big<R19_L1_MT, R24_KS, R19_L1_BATCH, true, 6, kR24BPF>(
                        a_smem, fbase, S.mt_total, S.a->w1, e, full, count,
                        kHidden, (2 * S.a->l1_k_out), kL1BlockN, lane, warp,
                        kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2,
                        out_col_base, S.a->l1_k_out);
                else if (l1_act == 7)
                    r24_l1_big<R19_L1_MT, R24_KS, R19_L1_BATCH, true, 7, kR24BPF>(
                        a_smem, fbase, S.mt_total, S.a->w1, e, full, count,
                        kHidden, (2 * S.a->l1_k_out), kL1BlockN, lane, warp,
                        kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2,
                        out_col_base, S.a->l1_k_out);
                else
                    r24_l1_big<R19_L1_MT, R24_KS, R19_L1_BATCH, true, R19_L1_MT, kR24BPF>(
                        a_smem, fbase, S.mt_total, S.a->w1, e, full, count,
                        kHidden, (2 * S.a->l1_k_out), kL1BlockN, lane, warp,
                        kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2,
                        out_col_base, S.a->l1_k_out);
            }
        } else
            // [R27·delivered] 非 async 回退固定 MT4/B4：避免与 async 的 MT8 档
            // 叠成两份大 acc 实例（同一 noinline 任务体寄存器/栈按最坏路径分配，
            // MT8 全档回退实测栈 1880B→2688B、71 机 TPR=128 回归 +4.7%；固定
            // MT4 回退后栈 2072B、TPR=128 回归收窄至 +2.8%）。
            r19_l1_big_iso<4, R19_L1_PF, 4>(
                pool_w, fbase, S.mt_total, S.a->w1, e, count, kHidden,
                (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps, S.l1_chk,
                g0_l1, g_l1, ffn_slot, ffn2, out_col_base, S.a->l1_k_out);
        }   // end L1_SPLITK else（原实例选择链）
    }   // end if (CM_L1_GATE)
    }   // end if constexpr (kPhase != MegaPhase::L2)  —— L1 段
#if R24_DIAG_PHASE
    if (tid == 0)
        atomicAdd(S.pt + (size_t)kGrid * 8 + (size_t)blockIdx.x * 8,
                  clock64() - t_l1s);
#endif
    // [L1L2-SPLIT] l1_prog 发布+等待是合并路径内的 L1→L2 依赖。拆分后由
    //   L1 kernel 的 slice_flags + kernel 边界保证 → 编译期跳过。
    // [BALANCE·docs/20 §4.2b] K 切：仅 owner（kseg==0）等 l1_prog（不发布，发布由
    //   各 slice 的最后 K 段完成）；非 owner 已由下方 return 结束。
    if constexpr (kPhase == MegaPhase::Both) {
#if R19_BLOCK_KSPLIT
        if (ksplit)
            { if (kseg == 0) r19_l1_wait_all(S, pool_slot, w); }
        else
#endif
            r19_l1_pub_wait(S, task, count, slice, epoch, pool_slot, w);
    }   // end if constexpr (kPhase == MegaPhase::Both)
#if R19_BLOCK_KSPLIT
    // 非 owner 的 K 段只做部分和：到此结束（不跑 L2/tail，不发布 l2_done）。
    if constexpr (kPhase != MegaPhase::L1) {
        if (ksplit && kseg != 0) return;
    }
#endif
#if R24_DIAG_PHASE
    const unsigned long long t_l2s = (tid == 0) ? clock64() : 0ull;
#endif
    // ---- L2：M 档位（K=I、N=H），BF16 写每块 staging 槽 ----
    if constexpr (kPhase != MegaPhase::L1) {
    const int g_l2 = (S.a->l2_k_out / kL1BlockN) / S.nslice_eff;
    const int g0_l2 = slice * g_l2;
    // [集71] L2 阶段执行次数（count>0 的任务，tid0 计一次）
    if (tid == 0 && count > 0) MX_CTR_ADD(S.ctr, kCtrL2Exec, 1);
    if (CM_L2_GATE) {
        const uint8_t* ffn_slot =
            (S.a->ffn_hidden != nullptr)
                ? S.a->ffn_hidden + (size_t)pool_slot * S.a->win_cap * S.a->l1_k_out
                : nullptr;
        uint16_t* out_slot = (S.a->l2_out != nullptr)
            ? S.a->l2_out + (size_t)blockIdx.x * S.a->win_cap * S.a->l2_k_out
            : nullptr;
        const uint8_t* w2_e = S.a->w2 + (size_t)e * S.a->l2_k_out * S.a->l1_k_out;
        // L2 的 A 走 local_data.frag2 的 fragment 副本
        const uint8_t* f2 = nullptr;
        if (CM_FRAG_L2 && S.frag_all && count > 0) {
            const size_t f2sz =
                (size_t)(S.a->l1_k_out / 32) * (size_t)S.mt_total * 512;
            if (CM_FRAG_A2)
                f2 = S.frag2 + (size_t)pool_slot * f2sz;   // [L1L2-SPLIT] 数据槽
        }
        // [R24/R34] L2 A-smem 路径（同 L1：count>64 才走，≤64 保持回退）
        // [R35] 去掉 g_l2%R19_L2_BATCH==0 门控：奇数 g_l2（NSLICE=8 → 24/8=3）
        //   由 r24_l2_big 的 kBatch=1 余量路径补齐，不再回退 MT4 慢路径。
        const bool kAsyncL2 = (f2 != nullptr) && (count > 64) &&
                              ((S.a->l1_k_out / 32) % R24_KS == 0) && (g_l2 >= 1);
        // [R29] 融合激活：仅当 tail 支持 + r24 路径 + staging 有效
#if R29_COMBINE_FUSE
        fuse.on = fuse.capable && kAsyncL2 && (out_slot != nullptr);
#else
        fuse.on = false;   // 回退：L2 不推送，任务尾走整段推送
#endif
        if (count <= 32 && g_l2 % 6 == 0) {
            // [R34] 同 L1：小档单批（A 只装一遍，B 一遍）
#if CM_FRAG_L2
            if (f2 != nullptr)
                l2_gemm_body_iso<CM_L2_PF, kNTilesPerWarp, 2, 6, true>(
                    ffn_slot, w2_e, count, S.a->l1_k_out, S.a->l2_k_out,
                    kL1BlockN, lane, warp, kComputeWarps, S.l2_chk,
                    g0_l2, g_l2, out_slot, S.a->l2_k_out, f2, S.mt_total);
            else
#endif
                l2_gemm_body_iso<CM_L2_PF, kNTilesPerWarp, 2, 6, false>(
                    ffn_slot, w2_e, count, S.a->l1_k_out, S.a->l2_k_out, kL1BlockN,
                    lane, warp, kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                    S.a->l2_k_out, nullptr, 0);
        } else if (count <= 32 && g_l2 % 3 == 0 && g_l2 >= 3) {
            // [R35] NSLICE=8 等 g_l2=3 几何：单批 3 组小档（A 一遍、MT2 裁剪），
            //   避免落入 r19_l2_big<4,PF,3> 的 MT4 档（M 空转）。
#if CM_FRAG_L2
            if (f2 != nullptr)
                l2_gemm_body_iso<2, kNTilesPerWarp, 2, 3, true>(
                    ffn_slot, w2_e, count, S.a->l1_k_out, S.a->l2_k_out, kL1BlockN,
                    lane, warp, kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                    S.a->l2_k_out, f2, S.mt_total);
            else
#endif
                l2_gemm_body_iso<2, kNTilesPerWarp, 2, 3>(
                    ffn_slot, w2_e, count, S.a->l1_k_out, S.a->l2_k_out, kL1BlockN,
                    lane, warp, kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                    S.a->l2_k_out);
        } else if (count <= 48 && g_l2 % 6 == 0) {
            // [OPT-33] 33..48 档：MT3 单批（ceil(count/16)=3 tile），acc[3][3][2]；
            //   原 MT4/3 分两批（acc[3][4][2]=24）→ MT3 更小且只一次调用。
            bool med_r24 = false;
#if CM_MED_R24
            if (f2 != nullptr && ((S.a->l1_k_out / 32) % 16 == 0)) {
                r24_l2_big<3, 16, 3, true, 3, kR24BPF, kBsmB>(
                    a_smem, f2, S.mt_total, w2_e, 0, count, S.a->l1_k_out,
                    S.a->l2_k_out, kL1BlockN, lane, warp, kComputeWarps,
                    S.l2_chk, g0_l2, g_l2, out_slot, S.a->l2_k_out, nullptr);
                med_r24 = true;
            }
#endif
            if (!med_r24) {
#if CM_FRAG_L2
            if (f2 != nullptr)
                l2_gemm_body_iso<CM_L2_PF, kNTilesPerWarp, 3, 3, true>(
                    ffn_slot, w2_e, count, S.a->l1_k_out, S.a->l2_k_out,
                    kL1BlockN, lane, warp, kComputeWarps, S.l2_chk,
                    g0_l2, g_l2, out_slot, S.a->l2_k_out, f2, S.mt_total);
            else
#endif
                l2_gemm_body_iso<CM_L2_PF, kNTilesPerWarp, 3, 3, false>(
                    ffn_slot, w2_e, count, S.a->l1_k_out, S.a->l2_k_out, kL1BlockN,
                    lane, warp, kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                    S.a->l2_k_out, nullptr, 0);
            }
        } else if (count > 48 && count <= 64 && f2 != nullptr &&
                   (count % (4 * kM) == 0) &&
                   ((S.a->l1_k_out / 32) % R24_KS == 0)) {
            // [OPT-64] 49..64 档 L2 同样走 r24 A-smem（MT4 无掩码）
            r24_l2_big<4, R24_KS, 3, false, 4, kR24BPF, kBsmB>(
                a_smem, f2, S.mt_total, w2_e, 0, count, S.a->l1_k_out,
                S.a->l2_k_out, kL1BlockN, lane, warp, kComputeWarps,
                S.l2_chk, g0_l2, g_l2, out_slot, S.a->l2_k_out, nullptr);
        } else if (kAsyncL2) {
            // [R34] 同 L1：全块 + 尾块（尾块掩码；融合推送对两段都生效）
            const int full = count - count % (R19_L2_MT * kM);
            if (full > 0)
                r24_l2_big<R19_L2_MT, R24_KS, R19_L2_BATCH, false, R19_L2_MT, kR24BPF>(
                    a_smem, f2, S.mt_total, w2_e, 0, full, S.a->l1_k_out,
                    S.a->l2_k_out, kL1BlockN, lane, warp, kComputeWarps,
                    S.l2_chk, g0_l2, g_l2, out_slot, S.a->l2_k_out,
                    fuse.on ? &fuse : nullptr);
            if (full < count) {
                const bool kTiny = (count - full <= kM) &&
                                   ((S.a->l1_k_out / 32) % 16 == 0);
                const int l2_act = (count - full + kM - 1) / kM;
                if (kTiny)
                    r24_l2_big<1, 16, R19_L2_BATCH, true, 1, kR24BPF>(
                        a_smem, f2, S.mt_total, w2_e, full, count,
                        S.a->l1_k_out, S.a->l2_k_out, kL1BlockN, lane, warp,
                        kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                        S.a->l2_k_out, fuse.on ? &fuse : nullptr);
                // [TAIL-MT] 同 L1：按精确活跃 tile 选档，消除 count∈65..127 的 MTAct=8 白算
                else if (l2_act == 2)
                    r24_l2_big<R19_L2_MT, R24_KS, R19_L2_BATCH, true, 2, kR24BPF>(
                        a_smem, f2, S.mt_total, w2_e, full, count,
                        S.a->l1_k_out, S.a->l2_k_out, kL1BlockN, lane, warp,
                        kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                        S.a->l2_k_out, fuse.on ? &fuse : nullptr);
                else if (l2_act == 3)
                    r24_l2_big<R19_L2_MT, R24_KS, R19_L2_BATCH, true, 3, kR24BPF>(
                        a_smem, f2, S.mt_total, w2_e, full, count,
                        S.a->l1_k_out, S.a->l2_k_out, kL1BlockN, lane, warp,
                        kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                        S.a->l2_k_out, fuse.on ? &fuse : nullptr);
                else if (l2_act == 4)
                    r24_l2_big<R19_L2_MT, R24_KS, R19_L2_BATCH, true, 4, kR24BPF>(
                        a_smem, f2, S.mt_total, w2_e, full, count,
                        S.a->l1_k_out, S.a->l2_k_out, kL1BlockN, lane, warp,
                        kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                        S.a->l2_k_out, fuse.on ? &fuse : nullptr);
                else if (l2_act == 5)
                    r24_l2_big<R19_L2_MT, R24_KS, R19_L2_BATCH, true, 5, kR24BPF>(
                        a_smem, f2, S.mt_total, w2_e, full, count,
                        S.a->l1_k_out, S.a->l2_k_out, kL1BlockN, lane, warp,
                        kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                        S.a->l2_k_out, fuse.on ? &fuse : nullptr);
                else if (l2_act == 6)
                    r24_l2_big<R19_L2_MT, R24_KS, R19_L2_BATCH, true, 6, kR24BPF>(
                        a_smem, f2, S.mt_total, w2_e, full, count,
                        S.a->l1_k_out, S.a->l2_k_out, kL1BlockN, lane, warp,
                        kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                        S.a->l2_k_out, fuse.on ? &fuse : nullptr);
                else if (l2_act == 7)
                    r24_l2_big<R19_L2_MT, R24_KS, R19_L2_BATCH, true, 7, kR24BPF>(
                        a_smem, f2, S.mt_total, w2_e, full, count,
                        S.a->l1_k_out, S.a->l2_k_out, kL1BlockN, lane, warp,
                        kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                        S.a->l2_k_out, fuse.on ? &fuse : nullptr);
                else
                    r24_l2_big<R19_L2_MT, R24_KS, R19_L2_BATCH, true, R19_L2_MT, kR24BPF>(
                        a_smem, f2, S.mt_total, w2_e, full, count,
                        S.a->l1_k_out, S.a->l2_k_out, kL1BlockN, lane, warp,
                        kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                        S.a->l2_k_out, fuse.on ? &fuse : nullptr);
            }
        } else
            // [R27·delivered] 同上：回退固定 MT4/B3
            r19_l2_big<4, R19_L2_PF, 3>(
                ffn_slot, f2, S.mt_total, w2_e, count, S.a->l1_k_out,
                S.a->l2_k_out, kL1BlockN, lane, warp, kComputeWarps, S.l2_chk,
                g0_l2, g_l2, out_slot, S.a->l2_k_out);
    }   // end if (CM_L2_GATE)
    }   // end if constexpr (kPhase != MegaPhase::L1)  —— L2 段
    __syncthreads();
#if R24_DIAG_PHASE
    if (tid == 0)
        atomicAdd(S.pt + (size_t)kGrid * 8 + (size_t)blockIdx.x * 8 + 1,
                  clock64() - t_l2s);
#endif
    // [L1L2-SPLIT] COMBINE tail 依赖 L2 输出 → L1 阶段不执行
    if constexpr (kPhase != MegaPhase::L1)
        r19_tail_common(S, cctx, fuse.on, task, e, w, count, slice, epoch, pool_slot);
}
// ============================================================================
// 小 count（count<=kM）任务体：与基线逐字一致的热路径，独立 __noinline__
// 声明以隔离大任务的寄存器分配（探针：空体调用点对 TPR=16 中性）。
// ============================================================================
// [L1L2-SPLIT] 同 r19_big_task：kPhase 控制任务体的阶段范围（Both/L1/L2）。
template <MegaPhase kPhase = MegaPhase::Both>
__device__ R19_SMALL_ATTR void r19_small_task(
        const R19Sched& S, int task, int e, int w, int count, int slice,
        int kseg, unsigned epoch, int pool_slot, const int* pfx,
        uint8_t* a_smem, uint8_t* b_smem) {
    const int tid = threadIdx.x;
    const int warp = tid / 64;
    const int lane = tid % 64;
    const bool ksplit = R19_BLOCK_KSPLIT && (S.a->win_cap <= R19_BLOCK_KSPLIT_WINMAX) &&
                        (kNK > 1) && S.frag_all;
    (void)a_smem;
    const CombineCtx cctx = cm_build_ctx(S);
    (void)b_smem;
    // [L1L2-SPLIT] pull 恒调用：其内部按 kPhase 决定搬数据（L1/Both）还是
    //   只重算 wt/meta staging（L2）；L2 的 COMBINE tail 依赖此 staging。
    r19_pull_flags<kPhase>(S, task, e, w, count, slice, epoch, pool_slot, pfx, kseg);
    // ---- L1：小 count 档位（与基线实例选择一致）----
    if constexpr (kPhase != MegaPhase::L2) {
    const uint8_t* pool = S.pool_base;
    const uint8_t* pool_w = pool + (size_t)pool_slot * S.a->win_cap * kHidden;
    const int g_l1 = ((2 * S.a->l1_k_out) / kL1BlockN) / S.nslice_eff;
    const int g0_l1 = slice * g_l1;
    // [集71] L1 阶段执行次数（count>0 的任务，tid0 计一次）
    if (tid == 0 && count > 0) MX_CTR_ADD(S.ctr, kCtrL1Exec, 1);
    if (CM_L1_GATE) {
        uint8_t* ffn_slot = (S.a->ffn_hidden != nullptr)
            ? S.a->ffn_hidden + (size_t)pool_slot * S.a->win_cap * S.a->l1_k_out
            : nullptr;
        const int out_col_base = slice * (S.a->l1_k_out / S.nslice_eff);
        // [FRAG-ALL] 小档同样走 fragment：A 从 frag_a、输出写 frag2
        const bool kFragNew = S.frag_all && count > 0;
        const uint8_t* fbase = nullptr;
        uint8_t* ffn2 = nullptr;
        if (kFragNew) {
            const size_t asz = (size_t)(kHidden / 32) * (size_t)S.mt_total * 512;
            const size_t f2sz = (size_t)(S.a->l1_k_out / 32) * (size_t)S.mt_total * 512;
            const size_t soff = (size_t)pool_slot;   // [L1L2-SPLIT] 数据槽
            if (CM_FRAG_A) fbase = S.frag_a + soff * asz;
            if (CM_FRAG_L2 && CM_FRAG_F2) ffn2 = S.frag2 + soff * f2sz;
        }
        if ((kFragNew && CM_FRAG_L2) && pool_slot != 0) ffn_slot = nullptr;
#if R19_BLOCK_KSPLIT
        // [BALANCE·docs/20 §4.2b] 跨 block K 切（count<=16 → MT1）。
        if (ksplit && fbase != nullptr) {
            const int part_slot = e * kRing + (w % kRing);
            float* part = S.l1_partial +
                ((size_t)part_slot * kNSlice + slice) * kMaxTPR * kL1SliceCols;
            uint32_t* cnt = S.l1_part_cnt + (size_t)part_slot * kNSlice + slice;
            const unsigned gen =
                epoch + (unsigned)(w / (kRing > 0 ? kRing : 1));
            r19_l1_ksplit_iso<1>(
                pool_w, fbase, S.mt_total, S.a->w1, e, count, kHidden,
                (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps,
                g0_l1, g_l1, out_col_base, S.a->l1_k_out, part, cnt, ffn_slot,
                ffn2, kseg, S.bar_flags, (int)S.a->l1_prog_idx, pool_slot,
                slice, gen, (int)S.nslice_eff, S.a->max_spins, S.dbg);
        } else
#endif
#if L1_SPLITK
        // [BALANCE·docs/20 §4.2] 小档 split-K：同 big（fbase 必需）。
        if (fbase != nullptr) {
            const int part_slot = e * kRing + (w % kRing);
            float* part = S.l1_partial +
                ((size_t)part_slot * kNSlice + slice) * kMaxTPR * kL1SliceCols;
            uint32_t* cnt =
                S.l1_part_cnt + (size_t)part_slot * kNSlice + slice;
            if (g_l1 % 4 == 0)
                r19_l1_splitk_iso<2, R19_L1_PF, 4>(
                    pool_w, fbase, S.mt_total, S.a->w1, e, count, kHidden,
                    (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps,
                    g0_l1, g_l1, out_col_base, S.a->l1_k_out, part, cnt,
                    ffn_slot, ffn2);
            else
                r19_l1_splitk_iso<2, R19_L1_PF, 2>(
                    pool_w, fbase, S.mt_total, S.a->w1, e, count, kHidden,
                    (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps,
                    g0_l1, g_l1, out_col_base, S.a->l1_k_out, part, cnt,
                    ffn_slot, ffn2);
        } else
#endif
        {
        // 均衡场景 g_l1=4：一次 K 循环处理全部 4 组（A 装载减半）；
        // kPrefetch=2 是为把 32 个 B fragment 槽控制在寄存器预算内。
        if (g_l1 % 4 == 0) {
            // count<=16（热路径）档切 r19<1,2,4>：与 17-32 的 r19 统一，实测中性
            r19_l1_big<1, R19_L1_PF, 4>(
                pool_w, fbase, S.mt_total, S.a->w1, e, count, kHidden,
                (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps, S.l1_chk,
                g0_l1, g_l1, ffn_slot, ffn2, out_col_base, S.a->l1_k_out);
        } else
            // [b] g_l1=2（NSLICE=8）几何：paired<4,kNT,1,2> → r19<1,2,2>，
            //   彻底移除 small task 的 paired 实例（省其 PF4 预取环寄存器）
            r19_l1_big<1, R19_L1_PF, 2>(
                pool_w, fbase, S.mt_total, S.a->w1, e, count, kHidden,
                (2 * S.a->l1_k_out), kL1BlockN, lane, warp, kComputeWarps, S.l1_chk,
                g0_l1, g_l1, ffn_slot, ffn2, out_col_base, S.a->l1_k_out);
        }   // end L1_SPLITK else（原实例选择链）
    }   // end if (CM_L1_GATE)
    }   // end if constexpr (kPhase != MegaPhase::L2)  —— L1 段（small）
    if constexpr (kPhase == MegaPhase::Both) {
#if R19_BLOCK_KSPLIT
        if (ksplit)
            { if (kseg == 0) r19_l1_wait_all(S, pool_slot, w); }
        else
#endif
            r19_l1_pub_wait(S, task, count, slice, epoch, pool_slot, w);
    }   // end if constexpr (kPhase == MegaPhase::Both)
#if R19_BLOCK_KSPLIT
    if constexpr (kPhase != MegaPhase::L1) {
        if (ksplit && kseg != 0) return;
    }
#endif
    // ---- L2：小 count（与基线实例选择一致）----
    if constexpr (kPhase != MegaPhase::L1) {
    const int g_l2 = (S.a->l2_k_out / kL1BlockN) / S.nslice_eff;
    const int g0_l2 = slice * g_l2;
    // [集71] L2 阶段执行次数（count>0 的任务，tid0 计一次）
    if (tid == 0 && count > 0) MX_CTR_ADD(S.ctr, kCtrL2Exec, 1);
    if (CM_L2_GATE) {
        const uint8_t* ffn_slot =
            (S.a->ffn_hidden != nullptr)
                ? S.a->ffn_hidden + (size_t)pool_slot * S.a->win_cap * S.a->l1_k_out
                : nullptr;
        uint16_t* out_slot = (S.a->l2_out != nullptr)
            ? S.a->l2_out + (size_t)blockIdx.x * S.a->win_cap * S.a->l2_k_out
            : nullptr;
        const uint8_t* w2_e = S.a->w2 + (size_t)e * S.a->l2_k_out * S.a->l1_k_out;
        // [FRAG-ALL] L2 的 A 走 L1 的 frag2 fragment 副本
        const uint8_t* f2 = nullptr;
        if (CM_FRAG_L2 && S.frag_all && count > 0 && CM_FRAG_A2) {
            const size_t f2sz =
                (size_t)(S.a->l1_k_out / 32) * (size_t)S.mt_total * 512;
            f2 = S.frag2 + (size_t)pool_slot * f2sz;   // [L1L2-SPLIT] 数据槽
        }
#define L2CALL(KB)                                                             \
    do {                                                                       \
        if (f2 != nullptr)                                                     \
            l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 1, KB, true>(               \
                ffn_slot, w2_e, count, S.a->l1_k_out, S.a->l2_k_out, kL1BlockN,   \
                lane, warp, kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,    \
                S.a->l2_k_out, f2, S.mt_total);                                   \
        else                                                                   \
            l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 1, KB>(                     \
                ffn_slot, w2_e, count, S.a->l1_k_out, S.a->l2_k_out, kL1BlockN,   \
                lane, warp, kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,    \
                S.a->l2_k_out);                                                   \
    } while (0)
        if (g_l2 % 6 == 0) {
            L2CALL(6);
        } else if (g_l2 % 4 == 0) {
            L2CALL(4);
        } else if (g_l2 == 3) {
            L2CALL(3);
        } else if (g_l2 % 3 == 0) {
            // [O3] g_l2 为 3 的倍数（6/12/…）→ 3+3 两批对称
            const int gb = g_l2 / 2;
            if (f2 != nullptr) {
                l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 1, 3, true>(
                    ffn_slot, w2_e, count, S.a->l1_k_out, S.a->l2_k_out, kL1BlockN,
                    lane, warp, kComputeWarps, S.l2_chk, g0_l2, gb, out_slot,
                    S.a->l2_k_out, f2, S.mt_total);
                l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 1, 3, true>(
                    ffn_slot, w2_e, count, S.a->l1_k_out, S.a->l2_k_out, kL1BlockN,
                    lane, warp, kComputeWarps, S.l2_chk, g0_l2 + gb, g_l2 - gb,
                    out_slot, S.a->l2_k_out, f2, S.mt_total);
            } else {
                l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 1, 3>(
                    ffn_slot, w2_e, count, S.a->l1_k_out, S.a->l2_k_out, kL1BlockN,
                    lane, warp, kComputeWarps, S.l2_chk, g0_l2, gb, out_slot,
                    S.a->l2_k_out);
                l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 1, 3>(
                    ffn_slot, w2_e, count, S.a->l1_k_out, S.a->l2_k_out, kL1BlockN,
                    lane, warp, kComputeWarps, S.l2_chk, g0_l2 + gb, g_l2 - gb,
                    out_slot, S.a->l2_k_out);
            }
        } else {
            if (f2 != nullptr)
                l2_gemm_body<4, kNTilesPerWarp, 1, 2, true>(
                    ffn_slot, w2_e, count, S.a->l1_k_out, S.a->l2_k_out, kL1BlockN,
                    lane, warp, kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                    S.a->l2_k_out, f2, S.mt_total);
            else
                l2_gemm_body<4, kNTilesPerWarp, 1, 2>(
                    ffn_slot, w2_e, count, S.a->l1_k_out, S.a->l2_k_out, kL1BlockN,
                    lane, warp, kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                    S.a->l2_k_out);
        }
#undef L2CALL
    }
    }   // end if constexpr (kPhase != MegaPhase::L1)  —— L2 段（small）
    __syncthreads();
    // [L1L2-SPLIT] COMBINE tail 依赖 L2 输出 → L1 阶段不执行
    if constexpr (kPhase != MegaPhase::L1)
        r19_tail_common(S, cctx, false, task, e, w, count, slice, epoch, pool_slot);
}
