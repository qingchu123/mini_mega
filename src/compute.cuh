// ============================================================================
// compute.cuh — fp8xfp8 MoE Mega Kernel 计算链
//
// 依赖：common.cuh（场景常量）、flags.cuh（门/旗接口）。
//
// 导出：
//   dispatch_pull_body —— 任务循环驱动（每 block 一次）：领号/解码 → ring 门 →
//     K 片 pull（peer x → 本 rank pool）+ staging weights/meta → 零填充 → 分片旗
//     → L1+SwiGLU → l1_prog 发布+等本窗全切片 → L2 → tail(COMBINE) → l2_done。
//     任务体按 count 分派到 compute_big.cuh 的 r19_small_task/r19_big_task。
//   l1_gemm_swiglu_body_paired<pf,nt,mt,nb> —— N 组批量 FP8 GEMM + SwiGLU + FP8 量化。
//   l2_gemm_body<pf,nt,mt,nb> —— FP8 GEMM（K=I、N=H），BF16 输出。
//
// 权重契约（layout16b，host 须按此打包）：16B/lane 配对 tile，
//   supertile = W + ((out_col/32)*kIters + (k/32))*1024，一次 16B 得两个 N16 fragment；
//   W1 gate/up 配对：组 g 占 2I 行 [g*256,(g+1)*256)，配对 (2p,2p+1) 产出隐藏列
//   [p*256,(p+1)*256)；W2 expert stride = H*I。
// 地址/容量：窗口划分用 win_cap；pool 行地址用 pool_stride；src_index_table 用
//   tbl_cap；staging 行 stride = win_cap。
// 动态 smem 头部：EPR + (EPR+1) + EPR*(R+1) 个 int（全 experts src 前缀一次预载）。
// A 走 fragment（4-line）恒编译；协议（wait-all/分片旗/l1_prog/l2_done/checksum）
//   与基线一致。优化沿革与档位见 docs/12、docs/13、perf.md。
// ============================================================================
#pragma once

#include "common.cuh"
#include "flags.cuh"

// ---------------------------------------------------------------- 主 kernel --
// host -> device 参数（按值传结构体；全部为 POD）
struct MegaArgs {
    // ======================================================================
    // [1] grid 规划：决定 block/任务划分与行 stride 的几何量
    // ======================================================================
    int R, rank;               // rank 数 / 本 rank 号
    int tokens;                // 实际 token 数（<= slot_capacity）
    int slot_capacity;         // 专家 token 容量（= max_tokens；= src_index_table 行容量）
    int win_cap;               // 窗口容量（窗口划分 / pool / staging / ffn / l2 行 stride）
    // 已内联常量（原运行期传参，本子集固定）：
    //   k_in=kHidden、topk=kTopk、ring=kRing、l1_block_n=kL1BlockN、
    //   l1_n_total=2*kIntermediate、tbl_cap=slot_capacity、pool_stride=win_cap、
    //   publish_epoch=kPublishEpoch、recv_stride=kRecvStride

    // ======================================================================
    // [2] 调用入参：结构体/缓冲指针、偏移、flag 索引
    // ======================================================================
    // -- 结构体指针 --
    rwk_data* rwk;                // 本 rank 普通显存整块（lf + ld）
    unk_data* unk;                // 本 rank FG 整块（rf + rd；= unk_peers[rank]）
    unk_data* const* unk_peers;   // 各 rank unk 基址表（设备数组，含 self）
    // -- 对称区 / 输入 --
    const uint8_t* self;          // 本 rank 对称区基址
    uint8_t* const* peer_bases;   // 各 rank 对称区基址表（拉 x / weights）
    const int64_t* topk_idx;      // 本 rank 输入路由（对称区内视图）
    // -- 数据缓冲偏移（self / peer sym 内）--
    //   pool / prefix 已移入本 rank 私有 rwk->ld（kernel 从 a->rwk->ld 取）；
    //   [UNK-MERGE] token_route_weights/token_src_meta/prefold/combine_rows 已并入
    //   unk（kernel 经 a->unk->rd / a->unk_peers 取），不再经 sym 偏移。
    //   仅保留对外输入视图 x 与 topk_weights 的 sym 偏移。
    int64_t x_off, weights_off;
    int in_bf16;                  // [BF16-IN] 输入 x 是 bf16(1) / fp8(0)
    int skip_combine;             // [诊断] MX_SKIP_COMBINE=1 → 任务尾 COMBINE 空操作
    // -- 计算链缓冲 --
    // l1_k_out/l2_k_out 已内联为常量 kIntermediate/kHidden（本子集固定）。
    // [RWK-LD] w1/w2/ffn_hidden/l2_out 已删除：kernel 一律从本 rank 私有
    //   a->rwk->ld（local_data）派生（w1/w2 lane 序 tile、ffn_hidden FP8、
    //   l2_out BF16），不再经 host 传入，省 4 个指针（smem s_a −32B）。
    // -- flag 索引已内联为编译期常量 kBarXxxxIdx（common.cuh；镜像 layout 头）--

    // ======================================================================
    // [3] 输出
    // ======================================================================
    // 本 kernel（mega_kernel）不写 y；y 由独立 REDUCE 核经 host 侧指针写出
    // （MegaLaunchArgs::y）。[方向3] MXK_FUSE_REDUCE 已废弃，设备结构体不再携带 y。

    // ======================================================================
    // [4] 性能调优参数
    // ======================================================================
    int max_spins;                // 各同步门的自旋上限
    int skip_prologue;            // [R38] 前序复用：0 完整，>0 复用（见 kernel.cuh 说明）

    // ======================================================================
    // [5] 调试 / debug
    // ======================================================================
    // （控制/诊断状态 out_total/out_processed/dbg/ck_dbg/pre_ticks/l1_chk/l2_chk
    //   已迁入 local_flags（lf->...）。计数缓冲 exec_ctr/dbg_count 已移除：
    //   设备入口恒以 ctr=nullptr 运行，MX_CTR_ADD 展开为空。）
};


// [CM-SKIP] 分阶段消融诊断（默认 0，不影响基线；仅 perf-only 口径）：
//   CM_SKIP_GEMM=1 同时跳过 L1+L2（纯通信）；CM_SKIP_L1/L2 单独跳过。
#ifndef CM_SKIP_GEMM
#define CM_SKIP_GEMM 0
#endif
#ifndef CM_SKIP_L1
#define CM_SKIP_L1 0
#endif
#ifndef CM_SKIP_L2
#define CM_SKIP_L2 0
#endif
// [BLOCK-LPT] 代价感知的块级 LPT（默认 0 = 关闭，行为逐字不变）：
//   FCFS 仅均衡任务「个数」；同 rank 内各 block 的窗口行数（代价）差异可达
//   2.19×。开启后 tid0 在建立任务表后，按代价 = min(c_e-w*win_cap, win_cap)
//   （N/K 固定 → ∝ 行数）对 raw_tasks 个任务做降序稳定计数排序，写入 smem
//   s_order；领号仍用现有 atomicAdd(task_counter)，但取 task = s_order[counter]，
//   降序发放 + 动态领取 ≈ LPT；解码 (e,w,slice)/pool_slot/ring 门/协议不变。
#ifndef R19_BLOCK_LPT
#define R19_BLOCK_LPT 0
#endif
// [BALANCE·docs/20 §4.2b] 跨 block K 切（默认 0 = 关闭，行为逐字不变）：
//   把每个 (e,w,slice) 的 L1 再沿 K=H 切成 kNK 段，跨 block 派发
//   （raw_tasks = s_cum[kEPR]*kNSlice*kNK）。每段只算自己 K 段的 gate/up 部分和，
//   以 FP32 atomicAdd 累加到 l1_partial[slot][slice]；l1_part_cnt[slot][slice]
//   由各段 atomicAdd，最后一段（owner 之外）触发 r19_l1_reduce（SwiGLU→FP8）
//   并发布 l1_prog。pull/切片旗/L1 回写/L2/tail/COMBINE 每个 (e,w,slice) 仅由
//   owner（kseg==0）执行一次，其余段只做 K 部分和 —— 严禁每段都跑协议。
//   仅当 win_cap <= R19_BLOCK_KSPLIT_WINMAX 时启用（长尾/TK 小窗），大窗走原路径。
#ifndef R19_BLOCK_KSPLIT
#define R19_BLOCK_KSPLIT 0
#endif
#ifndef R19_BLOCK_KSPLIT_WINMAX
#define R19_BLOCK_KSPLIT_WINMAX 64
#endif
#if CM_SKIP_GEMM || CM_SKIP_L1
#define CM_L1_GATE (false)
#else
#define CM_L1_GATE (warp < kComputeWarps && count > 0)
#endif
#if CM_SKIP_GEMM || CM_SKIP_L2
#define CM_L2_GATE (false)
#else
#define CM_L2_GATE (warp < kComputeWarps && count > 0)
#endif

// ---------------------------------------------------------------- 类型别名 --
using Fp8MmaFragment = __NATIVE_VECTOR__(2, uint32_t);
// 16B/lane 宽装载：一次得到同 lane 的两个 native fragment（低/高 8B）
using Fp8MmaFragment2 = __NATIVE_VECTOR__(4, uint32_t);
using Fp8Accumulator = __NATIVE_VECTOR__(4, float);
using Fp32x2 = __NATIVE_VECTOR__(2, float);

// 计算 warp 数：launch 512（8 warp）。骨架配对路径用 8 warp × 2 tile（每 warp
// 32 列）；此前 4 warp × 4 tile 实测更慢。block_n 固定 256。
constexpr int kComputeWarps = 8;
constexpr int kNTilesPerWarp = 2;

// [OPT] 小 count 任务体 B 预取深度 / N 批宽（可 -D 扫描；PF 为 2 的幂）
#ifndef CM_L1_PF
#define CM_L1_PF 2
#endif
#ifndef CM_L1_BATCH
#define CM_L1_BATCH 4
#endif
#ifndef CM_L2_PF
#define CM_L2_PF 2
#endif
// [OPT-BPTR] B 装载指针强度削减：每次 k step 只 +1024B，去掉 64 位乘移
#ifndef CM_BPTR
#define CM_BPTR 1
#endif
// [OPT-BPTR-L2] L2 的 B k=0 基址预计算（省去每次装载的 64bit 乘加寻址）
#ifndef CM_BPTR_L2
#define CM_BPTR_L2 0
#endif
// [OPT-AFRAG] 子开关（诊断用）
#ifndef CM_FRAG_A
#define CM_FRAG_A 1
#endif
#ifndef CM_FRAG_F2
#define CM_FRAG_F2 1
#endif
#ifndef CM_FRAG_A2
#define CM_FRAG_A2 1
#endif
#ifndef CM_FRAG_L2
#define CM_FRAG_L2 1
#endif

// ------------------------------------------------------------ 装载/数学工具 --
// A（激活）native-map 直装：row = m_tile*16 + (lane&15)，k8 = (lane>>4)&3；
// 越界行（row >= count）返回零 fragment（尾部 M tile 安全）。
__device__ __forceinline__ Fp8MmaFragment cm_load_a(
        const uint8_t* __restrict__ pool, int k_stride, int k0, int count,
        int m_tile = 0) {
#ifdef CM_DIAG_NOA
    (void)pool; (void)k_stride; (void)k0; (void)count; (void)m_tile;
    return Fp8MmaFragment{0u, 0u};
#endif
    const int lane = threadIdx.x & 63;
    const int row = m_tile * kM + (lane & 15);
    Fp8MmaFragment f = {0u, 0u};
    if (row < count)
        f = *reinterpret_cast<const Fp8MmaFragment*>(
            pool + (size_t)row * k_stride + k0 + ((lane >> 4) & 3) * 8);
    return f;
}

// B（权重）16B/lane 配对 tile：一次 16B 装载 = 同 lane 的两个 native fragment。
// n 必须为 32 的倍数（调用点 nbase 均为 block_n/8 的整数倍：256/32 网格）。
// 低 8B = 逻辑行 n 的 fragment，高 8B = 逻辑行 n+16 的 fragment（本 K 段）。
__device__ __forceinline__ Fp8MmaFragment2 cm_load_b_lane16(
        const uint8_t* __restrict__ w, int n, int k, int k_hidden) {
#ifdef CM_DIAG_BFIX
    n = 0; k = 0;   // 诊断：固定地址（L1 命中），保留装载指令
#endif
    const int lane = threadIdx.x & 63;
    const size_t n_tiles = (size_t)(k_hidden >> 5);
    const uint8_t* tile =
        w + ((size_t)(n >> 5) * n_tiles + (size_t)(k >> 5)) * 1024;
    return *reinterpret_cast<const Fp8MmaFragment2*>(tile + (size_t)lane * 16);
}

__device__ __forceinline__ float cm_silu(float x) {
    return x / (1.0f + __expf(-x));
}

// BF16 打包（L2 输出；生产 pack_float8_to_bf16 同构）。
__device__ __forceinline__ uint32_t cm_pack_f2bf16(float a, float b) {
    const Fp32x2 v = {a, b};
    return __builtin_bit_cast(uint32_t, __builtin_mxc_cvt_pk_f32tobf16(v));
}

// [OPT-AFRAG] A/f2 fragment 装载（4-line/指令）；fbase 布局 [step][m][row][k]：
//   addr = (step*mt_total + m)*512 + (row&15)*32 + (k&31)
__device__ __forceinline__ Fp8MmaFragment cm_load_a_frag(
        const uint8_t* __restrict__ fbase, int kk, int m_abs, int mt_total,
        int row_limit) {
    const int lane = threadIdx.x & 63;
    const int row = m_abs * kM + (lane & 15);
    Fp8MmaFragment f = {0u, 0u};
    if (row < row_limit)
        f = *reinterpret_cast<const Fp8MmaFragment*>(
            fbase + ((size_t)kk * mt_total + m_abs) * 512 + (lane & 15) * 32 +
            ((lane >> 4) & 3) * 8);
    return f;
}

// [OPT-AFRAG] L1 输出 fragment 写（与 cm_load_a_frag 同布局，step=col/32）
__device__ __forceinline__ void cm_store_ffn_frag(
        uint8_t* __restrict__ f2, int row, int col, uint8_t v, int mt_total) {
    f2[((size_t)(col >> 5) * mt_total + (row >> 4)) * 512 + (row & 15) * 32 +
       (col & 31)] = v;
}

// ============================================================================
// L1：N 组批量共享 A（kBatch 组一个 K 循环）+ SwiGLU + FP8 量化回写。
//   组对 (g,g+1)=(gate,up)：kBatch 组共享 A（a_pipe 仅一份），A 装载次数 = 1/kBatch。
//   K 循环 kPrefetch 槽预取，首组按 K 片边界等分片旗。
//   acc lane 布局：row = m*16+(lane>>4)*4+i，col = warp*n_per_warp+t*16+(lane&15)。
// ============================================================================
template <int kPrefetch = 4, int kNTiles = kNTilesPerWarp, int kMTiles = 1,
          int kBatch = 4, bool kUseFrag = false>
__device__ __forceinline__ void l1_gemm_swiglu_body_paired(
        const uint8_t* pool,      // 本窗口 A 基址（pool_slot 已定位）
        const uint8_t* w1,        // W1 张量基址 [EPR][2I][H]
        int expert,               // 本任务专家号
        int count,                // 本窗口实际行数
        int k_hidden,             // H（K 维）
        int n_total,              // 2I（N 维）
        int block_n,              // N 组宽（256）
        int lane, int warp, int nwarps,
        float* checksum,          // 可选（nullptr 关闭防 DCE 校验和）
        int n_group_begin, int n_group_count,
        uint8_t* ffn_out,         // 本窗口 ffn_hidden 基址（可 nullptr）
        int out_col_base,         // 本 slice 的隐藏列起点
        int k_out,                // I（ffn_hidden 行 stride）
        const SliceGate* gate,    // 分片就绪门（nullptr → 关闭）
        const uint8_t* fbase = nullptr, int mt_total = 0,
        uint8_t* ffn2 = nullptr) {   // [OPT-AFRAG] A/f2 fragment（可 nullptr）
    constexpr int kKStep = 32;
    static_assert(kNTiles == 2, "layout16b 配对装载要求 kNTiles==2");
    const int k_iters = k_hidden / kKStep;
    const int n_per_warp = block_n / nwarps;
    // 运行期 M 裁剪：窗口容量是上限，实际只算 ceil(count/16) 个 M tile。
    const int m_tiles =
        kMTiles > 1
            ? ((count + kM - 1) / kM < kMTiles ? (count + kM - 1) / kM : kMTiles)
            : 1;
    const uint8_t* w_e = w1 + (size_t)expert * n_total * k_hidden;
    const bool gate_on =
        (gate != nullptr && gate->flags != nullptr && gate->k_iters_per_slice > 0);
    float acc_chk = 0.0f;
    // [OPT-AFRAG] A 装载：fbase 非空走 fragment（4-line），否则行主序池
    // [OPT-AFRAG] 编译期选择（运行时分支会破坏热路径调度：实测 3× 回退）
    auto lda = [&](int kk, int m) -> Fp8MmaFragment {
        return kUseFrag ? cm_load_a_frag(fbase, kk, m, mt_total, count)
                        : cm_load_a(pool, k_hidden, kk * kKStep, count, m);
    };
    // n_group_count 必须为 kBatch 的整数倍（组对）；调用点按余数选择实例。
    for (int g0 = n_group_begin; g0 + kBatch <= n_group_begin + n_group_count;
         g0 += kBatch) {
        const bool first_group = (g0 == n_group_begin);
        int nbase[kBatch];
        #pragma unroll
        for (int q = 0; q < kBatch; ++q)
            nbase[q] = (g0 + q) * block_n + warp * n_per_warp;
#if CM_BPTR
        // [OPT-BPTR] B 的 k=0 基址（含 lane 偏移）；k step 递进 = +1024B
        const uint8_t* b_base[kBatch];
        #pragma unroll
        for (int q = 0; q < kBatch; ++q)
#ifdef CM_DIAG_BFIX
            b_base[q] = w_e + (size_t)(lane & 63) * 16;
#else
            b_base[q] = w_e + (size_t)(nbase[q] >> 5) *
                                  (size_t)(k_hidden >> 5) * 1024 +
                        (size_t)(lane & 63) * 16;
#endif
#ifdef CM_DIAG_BFIX
#define CM_LDB_L1(q, step)                                                  \
    (*reinterpret_cast<const Fp8MmaFragment2*>(b_base[(q)] +                \
                                               (size_t)((step) & 7) * 1024))
#else
#define CM_LDB_L1(q, step)                                                  \
    (*reinterpret_cast<const Fp8MmaFragment2*>(b_base[(q)] +                \
                                               (size_t)(step) * 1024))
#endif
#else
#define CM_LDB_L1(q, step) cm_load_b_lane16(w_e, nbase[q], (step) * kKStep, k_hidden)
#endif
        Fp8MmaFragment a_pipe[kPrefetch][kMTiles];
        // [layout16b] 16B/槽：低/高 8B = t=0/t=1 两个 native fragment
        Fp8MmaFragment2 b_pair[kPrefetch][kBatch];
        #pragma unroll
        for (int sp = 0; sp < kPrefetch; ++sp) {
            if (sp >= k_iters) break;
            // 只在 K 片边界等门（sp<kPrefetch 通常同片；非边界 slice 不变）
            if (first_group && gate_on &&
                (sp == 0 || sp % gate->k_iters_per_slice == 0))
                slice_gate_wait(*gate, sp);
            #pragma unroll
            for (int m = 0; m < kMTiles; ++m) {
                if constexpr (kMTiles > 1) {
                    if (m >= m_tiles) break;
                }
                a_pipe[sp][m] = lda(sp, m);
            }
            // [layout16b] 一次 16B 装载覆盖 t=0/t=1 两个 N16 tile（免 shuffle）；
            // 指令序保持 q 反序（[H5] 物理落点分散）
            #pragma unroll
            for (int q = 0; q < kBatch; ++q)
                b_pair[sp][kBatch - 1 - q] = CM_LDB_L1(q, sp);
        }
        Fp8Accumulator acc[kBatch][kMTiles][kNTiles];
        #pragma unroll
        for (int q = 0; q < kBatch; ++q)
            #pragma unroll
            for (int m = 0; m < kMTiles; ++m)
                #pragma unroll
                for (int t = 0; t < kNTiles; ++t)
                    acc[q][m][t] = Fp8Accumulator{0.f, 0.f, 0.f, 0.f};
        // 显式 unroll：sp=kk&(kPrefetch-1) 变编译期常量，预取 ring 常驻寄存器
        //（分片门分支不得引入块级 barrier，否则 ring 退化 local memory）。
        #pragma unroll kPrefetch
        for (int kk = 0; kk < k_iters; ++kk) {
            const int sp = kk & (kPrefetch - 1);
            #pragma unroll
            for (int m = 0; m < kMTiles; ++m) {
                if constexpr (kMTiles > 1) {
                    if (m >= m_tiles) break;
                }
                #pragma unroll
                for (int q = 0; q < kBatch; ++q) {
                    // 低/高 8B 分别喂 t=0/t=1（native fragment，无 shuffle）
                    // [反序修正] 预取按 q 反序落 b_pair（[H5] 物理落点分散），
                    // 故此处同样按反序取回：acc[q] 恒对应 N 组 (g0+q)。原实现
                    // 此处取 b_pair[sp][q]，使 acc[q] 对应 (g0+kBatch-1-q)，与
                    // r24_l1_big/文档契约的 gate/up 配对相反；PRJ 测试权重值函数
                    // 周期 64（fill_w_lane）掩盖了该差异，真实权重下必须按契约
                    //（gate=N 组 2p、up=2p+1）计算。
                    const Fp8MmaFragment2 bp = b_pair[sp][kBatch - 1 - q];
                    const Fp8MmaFragment b0 = {bp[0], bp[1]};
                    const Fp8MmaFragment b1 = {bp[2], bp[3]};
                    acc[q][m][0] = __builtin_mxc_mma_16x16x32f8(
                        a_pipe[sp][m], b0, acc[q][m][0]);
                    acc[q][m][1] = __builtin_mxc_mma_16x16x32f8(
                        a_pipe[sp][m], b1, acc[q][m][1]);
                }
            }
            const int nk = kk + kPrefetch;
            if (nk < k_iters) {
                // 只在 K 片边界等门（非边界 slice 不变，避免每步 threadfence 开销）
                if (first_group && gate_on &&
                    nk % gate->k_iters_per_slice == 0)
                    slice_gate_wait(*gate, nk);
                #pragma unroll
                for (int m = 0; m < kMTiles; ++m) {
                    if constexpr (kMTiles > 1) {
                        if (m >= m_tiles) break;
                    }
                    a_pipe[sp][m] = lda(nk, m);
                }
                #pragma unroll
                for (int q = 0; q < kBatch; ++q)
                    b_pair[sp][kBatch - 1 - q] = CM_LDB_L1(q, nk);
            }
        }
        // SwiGLU + FP8 量化回写：批次内每 2 组（gate, up）产出一个隐藏列块
        #pragma unroll
        for (int pi = 0; pi < kBatch / 2; ++pi) {
            const int pair_idx = ((g0 - n_group_begin) >> 1) + pi;
            const int qa = 2 * pi, qb = 2 * pi + 1;
            #pragma unroll
            for (int m = 0; m < kMTiles; ++m) {
                if constexpr (kMTiles > 1) {
                    if (m >= m_tiles) break;
                }
                #pragma unroll
                for (int t = 0; t < kNTiles; ++t) {
                    Fp8Accumulator sw;
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const float gv = acc[qa][m][t][i];
                        const float uv = acc[qb][m][t][i];
                        sw[i] = cm_silu(gv) * uv;
                        acc_chk += sw[i];
                    }
                    if (ffn_out != nullptr || ffn2 != nullptr) {
                        const uint32_t packed = __builtin_mxc_cvt_pk4_f32tof8_cfg(
                            sw, _MACA_FROUND_TO_NEAREST_EVEN,
                            _MACA_DENORMAL_FLUSH_NONE, _MACA_SRC_NO_MODIFIER,
                            false, true);
                        const int col = out_col_base + pair_idx * block_n +
                                        warp * n_per_warp + t * 16 + (lane & 15);
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            const int row = m * kM + (lane >> 4) * 4 + i;
                            if (row < count) {
                                const uint8_t byte = (uint8_t)(packed >> (8 * i));
                                if (ffn_out != nullptr)
                                    ffn_out[(size_t)row * k_out + col] = byte;
                                if (ffn2 != nullptr)
                                    cm_store_ffn_frag(ffn2, row, col, byte,
                                                      mt_total);
                            }
                        }
                    }
                }
            }
        }
    }
    if (checksum != nullptr && lane == 0)
        atomicAdd(checksum, acc_chk + (float)count * 0.0f);
}

// ============================================================================
// L2：N 组批量共享 A + lane 序 B，BF16 输出（K=I、N=H）
//   A = ffn_hidden（L1 输出，FP8 行主序 [count][k_in]）、B = W2（lane 序 tile）。
//   kBatch 个 N 组共享一份 a_pipe：A 装载次数 = kBatch 分之一。
// ============================================================================

// ============================================================================
// L2：N 组批量共享 A + lane 序 B，BF16 输出（K=I、N=H）
//   A = ffn_hidden（L1 输出，FP8 行主序 [count][k_in]）、B = W2（lane 序 tile）。
//   kBatch 个 N 组共享一份 a_pipe：A 装载次数 = kBatch 分之一。
// ============================================================================
template <int kPrefetch = 4, int kNTiles = kNTilesPerWarp, int kMTiles = 1,
          int kBatch = 6, bool kUseFrag = false>
__device__ __forceinline__ void l2_gemm_body(
        const uint8_t* a_pool,    // 本窗口 ffn_hidden 基址
        const uint8_t* w2,        // W2 本 expert 基址
        int count,                // 本窗口实际行数
        int k_in,                 // I（K 维）
        int n_total,              // H（N 维）
        int block_n,              // N 组宽（256）
        int lane, int warp, int nwarps,
        float* checksum,          // 可选（nullptr 关闭）
        int n_group_begin, int n_group_count,
        uint16_t* out,            // 本块 staging 槽（可 nullptr）
        int k_out,                // H（行 stride）
        const uint8_t* f2 = nullptr, int mt_total = 0) {  // [OPT-AFRAG]
    constexpr int kKStep = 32;
    static_assert(kNTiles == 2, "layout16b 配对装载要求 kNTiles==2");
    const int k_iters = k_in / kKStep;
    const int n_per_warp = block_n / nwarps;
    const int m_tiles =
        kMTiles > 1
            ? ((count + kM - 1) / kM < kMTiles ? (count + kM - 1) / kM : kMTiles)
            : 1;
    float acc_chk = 0.0f;
    auto lda = [&](int kk, int m) -> Fp8MmaFragment {
        return kUseFrag ? cm_load_a_frag(f2, kk, m, mt_total, count)
                        : cm_load_a(a_pool, k_in, kk * kKStep, count, m);
    };
    for (int g0 = n_group_begin; g0 + kBatch <= n_group_begin + n_group_count;
         g0 += kBatch) {
        int nbase[kBatch];
        #pragma unroll
        for (int q = 0; q < kBatch; ++q)
            nbase[q] = (g0 + q) * block_n + warp * n_per_warp;
#if CM_BPTR_L2
        // [OPT-BPTR-L2] B 的 k=0 基址（含 lane 偏移）；k step 递进 = +1024B
        const uint8_t* b_base[kBatch];
        #pragma unroll
        for (int q = 0; q < kBatch; ++q)
            b_base[q] = w2 + (size_t)(nbase[q] >> 5) * (size_t)(k_in >> 5) *
                                 1024 +
                        (size_t)(lane & 63) * 16;
#define CM_LDB_L2(q, step)                                                  \
    (*reinterpret_cast<const Fp8MmaFragment2*>(b_base[(q)] +                \
                                               (size_t)(step) * 1024))
#else
#define CM_LDB_L2(q, step) cm_load_b_lane16(w2, nbase[q], (step) * kKStep, k_in)
#endif
        Fp8MmaFragment a_pipe[kPrefetch][kMTiles];
        Fp8MmaFragment2 b_pair[kPrefetch][kBatch];
        #pragma unroll
        for (int sp = 0; sp < kPrefetch; ++sp) {
            if (sp >= k_iters) break;
            #pragma unroll
            for (int m = 0; m < kMTiles; ++m) {
                if constexpr (kMTiles > 1) {
                    if (m >= m_tiles) break;
                }
                a_pipe[sp][m] = lda(sp, m);
            }
            #pragma unroll
            for (int q = 0; q < kBatch; ++q)
                b_pair[sp][q] = CM_LDB_L2(q, sp);
        }
        Fp8Accumulator acc[kBatch][kMTiles][kNTiles];
        #pragma unroll
        for (int q = 0; q < kBatch; ++q)
            #pragma unroll
            for (int m = 0; m < kMTiles; ++m)
                #pragma unroll
                for (int t = 0; t < kNTiles; ++t)
                    acc[q][m][t] = Fp8Accumulator{0.f, 0.f, 0.f, 0.f};
        #pragma unroll kPrefetch
        for (int kk = 0; kk < k_iters; ++kk) {
            const int sp = kk & (kPrefetch - 1);
            #pragma unroll
            for (int m = 0; m < kMTiles; ++m) {
                if constexpr (kMTiles > 1) {
                    if (m >= m_tiles) break;
                }
                #pragma unroll
                for (int q = 0; q < kBatch; ++q) {
                    const Fp8MmaFragment2 bp = b_pair[sp][q];
                    const Fp8MmaFragment b0 = {bp[0], bp[1]};
                    const Fp8MmaFragment b1 = {bp[2], bp[3]};
                    acc[q][m][0] = __builtin_mxc_mma_16x16x32f8(
                        a_pipe[sp][m], b0, acc[q][m][0]);
                    acc[q][m][1] = __builtin_mxc_mma_16x16x32f8(
                        a_pipe[sp][m], b1, acc[q][m][1]);
                }
            }
            const int nk = kk + kPrefetch;
            if (nk < k_iters) {
                #pragma unroll
                for (int m = 0; m < kMTiles; ++m) {
                    if constexpr (kMTiles > 1) {
                        if (m >= m_tiles) break;
                    }
                    a_pipe[sp][m] = lda(nk, m);
                }
                #pragma unroll
                for (int q = 0; q < kBatch; ++q)
                    b_pair[sp][q] = CM_LDB_L2(q, nk);
            }
        }
        // 输出 BF16（acc lane 布局同 L1；每组写自己的列块）
        #pragma unroll
        for (int q = 0; q < kBatch; ++q) {
            #pragma unroll
            for (int m = 0; m < kMTiles; ++m) {
                if constexpr (kMTiles > 1) {
                    if (m >= m_tiles) break;
                }
                #pragma unroll
                for (int t = 0; t < kNTiles; ++t) {
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int row = m * kM + (lane >> 4) * 4 + i;
                        const int col = (g0 + q) * block_n + warp * n_per_warp +
                                        t * 16 + (lane & 15);
                        acc_chk += acc[q][m][t][i];
                        if (out != nullptr && row < count) {
                            out[(size_t)row * k_out + col] =
                                (uint16_t)(cm_pack_f2bf16(acc[q][m][t][i],
                                                          0.0f) &
                                           0xFFFFu);
                        }
                    }
                }
            }
        }
    }
    if (checksum != nullptr && lane == 0) atomicAdd(checksum, acc_chk);
}

// [regspill] 中/小档 L2 GEMM body 隔离 shim：__noinline__ 独立寄存器/栈，避免其
//   acc（单批最多 96 floats）传染 r19_big_task。forceinline 的 l2_gemm_body 会内联
//   进本 shim，即该档独立帧。small task 的调用点仍走内联 l2_gemm_body。
//   -DR19_L2BODY_ISO=__forceinline__ 可回退成基线（shim 内联 = 无隔离）。
#ifndef R19_L2BODY_ISO
#define R19_L2BODY_ISO __noinline__
#endif
template <int kPrefetch = 4, int kNTiles = kNTilesPerWarp, int kMTiles = 1,
          int kBatch = 6, bool kUseFrag = false>
__device__ R19_L2BODY_ISO void l2_gemm_body_iso(
        const uint8_t* a_pool, const uint8_t* w2, int count, int k_in,
        int n_total, int block_n, int lane, int warp, int nwarps,
        float* checksum, int n_group_begin, int n_group_count, uint16_t* out,
        int k_out, const uint8_t* f2 = nullptr, int mt_total = 0) {
    l2_gemm_body<kPrefetch, kNTiles, kMTiles, kBatch, kUseFrag>(
        a_pool, w2, count, k_in, n_total, block_n, lane, warp, nwarps,
        checksum, n_group_begin, n_group_count, out, k_out, f2, mt_total);
}

// ------------------------------------------------------------- 任务信息 --

#include "combine.cuh"




// 大/小 count 任务体（独立 __noinline__ 函数；此位置已可见 L1/L2 body 定义）
#include "compute_big.cuh"

// ============================================================================
// DISPATCH：任务循环驱动（领号/解码 + ring 门 + 任务体分派）。
//   按 count 分派：count<=kM → r19_small_task；count>kM → r19_big_task（均
//   __noinline__，寄存器互不污染）。协议与基线一致。
//   [L1L2-SPLIT] kPhase：Both/L1/L2 由 if constexpr 编译期消除另一段。
// ============================================================================
template <MegaPhase kPhase = MegaPhase::Both>
__device__ __forceinline__ void dispatch_pull_body(
        const MegaArgs* a, int cycle, uint8_t* a_smem = nullptr,
        uint8_t* b_smem = nullptr, unsigned long long* ctr = nullptr) {
    // [S-SHRINK2] 去函数级循环不变量局部缓存：每使用点直接读 a->（s_a, smem）；
    //   仅保留编译期常量（rep/n_reps）；ring 用 kRing。降低 mega_kernel 跨 noinline
    //   任务体的寄存器保活压力。
    constexpr int rep = 0, n_reps = 1;
    extern __shared__ __align__(16) uint8_t smem_raw[];
    int* s_count = reinterpret_cast<int*>(smem_raw);   // EPR：每专家总行数
    int* s_cum = s_count + kEPR;                       // EPR+1：窗口前缀（任务解码）
    int* s_prefix = s_cum + kEPR + 1;                  // EPR*(R+1)：全 experts src 前缀
    __shared__ int s_task, s_expert, s_window, s_count_w, s_slice, s_kseg;
    __shared__ unsigned s_epoch;
    // prefix 表 = 本 rank 私有 rwk->ld.prefix_tbl（rank 内读写）
    const int* prefix = &a->rwk->ld.prefix_tbl[0][0];
    const int tid = threadIdx.x;
    const int warp = tid / 64;
    const int lane = tid % 64;
    (void)warp; (void)lane;
    // [O1] 全 experts src 前缀一次预载（s_prefix 扩容 EPR*(R+1)）
    for (int i = tid; i < kEPR * (a->R + 1); i += (int)blockDim.x)
        s_prefix[i] = prefix[i];
    // 每专家总行数 + 窗口前缀（前缀和以 win_cap 划分窗口）
    if (tid == 0) {
        int total = 0;
        s_cum[0] = 0;
        for (int e = 0; e < kEPR; ++e) {
            s_count[e] = prefix[e * (a->R + 1) + a->R];
            total += (s_count[e] + a->win_cap - 1) / a->win_cap;
            s_cum[e + 1] = total;
        }
    }
    __syncthreads();
#if R19_BLOCK_KSPLIT
    // [BALANCE·docs/20 §4.2b] 跨 block K 切：任务 = (e,w,slice,kseg)，共 x kNk。
    //   运行时再按 win_cap + fragment 可用门控（大窗/无 frag 走原整段 L1）：
    //   ksw 为真时 raw_tasks 含 kNK，否则退化回 kNSlice（解码按 ksw 分支）。
    const int mt_total_ks = (a->win_cap + 15) >> 4;
#if defined(CM_NO_FRAGALL)
    const bool frag_ok_ks = false;
#else
    const bool frag_ok_ks = (a->rwk->ld.frag_a[0] != nullptr) &&
                            (a->rwk->ld.frag2[0] != nullptr) &&
                            (kIntermediate > 0) && (mt_total_ks > 0) &&
                            (a->win_cap > 0);
#endif
    const bool ksw = (a->win_cap <= R19_BLOCK_KSPLIT_WINMAX) && (kNK > 1) &&
                     frag_ok_ks;
    const int kseg_n = ksw ? kNK : 1;
    const int raw_tasks = s_cum[kEPR] * kNSlice * kseg_n;
#else
    const int raw_tasks = s_cum[kEPR] * kNSlice;
#endif
    // merge 机制已去除：固定 merge=1（不粗化），nslice_eff=kNSlice。
    const int merge = 1;
    const int nslice_eff = kNSlice / merge;
    const int total_tasks = raw_tasks / merge;
#if R19_BLOCK_LPT
    // [BLOCK-LPT] 建排序表：s_order[raw_tasks]（任务 id 降序按代价）+ 临时 s_cost
    //   与 s_hist。紧凑放在 s_prefix 之后；smem 不足则 lpt_ok=false 回退基线。
    const int kLptHeadInts = kEPR + (kEPR + 1) + kEPR * (a->R + 1);
    constexpr int kLptMaxTasks = 640;   // 任务数上界（EPR*ceil(512/32)*kNSlice=512）
    constexpr int kLptSmemInts = 2048;  // prod_smem(~8448B)/4 的保守下界
    int* s_order = s_prefix + kEPR * (a->R + 1);
    int* s_cost = s_order + kLptMaxTasks;
    int* s_hist = s_cost + kLptMaxTasks;
    const bool lpt_ok = (raw_tasks > 0) && (raw_tasks <= kLptMaxTasks) &&
                        (a->win_cap >= 0) &&
                        (kLptHeadInts + 2 * kLptMaxTasks + (a->win_cap + 1) <=
                         kLptSmemInts);
    if constexpr (kPhase != MegaPhase::L2) {
        if (lpt_ok) {
            // 1) 每任务代价 = min(count_e - w*win_cap, win_cap)（∝ L1/L2 行数）
            for (int i = tid; i < raw_tasks; i += (int)blockDim.x) {
#if R19_BLOCK_KSPLIT
                const int tw = i / (nslice_eff * kseg_n);
#else
                const int tw = i / nslice_eff;
#endif
                int lo = 0, hi = kEPR - 1;
                while (lo < hi) {
                    const int mid = (lo + hi) >> 1;
                    if (s_cum[mid + 1] <= tw) lo = mid + 1;
                    else hi = mid;
                }
                const int wv = tw - s_cum[lo];
                const int remain = s_count[lo] - wv * a->win_cap;
                s_cost[i] = remain < a->win_cap ? remain : a->win_cap;
            }
            __syncthreads();
            // 2) 降序稳定计数排序（相等代价保持 task id 升序 → 保持 ring 窗口关系）
            if (tid == 0) {
                for (int c = 0; c <= a->win_cap; ++c) s_hist[c] = 0;
                for (int i = 0; i < raw_tasks; ++i) ++s_hist[s_cost[i]];
                int acc = 0;
                for (int c = a->win_cap; c >= 0; --c) {
                    const int h = s_hist[c];
                    s_hist[c] = acc;
                    acc += h;
                }
                for (int i = 0; i < raw_tasks; ++i)
                    s_order[s_hist[s_cost[i]]++] = i;
            }
            __syncthreads();
        }
    }
#else
    (void)0;
#endif
    // [集71] 任务循环进入 block 数
    // [L1L2-SPLIT] 循环统计只在 L2/Both 记（L1 阶段另跑一遍会重复计数）
    if (tid == 0 && kPhase != MegaPhase::L1)
        MX_CTR_ADD(ctr, kCtrLoopEnter, 1);
    // 任务体调度参数（循环不变量打包，按 const& 传给两个 noinline 任务体）
    // [S-SHRINK] S 只放派生量；纯 a-> 字段由任务体经 S.a（smem）直接读。
    R19Sched S;
    S.a = a;
    S.cycle = cycle;
    // [A-RWK] pool 基址 = 本 rank 私有 rwk->ld.expert_tokens（rank 内读写）
    S.pool_base = &a->rwk->ld.expert_tokens[0][0][0];
    S.bar_flags = a->rwk->lf.bar[cycle];
    S.dbg = a->rwk->lf.dbg;
    S.l1_chk = a->rwk->lf.l1_chk;
    S.l2_chk = a->rwk->lf.l2_chk;
    S.pt = a->rwk->lf.pre_ticks;
    S.frag_a = a->rwk->ld.frag_a[0];
    S.frag2 = a->rwk->ld.frag2[0];
    // [BALANCE·docs/20 §4.2] L1 split-K 部分和 / 段计数缓冲（kernel 内部派生）
    S.l1_partial = a->rwk->ld.l1_partial[0][0][0];
    S.l1_part_cnt = a->rwk->lf.l1_part_cnt[0];
    // [R34][FRAG-INT] fragment m 维 = ceil(win_cap/16)；槽 = local_data.frag_a/frag2。
    const int mt_total = (a->win_cap + 15) >> 4;
    S.mt_total = mt_total;
#if defined(CM_NO_FRAGALL)
    S.frag_all = false;
#else
    S.frag_all = (a->rwk->ld.frag_a[0] != nullptr) &&
                 (a->rwk->ld.frag2[0] != nullptr) &&
                 (kIntermediate > 0) && (mt_total > 0) && (a->win_cap > 0);
#endif
    S.epoch = (uint64_t)a->rwk->lf.current_time[0][1];
    S.merge = merge;
    S.nslice_eff = nslice_eff;
    // [集71] 计数缓冲下传任务体
    S.ctr = ctr;
    int processed = 0;
    while (true) {
        // ---- 领号 + 解码 + ring 门（仅 tid0）----
        if (tid == 0) {
            // [L1L2-SPLIT] L2 阶段用静态任务划分（block b → {b, b+grid, ...}）：
            //   不依赖 task_counter/快照，L2 kernel 无需 task_setup 重臂与计数；
            //   与 L1 的动态领号映射无关（L2 只读按 (e,w) 定位的数据槽，且
            //   wt/meta staging 由 L2 自行重算）。
            const int task = (kPhase == MegaPhase::L2)
                ? ((int)blockIdx.x + processed * (int)gridDim.x)
                : ((processed == 0)
                       ? (int)blockIdx.x
                        : atomicAdd(reinterpret_cast<int*>(
                                       a->rwk->lf.bar[cycle] +
                                       kBarTaskCounterIdx), 1));
#if R19_BLOCK_LPT
            // [BLOCK-LPT] 领到的号 counter 映射为降序排序后的任务 id（仅非 L2 路径）。
            const int task_lpt = (kPhase != MegaPhase::L2 && lpt_ok &&
                                  task < total_tasks)
                                     ? s_order[task]
                                     : task;
            s_task = task_lpt;
#else
            s_task = task;
#endif
            if (s_task < total_tasks) {
                // [R35] 物理任务 → 窗号/物理 slice（每任务覆盖 merge 个逻辑 slice）
                // [BALANCE·docs/20 §4.2b] K 切开启时任务再含 kseg（尾维）。
#if R19_BLOCK_KSPLIT
                const int base = s_task / kseg_n;
                const int kseg = s_task % kseg_n;
                const int tw = base / nslice_eff;
                const int slice = base % nslice_eff;
                s_kseg = kseg;
#else
                const int tw = s_task / nslice_eff;
                const int slice = s_task % nslice_eff;
#endif
                s_slice = slice;
                int lo = 0, hi = kEPR - 1;
                while (lo < hi) {
                    const int mid = (lo + hi) >> 1;
                    if (s_cum[mid + 1] <= tw) lo = mid + 1;
                    else hi = mid;
                }
                s_expert = lo;
                s_window = tw - s_cum[lo];
                const int remain = s_count[lo] - s_window * a->win_cap;
                s_count_w = remain < a->win_cap ? remain : a->win_cap;
                s_epoch = *(volatile const uint32_t*)(a->rwk->lf.bar[cycle] +
                                                      kBarTaskReadyIdx);
                // [L1L2-SPLIT] ring 门等前代 L2 完成；拆分下前代 L2 在下一个
                //   kernel（stream 顺序已保证），且数据槽不复用 → 编译期跳过。
                if constexpr (kPhase == MegaPhase::Both) {
                    if (s_window >= kRing || rep > 0) {
                        // ring 门：target=(rep + w/ring)*nslice_eff（每物理任务 +1），
                        // 槽复用前等前代 L2 完成。[R35] 用 nslice_eff 匹配物理任务数。
                        l2_gate_wait(a->rwk->lf.bar[cycle], (int)kBarL2DoneIdx,
                                     lo, s_window, rep, kRing, nslice_eff, slice,
                                     -1, 0u, a->rwk->lf.dbg, a->max_spins, ctr);
                    }
                }
            }
        }
        __syncthreads();
        const int task = s_task;
        if (task >= total_tasks) break;
        const int e = s_expert;
        const int w = s_window;
        const int count = s_count_w;
        const int slice = s_slice;
        const unsigned epoch = s_epoch;
        // [BALANCE·docs/20 §4.2b] K 段号（未启用时为 0）。
        int kseg = 0;
#if R19_BLOCK_KSPLIT
        if (ksw) kseg = s_kseg;
#endif
        // [L1L2-SPLIT] 旗槽（ring，容量 EPR*ring）与数据槽（拆分下每窗口独占）：
        //   slot_win = ceil(kMaxTPR/win_cap) 覆盖 count<=kMaxTPR 的全部窗口；
        //   Both 下两者相同（e*ring + w%ring），行为逐字不变。
        const int pool_slot = e * kRing + w % kRing;
        const int slot_win = (kMaxTPR + a->win_cap - 1) / a->win_cap;
        const int data_slot =
            (kPhase == MegaPhase::Both) ? pool_slot : (e * slot_win + w);
        const int* pfx = s_prefix + e * (a->R + 1);
        // ---- 任务体分派（count<=kM 小路径 / count>kM M 档位大路径）----
        if (count > kM) {
            r19_big_task<kPhase>(S, task, e, w, count, slice, kseg,
                                                 epoch, data_slot, pfx,
                                                 a_smem, b_smem);
        } else {
            r19_small_task<kPhase>(S, task, e, w, count, slice, kseg,
                                             epoch, data_slot, pfx,
                                             a_smem, b_smem);
        }
        // ---- 任务尾 l2_done 递增（仅后续窗口会复用该槽或多轮时）----
        // [H3] 不发布 l2_done 时省略全线程 threadfence（无消费者，协议语义不变）
        // [L1L2-SPLIT] 拆分下无环复用（数据槽每窗口独占）→ 跳过 ring 门与发布。
        bool h_pub = false;
        if constexpr (kPhase == MegaPhase::Both) {
            h_pub = (s_count[e] > (w + kRing) * a->win_cap) || n_reps > 1;
        }
        // [BALANCE·docs/20 §4.2b] K 切下仅 owner（kseg==0）发布 l2_done：每窗口
        //   仍恰好 nslice_eff 次，ring 门 target 语义不变；非 owner 不发布。
        if (kseg != 0) h_pub = false;
        if (h_pub) __threadfence();
        if (tid == 0) {
            if (h_pub)
                atomicAdd(reinterpret_cast<uint32_t*>(
                              a->rwk->lf.bar[cycle] + kBarL2DoneIdx +
                              (size_t)(e * kRing + w % kRing)),
                          1u);
            ++processed;
            a->rwk->lf.out_processed[blockIdx.x] = processed;
            // [集71] 任务数 / slice 数 / l2_done 发布次数
            // [L1L2-SPLIT] L1/L2 各自跑一遍任务循环 → 统计只在 L2/Both 记一次
            if constexpr (kPhase != MegaPhase::L1) {
                MX_CTR_ADD(ctr, kCtrTasks, 1);
                MX_CTR_ADD(ctr, kCtrSlices, nslice_eff);
                MX_CTR_ADD(ctr, kCtrL2DonePublish, h_pub ? 1 : 0);
            }
        }
        // 单波（任务数不超过 grid）做完首波即退出
        if (total_tasks <= (int)gridDim.x) break;
        __syncthreads();
    }
    // [集71] 任务循环结束 block 数 + 单 block 任务数 max/min
    if (tid == 0) {
        if constexpr (kPhase != MegaPhase::L1) {
            MX_CTR_ADD(ctr, kCtrLoopExit, 1);
            MX_CTR_MAX(ctr, kCtrTasksPerBlockMax, (unsigned long long)processed);
            MX_CTR_MIN(ctr, kCtrTasksPerBlockMin, (unsigned long long)processed);
        }
    }
    // 诊断：block0 任务循环结束（L2 + COMBINE + 尾）
    if (blockIdx.x == 0 && tid == 0)
        a->rwk->lf.pre_ticks[7] = clock64();
}
