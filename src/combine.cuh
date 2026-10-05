// ============================================================================
// combine.cuh — 收尾模块：COMBINE（__stwt 跨 rank 推送）+ REDUCE（topk 加权归约）
//
// 项目：zzzz/projs_deepseek/projs/01.fp8xfp8_moe_kernel（fp8xfp8 MoE mega kernel）
// 骨架参考（只读，不 include）：
//   .../00.fp8_mega_kernel_refactor/test/00.empty_kernel/empty_kernel_bench.cu
//   - CombineCtx / combine_push_pair : bench:1002-1046
//   - 独立 COMBINE（逐 slice 列范围）: bench:1874-1920
//   - reduce_kernel                  : bench:1283-1365
//
// 协议（docs/01 §4.7/§4.8）：
//   * COMBINE（全 grid、全块线程）：按 staging meta (src_rank, src_token, k)
//     把本 block 的 ffn_out 行（BF16）用 __stwt 推送到源 rank 的 combine_rows；
//     每 slice 只写自己的列范围，每 lane 8 列（16B uint4）直拷；每行推完后由
//     lane0 写完成标记 combine_sums[(k*slot_cap+src_token)*stride + slice] = epoch。
//   * REDUCE（独立 kernel / 全 grid；每 token 2 个 block 列分半）：tid0 轮询全部
//     有效 k 的 NSLICE 个完成标记（>= epoch）→ __threadfence → __syncthreads →
//     全块无条件并行 uint4 读（每线程 8 列 × topk 行；无效 k 指向安全行、由权重 0
//     屏蔽但照读，避免分支化读被编译成串行）→ BF16 解码 × topk 权重累加 → y(FP32)。
//
// 导出签名与语义（docs/04 §2.3 登记）：
//   1) struct CombineCtx
//        COMBINE 上下文：本 block staging 的 meta/weights 视图 + peer 对称区
//        基址表 + combine_rows/combine_sums 偏移 + epoch/slice/block_idx/
//        staging_cap/combine_cap。字段含义见结构体注释。
//   2) __device__ void combine_push_pair(const CombineCtx& c,
//          const uint8_t* out_base, int count, int lane, int warp,
//          int nwarps, int sums_stride)
//        本 block 的 COMBINE：warp 分摊 count 个窗口行；staging 权重为 0 的
//        行跳过；每有效行把本 slice 的列范围（BF16 uint4）__stwt 到
//        peer_bases[src] 的 combine_rows 行 (k*combine_cap+src_token)；lane0
//        随后写完成标记。out_base = ffn_out staging 全区基址（行内索引
//        block_idx*staging_cap + s2）。sums_stride = combine_sums 行槽数。
//   [R29] 融合推送（可选，默认开）：combine_push_chunk/CombineFuseCtx 允许 L2
//     在 r24 路径上每算完一个 (c0,g) 列块就推送该列块，任务尾只写完成标记
//     （combine_mark_rows）。标记语义/可见性顺序与 combine_push_pair 完全相同
//     （数据推送 → __threadfence → __syncthreads → lane0 标记）。
//   3) __device__ void reduce_body(const uint8_t* self, float* y, int R,
//          int tokens, int combine_cap, long long combine_rows_off,
//          long long combine_sums_off, long long topk_idx_off,
//          long long weights_off, const uint32_t* bar_flags,
//          long long task_ready_idx, int* dbg, int max_spins, int sums_stride)
//        REDUCE：block 映射 token = blockIdx.x/2、half = blockIdx.x%2（列分半）；
//        行完成轮询 → fence → 全块并行读 → topk 加权求和 → y（[tokens][kHidden]
//        FP32）。epoch 取 bar_flags[task_ready_idx]（volatile）。
//
// 与骨架差异（适配点，其余照抄）：
//   1) 行数据为 BF16（生产布局）：COMBINE 直接 16B 拷贝，不再 FP32→BF16 打包
//      （pack_f8bf16 仅存在于骨架 FusePush 变体）；REDUCE 按 uint16 解码、
//      行步长按 bf16 元素计。
//   2) combine_push_pair 的列范围由 slice/NSLICE 推导（只写本 slice），与骨架
//      独立 COMBINE（bench:1884-1910）一致，不再走 col_begin/col_count/last_pair。
//   3) combine_sums 行槽数 sums_stride 显式传参（骨架/生产 =
//      kMegaMoeMaxCombineSlices = 32）：本文件不 include 布局头、不引入新常量。
//   4) 容量显式两分（docs/03 §6 防"容量混用"）：staging 行容量 staging_cap
//      （= 窗口容量 win_capacity）与 combine_rows 行容量 combine_cap
//      （= max_tokens，生产布局 [topk][max_tokens][H]）分别传参；骨架因默认
//      场景两者相等（16）而共用 slot_capacity。
//   5) REDUCE 元数据外提位置改为 token 守卫内（避免 token>=tokens 的越界读），
//      外提与无条件并行读的优化语义不变。
// ============================================================================
#pragma once

#include "common.cuh"
#include "flags_defs.cuh"

// [R30] 推送流水实验开关（默认关：与 R29 交付逐字等价）
//   R30_PUSH_RUN  ：融合推送行分配改为 warp 连续段（减小跨目标 rank 的行步跳）
//   R30_PUSH_DEFER：融合推送延迟一个列块，在下一列块 K stage 循环内分片发出
#ifndef R30_PUSH_RUN
#define R30_PUSH_RUN 0
#endif
#ifndef R30_PUSH_DEFER
#define R30_PUSH_DEFER 0
#endif
// 诊断（仅归因，交付默认关）：
//   R30_DIAG_NOSTORE：保留 staging 读+地址计算，去掉 __stwt
//   R30_DIAG_NOREAD ：保留 __stwt（写常量），去掉 staging 读
#ifndef R30_DIAG_NOSTORE
#define R30_DIAG_NOSTORE 0
#endif
//   R30_DIAG_NOPUSH ：整段跳过推送（读/地址/存储全免；标记保留）= 同会话 nodata
//   R30_DIAG_NOWEIGHT：跳过批内 weights 读（融合列块行恒有效；多推 0 权行无害）
#ifndef R30_DIAG_NOPUSH
#define R30_DIAG_NOPUSH 0
#endif
#ifndef R30_DIAG_NOWEIGHT
#define R30_DIAG_NOWEIGHT 0
#endif
//   R30_FASTPUSH：多行一批并发载荷 + 统一 store（打破逐行 load→store 串行链）
//   R30_ST_PLAIN：批内 store 用普通写（默认 __stwt）
#ifndef R30_FASTPUSH
#define R30_FASTPUSH 1
#endif
#ifndef R30_ST_PLAIN
#define R30_ST_PLAIN 1
#endif
//   R30_PUSHSTRIDE：批内 4 行按 nwarps 步进（目标 rank 不同，摊开各 peer 队列）
#ifndef R30_PUSHSTRIDE
#define R30_PUSHSTRIDE 0
#endif
// [R33] 每 warp 只推自己写的列段（列段宽 block_n/nwarps），免 __syncthreads：
//   列块 epilogue 由 warp w 写 [w*n_per_warp, +n_per_warp)，推送同列段则
//   无需等其它 warp（__syncwarp 即可），末列块的推送与自身计算重叠、且
//   各 warp 推送时刻错开（fabric 队列更平滑）。协议序不变（尾 fence→sync→标记）。
#ifndef R33_WARPPUSH
#define R33_WARPPUSH 1
#endif
#ifndef R30_DIAG_NOREAD
#define R30_DIAG_NOREAD 0
#endif

// ---------------------------------------------------------------------------
// COMBINE 上下文（对应骨架 CombineCtx bench:1002-1013）
//   meta/weights 指向本 block 的 staging 区（DISPATCH lane0 写入）：
//     weights: [max(EPR,grid)][staging_cap] float，无效行 = 0
//     meta   : [max(EPR,grid)][staging_cap][3] int = (src_rank, src_token, k)
//   行索引均为 block_idx * staging_cap + row。
//   peer_bases[src] 为源 rank 对称区基址（含自映射），行数据 __stwt 推送。
//   容量两分（默认场景相等）：
//     staging_cap = 窗口容量 win_capacity（staging/ffn_out 行容量）
//     combine_cap = token 槽容量 max_tokens（combine_rows/sums 行容量）
// ---------------------------------------------------------------------------
struct CombineCtx {
    const int* meta = nullptr;          // staging 元数据视图（见上）
    const float* weights = nullptr;     // staging 权重视图（无效行 = 0）
    uint8_t* const* peer_bases = nullptr;  // 各 rank 对称区基址（含 self）
    long long rows_off = -1;            // 目标 rank combine_rows 区偏移（BF16）
    long long sums_off = -1;            // 目标 rank combine_sums 区偏移（uint32）
    uint32_t epoch = 0;                 // 本窗 epoch（= bar_flags[task_ready]）
    int slice = 0;                      // 本 block 任务切片号 [0, kNSlice)
    int block_idx = 0;                  // staging 块号（= blockIdx.x）
    int staging_cap = 0;                // staging 行容量（= win_capacity）
    int combine_cap = 0;                // combine_rows 行容量（= max_tokens）
    // [R35] 波量化粗化：本物理任务覆盖 merge 个逻辑 slice，几何用 nslice_eff；
    //   标记槽按逻辑 slice（slice*merge + i）写满 kNSlice。
    int nslice_eff = kNSlice;
    int merge = 1;
};

// BF16 → FP32（无符号左移 16 位即得 FP32 位模式；生产同构）
__device__ __forceinline__ float combine_bf16_to_float(uint16_t h) {
    return __uint_as_float((uint32_t)h << 16);
}

// ---------------------------------------------------------------------------
// [R29] COMBINE 与 L2 重叠（融合推送）
//   L2 每算完一个 (c0 行块, g 列批) 就把该块 staging 数据用 __stwt 推到目标
//   rank（fabric 排空与后续 L2 计算/其它 warp 的 push 重叠）；任务尾只写完成
//   标记。协议语义不变：标记仍在该 slice 全部列推完后写、值 = epoch。
//   capable：该 tail 类型支持融合（kernel.cuh 对 CombineTail 置位）；
//   on     ：本任务实际融合（任务体仅在 r24 L2 路径 + staging 有效时置位）。
// ---------------------------------------------------------------------------
struct CombineFuseCtx {
    bool capable = false;            // tail 具备融合能力（kernel 侧置位）
    bool on = false;                 // 本任务实际激活融合（任务体置位）
    const uint8_t* out_base = nullptr;  // ffn_out staging 全区基址（BF16）
    const int* meta = nullptr;       // staging 元数据视图
    const float* weights = nullptr;  // staging 权重视图（无效行 = 0）
    uint8_t* const* peer_bases = nullptr;  // 各 rank 对称区基址（含 self）
    long long rows_off = -1;         // 目标 rank combine_rows 区偏移（BF16）
    int combine_cap = 0;             // combine_rows 行容量（= max_tokens）
    int staging_cap = 0;             // staging 行容量（= win_capacity）
    int block_idx = 0;               // staging 块号（= blockIdx.x）
    int sums_stride = 0;             // combine_sums 每行槽数（= 32）
};

// 融合推送：把 f.out_base（staging 全区基址）的行 [row_lo,row_hi)、列
//   [col_lo,col_hi)（全局 H 维坐标；列范围 = 某 g 批次的列块）推送到目标
//   rank 的 combine_rows 对应行。不写完成标记（由任务尾 combine_mark_rows
//   统一写）。行索引 = block_idx*staging_cap + s2（与任务尾一致）。
// [R30] 单行推送体（含 weights 无效跳过；融合列块范围内的行恒有效，但保留
//   兼容语义：weights==0 即 dispatch 零填充行，跳过）
__device__ __forceinline__ void combine_push_one_row(
        const CombineFuseCtx& f, const uint16_t* out_h, int nu4, int s2,
        int col_lo, int lane) {
    const size_t stag = (size_t)f.block_idx * f.staging_cap + s2;
    if (f.weights[stag] == 0.0f) return;   // 无效行（dispatch 零填充）
    const int* m = f.meta + stag * 3;
    const int src = m[0];
    const int stok = m[1];
    const int kk = m[2];
    const uint16_t* srow = out_h + stag * kHidden + col_lo;
    uint16_t* drow_h = reinterpret_cast<uint16_t*>(
                           const_cast<uint8_t*>(f.peer_bases[src]) +
                           f.rows_off) +
                       ((size_t)kk * f.combine_cap + stok) * kHidden + col_lo;
    for (int i = lane; i < nu4; i += 64) {
#if R30_DIAG_NOREAD
        const uint4 v = make_uint4((unsigned)col_lo, (unsigned)s2,
                                   (unsigned)i, 0u);
#else
        const uint4 v = *reinterpret_cast<const uint4*>(srow + i * 8);
#endif
#if R30_DIAG_NOSTORE
        // 诊断：保留读与地址计算，不落远端（用不可判条件防 DCE）
        if (f.sums_stride == -12345)
            __stwt(reinterpret_cast<uint4*>(drow_h + i * 8), v);
#else
        __stwt(reinterpret_cast<uint4*>(drow_h + i * 8), v);
#endif
    }
}

__device__ __forceinline__ void combine_push_chunk_rows(
        const CombineFuseCtx& f, int row_lo,
        int row_hi, int col_lo, int col_hi, int lane, int warp, int nwarps) {
    const uint16_t* out_h = reinterpret_cast<const uint16_t*>(f.out_base);
    const int nu4 = (col_hi - col_lo) / 8;   // 本列块每行 uint4 数
    const int nr = row_hi - row_lo;
    if (nr <= 0) return;
#if R30_FASTPUSH
    // [R30] 4 行一批：并行发出 4 行的 weights/meta 与 staging 载荷，再统一
    //   store，打破热路径「逐行 load→(weights 分支)→地址→store」的串行
    //   延迟链（vR 诊断：仅保留读+地址就要 ~240µs；vW：仅 store 也要 ~300µs）。
#if R30_PUSHSTRIDE
    // 批内行按 nwarps 步进：4 行的目标 rank 不同（摊开各 peer 队列），
    // 残余行由单行路径补推。
    const int nfull = nr / (4 * nwarps);
    for (int mb = 0; mb < nfull; ++mb) {
        const int s2 = row_lo + warp + mb * 4 * nwarps;
#else
    const int wrun = (nr + nwarps - 1) / nwarps;
    int s2 = row_lo + warp * wrun;
    const int r1 = (s2 + wrun < row_hi) ? (s2 + wrun) : row_hi;
    for (; s2 + 4 <= r1; s2 += 4) {
#endif
        const size_t stag = (size_t)f.block_idx * f.staging_cap + s2;
#if R30_PUSHSTRIDE
        const size_t st1 = stag + nwarps;
        const size_t st2 = stag + 2 * nwarps;
        const size_t st3 = stag + 3 * nwarps;
        const int* m0 = f.meta + stag * 3;
        const int* m1 = f.meta + st1 * 3;
        const int* m2 = f.meta + st2 * 3;
        const int* m3 = f.meta + st3 * 3;
#else
        const size_t st1 = stag + 1, st2 = stag + 2, st3 = stag + 3;
        const int* m0 = f.meta + stag * 3;
        const int* m1 = m0 + 3;
        const int* m2 = m0 + 6;
        const int* m3 = m0 + 9;
#endif
#if R30_DIAG_NOWEIGHT
        const float w0 = 1.0f, w1 = 1.0f, w2 = 1.0f, w3 = 1.0f;
#else
        const float w0 = f.weights[stag];
        const float w1 = f.weights[st1];
        const float w2 = f.weights[st2];
        const float w3 = f.weights[st3];
#endif
        const int src0 = m0[0], stok0 = m0[1], kk0 = m0[2];
        const int src1 = m1[0], stok1 = m1[1], kk1 = m1[2];
        const int src2 = m2[0], stok2 = m2[1], kk2 = m2[2];
        const int src3 = m3[0], stok3 = m3[1], kk3 = m3[2];
        uint16_t* d0 = reinterpret_cast<uint16_t*>(
                           const_cast<uint8_t*>(f.peer_bases[src0]) +
                           f.rows_off) +
                       ((size_t)kk0 * f.combine_cap + stok0) * kHidden + col_lo;
        uint16_t* d1 = reinterpret_cast<uint16_t*>(
                           const_cast<uint8_t*>(f.peer_bases[src1]) +
                           f.rows_off) +
                       ((size_t)kk1 * f.combine_cap + stok1) * kHidden + col_lo;
        uint16_t* d2 = reinterpret_cast<uint16_t*>(
                           const_cast<uint8_t*>(f.peer_bases[src2]) +
                           f.rows_off) +
                       ((size_t)kk2 * f.combine_cap + stok2) * kHidden + col_lo;
        uint16_t* d3 = reinterpret_cast<uint16_t*>(
                           const_cast<uint8_t*>(f.peer_bases[src3]) +
                           f.rows_off) +
                       ((size_t)kk3 * f.combine_cap + stok3) * kHidden + col_lo;
        const uint16_t* s0 = out_h + stag * kHidden + col_lo;
#if R30_PUSHSTRIDE
        const uint16_t* s1 = out_h + st1 * kHidden + col_lo;
        const uint16_t* s2p = out_h + st2 * kHidden + col_lo;
        const uint16_t* s3 = out_h + st3 * kHidden + col_lo;
#else
        const uint16_t* s1 = s0 + kHidden;
        const uint16_t* s2p = s0 + 2 * kHidden;
        const uint16_t* s3 = s0 + 3 * kHidden;
#endif
        for (int i = lane; i < nu4; i += 64) {
#if R30_DIAG_NOREAD
            const uint4 v0 = make_uint4(0u, (unsigned)s2, (unsigned)i, 0u);
            const uint4 v1 = v0, v2 = v0, v3 = v0;
#else
            const uint4 v0 = *reinterpret_cast<const uint4*>(s0 + i * 8);
            const uint4 v1 = *reinterpret_cast<const uint4*>(s1 + i * 8);
            const uint4 v2 = *reinterpret_cast<const uint4*>(s2p + i * 8);
            const uint4 v3 = *reinterpret_cast<const uint4*>(s3 + i * 8);
#endif
#if R30_DIAG_NOSTORE
            if (f.sums_stride == -12345) {
                __stwt(reinterpret_cast<uint4*>(d0 + i * 8), v0);
                __stwt(reinterpret_cast<uint4*>(d1 + i * 8), v1);
                __stwt(reinterpret_cast<uint4*>(d2 + i * 8), v2);
                __stwt(reinterpret_cast<uint4*>(d3 + i * 8), v3);
            }
#elif R30_ST_PLAIN
            if (w0 != 0.0f) *reinterpret_cast<uint4*>(d0 + i * 8) = v0;
            if (w1 != 0.0f) *reinterpret_cast<uint4*>(d1 + i * 8) = v1;
            if (w2 != 0.0f) *reinterpret_cast<uint4*>(d2 + i * 8) = v2;
            if (w3 != 0.0f) *reinterpret_cast<uint4*>(d3 + i * 8) = v3;
#else
            if (w0 != 0.0f) __stwt(reinterpret_cast<uint4*>(d0 + i * 8), v0);
            if (w1 != 0.0f) __stwt(reinterpret_cast<uint4*>(d1 + i * 8), v1);
            if (w2 != 0.0f) __stwt(reinterpret_cast<uint4*>(d2 + i * 8), v2);
            if (w3 != 0.0f) __stwt(reinterpret_cast<uint4*>(d3 + i * 8), v3);
#endif
        }
    }
#if R30_PUSHSTRIDE
    for (int s2 = row_lo + warp + nfull * 4 * nwarps; s2 < row_hi;
         s2 += nwarps)
        combine_push_one_row(f, out_h, nu4, s2, col_lo, lane);
#else
    for (; s2 < r1; ++s2)
        combine_push_one_row(f, out_h, nu4, s2, col_lo, lane);
#endif
#elif R30_PUSH_RUN
    // [R30] warp 连续行段：每个 warp 连续处理 wrun 行（同一目标 rank 段更长），
    // 避免逐行跨 rank 步跳（8 行一组 = 同源 rank）打散 fabric 队列/TLB。
    const int wrun = (nr + nwarps - 1) / nwarps;
    const int r0 = row_lo + warp * wrun;
    const int r1 = (r0 + wrun < row_hi) ? (r0 + wrun) : row_hi;
    for (int s2 = r0; s2 < r1; ++s2)
        combine_push_one_row(f, out_h, nu4, s2, col_lo, lane);
#else
    for (int s2 = row_lo + warp; s2 < row_hi; s2 += nwarps)
        combine_push_one_row(f, out_h, nu4, s2, col_lo, lane);
#endif
}

// [R33] 每 warp 推送自己写的列段（同一列块内）：
//   行区间 [row_lo,row_hi)，列 = 每 q 组 [g0*block_n + warp*n_per_warp, +n_per_warp)。
//   调用前该 warp 已完成本列块 epilogue（__syncwarp 保证同 warp lane 间可见）。
//   行/列分配：lane -> (行组内偏移 lane/nu4, uint4 下标 lane%nu4)，64 lane 覆盖
//   (64/nu4) 行 × nu4 个 uint4；数据与目标地址结构与 combine_push_chunk_rows 一致。
__device__ __forceinline__ void combine_push_warp_cols(
        const CombineFuseCtx& f, int row_lo, int row_hi, int g0, int nb,
        int block_n, int n_per_warp, int lane, int warp) {
    const int nu4 = n_per_warp / 8;          // 本 warp 列段每行 uint4 数
    if (nu4 <= 0) return;
    const int rstep = 64 / nu4;              // 每次覆盖的行数
    if (rstep <= 0) return;
    const int dr = lane / nu4;               // 行组内偏移
    const int j = lane % nu4;                // 列段内 uint4 下标
    const uint16_t* out_h = reinterpret_cast<const uint16_t*>(f.out_base);
    for (int r0 = row_lo; r0 < row_hi; r0 += rstep) {
        const int r = r0 + dr;
        if (r >= row_hi) continue;
        const size_t stag = (size_t)f.block_idx * f.staging_cap + r;
        if (f.weights[stag] == 0.0f) continue;
        const int* m = f.meta + stag * 3;
        const uint16_t* srow = out_h + stag * kHidden;
        uint16_t* drow = reinterpret_cast<uint16_t*>(
                             const_cast<uint8_t*>(f.peer_bases[m[0]]) +
                             f.rows_off) +
                         ((size_t)m[2] * f.combine_cap + m[1]) * kHidden;
        for (int q = 0; q < nb; ++q) {
            const int col = (g0 + q) * block_n + warp * n_per_warp + j * 8;
            const uint4 v = *reinterpret_cast<const uint4*>(srow + col);
            *reinterpret_cast<uint4*>(drow + col) = v;
        }
    }
}

// 融合推送（整列块，与 R29 逐字一致）
__device__ __forceinline__ void combine_push_chunk(
        const CombineFuseCtx& f, int row_lo,
        int row_hi, int col_lo, int col_hi, int lane, int warp, int nwarps) {
    combine_push_chunk_rows(f, row_lo, row_hi, col_lo, col_hi, lane, warp,
                            nwarps);
}

// 融合模式任务尾：数据已在 L2 各列块推完，这里只写完成标记。
//   发布序：__threadfence（各 lane 的数据推送）→ __syncthreads（全块汇合，
//   保证所有 warp 已推送）→ lane0 写标记。REDUCE 侧 epoch 协议不变。
__device__ __forceinline__ void combine_mark_rows(
        const CombineCtx& c, int count, int lane, int warp, int nwarps,
        int sums_stride) {
    __threadfence();
    __syncthreads();
    for (int s2 = warp; s2 < count; s2 += nwarps) {
        const size_t stag = (size_t)c.block_idx * c.staging_cap + s2;
        if (c.weights[stag] == 0.0f) continue;
        const int* m = c.meta + stag * 3;
        const int src = m[0];
        const int stok = m[1];
        const int kk = m[2];
        if (lane == 0)
            __stwt(reinterpret_cast<uint32_t*>(
                       const_cast<uint8_t*>(c.peer_bases[src]) + c.sums_off) +
                       ((size_t)kk * c.combine_cap + stok) * sums_stride +
                       c.slice,
                   c.epoch);
    }
}

// ---------------------------------------------------------------------------
// COMBINE：把本 block 的 ffn_out 有效行按 meta 推送到源 rank 的 combine_rows。
//   执行者约束：COMBINE 相位全 grid 全块线程（warp 分摊行、lane 分摊列）。
//   out_base  ：本 rank ffn_out staging 全区基址（BF16 行主序，行内索引
//               block_idx*staging_cap+s2；调用方传全区基址，不预先加块偏移）。
//   count     ：本任务窗口行数（无效行 staging 权重 = 0，跳过）。
//   lane/warp/nwarps：MACA warp=64；行按 warp 轮转、列按 lane 步进 64。
//   sums_stride：combine_sums 每行槽数（host 分配与本参数必须一致；
//               骨架/生产 = kMegaMoeMaxCombineSlices = 32）。
// ---------------------------------------------------------------------------
__device__ __forceinline__ void combine_push_pair(
        const CombineCtx& c, const uint8_t* out_base, int count, int lane,
        int warp, int nwarps, int sums_stride) {
    // 本物理 slice 的列范围（BF16 元素；c0s 起、cols_per_slice 列）
    // [R35] 粗化后每个物理任务覆盖合并后的列段（kHidden/nslice_eff）
    const int cols_per_slice = kHidden / c.nslice_eff;
    const int c0s = c.slice * cols_per_slice;
    const int n8s = cols_per_slice / 8;   // 本 slice 的 uint4 数（每 lane 8 列）
    const uint16_t* out_h = reinterpret_cast<const uint16_t*>(out_base);
    // [OPT-PUSH] warp 连续行段（原按 warp 交错）：同行段局部性更好
    const int wchunk = (count + nwarps - 1) / nwarps;
    const int row_lo = warp * wchunk;
    const int row_hi = (row_lo + wchunk < count) ? (row_lo + wchunk) : count;
    for (int s2 = row_lo; s2 < row_hi; ++s2) {
        const size_t stag = (size_t)c.block_idx * c.staging_cap + s2;
        if (c.weights[stag] == 0.0f) continue;   // 无效行（dispatch 零填充）
        const int* m = c.meta + stag * 3;
        const int src = m[0];    // 目标（源）rank
        const int stok = m[1];   // 源 token
        const int kk = m[2];     // topk 槽
        const uint16_t* srow = out_h + stag * kHidden;
        uint16_t* drow_h = reinterpret_cast<uint16_t*>(
                               const_cast<uint8_t*>(c.peer_bases[src]) +
                               c.rows_off) +
                           ((size_t)kk * c.combine_cap + stok) * kHidden;
        for (int i = lane; i < n8s; i += 64) {   // 每 lane 3 个 uint4（1536 列）
            const int col = c0s + i * 8;
            const uint4 v = *reinterpret_cast<const uint4*>(srow + col);
            __stwt(reinterpret_cast<uint4*>(drow_h + col), v);
        }
        // slice 完成标记：槽 = 物理 slice，值 = epoch（REDUCE 以 >= epoch 放行）
        // [R37] 粒度=物理 slice（每任务 1 个标记），REDUCE 按 nslice_eff 等待
        if (lane == 0)
            __stwt(reinterpret_cast<uint32_t*>(
                       const_cast<uint8_t*>(c.peer_bases[src]) + c.sums_off) +
                       ((size_t)kk * c.combine_cap + stok) * sums_stride +
                       c.slice,
                   c.epoch);
    }
}

// ---------------------------------------------------------------------------
// REDUCE：每 token 2 个 block（列分半）协作完成 topk 加权求和 → y（FP32）。
//   self    ：本 rank 对称区基址（combine_rows/sums/topk_idx/weights 的视图）。
//   y       ：输出 [tokens][kHidden] FP32。
//   R       ：rank 数（有效专家判据 e < R*kEPR）。
//   combine_cap：combine_rows/sums 行容量（= max_tokens；行索引
//               (k*combine_cap+token)）。
//   其余偏移与 bar_flags 语义同骨架 reduce_kernel（bench:1283-1365）。
//   等待放行判据：sums 标记 >= epoch（单调）；无效 k 不等待、不累加。
// ---------------------------------------------------------------------------
__device__ __forceinline__ void reduce_body(
        const uint8_t* self, float* y, int R, int tokens, int combine_cap,
        long long combine_rows_off, long long combine_sums_off,
        long long topk_idx_off, long long weights_off,
        const local_flags* lf, long long task_ready_idx, int* dbg,
        int max_spins, int sums_stride, unsigned long long* ctr = nullptr) {
    const uint32_t* bar_flags = lf->bar;
    // [集71] REDUCE 启动次数（block0 tid0）+ 参与 block 数（tid0）
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) MX_CTR_ADD(ctr, kCtrReduceLaunch, 1);
        MX_CTR_ADD(ctr, kCtrReduceBlock, 1);
    }
    const uint32_t epoch =
        *reinterpret_cast<volatile const uint32_t*>(bar_flags + task_ready_idx);
    const int64_t* topk_idx =
        reinterpret_cast<const int64_t*>(self + topk_idx_off);
    const float* weights = reinterpret_cast<const float*>(self + weights_off);
    const uint32_t* sums =
        reinterpret_cast<const uint32_t*>(self + combine_sums_off);
    const uint16_t* rows_h =
        reinterpret_cast<const uint16_t*>(self + combine_rows_off);
    const int tid = (int)threadIdx.x;
    // 每 token 2 个 block（列分半），grid=kGrid=32 覆盖 TPR=16 的 32 个半区
    const int token = (int)blockIdx.x / 2;
    const int half = (int)blockIdx.x % 2;
    const int cols_h = kHidden / 2;
    const int c_base = half * cols_h;
    if (token < tokens) {
        // 元数据外提进寄存器：循环内重复读 FG 会把 ~1.5µs 读延迟串行累加
        int64_t e8[kTopk];
        float w8[kTopk];
        bool ok8[kTopk];
        #pragma unroll
        for (int k = 0; k < kTopk; ++k) {
            e8[k] = topk_idx[(size_t)token * kTopk + k];
            ok8[k] = (e8[k] >= 0 && e8[k] < (int64_t)(R * kEPR));
            w8[k] = ok8[k] ? weights[(size_t)token * kTopk + k] : 0.0f;
        }
        // 行完成轮询（[R33] 并行化：kTopk*kNSlice 个线程各等一个 (k,slice)
        // 标记）。旧写法 tid0 串行等 32 个 volatile 读，每读一次 L2 往返
        // （~10µs+/block，16 block/SM 时明显）；并行等待后每线程只做一次
        // 观察，观察后各自 __threadfence 再 barrier 的发布序语义不变。
        // [R37] 等待槽数按运行期 nslice_eff（dispatch 广播于 kBarNsliceEffIdx）
        int ns_eff = (int)*reinterpret_cast<volatile const uint32_t*>(
            bar_flags + kBarNsliceEffIdx);
        if (ns_eff < 1) ns_eff = 1;
        if (ns_eff > kNSlice) ns_eff = kNSlice;
        if (tid < kTopk * ns_eff) {
            const int k = tid / ns_eff;
            const int s2 = tid % ns_eff;
            if (ok8[k]) {
                const volatile uint32_t* part =
                    sums + ((size_t)k * combine_cap + token) * sums_stride;
                long long spin = 0;
                // [集71] REDUCE 等待轮次（每 (k,slice) 标记一次）
                MX_CTR_ADD(ctr, kCtrReduceWait, 1);
                while (part[s2] < epoch) {
                    if (++spin > max_spins) {
                        if (dbg) atomicAdd(dbg + 4, 1);
                        break;
                    }
                }
                // [集71] REDUCE 自旋轮数 / 超时次数
                MX_CTR_ADD(ctr, kCtrReduceSpin, spin);
                if (spin > max_spins) MX_CTR_ADD(ctr, kCtrReduceTimeout, 1);
                __threadfence();
            }
        }
        __syncthreads();
        // BF16 读（字节减半）+ 每线程 8 列 × topk 行独立 uint4 读（fence 后
        // 普通读，可并行/预取）；无效 k 用权重 0 屏蔽但照读（指向 k=0 安全行）
        const int n8h = cols_h / 8;
        for (int i = c_base / 8 + tid; i < c_base / 8 + n8h;
             i += (int)blockDim.x) {
            const int c = i * 8;
            const uint16_t* ptrs[kTopk];
            #pragma unroll
            for (int k = 0; k < kTopk; ++k)
                ptrs[k] = rows_h + ((size_t)(ok8[k] ? k : 0) * combine_cap +
                                    token) * kHidden + c;
            uint4 raw[kTopk];
            #pragma unroll
            for (int k = 0; k < kTopk; ++k)
                raw[k] = *reinterpret_cast<const uint4*>(ptrs[k]);
            float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f,
                            0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int k = 0; k < kTopk; ++k) {
                if (!ok8[k]) continue;
                const uint16_t* hf = reinterpret_cast<const uint16_t*>(&raw[k]);
                const float wv = w8[k];
                #pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[j] += combine_bf16_to_float(hf[j]) * wv;
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j)
                y[(size_t)token * kHidden + c + j] = acc[j];
        }
        __syncthreads();
    }
}
