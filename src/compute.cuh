// ============================================================================
// compute.cuh — fp8xfp8 MoE Mega Kernel 计算链（T2：DISPATCH pull + L1+SwiGLU + L2）
//
// 依赖：common.cuh（场景常量）、flags.cuh（T1 冻结接口：SliceGate /
//       slice_gate_wait / l2_gate_wait / l1_progress_publish）。
// 参考骨架（移植+适配）：empty_kernel_bench.cu
//   dispatch_pull_body            bench:1573-1939
//   l1_gemm_swiglu_body_paired    bench:850-980
//   l2_gemm_body                  bench:1052-1168
//
// ---------------------------------------------------------------- 导出函数 --
// 1) dispatch_pull_body(...)  —— 任务循环驱动（每 block 调用一次），每个任务：
//      领号（首波 task=blockIdx，其后 atomicAdd(task_counter)）
//      → 解码 (e, w, slice) → ring 门（l2_gate_wait）
//      → 行均分 K 片 pull（peer x 视图 → 本 rank pool 槽）+ staging weights/meta
//      → 非法行零填充 → 分片就绪旗（本 slice）
//      → L1（l1_gemm_swiglu_body_paired，4 warp）
//      → 块同步 → l1_progress_publish → 等本窗全部 NSLICE 个 l1_prog 旗
//      → L2（l2_gemm_body，4 warp）→ 块同步
//      → tail(TaskDesc)（T3 COMBINE 钩子）→ 任务尾 l2_done 递增。
//    参数（按骨架顺序，已裁剪实验开关）：
//      self/peer_bases/R/rank     对称区与 peer 视图
//      win_cap                    逻辑窗口容量（窗口划分/行数/staging 行 stride）
//      pool_stride                pool 槽地址 stride（= win_cap + win_pad）
//      tbl_cap                    src_index_table 行容量（= max_tokens；非法行判据）
//      k_in/topk/ring             K（H）/ topk / 窗口 ring 槽数
//      *_off                      prefix/src/x/weights/pool/w/meta 在对称区内偏移
//      bar_flags + *_idx          task_counter/l2_done/task_ready/slice_flags/l1_prog
//      out_processed/dbg/max_spins 诊断与自旋保护
//      w1/l1_n_total/l1_block_n/l1_k_out/ffn_hidden/l1_chk   L1 通路
//      w2/l2_out/l2_k_out/l2_chk                            L2 通路
//      rep/n_reps                 轮次（单轮 rep=0, n_reps=1；多轮时 ring 门每轮等）
//      tail                       任务尾钩子（T3 COMBINE；默认 NoTaskTail 空钩子）
//      prefix_ptr                 可选独立前缀表缓冲（与 task_setup_body 的
//                                 prefix_ptr 对齐；nullptr 时用 self+prefix_off）
// 2) l1_gemm_swiglu_body_paired<pf, nt, mt, nb>(pool, w1, expert, count, k_hidden,
//      n_total, block_n, lane, warp, nwarps, checksum, n_group_begin,
//      n_group_count, ffn_out, out_col_base, k_out, gate)
//    —— N 组批量（nb 组共享 A）FP8 GEMM + SwiGLU + FP8 量化回写 ffn_hidden；
//       K 循环按 K 片边界调 slice_gate_wait（仅首组）。调用者保证：
//       n_group_count 为 nb 的整数倍（组对 = (gate, up)），ffn_out 非空时写行 < count。
// 3) l2_gemm_body<pf, nt, mt, nb>(a_pool, w2, count, k_in, n_total, block_n, lane,
//      warp, nwarps, checksum, n_group_begin, n_group_count, out, k_out)
//    —— FP8 GEMM（K=I、N=H），N 组批量（nb 组共享 A），BF16 输出到每块 staging 槽。
//
// ------------------------------------------------------- 权重 / 地址契约 --
// * 权重 16B/lane 配对 packed tile（零 shuffle，layout16b）：
//       supertile 基址 = W + ((out_col/32) * kIters + (k/32)) * 1024，
//       kIters = K/32；supertile 内 64 lane × 16B 即两个 native fragment：
//       lane L 低 8B 覆盖逻辑行 (out_col/32)*32 + L%16、K 段 [(L/16)*8, +8)，
//       高 8B 覆盖同 K 段的逻辑行 +16（即 t=0/t=1 两个 N16 tile）。
//       一次 16B 装载直接得到两个 fragment，免 shuffle/重分发。
// * W1 [EPR][2I][H] 的 gate/up 配对布局（host 必须按此打包）：
//       组 g 对应 2I 行空间的 [g*256, (g+1)*256)；配对 (2p, 2p+1) 产出隐藏列
//       [p*256, (p+1)*256)，即 gate 行 = (h/256)*512 + h%256、up 行 = +256。
// * W2 [EPR][H][I]：expert stride = H*I 字节。
// * 地址/容量三元组分离：窗口划分用 win_cap；pool 行地址用 pool_stride；
//   src_index_table 索引与非法行判据用 tbl_cap；staging 行 stride = win_cap
//   （token_route_weights / token_src_meta / ffn_hidden / l2_out）。
// * 本文件使用动态 smem 头部 EPR + (EPR+1) + EPR*(R+1) 个 int
//   （前两块与骨架 dispatch 一致；第三块为全 experts 的 src 前缀一次性预载，
//    见下方"本轮优化"；task_setup 已完成后再进入本函数）。
//
// ------------------------------------------------------------- 本轮优化 --
// O1) src 前缀整体预载：启动时把 prefix[EPR][R+1] 一次性搬入 smem，省去每任务
//     的 global 重读与紧随其后的一次 __syncthreads（原 s_prefix 仅 R+1 大）。
// O2) 权重远端读前移：pull 每行先发 weights 远端读再搬运整行，使读延迟与行
//     搬运的 store 重叠（原顺序先搬运后读，延迟暴露在任务尾 fence 前）。
// O3) L2 N 组两批由 4+2 改 3+3（g_l2 为 3 的倍数时，本场景 g_l2=6）：两批
//     规模对称，第二批不再退化为 kBatch=2 的小 ring（A 装载次数不变）。
// 集14) hot/mixed-invalid 路由专项（test/14.hot_perf/）：大 count（>kM）路径
//      由"M4 单实例（kMTiles=4）"改为"M1 分块循环（每块 16 行，单一
//      <2,2,1,2> 实例）"。根因：M4 实例使整 kernel 顶格 256 MTregs 且栈帧
//      2032B（spill 到 local），污染全部路径（wc16 的 M1 任务 L1/L2 各慢
//      ~10%）；去掉后栈帧 176B，graph 链三形态 -18~26%（同会话背靠背）。
//      数值等价（行独立 GEMM 分块），并顺带修正原 M4 对 count>64 的
//      m_tiles 截断（原语义只算前 64 行）。协议/同步/标志/checksum 全部未动。
// 集19) TPR=128 专项（test/19.mtile_big/）：任务级编译隔离 + M 档位：
//      - 调度循环只做领号/解码/ring 门，任务体按 count 分派到 compute_big.cuh
//        的两个 __noinline__ 独立函数（r19_small_task / r19_big_task）；
//        探针证明"空体 noinline 调用点"对 TPR=16 中性（298.6 vs 297.8µs），
//        回归只来自大任务体的寄存器需求，任务级隔离后 TPR=16 +8.3%（71 机）。
//      - 大 count 走 M 档位（L1 kMT=4/kBatch=4、L2 kMT=4/kBatch=3、kPF=1）：
//        外层 c0 按 kMT*16 行分组、组内 B 在 M tile 间共享 → 权重读 8 遍→2 遍。
//      - 结果（71 机同会话）：TPR=128 2453→1406µs（−42.7%），TPR=64 −40.7%。
// 集21) A fragment 布局专项（test/21.mtile_big2/）：见 compute_big.cuh 头部。
//      大 count 任务在单波/单窗条件下把 A（池）与 L2 的 A（ffn_hidden）各写一份
//      [step][m][row][32B] 分块副本（放 ffn_hidden 空槽），使 L1/L2 的 A 装载从
//      16 line/指令降到 4 line/指令，并消除 L2 的 warp 末尾 straggler；L1 kPF=2。
//      71 机同会话：TPR=128 1405→1246µs，TPR=64 727→666，TPR=16 +1.7%。
// 说明：O1~O3、集14、集19、集21 均不改变同步/标志协议（wait-all、slice 旗、
//      l1_prog、l2_done、checksum、REDUCE epoch）；仅调整数据搬运顺序、
//      GEMM 批次切分与 A 的本地副本布局。
//
// ---------------------------------------------------------------- 与骨架差异 --
// a) 4 warp 计算：nwarps=4、每 warp 4 个 N16 tile（kNTilesPerWarp=4）。
//    骨架 kL1Paired 路径用 8 warp×2 tile，4 warp 变体在其非配对路径实测最优；
//    此处组合"配对 + 4 warp"（block_n 固定 256 = 4 warp × 4 tile × 16）。
// b) L1→L2 依赖等本窗全部 NSLICE 个 l1_prog 旗（骨架末版只等本 slice）。
//    L2 的 K = I 覆盖 ffn_hidden 全部隐藏列，只等本 slice 会读到未完成输出；
//    results.md §7.8 的已验证实现即为"等全部旗"，本文件沿用。
// c) 非法行零填充：pool 本 K 片写零 + staging weight=0/meta=0（骨架遗留旧值）。
// d) W2 expert stride 取 e*H*I（骨架 dispatch 路径误用 e*l2_k_out）。
// e) M 档位：count<=16 走单档 M1 路径；count>16 走 compute_big.cuh 的 M 档位
//    任务体（[集19]，见"本轮优化"；集14 的 M1 分块循环已被其取代）。
// f) COMBINE 不在本模块：任务尾通过 tail 钩子交给 T3。
// g) 分片流水门固定启用（production 口径）：pull 后即时发布分片旗，L1 首组
//    的 K 片边界调 slice_gate_wait（自 slice 立即通过）。SliceGate 的 T1 扩展
//    字段 seq/dbg/max_spins 由本模块填充（seq=本窗 epoch，否则门恒关闭）。
// h) 补 prefix_ptr 可选参数，与 T1 task_setup_body 的独立前缀表路径对齐。
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
    //   l1_n_total=2*l1_k_out、tbl_cap=slot_capacity、pool_stride=win_cap、
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
    // -- 可选独立缓冲（nullptr → self 内对应偏移）--
    uint8_t* pool_ptr;            // [A-RWK] pool 缓冲（→ Self+pool_off）
    int* prefix_ptr;              // 前缀表缓冲（→ self+prefix_off）
    // -- 前序 / 收尾偏移（self 内）--
    int64_t prefix_off;           // prefix_tbl 在 self 内偏移（prefix_ptr 为空时用）
    // -- 数据缓冲偏移（self / peer sym 内）--
    int64_t x_off, weights_off, pool_off, w_off, meta_off;
    int64_t combine_rows_off;     // 目标 rank combine_rows（BF16，<0 关闭 COMBINE）
    // -- 计算链缓冲 --
    int l1_k_out;                 // L1 输出隐藏维 I（l1_n_total 由 2*l1_k_out 派生）
    int l2_k_out;                 // L2 输出维 H
    const uint8_t* w1;            // [EPR][2I][H] lane 序 packed tile（gate/up 配对）
    const uint8_t* w2;            // [EPR][H][I] lane 序 packed tile
    uint8_t* ffn_hidden;          // [EPR][ring][win_cap][I] FP8
    uint16_t* l2_out;             // [kGrid][win_cap][H] BF16
    // -- flag 索引（layout 头经 host 传入；kernel 不 include 该头）--
    int64_t xrelease_idx, task_ready_idx, task_counter_idx, l2_done_idx;
    int64_t slice_flags_idx, l1_prog_idx;

    // ======================================================================
    // [3] 输出
    // ======================================================================
    // 本 kernel（mega_kernel）不写最终 y；y 由 REDUCE 经 host 侧指针写出。

    // ======================================================================
    // [4] 性能调优参数
    // ======================================================================
    int max_spins;                // 各同步门的自旋上限
    int skip_prologue;            // [R38] 前序复用：0 完整，>0 复用（见 kernel.cuh 说明）

    // ======================================================================
    // [5] 调试 / debug（MX_DBG_COUNT，编译期默认关）
    // ======================================================================
    // （控制/诊断状态 out_total/out_processed/dbg/ck_dbg/pre_ticks/l1_chk/l2_chk
    //   已迁入 local_flags（lf->...），不在此结构体。）
#if MX_DBG_COUNT
    unsigned long long* exec_ctr;  // [kExecCtrSlots] 各计数槽（nullptr = 不统计）
    int dbg_count;                 // 运行期开关（host 读 env MXKERNEL_COUNT 填入）
#endif
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
// L1：N 组批量共享 A（kBatch 组一个 K 循环）+ SwiGLU + FP8 量化回写
//
// 组对 (g, g+1) = (gate, up)：kBatch 个 N 组在同一 K 循环内共享 A（a_pipe 仅
// 一份），B 每步 kBatch × kNTiles；A 装载次数由 kBatch 决定（kBatch=4 时每个
// warp 每 k-step 只读一次 A，是 kBatch=2 的一半）。K 循环 kPrefetch 槽预取；
// 首组按 K 片边界等分片旗。
// 累加器 lane 布局：row = m*16 + (lane>>4)*4 + i，
//                  col = warp*n_per_warp + t*16 + (lane&15)。
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

// ------------------------------------------------------------- 任务信息 --

#include "combine.cuh"




// 大/小 count 任务体（独立 __noinline__ 函数；此位置已可见 L1/L2 body 定义）
#include "compute_big.cuh"

// ============================================================================
// DISPATCH：任务循环驱动（领号/解码 + ring 门 + 任务体分派）
//   任务体（pull → 分片旗 → wait-all → L1 → l1_prog → L2 → tail）按 count
//   分派到 compute_big.cuh 的两个 __noinline__ 独立函数：
//     count <= kM → r19_small_task（热路径，与基线逐字一致）
//     count >  kM → r19_big_task（M 档位，B 在 M tile 间共享）
//   两者独立寄存器分配，互不污染；协议语义（wait-all、分片旗、l1_prog、
//   l2_done、checksum、REDUCE epoch、任务编号）与集14 基线逐字一致。
// ============================================================================
// [R31] kSmallOnly=true：编译期删除大 count（M 档位）任务体分支 → 生成的 kernel
//   不含重型 GEMM 实例（246 MTregs vs 256），TPR=16 恢复 ~24µs（见
//   test/31.tpr16_regress/results.md）。由 host 在 slot_capacity<=kM 时选择该实例；
//   语义要求 win_cap<=slot_capacity<=kM（此时 count>kM 分支恒不可达，协议不变）。
// [L1L2-SPLIT] kPhase：Both=原合并路径；L1=只跑 DISPATCH+L1；L2=只跑 L2+tail。
//   拆分下数据槽 e*slot_win+w 每窗口独占（ring 门/l2_done 不再需要）；旗槽
//   （slice_flags/l1_prog/l2_done）保持 ring 索引不变（旗区容量 256 不动）。
template <bool kSmallOnly = false,
          int kCap = 0, MegaPhase kPhase = MegaPhase::Both>
__device__ __forceinline__ void dispatch_pull_body(
        const MegaArgs* a, int cycle, uint8_t* a_smem = nullptr,
        uint8_t* b_smem = nullptr, unsigned long long* ctr = nullptr) {
    const uint8_t* self = a->self;
    const ranks_data* remote = &a->unk->rd;
    uint8_t* const* peer_bases = a->peer_bases;
    const int R = a->R, rank = a->rank, win_cap = a->win_cap;
    const int pool_stride = a->win_cap, tbl_cap = a->slot_capacity;
    const int k_in = kHidden, topk = kTopk, ring = kRing;
    const int64_t prefix_off = a->prefix_off, x_off = a->x_off;
    const int64_t weights_off = a->weights_off, pool_off = a->pool_off;
    const int64_t w_off = a->w_off, meta_off = a->meta_off;
    uint8_t* pool_ptr = a->pool_ptr;
    uint32_t* bar_flags = a->rwk->lf.bar[cycle];
    const int64_t task_counter_idx = a->task_counter_idx;
    const int64_t l2_done_idx = a->l2_done_idx;
    const int64_t task_ready_idx = a->task_ready_idx;
    const int64_t slice_flags_idx = a->slice_flags_idx;
    const int64_t l1_prog_idx = a->l1_prog_idx;
    int* out_processed = a->rwk->lf.out_processed;
    int* dbg = a->rwk->lf.dbg;
    const int max_spins = a->max_spins;
    const uint8_t* w1 = a->w1;
    const int l1_n_total = 2 * a->l1_k_out, l1_block_n = kL1BlockN;
    const int l1_k_out = a->l1_k_out;
    uint8_t* ffn_hidden = a->ffn_hidden;
    float* l1_chk = a->rwk->lf.l1_chk;
    const uint8_t* w2 = a->w2;
    uint16_t* l2_out = a->l2_out;
    const int l2_k_out = a->l2_k_out;
    float* l2_chk = a->rwk->lf.l2_chk;
    const int rep = 0, n_reps = 1;
    const int64_t combine_rows_off = a->combine_rows_off;
    unk_data* const* unk_peers = a->unk_peers;
    const uint64_t epoch = (uint64_t)a->rwk->lf.current_time[0][1];
    const int* prefix_ptr = a->prefix_ptr;
    unsigned long long* pt = a->rwk->lf.pre_ticks;
    uint8_t* frag_a = a->rwk->ld.frag_a[0];
    uint8_t* frag2 = a->rwk->ld.frag2[0];
    extern __shared__ __align__(16) uint8_t smem_raw[];
    int* s_count = reinterpret_cast<int*>(smem_raw);   // EPR：每专家总行数
    int* s_cum = s_count + kEPR;                       // EPR+1：窗口前缀（任务解码）
    int* s_prefix = s_cum + kEPR + 1;                  // EPR*(R+1)：全 experts src 前缀
    __shared__ int s_task, s_expert, s_window, s_count_w, s_row_off, s_slice;
    __shared__ unsigned s_epoch;
    // prefix_ptr 非空时用独立前缀表缓冲（与 task_setup_body 的 prefix_ptr 对齐）
    const int* prefix = (prefix_ptr != nullptr)
        ? prefix_ptr
        : reinterpret_cast<const int*>(self + prefix_off);
    const int tid = threadIdx.x;
    const int warp = tid / 64;
    const int lane = tid % 64;
    (void)rank;   // 预留：本 rank 号（当前寻址全部经 self/peer_bases 完成）
    (void)warp; (void)lane;
    // 相位计时（可选，默认 nullptr 不改变原行为）：仅 block0 的首个任务由 tid0 记录
    const bool kTick = (pt != nullptr) && (blockIdx.x == 0) && (tid == 0);
    // [O1] 全 experts src 前缀一次预载（s_prefix 扩容 EPR*(R+1)）
    for (int i = tid; i < kEPR * (R + 1); i += (int)blockDim.x)
        s_prefix[i] = prefix[i];
    // 每专家总行数 + 窗口前缀（前缀和以 win_cap 划分窗口）
    if (tid == 0) {
        int total = 0;
        s_cum[0] = 0;
        for (int e = 0; e < kEPR; ++e) {
            s_count[e] = prefix[e * (R + 1) + R];
            total += (s_count[e] + win_cap - 1) / win_cap;
            s_cum[e + 1] = total;
        }
    }
    __syncthreads();
    // [R35] 波量化自适应粗化：逻辑任务数（窗数×kNSlice）超过 grid 时，按
    //   merge（2 的幂，整除 kNSlice）个逻辑 slice 合并为一个物理任务，
    //   使物理任务数 ≤ grid（单波）。收益：
    //   a) A 读总量 ∝ 物理任务数，粗化后不再随 NSLICE 放大；
    //   b) 消除第二波串行与跨波 wait-all 悬挂；
    //   c) 合并任务内 K/N 范围连续（u4 连续、列连续），不破坏任何 flag 协议
    //      （slice_flags/l1_prog 仍逐逻辑 slice 写满，wait-all 仍遍历 kNSlice）。
    //   任务数 ≤ grid 时 merge=1，行为与改动前逐字一致（NSLICE=4 不受影响）。
    const int raw_tasks = s_cum[kEPR] * kNSlice;
    // [步骤20a] 去掉 merge 机制：固定 merge=1（不粗化），nslice_eff=kNSlice。
    const int merge = 1;
    const int nslice_eff = kNSlice / merge;
    const int total_tasks = raw_tasks / merge;
    // [集71] 任务循环进入 block 数
    // [L1L2-SPLIT] 循环统计只在 L2/Both 记（L1 阶段另跑一遍会重复计数）
    if (tid == 0 && kPhase != MegaPhase::L1)
        MX_CTR_ADD(ctr, kCtrLoopEnter, 1);
    // 任务体调度参数（循环不变量打包，按 const& 传给两个 noinline 任务体）
    R19Sched S;
    S.self = self;
    S.remote = remote;
    S.cycle = cycle;
    S.peer_bases = peer_bases;
    S.R = R;
    S.rank = rank;
    S.win_cap = win_cap;
    S.pool_stride = pool_stride;
    S.tbl_cap = tbl_cap;
    S.k_in = k_in;
    S.topk = topk;
    S.ring = ring;
    S.x_off = x_off;
    S.weights_off = weights_off;
    S.pool_off = pool_off;
    // [A-RWK] pool 基址：显式传入（普通显存）或回退对称区 self+pool_off
    S.pool_base = (pool_ptr != nullptr)
                      ? pool_ptr
                      : const_cast<uint8_t*>(self) + pool_off;
    S.w_off = w_off;
    S.meta_off = meta_off;
    S.bar_flags = bar_flags;
    S.l2_done_idx = l2_done_idx;
    S.task_ready_idx = task_ready_idx;
    S.slice_flags_idx = slice_flags_idx;
    S.l1_prog_idx = l1_prog_idx;
    S.dbg = dbg;
    S.max_spins = max_spins;
    // [集71] 计数缓冲下传任务体（L1/L2/COMBINE/wait-all 统计）
    S.ctr = ctr;
    S.w1 = w1;
    S.l1_n_total = l1_n_total;
    S.l1_block_n = l1_block_n;
    S.l1_k_out = l1_k_out;
    S.ffn_hidden = ffn_hidden;
    S.frag_a = frag_a;
    S.frag2 = frag2;
    S.l1_chk = l1_chk;
    S.w2 = w2;
    S.l2_out = l2_out;
    S.l2_k_out = l2_k_out;
    S.l2_chk = l2_chk;
    S.pt = pt;
    S.combine_rows_off = combine_rows_off;
    S.unk_peers = unk_peers;
    S.epoch = epoch;
    // [R21] A fragment 副本（放在 ffn_hidden 的空槽）启用条件：
    //   单波（每 block 恰 1 任务，sibling 组固定）+ 每 expert 单窗（win_cap
    // [R34][FRAG-INT] fragment 的 m 维长度 = ceil(win_cap/16)（win_cap 允许非 16
    //   整除：尾块由 row<cc 掩码，fragment 尾 tile 折叠到最后一 tile 读取）。
    //   A/F2 fragment 槽 = local_data.frag_a/frag2[(e*ring + w%ring)]（恒存在）。
    const int mt_total = (win_cap + 15) >> 4;
    S.mt_total = mt_total;
    // [FRAG-INT] A/f2 fragment 专用缓冲恒存在（local_data.frag_a/frag2，非 host 传入）。
    //   取消 afrag_ok（借 ffn_hidden 空槽）路径：一律走专用缓冲，不再有 launch 级分支。
#if defined(CM_NO_FRAGALL)
    S.frag_all = false;
#else
    S.frag_all = (frag_a != nullptr) && (frag2 != nullptr) &&
                 (l1_k_out > 0) && (mt_total > 0) && (win_cap > 0);
#endif
    S.merge = merge;
    S.nslice_eff = nslice_eff;
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
                                       bar_flags + task_counter_idx), 1));
            s_task = task;
            if (task < total_tasks) {
                // [R35] 物理任务 → 窗号/物理 slice（每任务覆盖 merge 个逻辑 slice）
                const int tw = task / nslice_eff;
                const int slice = task % nslice_eff;
                s_slice = slice;
                int lo = 0, hi = kEPR - 1;
                while (lo < hi) {
                    const int mid = (lo + hi) >> 1;
                    if (s_cum[mid + 1] <= tw) lo = mid + 1;
                    else hi = mid;
                }
                s_expert = lo;
                s_window = tw - s_cum[lo];
                const int remain = s_count[lo] - s_window * win_cap;
                s_count_w = remain < win_cap ? remain : win_cap;
                s_row_off = s_window * win_cap;
                s_epoch = *(volatile const uint32_t*)(bar_flags + task_ready_idx);
                // [L1L2-SPLIT] ring 门等前代 L2 完成；拆分下前代 L2 在下一个
                //   kernel（stream 顺序已保证），且数据槽不复用 → 编译期跳过。
                if constexpr (kPhase == MegaPhase::Both) {
                    if (s_window >= ring || rep > 0) {
                        // ring 门：target=(rep + w/ring)*nslice_eff（每物理任务 +1），
                        // 槽复用前等前代 L2 完成。[R35] 用 nslice_eff 匹配物理任务数。
                        l2_gate_wait(bar_flags, (int)l2_done_idx, lo, s_window,
                                     rep, ring, nslice_eff, slice, -1, 0u, dbg,
                                     max_spins, ctr);
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
        // [L1L2-SPLIT] 旗槽（ring，容量 EPR*ring）与数据槽（拆分下每窗口独占）：
        //   slot_win = ceil(kMaxTPR/win_cap) 覆盖 count<=kMaxTPR 的全部窗口；
        //   Both 下两者相同（e*ring + w%ring），行为逐字不变。
        const int pool_slot = e * ring + w % ring;
        const int slot_win = (kMaxTPR + win_cap - 1) / win_cap;
        const int data_slot =
            (kPhase == MegaPhase::Both) ? pool_slot : (e * slot_win + w);
        const int* pfx = s_prefix + e * (R + 1);
        // ---- 任务体分派（count<=kM 小路径 / count>kM M 档位大路径）----
        // [R31] kSmallOnly 实例：大任务体不实例化（小核专用路径）
        if constexpr (kSmallOnly) {
            r19_small_task<kPhase>(S, task, e, w, count, slice,
                                             epoch, data_slot, pfx,
                                             a_smem, b_smem);
        } else if (count > kM) {
            r19_big_task<kCap, kPhase>(S, task, e, w, count, slice,
                                                 epoch, data_slot, pfx,
                                                 a_smem, b_smem);
        } else {
            r19_small_task<kPhase>(S, task, e, w, count, slice,
                                             epoch, data_slot, pfx,
                                             a_smem, b_smem);
        }
        // ---- 任务尾 l2_done 递增（仅后续窗口会复用该槽或多轮时）----
        // [H3] 不发布 l2_done 时省略全线程 threadfence（无消费者，协议语义不变）
        // [L1L2-SPLIT] 拆分下无环复用（数据槽每窗口独占）→ 跳过 ring 门与发布。
        bool h_pub = false;
        if constexpr (kPhase == MegaPhase::Both) {
            h_pub = (s_count[e] > (w + ring) * win_cap) || n_reps > 1;
        }
        if (h_pub) __threadfence();
        if (tid == 0) {
            if (h_pub)
                atomicAdd(reinterpret_cast<uint32_t*>(
                              bar_flags + l2_done_idx +
                              (size_t)(e * ring + w % ring)),
                          1u);
            ++processed;
            if (out_processed != nullptr) out_processed[blockIdx.x] = processed;
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
    if (pt != nullptr && blockIdx.x == 0 && tid == 0)
        pt[7] = clock64();
}
