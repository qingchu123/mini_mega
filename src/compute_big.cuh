// ============================================================================
// compute_big.cuh — 大 count（>kM）任务独立编译单元（集19 / 子 agent R）
//
// 背景（test/17、test/18 结论）：
//   基线的 M1 分块循环（16 行/块）在 TPR=128 时把权重读 8 遍，是绝对主成本
//   （诊断：删 A+B 后全链 653.6µs，B 单独≈901µs）；唯一结构性杠杆是 M 档位
//   （B 在 kMT 个 M tile 间共享，权重读 TPR/(kMT*16) 遍）。
//   但 M 档位加入同一全内联大函数会扰动 count=16 热路径的寄存器分配/调度
//   （P: +33%、Q: +45%）。
//
// 本文件方案（R19）：
//   1) 任务级编译隔离：把"大 count 任务"的整体（pull + 分片旗 + L1 M 档位 +
//      l1_prog + L2 M 档位 + tail）放进 __noinline__ 独立函数 r19_big_task；
//      count=16 热路径任务走 compute.cuh 的 r19_small_task（同样 __noinline__，
//      彼此不共享寄存器分配）。两项经探针验证：仅"空体调用点"对 TPR=16 中性
//      （298.6µs vs 基线 297.8µs）。
//   2) M 档位结构：外层 c0 按 kMT*16 行分组；组内 K 循环外层、M 块内层，
//      B fragment 在 kMT 个 M tile 间共享（B 只读一遍/组）。
//      参考 test/17 骨架/生产同构实现：`acc[kBatch][kMT][kNT]` 常驻，
//      `a_pipe[kPF][kMT]`、`b_pair[kPF][kBatch]` 预取环。
//   3) 档位可配：R19_L1_MT/PF/BATCH、R19_L2_MT/PF/BATCH（-D 扫描）。
//
// 协议语义：与集14 基线逐字一致（wait-all、分片旗、l1_prog、l2_done、
//   checksum、REDUCE epoch、任务编号）；pull/staging/零填充代码自 compute.cuh
//   原样搬移，仅变量改经 R19Sched 传入。
//
// 集21（R21，test/21.mtile_big2）：A 装载 fragment 布局专项
//   1) 归因：诊断证明 A 的 327µs 成本几乎全在"16 行×32B/指令（16 line）"
//      的散射装载 + 由此引发的 L2 A 装载 straggler（末位 warp 落后 ~120µs），
//      与字节量无关（固定地址/L1 常驻即消失）。
//   2) F_A：pull 阶段在池行主序之外，向本 expert 的 ffn_hidden 空槽
//      （e*ring+1..3）写一份 A 的 [step][m][row][32B] 分块副本；L1 的 A 装载
//      变为连续 512B/指令（4 line）。
//   3) F2：L1 epilogue 同时按字节写 ffn_hidden 的 fragment 副本（e*ring+4 槽），
//      L2 的 A（ffn_hidden）装载同样变 4 line，消除 L2 的 warp 末尾 straggler
//      （130µs → 16µs）。行主序 ffn_hidden 仅保留 slot0（bench 数值校验读取）。
//   4) 仅在"单波 + 每 expert 单窗 + 空槽足够"时启用（afrag_ok），否则回退
//      行主序路径（协议/数值等价）。L1 kPF 由 1 升 2（acc 128 下刚好容纳）。
//   5) 结果（71 机同会话）：TPR=128 1405→1246µs（−11.3%）、TPR=64 727→666
//      （−8.4%）、TPR=16 313→318.5（+1.7%）；正确性/数值全 PASS。
//
// 集27（R27，test/27.tpr256_calc）：TPR=256 权重遍数专项
//   1) 归因（本地 TPR=256 诊断）：B（权重）占 L1 约 700µs / L2 约 260µs，
//      是 compute 主成本；B 字节 = count/(kMT*16) × 切片列数 × K，即 M 遍数。
//      count=256 时 MT4 → 4 遍（L1 25.2MB/block、L2 12.6MB/block）。
//   2) 交付：大 count 档位升为 L1 MT8/B2、L2 MT8/B2（acc 仍 128 floats/线程
//      顶格：kBatch*kMT*kNT*4=2*8*2*4），R24_KS 16→8 保持 A smem 64KB 双缓冲；
//      非 async 回退固定 MT4（避免双份大 acc 实例抬高栈）。B 遍数减半
//      （L1/L2 各减 50% 权重读）。
//   3) 结果（71 机同会话配对 ×2）：R=8 TPR=256 full 1921→1839µs（−4.3%），
//      L1 909→713µs（−22%）；TPR=128 +2.8%、TPR=16 +0.5%（≤5% 容忍内）；
//      正确性三形态 + 三行数值 PASS（bit-exact 不变）。
//   4) 未达 <1600：task-loop 已由 1972→1470µs（本地），但 wall 被 COMBINE
//      出/入向 fabric 排空（每 rank ~25MB，~60GB/s 级）+ REDUCE 行旗等待
//      串死（节点内 full−chain ~340µs，nodes=1 同样存在）——该部分在
//      combine.cuh/协议侧，compute 两文件内无杠杆（早推实验见 results.md）。
//
// 集34（R34，test/40.nondiv_opt）：非整除 TPR 回退路径专项
//   1) 归因：kAsync 原来要求 count%(MT*16)==0 且 win_cap%16==0 → 129/257/513
//      整段掉非 async 回退（MT4/B4，权重 ceil(count/64) 遍）；且 F_A 槽位条件
//      (win_cap&15)==0 使 win_cap 非整除时连 F 布局都不可用。
//   2) 交付：F_A/F2 按 mt_total=ceil(win_cap/16) 布局、f2_slot 动态紧随 F；
//      r24_*_big 改行区间 [c0_begin,c0_end) + kMask 模板（整块无掩码快路径、
//      尾块掩码）；权重遍数 = ceil(count/(MT*16))。尾块 ≤16 行走 MT1/KS16 小
//      档（A/MMA 1/8、stage 同步减半）；count∈(16,32] 走小档 body（runtime M
//      裁剪 + 单批 N）；count>64 才启用 async（≤64 时 async 无权重收益且更慢）。
//   3) 结果（本机同会话背靠背）：R=32 129 2038.6→1293.9、257 3850.9→2153.2、
//      513 7715.1→3848.2；R=8 129 1925.8→1195.3、257 3563.0→1924.7、
//      513 7177.4→3440.7；整除档全面 −4%（尾块掩码谓词在旧快路径的消除）。
//      正确性三形态 + 数值三行 PASS，bit-exact 不变。
//
// 集29（R29，test/29.combine_reduce）：COMBINE×L2 融合推送（尾排空隐匿）
//   1) 归因（诊断 nodata：保留完成标记、只删数据推送）：R=8 TPR=256
//      full 1843→1426、R=32 2189→1742 —— 数据推送的「任务尾串行 + 内核退出
//      前 fabric 排空」共 ~420-450µs，是尾段绝对主成本。
//   2) 交付：L2 的 r24 路径每算完一个 (c0 行块, g 列批) 就 __stwt 推送该列块
//      （combine.cuh 的 CombineFuseCtx/combine_push_chunk），任务尾只写完成
//      标记（combine_mark_rows，保留 fence→sync→lane0 发布序）。协议语义
//      （每 slice 列范围/标记值 epoch/REDUCE 放行）逐字不变。
//   3) 结果（71 机同会话）：R=8 TPR=256 1843→1726µs（−6.5%）；TPR=128
//      981→922（−6.0%）；R=32 TPR=256 2189→1996µs（−8.9%）；正确性/数值
//      （R=8 256/128/16、R=32 256）全 PASS（bit-exact 不变）。
//   4) 剩余瓶颈（R=32 <1900 未达，差 ~95µs）：nodata 地板 1743µs 与融合后
//      1996 之差 ≈ 253µs 是「推送带宽在 L2 内存忙期内只有 ~100GB/s 边际」
//      + L2 相 ~+70µs 挤出；而 full−chain ≈ 210µs 与 nodata 地板同值，
//      为 rank 8~31 的固有偏斜（Y 归因驱动/仲裁层，非 COMBINE 可动）。
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

// [R29] tail → 融合上下文获取/回写（默认无融合；kernel.cuh 对 CombineTail 特化）。
//   任务体在进入 r24 L2 路径且 staging 有效时激活 fuse.on，并在任务尾把该状态
//   写回 tail，供 CombineTail 选择"只写标记"路径。
template <typename T>
struct R29TailFuse {
    static __device__ __forceinline__ CombineFuseCtx get(const T&) {
        return CombineFuseCtx{};
    }
};
template <typename T>
struct R29TailFuseSet {
    static __device__ __forceinline__ void on(T&, const CombineFuseCtx&) {}
};

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
// [R21] A fragment 布局（消除 A 装载的 16-line/指令 散射）：
//   池中 A 仍按行主序写（保持 bench 数值校验 dump 不变），另在本 expert 的
//   ffn_hidden 空槽（e*ring+1 起）写一份 A 的 K-step×M-tile 分块副本：
//       F[step][m][row_in_tile][k_in_step]   step=k/32, m=row/16
//   地址 = ((step*mt_total + m)*16 + (row&15))*32 + (k&31)，共 win_cap*k_in 字节。
//   L1 一次 warp 装载的 64 lane 落在连续 512B（4 line）而非 16 行×32B（16 line）。
//   安全性：仅在"单波（每 block 1 任务）+ 每 expert 单窗（win_cap>=tbl_cap）"
//   时启用，此时 e*ring+1..ring-1 槽空闲（F 占 3 槽），L2 输出槽不受影响。
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

// [R21] L2 的 A（ffn_hidden）fragment 副本：同一 [step][m][16][32B] 分块，
// step = k/32（I=2048 → 64 步），m = row/16（win_cap/16 个）。L1 epilogue 按
// 字节写入；L2 一次 warp 装载落连续 512B（4 line）。
__device__ __forceinline__ void r19_store_ffn_frag(
        uint8_t* __restrict__ f2, int row, int col, uint8_t v, int mt_total) {
    f2[((size_t)(col >> 5) * mt_total + (row >> 4)) * 512 + (row & 15) * 32 +
       (col & 31)] = v;
}

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
    const uint8_t* self;
    uint8_t* const* peer_bases;
    int R, rank;
    int win_cap, pool_stride, tbl_cap, k_in, topk, ring;
    int64_t src_off, x_off, weights_off, pool_off, w_off, meta_off;
    uint32_t* bar_flags;
    int64_t l2_done_idx, task_ready_idx, slice_flags_idx, l1_prog_idx;
    int* dbg;
    int max_spins;
    const uint8_t* w1;
    int l1_n_total, l1_block_n, l1_k_out;
    uint8_t* ffn_hidden;
    uint8_t* frag_a;    // [OPT-AFRAG] A 输入 fragment 缓冲（nullptr=关闭）
    uint8_t* frag2;     // [OPT-AFRAG] L1 输出 fragment 缓冲（nullptr=关闭）
    bool frag_all;      // [OPT-AFRAG] 多窗口下的 fragment 布局启用
    float* l1_chk;
    const uint8_t* w2;
    uint16_t* l2_out;
    int l2_k_out;
    float* l2_chk;
    unsigned long long* pt;
    bool afrag_ok;   // [R21] 单波 + 每 expert 单窗 + 空槽足够 → 启用 A fragment 布局
    bool poolfrag_ok; // [R23] pool 槽直接按 native-fragment 序落盘（免 F_A 副本）
    // [R35] 波量化粗化：本任务的逻辑 slice 数（merge）与物理 slice 数
    //   （nslice_eff = kNSlice/merge）。几何公式用 nslice_eff；
    //   [R36b] slice_flags/l1_prog 粒度=物理 slice（stride=nslice_eff），
    //   combine 完成标记仍按逻辑 slice（slice*merge + i）写满 sums_stride。
    int merge;
    int nslice_eff;
    // [R34] F_A/F2 的 m 维长度 = ceil(win_cap/16)（win_cap 可 16 非整除：尾块
    //   按 row<cc 掩码）；f2_slot = F 之后的空闲槽号（F 占 1..f_slots）
    int mt_total;
    int f2_slot;
    // [集71] 内部执行次数统计（编译期关闭时为空操作；运行期 nullptr 关闭）
    unsigned long long* ctr = nullptr;
};

// [R21] F 槽基址：本 expert 的 ffn_hidden 空槽（e*ring+1 起；仅在"每 expert
// 单窗"时启用，此时 e*ring+1..7 均空闲，F 需 ceil(k_in/l1_k_out) 个槽）。
__device__ __forceinline__ uint8_t* r19_afrag_base(const R19Sched& S, int e) {
    return S.ffn_hidden + ((size_t)e * S.ring + 1) * S.win_cap * S.l1_k_out;
}

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
         ? r19_load_a_frag(fbase, (step), m_abs0 + (m), mt_total,           \
                           m_abs0 * kM + count)                             \
         : r19_load_a(pool, k_hidden, (step) * kKStep, count, (m)))
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

template <int kMT, int kPF, int kBatch>
__device__ __forceinline__ void r19_l1_tile(
        const uint8_t* pool, const uint8_t* fbase, int m_abs0, int mt_total,
        const uint8_t* w_e, int count, int k_hidden,
        int block_n, int lane, int warp, int nwarps, int g0, int gbase,
        uint8_t* ffn_out, uint8_t* ffn2, int out_col_base, int k_out,
        float& acc_chk) {
    constexpr int kKStep = 32;
    constexpr int kNT = 2;   // 每 warp 2 个 N16 tile（layout16b 配对）
    const int k_iters = k_hidden / kKStep;
    const int n_per_warp = block_n / nwarps;
    (void)fbase; (void)m_abs0; (void)mt_total;
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
                R19_LDB(w_e, nbase[q], sp * kKStep, k_hidden);
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
        // [反序修正] 与 compute.cuh 同因：预取按 q 反序落 b_pair，
        // 此处按反序取回，acc[q] 恒对应 N 组 (g0+q)（契约 gate=2p / up=2p+1）。
#ifdef R19_L1_QOUT
        #pragma unroll
        for (int q = 0; q < kBatch; ++q) {
            const R19Frag2 bp = b_pair[sp][kBatch - 1 - q];
            const R19Frag b0 = {bp[0], bp[1]};
            const R19Frag b1 = {bp[2], bp[3]};
            #pragma unroll
            for (int m = 0; m < kMT; ++m) {
                acc[q][m][0] = __builtin_mxc_mma_16x16x32f8(
                    a_pipe[sp][m], b0, acc[q][m][0]);
                acc[q][m][1] = __builtin_mxc_mma_16x16x32f8(
                    a_pipe[sp][m], b1, acc[q][m][1]);
            }
        }
#else
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
#endif
        const int nk = kk + kPF;
        if (nk < k_iters) {
            #pragma unroll
            for (int m = 0; m < kMT; ++m)
                a_pipe[sp][m] = R21_LDA(nk, m);
            #pragma unroll
            for (int q = 0; q < kBatch; ++q)
                b_pair[sp][kBatch - 1 - q] =
                    R19_LDB(w_e, nbase[q], nk * kKStep, k_hidden);
        }
    }
    // SwiGLU + FP8 量化回写（acc lane 布局同 compute.cuh）
    #pragma unroll
    for (int pi = 0; pi < kBatch / 2; ++pi) {
        const int qa = 2 * pi, qb = 2 * pi + 1;
        #pragma unroll
        for (int m = 0; m < kMT; ++m) {
            #pragma unroll
            for (int t = 0; t < kNT; ++t) {
                R19Acc sw;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float gv = acc[qa][m][t][i];
                    const float uv = acc[qb][m][t][i];
                    sw[i] = gv / (1.0f + __expf(-gv)) * uv;
                    acc_chk += sw[i];
                }
                if (ffn_out != nullptr || ffn2 != nullptr) {
                    const uint32_t packed = __builtin_mxc_cvt_pk4_f32tof8_cfg(
                        sw, _MACA_FROUND_TO_NEAREST_EVEN,
                        _MACA_DENORMAL_FLUSH_NONE, _MACA_SRC_NO_MODIFIER,
                        false, true);
                    const int col = out_col_base +
                                    (((g0 - gbase) >> 1) + pi) * block_n +
                                    warp * n_per_warp + t * 16 + (lane & 15);
                    // [R21] L2 A fragment 副本基址（4 字节行相邻，立即数偏移）
                    uint8_t* f2p = nullptr;
                    if (ffn2 != nullptr)
                        f2p = ffn2 +
                              ((size_t)(col >> 5) * mt_total + (m_abs0 + m)) *
                                  512 +
                              (lane >> 4) * 4 * 32 + (col & 31);
                    #pragma unroll
                    for (int i = 0; i < 4; ++i) {
                        const int row = m * kM + (lane >> 4) * 4 + i;
                        if (row < count) {
                            const uint8_t byte = (uint8_t)(packed >> (8 * i));
                            if (ffn_out != nullptr)
                                ffn_out[(size_t)row * k_out + col] = byte;
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

// 大 count L1：外层 c0 按 kMT*16 行分组，组内 B 只读一遍。
// [R21] fbase = A fragment 副本基址（nullptr 走池行主序）；m_abs0 = 本 c0 的
// 起始 M tile 号；mt_total = win_cap/16（F 的 m 维长度）。
template <int kMT, int kPF, int kBatch>
__device__ __forceinline__ void r19_l1_big(
        const uint8_t* pool, const uint8_t* fbase, int mt_total,
        const uint8_t* w1, int expert, int count,
        int k_hidden, int n_total, int block_n, int lane, int warp, int nwarps,
        float* checksum, int n_group_begin, int n_group_count,
        uint8_t* ffn_out, uint8_t* ffn2, int out_col_base, int k_out) {
    const uint8_t* w_e = w1 + (size_t)expert * n_total * k_hidden;
    float acc_chk = 0.0f;
    for (int c0 = 0; c0 < count; c0 += kMT * kM) {
        const int cc = (count - c0 < kMT * kM) ? (count - c0) : kMT * kM;
        const uint8_t* a_c = pool + (size_t)c0 * k_hidden;
        uint8_t* f_c = (ffn_out != nullptr) ? ffn_out + (size_t)c0 * k_out
                                            : nullptr;
        int g0 = n_group_begin;
        for (; g0 + kBatch <= n_group_begin + n_group_count; g0 += kBatch) {
            r19_l1_tile<kMT, kPF, kBatch>(
                a_c, fbase, c0 >> 4, mt_total, w_e, cc, k_hidden, block_n,
                lane, warp, nwarps, g0, n_group_begin, f_c, ffn2,
                out_col_base, k_out, acc_chk);
        }
        // 余数组（组对语义下必为偶数）：kBatch=2 收尾，兼容非生产几何
        if (g0 + 2 <= n_group_begin + n_group_count) {
            r19_l1_tile<kMT, kPF, 2>(a_c, fbase, c0 >> 4, mt_total, w_e, cc,
                                     k_hidden, block_n, lane, warp, nwarps, g0,
                                     n_group_begin, f_c, ffn2, out_col_base,
                                     k_out, acc_chk);
        }
    }
    if (checksum != nullptr && lane == 0)
        atomicAdd(checksum, acc_chk + (float)count * 0.0f);
}

// ============================================================================
// [R24] L1 的 A 走 smem：warp 协作 bsm 异步搬运（__builtin_mxc_ldg_b128_bsm）
//   - 每个 stage = kKS 个 K step（kKS*32B/行）× kMT 个 M-tile（kMT*16 行）
//   - smem 布局与 F_A 逐字一致：[j][m][row][32B]，LDS 直接读 native fragment
//   - 双缓冲：stage st 消费时，stage st+1 的 bsm 在飞；arrive_gvmcnt(0) +
//     __syncthreads 发布当前 buffer
//   - B 仍直装（ldg 16B/lane），预取环 2 槽跨 stage 连续
//   - [R34] 支持任意 count（行区间 [c0_begin,c0_end)）：整块 kMask=false 无
//     掩码快路径，尾块 kMask=true（F tile 越界折叠 + row<cc 掩码）；k_iters
//     仍须为 kKS 整数倍，否则调用点回退 r19_l1_big
// ============================================================================
// [R27·delivered] A-smem 阶段步长由 16 降为 8：MT8 下 smem 双缓冲保持
//   2*8*8*512=64KB（KS=16 会到 128KB，加动态 smem 后超设备 131072B/block
//   上限，实测启动失败）。KS=8 在 71 机与 KS=16 中性，本地略优。
// [OPT-TK-CAP64] tk_first 扫描发现：MT8 大档 acc（kBatch*kMT*kNT*4）把
//   r19_big_task 的寄存器/栈顶高，拖慢同函数内 count<=64 的小/中档路径。
//   win_cap<=64 时 count>64 不可达 → 用 kCap64 实例（见 megakernel.cu）
//   编译期删除 MT8 异步档，前 50 launch 平均 ~-4%，大档 TPR=256 不变。
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

template <int kMT, int kKS, int kBatch, bool kMask = false, int kMTAct = kMT,
          int kBPF = R24_BPF>
__device__ __forceinline__ void r24_l1_big(
        uint8_t* s_mem, const uint8_t* fbase, int mt_total, const uint8_t* w1,
        int expert, int c0_begin, int c0_end, int k_hidden, int n_total,
        int block_n, int lane, int warp, int nwarps, float* checksum,
        int n_group_begin, int n_group_count, uint8_t* ffn_out, uint8_t* ffn2,
        int out_col_base, int k_out) {
    constexpr int kKStep = 32;
    constexpr int kNT = 2;
    constexpr int kBufBytes = kMT * kKS * 512;
    const int k_iters = k_hidden / kKStep;
    const int nstage = k_iters / kKS;
    const int n_per_warp = block_n / nwarps;
    const uint8_t* w_e = w1 + (size_t)expert * n_total * k_hidden;
    float acc_chk = 0.0f;
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
        r24_issue_a<kMT, kKS, kMask, kMTAct>(s_mem, fbase, 0, mt_total,
                                        m_abs0);
        R19Frag2 b_pair[kBPF][kBatch];
        #pragma unroll
        for (int sp = 0; sp < kBPF; ++sp) {
            if (sp >= k_iters) break;
            #pragma unroll
            for (int q = 0; q < kBatch; ++q)
                b_pair[sp][q] = R19_LDB(w_e, nbase[q], sp * kKStep, k_hidden);
        }
        R19Acc acc[kBatch][kMT][kNT];
        #pragma unroll
        for (int q = 0; q < kBatch; ++q)
            #pragma unroll
            for (int m = 0; m < kMT; ++m)
                #pragma unroll
                for (int t = 0; t < kNT; ++t)
                    acc[q][m][t] = R19Acc{0.f, 0.f, 0.f, 0.f};
        for (int st = 0; st < nstage; ++st) {
            // 等 stage st 的 bsm 完成 + 发布
            R24_WAIT_BSM();
            __syncthreads();
            if (st + 1 < nstage)
                r24_issue_a<kMT, kKS, kMask, kMTAct>(
                    s_mem + ((st + 1) & 1) * kBufBytes, fbase,
                    (st + 1) * kKS, mt_total, m_abs0);
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
                        b_pair[sp][q] = R19_LDB(w_e, nbase[q], nk * kKStep,
                                                k_hidden);
                }
            }
        }
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
          int kBPF = R24_BPF>
__device__ __forceinline__ void r24_l2_big(
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
            R19Frag2 b_pair[kBPF][kBatch];
            #pragma unroll
            for (int sp = 0; sp < kBPF; ++sp) {
                if (sp >= k_iters) break;
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
__device__ __forceinline__ void r19_pull_flags(
        const R19Sched& S, int task, int e, int w, int count, int slice,
        unsigned epoch, int pool_slot, const int* pfx) {
    const int tid = threadIdx.x;
    const int warp = tid / 64;
    const int lane = tid % 64;
    const int nwarps = (int)(blockDim.x / 64);
    const int s_row_off = w * S.win_cap;
    // [R35] 物理任务覆盖 merge 个连续逻辑 slice：K 片为连续 merge 段
    const int u4_slice = (S.k_in / 16) / S.nslice_eff;   // 本任务 uint4 数
    const int u4_off = slice * u4_slice;
    uint8_t* pool = const_cast<uint8_t*>(S.self) + S.pool_off;
    float* wt = reinterpret_cast<float*>(const_cast<uint8_t*>(S.self) + S.w_off);
    int* meta = reinterpret_cast<int*>(const_cast<uint8_t*>(S.self) + S.meta_off);
    const uint32_t* src_tbl =
        reinterpret_cast<const uint32_t*>(S.self + S.src_off);
    const bool kTick = (S.pt != nullptr) && (blockIdx.x == 0) && (task == 0) &&
                       (tid == 0);
    // ---- 行均分 K 片 pull：半 warp 一行（8 warp × 2 行并行），staging 写 weights/meta
    // [H1] 每个 warp 的两个半 warp 各负责一行，且相邻行配对（同源 rank）
    const int half = lane >> 5;           // 半 warp 号（0/1）
    const int ll = lane & 31;             // 半 warp 内 lane
    // [R21] A fragment 副本（拉取时顺带写 F_A：pool 仍行主序，供 bench 校验）
    // [R34] count<=32 走小档（runtime M 裁剪 + 小档 body），无需 F/F2 副本
    const bool kAfrag = (S.afrag_ok && count > 32) || (S.frag_all && count > 0);
    // [OPT-AFRAG] frag_all 时 L1 只读 fragment，行主序 pool 仅 bench 校验用，
    //   故跳过 pool 写（省 A 写带宽；fragment 写已含同数据）
    const bool kSkipPool = false;   // [OPT-AFRAG] 实测跳过 pool 写更慢，保留
    uint8_t* frag_base = nullptr;
    if (kAfrag) {
        if (S.afrag_ok) {
            frag_base = r19_afrag_base(S, e);
        } else {
            // [OPT-AFRAG] 多窗口：按 (expert, window-slot) 定位专用 A fragment
            const size_t fsz =
                (size_t)(S.k_in / 32) * (size_t)S.mt_total * 512;
            frag_base = S.frag_a + (size_t)(e * S.ring + (w % S.ring)) * fsz;
        }
    }
    const int mt_total = S.mt_total;   // [R34] ceil(win_cap/16)
    // [R26] warp 独占一段连续行（= 同源 rank 段）：每个 warp 的 8 次迭代
    // 始终读同一 peer，消除跨 peer 轮转的 fabric 排队惩罚（71 机 TPR=128
    // 同会话 −60µs；dispatch 慢 rank 160~210→110~125µs）。数据放置不变。
    const int wchunk = (count + nwarps - 1) / nwarps;
    const int wlo = warp * wchunk;
    const int whi = (wlo + wchunk < count) ? (wlo + wchunk) : count;
    for (int s2 = wlo + half; s2 < whi; s2 += 2) {
        const int global_row = s_row_off + s2;
        int src = 0;
        while (src + 1 < S.R && pfx[src + 1] <= global_row) ++src;
        const int slot_in_rank = global_row - pfx[src];
        const uint32_t encoded =
            src_tbl[(size_t)(e * S.R + src) * S.tbl_cap + slot_in_rank];
        const int src_token = (int)(encoded / (uint32_t)S.topk);
        const bool valid = src_token >= 0 && src_token < S.tbl_cap;
        uint4* dstp = reinterpret_cast<uint4*>(
            pool + ((size_t)pool_slot * S.pool_stride + s2) * S.k_in);
        if (valid) {
            const uint4* srcp = reinterpret_cast<const uint4*>(
                                    S.peer_bases[src] + S.x_off) +
                                (size_t)src_token * (S.k_in / 16) + u4_off;
            // [O2] 权重远端读先于整行搬运发出：与搬运 store 并行，
            // 避免原顺序（先搬运后读权重）把远端读延迟暴露在任务尾。
            float wv = 0.0f;
            if (ll == 0)
                wv = *(reinterpret_cast<const float*>(
                           S.peer_bases[src] + S.weights_off) + encoded);
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
            if (ll == 0) {
                wt[(size_t)blockIdx.x * S.win_cap + s2] = wv;
                meta[((size_t)blockIdx.x * S.win_cap + s2) * 3 + 0] = src;
                meta[((size_t)blockIdx.x * S.win_cap + s2) * 3 + 1] = src_token;
                meta[((size_t)blockIdx.x * S.win_cap + s2) * 3 + 2] =
                    (int)(encoded % (uint32_t)S.topk);
            }
        } else {
            // 非法行零填充（本 K 片 + staging），保证确定性与 COMBINE 跳过
            const uint4 z4{0u, 0u, 0u, 0u};
            for (int i = ll; i < u4_slice; i += 32) {
                if (!kSkipPool) dstp[u4_off + i] = z4;
                if (kAfrag)
                    r19_store_a_frag(frag_base, s2, (u4_off + i) * 16, z4,
                                     mt_total);
            }
            if (ll == 0) {
                wt[(size_t)blockIdx.x * S.win_cap + s2] = 0.0f;
                meta[((size_t)blockIdx.x * S.win_cap + s2) * 3 + 0] = 0;
                meta[((size_t)blockIdx.x * S.win_cap + s2) * 3 + 1] = 0;
                meta[((size_t)blockIdx.x * S.win_cap + s2) * 3 + 2] = 0;
            }
        }
    }
    __threadfence();
    __syncthreads();
    // ---- 分片就绪旗（本 slice 的 K 片已落 pool；L1 消费）----
    if (tid == 0) {
        __threadfence();
        // 同设备分片旗：普通写 + fence（__stwt 在普通显存上不更新 L2）
        // [R36b] flag 粒度=物理 slice（nslice_eff）：粗化后 stride≤4，区容量
        //   EPR×ring×nslice_eff ≤ 256 恒成立（不再随 kNSlice 放大）；
        //   kNSlice=16 因此无需改布局常量即可用。
        *reinterpret_cast<volatile uint32_t*>(
            S.bar_flags + S.slice_flags_idx +
            (size_t)pool_slot * S.nslice_eff + slice) = epoch;
        __threadfence();
    }
    __syncthreads();
    // 诊断：block0 首任务 DISPATCH 完成（分片旗已写）
    if (kTick) S.pt[4] = clock64();
    // ---- L1 入口 wait-all：一次等齐本窗全部分片就绪旗 ----
    // 骨架结论：packed+配对下 wait-all 优于 K 循环内每步分片门；任务按窗口连续
    // 编号，同窗全部 slice 旗由本窗 nslice_eff 个物理任务写满。
    if (count > 0 && tid == 0) {
        for (int s2 = 0; s2 < S.nslice_eff; ++s2) {
            if (s2 == slice) continue;
            long long spin = 0;
            // [集71] wait-all（分片就绪旗）等待轮次
            MX_CTR_ADD(S.ctr, kCtrWaitAllSlice, 1);
            while (*(volatile const uint32_t*)(S.bar_flags + S.slice_flags_idx +
                                               (size_t)pool_slot *
                                                   S.nslice_eff +
                                               s2) != epoch) {
                if (++spin > S.max_spins) {
                    if (S.dbg != nullptr) atomicAdd(S.dbg + 4, 1);
                    break;
                }
            }
        }
        __threadfence();
    }
    __syncthreads();
}

__device__ __forceinline__ void r19_l1_pub_wait(
        const R19Sched& S, int task, int count, int slice, unsigned epoch,
        int pool_slot) {
    const int tid = threadIdx.x;
    const bool kTick = (S.pt != nullptr) && (blockIdx.x == 0) && (task == 0) &&
                       (tid == 0);
    __syncthreads();
    // ---- l1_prog 发布 + L1→L2 依赖（本窗全部 NSLICE 个旗）----
    if (tid == 0) {
        __threadfence();
        // [R36b] 同 slice 旗：flag 粒度=物理 slice（stride=nslice_eff）
        l1_progress_publish(S.bar_flags, (int)S.l1_prog_idx, pool_slot, slice,
                            epoch, (int)S.nslice_eff);
        for (int s2 = 0; s2 < S.nslice_eff; ++s2) {
            if (s2 == slice) continue;
            long long spin = 0;
            // [集71] wait-all（l1_prog 旗）等待轮次
            MX_CTR_ADD(S.ctr, kCtrWaitAllL1Prog, 1);
            while (*(volatile const uint32_t*)(S.bar_flags + S.l1_prog_idx +
                                               (size_t)pool_slot *
                                                   S.nslice_eff +
                                               s2) != epoch) {
                if (++spin > S.max_spins) {
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

template <typename TaskTail>
__device__ __forceinline__ void r19_tail_common(
        const R19Sched& S, TaskTail tail, int task, int e, int w, int count,
        int slice, unsigned epoch, int pool_slot) {
    (void)S;
    TaskDesc td;
    td.task = task;
    td.e = e;
    td.w = w;
    td.slice = slice;
    td.count = count;
    td.pool_slot = pool_slot;
    td.epoch = epoch;
    td.nslice_eff = S.nslice_eff;   // [R35]
    td.merge = S.merge;
    tail(td);
    __syncthreads();
}

// ============================================================================
// 大 count 任务体（__noinline__ 独立寄存器分配）
//   自包含：pull → 分片旗 → wait-all → L1（M 档位）→ l1_prog 发布/等待
//   → L2（M 档位）→ tail。调用者（dispatch_pull_body 循环）只做领号/解码与
//   任务尾 l2_done/processed 计数（与基线逐字一致）。
// ============================================================================
template <typename TaskTail, bool kFragCap = false, int kCap = 0>
__device__ R19_BIG_ATTR void r19_big_task(
        const R19Sched& S, int task, int e, int w, int count, int slice,
        unsigned epoch, int pool_slot, const int* pfx, TaskTail tail,
        uint8_t* a_smem, uint8_t* b_smem) {
    const int tid = threadIdx.x;
    const int warp = tid / 64;
    const int lane = tid % 64;
    // [OPT-TK-CAP] 编译期任务 count 上界（0=不限）：win_cap<=kCap 时相应重档
    //   不可达，用 if constexpr 删除其模板实例以缩寄存器/栈。
    constexpr bool kCap64 = (kCap > 0);
    constexpr bool kCap48 = (kCap > 0 && kCap <= 48);
    // [R38] tk/长尾实例（kCap>0）用更深 B 预取环（R24_BPF=4）：这些实例
    //   count<=64、寄存器预算有裕量；均衡大档（kCap=0）保持 2，避免 TPR=256 回退。
    constexpr int kR24BPF = (kCap > 0) ? 4 : R24_BPF;
    (void)a_smem;  // [R24] kernel 作用域 A-smem（由调用方声明；小任务不用）
    // [R29] 融合推送上下文（capable=tail 支持；on 在本任务 L2 段按条件激活）
    CombineFuseCtx fuse = R29TailFuse<TaskTail>::get(tail);
    r19_pull_flags(S, task, e, w, count, slice, epoch, pool_slot, pfx);
#if R24_DIAG_PHASE
    const unsigned long long t_l1s = (tid == 0) ? clock64() : 0ull;
#endif
    // ---- L1：M 档位（kMT 个 16 行 tile 共享 B 一遍）----
    const uint8_t* pool = const_cast<uint8_t*>(S.self) + S.pool_off;
    const uint8_t* pool_w = pool + (size_t)pool_slot * S.pool_stride * S.k_in;
    const int g_l1 = (S.l1_n_total / S.l1_block_n) / S.nslice_eff;
    const int g0_l1 = slice * g_l1;
    // [集71] L1 阶段执行次数（count>0 的任务，tid0 计一次）
    if (tid == 0 && count > 0) MX_CTR_ADD(S.ctr, kCtrL1Exec, 1);
    if (CM_L1_GATE) {
        uint8_t* ffn_slot = (S.ffn_hidden != nullptr)
            ? S.ffn_hidden + (size_t)pool_slot * S.win_cap * S.l1_k_out
            : nullptr;
        const int out_col_base = slice * (S.l1_k_out / S.nslice_eff);
        // [R21] A fragment 副本（单波大任务）；[OPT-AFRAG] 多窗口走专用缓冲
        // [R34] count<=32 不用 F/F2（小档 runtime M 裁剪，行主序直接算）
        const bool kFragOld = S.afrag_ok && count > 32 &&
                              S.ffn_hidden != nullptr;
        const bool kFragNew = kFragCap && S.frag_all && count > 0;
        const uint8_t* fbase = nullptr;
        uint8_t* ffn2 = nullptr;
        if (kFragOld) {
            fbase = r19_afrag_base(S, e);
            ffn2 = S.ffn_hidden +
                   ((size_t)e * S.ring + S.f2_slot) * S.win_cap * S.l1_k_out;
        } else if (kFragNew) {
            const size_t asz =
                (size_t)(S.k_in / 32) * (size_t)S.mt_total * 512;
            const size_t f2sz =
                (size_t)(S.l1_k_out / 32) * (size_t)S.mt_total * 512;
            const size_t soff = (size_t)(e * S.ring + (w % S.ring));
            if (CM_FRAG_A) fbase = S.frag_a + soff * asz;
            if (CM_FRAG_L2 && CM_FRAG_F2) ffn2 = S.frag2 + soff * f2sz;
        }
        // 行主序 ffn_hidden 仅 bench 数值校验读 slot0；fragment 时其余槽只写 F2
        if ((kFragOld || (kFragNew && CM_FRAG_L2)) && pool_slot != 0)
            ffn_slot = nullptr;
        // [R24/R34] A-smem 路径：F 可用 + kKS 整除 + 单批（g_l1==kBatch）。
        // [R34] 去掉 count%(MT*16)==0 门控：count>64 时走"全块 + 尾块"两段式
        // （尾块按 row<cc 掩码），权重遍数 = ceil(count/(MT*16))（旧整段 MT4
        // 回退为 ceil(count/64)）。count<=64 时权重同为 1 遍而 async 多一遍
        // stage 同步且尾块掩码有开销，实测更慢 → 保持非 async 回退。
        const bool kAsync = (fbase != nullptr) && (count > 64) &&
                            ((S.k_in / 32) % R24_KS == 0) &&
                            (g_l1 % R19_L1_BATCH == 0);
        if (count <= 32 && g_l1 % 4 == 0) {
            // [R34] count∈(kM,32]：小档同构（runtime M 裁剪，空 tile 不参与
            //   MMA/A 装载；单批 N 组）。旧 MT4/B4 对 17 行空转 4 tile(64 行)
            //   且 L2 三批 A，是 TPR=17 L1/L2 各近翻倍的主因。
#if CM_USE_BSM
            l1_gemm_swiglu_body_bsm<2, CM_L1_BATCH, CM_BR>(
                b_smem, pool_w, S.w1, e, count, S.k_in, S.l1_n_total,
                S.l1_block_n, lane, warp, kComputeWarps, S.l1_chk, g0_l1,
                g_l1, ffn_slot, out_col_base, S.l1_k_out);
#else
            if constexpr (kFragCap) {
                if (fbase != nullptr)
                    l1_gemm_swiglu_body_paired<CM_L1_PF, kNTilesPerWarp, 2,
                                               CM_L1_BATCH, true>(
                        pool_w, S.w1, e, count, S.k_in, S.l1_n_total,
                        S.l1_block_n, lane, warp, kComputeWarps, S.l1_chk,
                        g0_l1, g_l1, ffn_slot, out_col_base, S.l1_k_out, nullptr,
                        fbase, S.mt_total, ffn2);
                else
                    l1_gemm_swiglu_body_paired<CM_L1_PF, kNTilesPerWarp, 2,
                                               CM_L1_BATCH, false>(
                        pool_w, S.w1, e, count, S.k_in, S.l1_n_total,
                        S.l1_block_n, lane, warp, kComputeWarps, S.l1_chk,
                        g0_l1, g_l1, ffn_slot, out_col_base, S.l1_k_out, nullptr,
                        nullptr, 0, ffn2);
            } else {
                l1_gemm_swiglu_body_paired<CM_L1_PF, kNTilesPerWarp, 2,
                                           CM_L1_BATCH, false>(
                    pool_w, S.w1, e, count, S.k_in, S.l1_n_total, S.l1_block_n,
                    lane, warp, kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot,
                    out_col_base, S.l1_k_out, nullptr, nullptr, 0, ffn2);
            }
#endif
        } else if (count <= 32 && g_l1 % 2 == 0 && g_l1 >= 2) {
            // [R35] NSLICE=8 等 g_l1=2 几何：单批 2 组小档（A 装载一遍、MT2
            //   runtime 裁剪），与 NSLICE=4 的 <2,kNT,2,4> 同构按比例缩小；
            //   避免落入 r19_l1_big<4,...> 的 MT4 档（A 读 2× + M 空转）。
            l1_gemm_swiglu_body_paired<2, kNTilesPerWarp, 2, 2>(
                pool_w, S.w1, e, count, S.k_in, S.l1_n_total, S.l1_block_n,
                lane, warp, kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot,
                out_col_base, S.l1_k_out, nullptr);
        } else if (count <= 48 && g_l1 % 4 == 0) {
            // [OPT-33] 33..48 档：MT3（恰好 ceil(count/16)=3 个 tile），acc 比
            //   MT4 小（[4][3][2] → [4][3][2] 不变，但省 1 个 M tile 的寄存器/
            //   发射槽），减少尾块 M 空转（原 MT4 对 33..48 仍按 4 tile 分配）。
            bool med_r24 = false;
#if CM_MED_R24
            if (fbase != nullptr && ((S.k_in / 32) % 16 == 0)) {
                r24_l1_big<3, 16, CM_L1_BATCH, true, 3, kR24BPF>(
                    a_smem, fbase, S.mt_total, S.w1, e, 0, count, S.k_in,
                    S.l1_n_total, S.l1_block_n, lane, warp, kComputeWarps,
                    S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2, out_col_base,
                    S.l1_k_out);
                med_r24 = true;
            }
#endif
            if (!med_r24) {
#if CM_USE_BSM
            l1_gemm_swiglu_body_bsm<3, CM_L1_BATCH, CM_BR>(
                b_smem, pool_w, S.w1, e, count, S.k_in, S.l1_n_total,
                S.l1_block_n, lane, warp, kComputeWarps, S.l1_chk, g0_l1,
                g_l1, ffn_slot, out_col_base, S.l1_k_out);
#else
            if constexpr (kFragCap) {
                if (fbase != nullptr)
                    l1_gemm_swiglu_body_paired<CM_L1_PF, kNTilesPerWarp, 3,
                                               CM_L1_BATCH, true>(
                        pool_w, S.w1, e, count, S.k_in, S.l1_n_total,
                        S.l1_block_n, lane, warp, kComputeWarps, S.l1_chk,
                        g0_l1, g_l1, ffn_slot, out_col_base, S.l1_k_out, nullptr,
                        fbase, S.mt_total, ffn2);
                else
                    l1_gemm_swiglu_body_paired<CM_L1_PF, kNTilesPerWarp, 3,
                                               CM_L1_BATCH, false>(
                        pool_w, S.w1, e, count, S.k_in, S.l1_n_total,
                        S.l1_block_n, lane, warp, kComputeWarps, S.l1_chk,
                        g0_l1, g_l1, ffn_slot, out_col_base, S.l1_k_out, nullptr,
                        nullptr, 0, ffn2);
            } else {
                l1_gemm_swiglu_body_paired<CM_L1_PF, kNTilesPerWarp, 3,
                                           CM_L1_BATCH, false>(
                    pool_w, S.w1, e, count, S.k_in, S.l1_n_total, S.l1_block_n,
                    lane, warp, kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot,
                    out_col_base, S.l1_k_out, nullptr, nullptr, 0, ffn2);
            }
#endif
            }
        } else if (count <= 48 && g_l1 % 2 == 0 && g_l1 >= 2) {
            // [OPT-NS8] NSLICE=8 时 g_l1=2：33..48 档用 MT3/B2（A 装一遍、
            //   runtime M 裁剪），避免落入 r19_l1_big<4,...> 的 MT4 慢档。
            l1_gemm_swiglu_body_paired<2, kNTilesPerWarp, 3, 2>(
                pool_w, S.w1, e, count, S.k_in, S.l1_n_total, S.l1_block_n,
                lane, warp, kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot,
                out_col_base, S.l1_k_out, nullptr);
        } else if (count > 48 && count <= 64 && fbase != nullptr &&
                   (count % (4 * kM) == 0) &&
                   ((S.k_in / 32) % R24_KS == 0)) {
            // [OPT-64] 49..64 档改走 r24 A-smem（MT4 无掩码）：A 经 bsm 异步搬运
            r24_l1_big<4, R24_KS, R19_L1_BATCH, false, 4, kR24BPF>(
                a_smem, fbase, S.mt_total, S.w1, e, 0, count, S.k_in,
                S.l1_n_total, S.l1_block_n, lane, warp, kComputeWarps,
                S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2, out_col_base,
                S.l1_k_out);
        } else if constexpr (kCap48) {
            // [OPT-TK-CAP48] win_cap<=48：count<=48，上述小/中档分支对生产
            //   g_l1 几何恒命中；此处仅为兜底（保持正确），用已实例化的 paired
            //   MT3 替换 r19_l1_big<4> 重档 → 不实例化大 GEMM（省寄存器/栈）。
            l1_gemm_swiglu_body_paired<CM_L1_PF, kNTilesPerWarp, 3, CM_L1_BATCH,
                                       false>(
                pool_w, S.w1, e, count, S.k_in, S.l1_n_total, S.l1_block_n, lane,
                warp, kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot,
                out_col_base, S.l1_k_out, nullptr, nullptr, 0, ffn2);
        } else if constexpr (kCap64) {
            // [OPT-TK-CAP64] cap64 变体：count<=64，MT8 异步档不可达 → 直接
            //   走固定 MT4 回退；if constexpr 保证不实例化 MT8 模板（不占栈）。
            r19_l1_big<4, R19_L1_PF, 4>(
                pool_w, fbase, S.mt_total, S.w1, e, count, S.k_in,
                S.l1_n_total, S.l1_block_n, lane, warp, kComputeWarps, S.l1_chk,
                g0_l1, g_l1, ffn_slot, ffn2, out_col_base, S.l1_k_out);
        } else if (kAsync) {
            // [R34] 全块（无掩码，原 async 快路径）+ 尾块（kMask=true）两段式：
            //   尾块 A 读取折叠到 F 最后一 tile、写/checksum 按 row<cc 掩码，
            //   权重遍数 = ceil(count/(MT*16))（旧整段 MT4 回退为 ceil(count/64)）
            const int full = count - count % (R19_L1_MT * kM);
            if (full > 0)
                r24_l1_big<R19_L1_MT, R24_KS, R19_L1_BATCH, false, R19_L1_MT, kR24BPF>(
                    a_smem, fbase, S.mt_total, S.w1, e, 0, full, S.k_in,
                    S.l1_n_total, S.l1_block_n, lane, warp, kComputeWarps,
                    S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2, out_col_base,
                    S.l1_k_out);
            if (full < count) {
                // [R34] 尾块只占 1 tile（r<=16，TPR=129/257/513 皆 r=1）时用
                //   MT1/KS16 小档：A/MMA 降到 1/8（B 遍数不变），并省一半 stage
                //   同步；几何不匹配时回退 MT8 掩码档
                const bool kTiny = (count - full <= kM) &&
                                   ((S.k_in / 32) % 16 == 0);
                // [R36] 尾块活跃 tile 数（编译期裁剪 A/MMA/epilogue）
                const int l1_act = (count - full + kM - 1) / kM;
                if (kTiny)
                    r24_l1_big<1, 16, R19_L1_BATCH, true, 1, kR24BPF>(
                        a_smem, fbase, S.mt_total, S.w1, e, full, count,
                        S.k_in, S.l1_n_total, S.l1_block_n, lane, warp,
                        kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2,
                        out_col_base, S.l1_k_out);
                else if (l1_act == 5)
                    r24_l1_big<R19_L1_MT, R24_KS, R19_L1_BATCH, true, 5, kR24BPF>(
                        a_smem, fbase, S.mt_total, S.w1, e, full, count,
                        S.k_in, S.l1_n_total, S.l1_block_n, lane, warp,
                        kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2,
                        out_col_base, S.l1_k_out);
                else
                    r24_l1_big<R19_L1_MT, R24_KS, R19_L1_BATCH, true, R19_L1_MT, kR24BPF>(
                        a_smem, fbase, S.mt_total, S.w1, e, full, count,
                        S.k_in, S.l1_n_total, S.l1_block_n, lane, warp,
                        kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot, ffn2,
                        out_col_base, S.l1_k_out);
            }
        } else
            // [R27·delivered] 非 async 回退固定 MT4/B4：避免与 async 的 MT8 档
            // 叠成两份大 acc 实例（同一 noinline 任务体寄存器/栈按最坏路径分配，
            // MT8 全档回退实测栈 1880B→2688B、71 机 TPR=128 回归 +4.7%；固定
            // MT4 回退后栈 2072B、TPR=128 回归收窄至 +2.8%）。
            r19_l1_big<4, R19_L1_PF, 4>(
                pool_w, fbase, S.mt_total, S.w1, e, count, S.k_in,
                S.l1_n_total, S.l1_block_n, lane, warp, kComputeWarps, S.l1_chk,
                g0_l1, g_l1, ffn_slot, ffn2, out_col_base, S.l1_k_out);
    }
#if R24_DIAG_PHASE
    if (tid == 0)
        atomicAdd(S.pt + (size_t)kGrid * 8 + (size_t)blockIdx.x * 8,
                  clock64() - t_l1s);
#endif
    r19_l1_pub_wait(S, task, count, slice, epoch, pool_slot);
#if R24_DIAG_PHASE
    const unsigned long long t_l2s = (tid == 0) ? clock64() : 0ull;
#endif
    // ---- L2：M 档位（K=I、N=H），BF16 写每块 staging 槽 ----
    const int g_l2 = (S.l2_k_out / S.l1_block_n) / S.nslice_eff;
    const int g0_l2 = slice * g_l2;
    // [集71] L2 阶段执行次数（count>0 的任务，tid0 计一次）
    if (tid == 0 && count > 0) MX_CTR_ADD(S.ctr, kCtrL2Exec, 1);
    if (CM_L2_GATE) {
        const uint8_t* ffn_slot =
            (S.ffn_hidden != nullptr)
                ? S.ffn_hidden + (size_t)pool_slot * S.win_cap * S.l1_k_out
                : nullptr;
        uint16_t* out_slot = (S.l2_out != nullptr)
            ? S.l2_out + (size_t)blockIdx.x * S.win_cap * S.l2_k_out
            : nullptr;
        const uint8_t* w2_e = S.w2 + (size_t)e * S.l2_k_out * S.l1_k_out;
        // [R21] L2 的 A 走 ffn_hidden 的 fragment 副本（同一单波大任务条件）
        const uint8_t* f2 = nullptr;
        if (S.afrag_ok && count > 32 && S.ffn_hidden != nullptr) {
            f2 = S.ffn_hidden +
                 ((size_t)e * S.ring + S.f2_slot) * S.win_cap * S.l1_k_out;
        } else if (CM_FRAG_L2 && S.frag_all && count > 0) {
            const size_t f2sz =
                (size_t)(S.l1_k_out / 32) * (size_t)S.mt_total * 512;
            if (CM_FRAG_A2)
                f2 = S.frag2 + (size_t)(e * S.ring + (w % S.ring)) * f2sz;
        }
        // [R24/R34] L2 A-smem 路径（同 L1：count>64 才走，≤64 保持回退）
        // [R35] 去掉 g_l2%R19_L2_BATCH==0 门控：奇数 g_l2（NSLICE=8 → 24/8=3）
        //   由 r24_l2_big 的 kBatch=1 余量路径补齐，不再回退 MT4 慢路径。
        const bool kAsyncL2 = (f2 != nullptr) && (count > 64) &&
                              ((S.l1_k_out / 32) % R24_KS == 0) && (g_l2 >= 1);
        // [R29] 融合激活：仅当 tail 支持 + r24 路径 + staging 有效
#if R29_COMBINE_FUSE
        fuse.on = fuse.capable && kAsyncL2 && (out_slot != nullptr);
#else
        fuse.on = false;   // 回退：L2 不推送，任务尾走整段推送
#endif
        R29TailFuseSet<TaskTail>::on(tail, fuse);
        if (count <= 32 && g_l2 % 6 == 0) {
            // [R34] 同 L1：小档单批（A 只装一遍，B 一遍）
#if CM_FRAG_L2
            if constexpr (kFragCap) {
                if (f2 != nullptr)
                    l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 2, 6, true>(
                        ffn_slot, w2_e, count, S.l1_k_out, S.l2_k_out,
                        S.l1_block_n, lane, warp, kComputeWarps, S.l2_chk,
                        g0_l2, g_l2, out_slot, S.l2_k_out, f2, S.mt_total);
                else
                    l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 2, 6, false>(
                        ffn_slot, w2_e, count, S.l1_k_out, S.l2_k_out,
                        S.l1_block_n, lane, warp, kComputeWarps, S.l2_chk,
                        g0_l2, g_l2, out_slot, S.l2_k_out, nullptr, 0);
            } else
#endif
                l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 2, 6, false>(
                    ffn_slot, w2_e, count, S.l1_k_out, S.l2_k_out, S.l1_block_n,
                    lane, warp, kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                    S.l2_k_out, nullptr, 0);
        } else if (count <= 32 && g_l2 % 3 == 0 && g_l2 >= 3) {
            // [R35] NSLICE=8 等 g_l2=3 几何：单批 3 组小档（A 一遍、MT2 裁剪），
            //   避免落入 r19_l2_big<4,PF,3> 的 MT4 档（M 空转）。
            l2_gemm_body<2, kNTilesPerWarp, 2, 3>(
                ffn_slot, w2_e, count, S.l1_k_out, S.l2_k_out, S.l1_block_n,
                lane, warp, kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                S.l2_k_out);
        } else if (count <= 48 && g_l2 % 6 == 0) {
            // [OPT-33] 33..48 档：MT3 单批（ceil(count/16)=3 tile），acc[3][3][2]；
            //   原 MT4/3 分两批（acc[3][4][2]=24）→ MT3 更小且只一次调用。
            bool med_r24 = false;
#if CM_MED_R24
            if (f2 != nullptr && ((S.l1_k_out / 32) % 16 == 0)) {
                r24_l2_big<3, 16, 3, true, 3, kR24BPF>(
                    a_smem, f2, S.mt_total, w2_e, 0, count, S.l1_k_out,
                    S.l2_k_out, S.l1_block_n, lane, warp, kComputeWarps,
                    S.l2_chk, g0_l2, g_l2, out_slot, S.l2_k_out, nullptr);
                med_r24 = true;
            }
#endif
            if (!med_r24) {
#if CM_FRAG_L2
            if constexpr (kFragCap) {
                if (f2 != nullptr)
                    l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 3, 3, true>(
                        ffn_slot, w2_e, count, S.l1_k_out, S.l2_k_out,
                        S.l1_block_n, lane, warp, kComputeWarps, S.l2_chk,
                        g0_l2, g_l2, out_slot, S.l2_k_out, f2, S.mt_total);
                else
                    l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 3, 3, false>(
                        ffn_slot, w2_e, count, S.l1_k_out, S.l2_k_out,
                        S.l1_block_n, lane, warp, kComputeWarps, S.l2_chk,
                        g0_l2, g_l2, out_slot, S.l2_k_out, nullptr, 0);
            } else
#endif
                l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 3, 3, false>(
                    ffn_slot, w2_e, count, S.l1_k_out, S.l2_k_out, S.l1_block_n,
                    lane, warp, kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                    S.l2_k_out, nullptr, 0);
            }
        } else if (count > 48 && count <= 64 && f2 != nullptr &&
                   (count % (4 * kM) == 0) &&
                   ((S.l1_k_out / 32) % R24_KS == 0)) {
            // [OPT-64] 49..64 档 L2 同样走 r24 A-smem（MT4 无掩码）
            r24_l2_big<4, R24_KS, 3, false, 4, kR24BPF>(
                a_smem, f2, S.mt_total, w2_e, 0, count, S.l1_k_out,
                S.l2_k_out, S.l1_block_n, lane, warp, kComputeWarps,
                S.l2_chk, g0_l2, g_l2, out_slot, S.l2_k_out, nullptr);
        } else if constexpr (kCap48) {
            // [OPT-TK-CAP48] 兜底：count<=48，用已实例化的 paired MT3 替换
            //   r19_l2_big<4> 重档（不实例化大 GEMM）。
            l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 3, 3, false>(
                ffn_slot, w2_e, count, S.l1_k_out, S.l2_k_out, S.l1_block_n, lane,
                warp, kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot, S.l2_k_out,
                nullptr, 0);
        } else if constexpr (kCap64) {
            // [OPT-TK-CAP64] cap64 变体：count<=64，MT8 异步档不可达 → 固定
            //   MT4/B3，if constexpr 保证不实例化 MT8 模板。
            r19_l2_big<4, R19_L2_PF, 3>(
                ffn_slot, f2, S.mt_total, w2_e, count, S.l1_k_out,
                S.l2_k_out, S.l1_block_n, lane, warp, kComputeWarps, S.l2_chk,
                g0_l2, g_l2, out_slot, S.l2_k_out);
        } else if (kAsyncL2) {
            // [R34] 同 L1：全块 + 尾块（尾块掩码；融合推送对两段都生效）
            const int full = count - count % (R19_L2_MT * kM);
            if (full > 0)
                r24_l2_big<R19_L2_MT, R24_KS, R19_L2_BATCH, false, R19_L2_MT, kR24BPF>(
                    a_smem, f2, S.mt_total, w2_e, 0, full, S.l1_k_out,
                    S.l2_k_out, S.l1_block_n, lane, warp, kComputeWarps,
                    S.l2_chk, g0_l2, g_l2, out_slot, S.l2_k_out,
                    fuse.on ? &fuse : nullptr);
            if (full < count) {
                const bool kTiny = (count - full <= kM) &&
                                   ((S.l1_k_out / 32) % 16 == 0);
                const int l2_act = (count - full + kM - 1) / kM;
                if (kTiny)
                    r24_l2_big<1, 16, R19_L2_BATCH, true, 1, kR24BPF>(
                        a_smem, f2, S.mt_total, w2_e, full, count,
                        S.l1_k_out, S.l2_k_out, S.l1_block_n, lane, warp,
                        kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                        S.l2_k_out, fuse.on ? &fuse : nullptr);
                else if (l2_act == 5)
                    r24_l2_big<R19_L2_MT, R24_KS, R19_L2_BATCH, true, 5, kR24BPF>(
                        a_smem, f2, S.mt_total, w2_e, full, count,
                        S.l1_k_out, S.l2_k_out, S.l1_block_n, lane, warp,
                        kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                        S.l2_k_out, fuse.on ? &fuse : nullptr);
                else
                    r24_l2_big<R19_L2_MT, R24_KS, R19_L2_BATCH, true, R19_L2_MT, kR24BPF>(
                        a_smem, f2, S.mt_total, w2_e, full, count,
                        S.l1_k_out, S.l2_k_out, S.l1_block_n, lane, warp,
                        kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                        S.l2_k_out, fuse.on ? &fuse : nullptr);
            }
        } else
            // [R27·delivered] 同上：回退固定 MT4/B3
            r19_l2_big<4, R19_L2_PF, 3>(
                ffn_slot, f2, S.mt_total, w2_e, count, S.l1_k_out,
                S.l2_k_out, S.l1_block_n, lane, warp, kComputeWarps, S.l2_chk,
                g0_l2, g_l2, out_slot, S.l2_k_out);
    }
    __syncthreads();
#if R24_DIAG_PHASE
    if (tid == 0)
        atomicAdd(S.pt + (size_t)kGrid * 8 + (size_t)blockIdx.x * 8 + 1,
                  clock64() - t_l2s);
#endif
    r19_tail_common(S, tail, task, e, w, count, slice, epoch, pool_slot);
}

// ============================================================================
// 小 count（count<=kM）任务体：与基线逐字一致的热路径，独立 __noinline__
// 声明以隔离大任务的寄存器分配（探针：空体调用点对 TPR=16 中性）。
// ============================================================================
template <typename TaskTail, bool kFragCap = false>
__device__ R19_SMALL_ATTR void r19_small_task(
        const R19Sched& S, int task, int e, int w, int count, int slice,
        unsigned epoch, int pool_slot, const int* pfx, TaskTail tail,
        uint8_t* a_smem, uint8_t* b_smem) {
    const int tid = threadIdx.x;
    const int warp = tid / 64;
    const int lane = tid % 64;
    (void)b_smem;
    r19_pull_flags(S, task, e, w, count, slice, epoch, pool_slot, pfx);
    // ---- L1：小 count 档位（与基线实例选择一致）----
    const uint8_t* pool = const_cast<uint8_t*>(S.self) + S.pool_off;
    const uint8_t* pool_w = pool + (size_t)pool_slot * S.pool_stride * S.k_in;
    const int g_l1 = (S.l1_n_total / S.l1_block_n) / S.nslice_eff;
    const int g0_l1 = slice * g_l1;
    // [集71] L1 阶段执行次数（count>0 的任务，tid0 计一次）
    if (tid == 0 && count > 0) MX_CTR_ADD(S.ctr, kCtrL1Exec, 1);
    if (CM_L1_GATE) {
        uint8_t* ffn_slot = (S.ffn_hidden != nullptr)
            ? S.ffn_hidden + (size_t)pool_slot * S.win_cap * S.l1_k_out
            : nullptr;
        const int out_col_base = slice * (S.l1_k_out / S.nslice_eff);
        // 均衡场景 g_l1=4：一次 K 循环处理全部 4 组（A 装载减半）；
        // kPrefetch=2 是为把 32 个 B fragment 槽控制在寄存器预算内。
        if (g_l1 % 4 == 0) {
#if CM_SMALL_BSM
            // [OPT-SMALL-BSM] B 经 smem 分级异步流水（kBR 级），A 仍直装。
            //   提前 step 数 = kBR；smem = kBR*kBatch*8KB。函数内部按批宽
            //   循环覆盖 g_l1 组（批宽为偶，gate/up 配对）。
            l1_gemm_swiglu_body_bsm<1, CM_SMALL_BATCH, CM_SMALL_BR>(
                b_smem, pool_w, S.w1, e, count, S.k_in, S.l1_n_total,
                S.l1_block_n, lane, warp, kComputeWarps, S.l1_chk, g0_l1, g_l1,
                ffn_slot, out_col_base, S.l1_k_out
#ifdef CM_SMALL_BSM_PROF
                , S.pt + (size_t)kGrid * 8 + (size_t)blockIdx.x * 8 + 2
#endif
            );
#else
            l1_gemm_swiglu_body_paired<CM_L1_PF, kNTilesPerWarp, 1, CM_L1_BATCH>(
                pool_w, S.w1, e, count, S.k_in, S.l1_n_total, S.l1_block_n,
                lane, warp, kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot,
                out_col_base, S.l1_k_out, nullptr);
#endif
        } else
            l1_gemm_swiglu_body_paired<4, kNTilesPerWarp, 1, 2>(
                pool_w, S.w1, e, count, S.k_in, S.l1_n_total, S.l1_block_n,
                lane, warp, kComputeWarps, S.l1_chk, g0_l1, g_l1, ffn_slot,
                out_col_base, S.l1_k_out, nullptr);
    }
    r19_l1_pub_wait(S, task, count, slice, epoch, pool_slot);
    // ---- L2：小 count（与基线实例选择一致）----
    const int g_l2 = (S.l2_k_out / S.l1_block_n) / S.nslice_eff;
    const int g0_l2 = slice * g_l2;
    // [集71] L2 阶段执行次数（count>0 的任务，tid0 计一次）
    if (tid == 0 && count > 0) MX_CTR_ADD(S.ctr, kCtrL2Exec, 1);
    if (CM_L2_GATE) {
        const uint8_t* ffn_slot =
            (S.ffn_hidden != nullptr)
                ? S.ffn_hidden + (size_t)pool_slot * S.win_cap * S.l1_k_out
                : nullptr;
        uint16_t* out_slot = (S.l2_out != nullptr)
            ? S.l2_out + (size_t)blockIdx.x * S.win_cap * S.l2_k_out
            : nullptr;
        const uint8_t* w2_e = S.w2 + (size_t)e * S.l2_k_out * S.l1_k_out;
        // 均衡场景 g_l2=6：[集33] 单批 <2,2,1,6>（A 只装一遍、K 循环一次；
        //   B 字节不变，acc 48 floats。71 机 TPR=16 −10µs，TPR=256 中性）。
        //   原 [O3] 3+3 两批保留为 g_l2%3==0 的回退几何（g_l2=6/12 优先走单批）。
        if (g_l2 % 6 == 0) {
            l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 1, 6>(
                ffn_slot, w2_e, count, S.l1_k_out, S.l2_k_out, S.l1_block_n,
                lane, warp, kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                S.l2_k_out);
        } else if (g_l2 % 4 == 0) {
            l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 1, 4>(
                ffn_slot, w2_e, count, S.l1_k_out, S.l2_k_out, S.l1_block_n,
                lane, warp, kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                S.l2_k_out);
        } else if (g_l2 % 3 == 0) {
            // [O3] g_l2 为 3 的倍数（6/12/…）→ 3+3 两批对称
            const int gb = g_l2 / 2;
            l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 1, 3>(
                ffn_slot, w2_e, count, S.l1_k_out, S.l2_k_out, S.l1_block_n,
                lane, warp, kComputeWarps, S.l2_chk, g0_l2, gb, out_slot,
                S.l2_k_out);
            l2_gemm_body<CM_L2_PF, kNTilesPerWarp, 1, 3>(
                ffn_slot, w2_e, count, S.l1_k_out, S.l2_k_out, S.l1_block_n,
                lane, warp, kComputeWarps, S.l2_chk, g0_l2 + gb, g_l2 - gb,
                out_slot, S.l2_k_out);
        } else {
            l2_gemm_body<4, kNTilesPerWarp, 1, 2>(
                ffn_slot, w2_e, count, S.l1_k_out, S.l2_k_out, S.l1_block_n,
                lane, warp, kComputeWarps, S.l2_chk, g0_l2, g_l2, out_slot,
                S.l2_k_out);
        }
    }
    __syncthreads();
    r19_tail_common(S, tail, task, e, w, count, slice, epoch, pool_slot);
}
