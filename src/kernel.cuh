// ============================================================================
// kernel.cuh — fp8xfp8 MoE Mega Kernel 设备侧入口与编排
//
// 单轮（kNodeReps=1）相位顺序：
//   PROLOGUE(ROUTE_BUILD + PUBLISH_COUNTS，仅 block0)
//   -> XBAR_ROUTE_READY（全 grid，mode3）
//   -> TASK_SETUP（block0 leader 建表，其余等 task_ready）
//   -> 任务循环 { DISPATCH -> L1 -> L2 -> tail(COMBINE) }（全 grid）
//   -> REDUCE（独立 kernel，每 token 2 block）
// ============================================================================
#pragma once

#include "common.cuh"
#include "flags.cuh"
#include "prologue.cuh"
#include "compute.cuh"
#include "combine.cuh"

// 空 kernel（M0 基准：测 launch/graph 地板）
// __launch_bounds__ 覆盖 bench 全部启动线程数（G0=32 / G1/G2=512 或 1024），
// 固定 AOT 编译上限，避免按实际 block 大小重复编译。
__global__ void __launch_bounds__(1024) mega_empty_kernel() {}

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

// 计算链任务循环（tail 类型由调用点按运行期开关选择）
// [FRAG-ALWAYS] 去掉 kFragCap：fragment（A/f2 4-line 快路径）分支恒编译。
// [FRAG-INT] fragment 缓冲改为 kernel 内部（rwk->ld.frag_a/frag2），不再由 host 传入；
//   运行期是否真走由 S.frag_all 决定（见 compute.cuh 的 S.frag_all）。
template <bool kSmallOnly = false, int kCap = 0,
          MegaPhase kPhase = MegaPhase::Both>
__device__ __forceinline__ void mega_dispatch_loop(const MegaArgs& a,
                                                   int cycle,
                                                   uint8_t* a_smem = nullptr,
                                                   uint8_t* b_smem = nullptr,
                                                   unsigned long long* ctr = nullptr) {
    dispatch_pull_body<kSmallOnly, kCap, kPhase>(
        a.self, &a.unk->rd, cycle, a.peer_bases, a.R, a.rank, a.win_cap,
        a.win_cap /*pool_stride*/, a.slot_capacity /*tbl_cap*/,
        kHidden /*k_in*/, kTopk /*topk*/, kRing /*ring*/, a.prefix_off,
        a.x_off, a.weights_off, a.pool_off, a.w_off, a.meta_off,
        a.pool_ptr, a.rwk->lf.bar[cycle],
        a.task_counter_idx, a.l2_done_idx, a.task_ready_idx, a.slice_flags_idx,
        a.l1_prog_idx, a.rwk->lf.out_processed, a.rwk->lf.dbg, a.max_spins, a.w1,
        2 * a.l1_k_out /*l1_n_total*/, kL1BlockN /*l1_block_n*/, a.l1_k_out,
        a.ffn_hidden, a.rwk->lf.l1_chk, a.w2,
        a.l2_out, a.l2_k_out, a.rwk->lf.l2_chk, /*rep=*/0, /*n_reps=*/1,
        a.combine_rows_off, a.unk_peers, (uint64_t)a.rwk->lf.current_time[0][1],
        a.prefix_ptr, a.rwk->lf.pre_ticks, a_smem, b_smem,
        a.rwk->ld.frag_a[0], a.rwk->ld.frag2[0], ctr);
}

// 主 kernel（单轮）。
//   kWithCompute=false：只跑前序（M1 验证）；true：完整任务循环（M2/M3）；
//   kWithCompute=true 且 combine_rows_off>=0：任务尾接 COMBINE（M3）
// [L1L2-SPLIT] kPhase：Both=原合并；L1=只跑前序+DISPATCH+L1；L2=只跑 L2+tail
//   （prologue 行为由 a.skip_prologue 驱动：L2 发射时 host 置 2 = 跳过
//   route/publish/xbar、轻量 task_setup）。
template <bool kWithCompute,
          bool kSmallOnly = false, int kCap = 0,
          MegaPhase kPhase = MegaPhase::Both>
__global__ void __launch_bounds__(512) mega_kernel(MegaArgs a) {
    // 相位时间戳（block0 tid0；[6]=入口 [0]=route后 [1]=publish后 [2]=xbar后 [3]=setup后）
    const bool kTick = (blockIdx.x == 0) && (threadIdx.x == 0);
    // [集71] 计数缓冲归一（运行期开关折叠成单指针判空；编译期关闭时恒 nullptr，
    //   该局部量无人读取即被优化器消除，不进入任何 kernel 参数/寄存器）
#if MX_DBG_COUNT
    unsigned long long* ctr = mx_ctr_open(a.dbg_count, a.exec_ctr);
#else
    unsigned long long* ctr = nullptr;
#endif
    // [集71] kernel 启动次数（block0 tid0）+ 参与 block 数（每 block tid0）
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) MX_CTR_ADD(ctr, kCtrKernelLaunch, 1);
        MX_CTR_ADD(ctr, kCtrKernelBlocks, 1);
    }
    // [R24] A-smem 双缓冲（kernel 作用域）；L1 B-smem 流水已删除。
    // dispatch 参数中的 b_smem 暂保留，供调用边界兼容，但始终为 nullptr。
    union __align__(16) ABShared {
        uint8_t a[2][R24_MT * R24_KS * 512];
    };
    __shared__ ABShared s_ab;
    if (kTick) a.rwk->lf.pre_ticks[6] = clock64();
    // 0) [M_CYCLE] 预清 ct2=(c+2)%M_CYCLE 槽（本轮 copy c 尚未定，用 current_time
    //   [b][0] 派生；入口全 block 同为 L ⇒ c 一致）。ct1 留给下一轮，本轮不碰。
    mx_preclear_ct2(&a.rwk->lf, &a.unk->rf, /*skip_xbar=*/a.skip_prologue >= 2);
    __syncthreads();
    if (!a.skip_prologue) {
        // 1) PROLOGUE：ROUTE_BUILD + PUBLISH_COUNTS（仅 block0）
        if (blockIdx.x == 0) {
            const int route_cycle =
                (int)(a.rwk->lf.current_time[0][0] % M_CYCLE);
            route_build_body(a.topk_idx, &a.rwk->ld.route_slot_counts[0],
                             a.unk_peers, route_cycle, a.R, a.rank,
                             a.tokens, a.slot_capacity, 0, 0, ctr);
            if (kTick) a.rwk->lf.pre_ticks[0] = clock64();
            publish_counts_body(&a.rwk->ld.route_slot_counts[0], a.unk_peers,
                                route_cycle, a.R, a.rank, kPublishEpoch,
                                kRecvStride, 0, ctr);
            if (kTick) a.rwk->lf.pre_ticks[1] = clock64();
        }
    } else {
        // [R38] 复用前序：路由/计数持久，跳过 ROUTE/PUBLISH（数据与 node0 相同）。
        if (kTick) a.rwk->lf.pre_ticks[0] = a.rwk->lf.pre_ticks[1] = clock64();
    }
    // 2) XBAR_ROUTE_READY（全 grid，mode3：转置 + warp 并行发布）：每节点保留，
    //    既是 route/publish 可见性门，也是跨 rank 的每节点对齐（防漂移）。
    if (a.skip_prologue >= 2) {
        // 诊断模式：连 xbar 也跳过，仅推进 xbar 相位计数。
        // [L1L2-SPLIT] L2 阶段：xbar 相位计数不由 L2 推进 —— 每 round 仅 L1 前序
        //   前进一次，使 ct0/ct1 单步同步（combine/REDUCE 的 cyc/epoch 对齐）。
        if constexpr (kPhase != MegaPhase::L2) {
            if (threadIdx.x == 0) a.rwk->lf.current_time[blockIdx.x][0] += 1;
        }
        __syncthreads();
    } else {
        xbar_route_ready_body(&a.unk->rf, a.unk_peers, &a.rwk->lf, a.R, a.rank,
                              a.xrelease_idx, a.rwk->lf.dbg, a.max_spins, 0, ctr);
    }
    if (kTick) a.rwk->lf.pre_ticks[2] = clock64();
    // 3) TASK_SETUP（block0 leader：checksum -> prefix -> 窗口表 -> task_ready）
    // [L1L2-SPLIT] L2 阶段：改用静态任务划分（见 dispatch_pull_body），
    //   不重臂 task_counter/task_ready、不推进 current_time[1]（每 round 仅 L1
    //   前序 +1）；prefix 由本 round L1 的 task_setup 持久化，L2 只读。
    if constexpr (kPhase == MegaPhase::L2) {
        if (kTick) a.rwk->lf.pre_ticks[3] = clock64();
    } else if (a.skip_prologue) {
        // [R38] 复用前序：prefix/路由/计数持久，仅轻量重臂 flag（省 checksum
        //   轮询与 prefix 重建）。
        task_setup_light_body(&a.rwk->lf, a.task_ready_idx, a.task_counter_idx,
                              a.l2_done_idx, a.max_spins, a.rwk->lf.dbg);
    } else {
        task_setup_body(a.self, &a.unk->rd, a.R, a.rank, a.slot_capacity,
                        a.prefix_off, &a.rwk->lf, a.task_ready_idx,
                        a.task_counter_idx, a.l2_done_idx,
                        a.rwk->lf.out_total, a.rwk->lf.dbg, a.max_spins,
                        kRecvStride, /*reset_ring=*/true, /*tid_base=*/0,
                        /*sb_cnt=*/nullptr, /*smem_off=*/0, a.prefix_ptr,
                        a.rwk->lf.ck_dbg, ctr);
    }
    if (kTick) a.rwk->lf.pre_ticks[3] = clock64();
    // 4) 任务循环（DISPATCH -> L1 -> L2 -> tail(COMBINE)）
    if (kWithCompute) {
        // [M_CYCLE] 任务循环 / COMBINE / REDUCE 用的 copy c：task_setup 结束时
        //   current_time[b][1] 已自增到 L+1，故 (ct-1)%M_CYCLE = L%M_CYCLE，与
        //   xbar/task_setup 所用 c 一致。COMBINE 完成标记与 REDUCE 等待须同片。
        const int cyc = (int)((a.rwk->lf.current_time[blockIdx.x][1] - 1) % M_CYCLE);
        // [集33] tail 实例统一：任务循环恒用 CombineTail，combine 关闭时
        //   （combine_rows_off<0）融合能力置 false 且 tail 为空操作。这样
        //   mega_kernel<..., NoTaskTail> 不再实例化第二份重型任务体
        //   （原 NoTaskTail 运行期从不执行，却占整核寄存器/代码布局 collateral；
        //   71 机 TPR=16 −7µs，TPR=256 中性）。
        mega_dispatch_loop<kSmallOnly, kCap, kPhase>(
            a, cyc, &s_ab.a[0][0], nullptr, ctr);
    }
    if (kTick) a.rwk->lf.pre_ticks[15] = clock64();
}

// ---------------------------------------------------------------- REDUCE --
// 独立 kernel：每 token 2 个 block 列分半（grid=32 覆盖 TPR=16）
// [集71] 末尾两个参数为带默认值的可选项：老调用点（bench.cu）不传即 nullptr，
//        编译期 MX_DBG_COUNT=0 时统计代码整体消除。
__global__ void __launch_bounds__(256) mega_reduce_kernel(
                                   const uint8_t* self, const ranks_data* remote,
                                   float* y, int R,
                                   int tokens,
                                   int64_t rows_off,
                                   int64_t topk_idx_off, int64_t weights_off,
                                   local_flags* lf, ranks_flags* rf,
                                   int64_t task_ready_idx,
                                   int max_spins
#if MX_DBG_COUNT
                                   ,
                                   unsigned long long* exec_ctr = nullptr,
                                   int dbg_count = 0
#endif
) {
    reduce_body(self, remote, y, R, tokens, rows_off,
                topk_idx_off, weights_off, lf, rf, task_ready_idx,
                max_spins
#if MX_DBG_COUNT
                , mx_ctr_open(dbg_count, exec_ctr)
#endif
    );
}

// =============================================================================
// [集71] host 侧接口：计数缓冲清零 / 打印
//
// 语义（重要，配合 CUDA Graph）：
//   1) mx_count_enabled()：读运行期 env MXKERNEL_COUNT（"1"/非 0 即开），
//      host 侧填入 MegaArgs::dbg_count。
//   2) mx_count_alloc(dev_ptr)：分配 [kExecCtrSlots] 个 unsigned long long。
//   3) mx_count_clear(dev_ptr)：**必须在 graph capture 之前、在捕获流之外**
//      调用。capture 流内做依赖 host 同步（mcMemcpy）的清零会破坏捕获合法性；
//      若需要「每次 replay 从 0 开始」，正确做法是在 capture 之前清零，
//      接受跨 replay 累积，或用「清零 -> capture -> 立刻 replay 一次 -> 读回」
//      的方式把捕获流与清零解耦。
//   4) mx_count_print(dev_ptr, tag)：同步回读并按中文标签打印所有槽位。
// 编译期关闭（MX_DBG_COUNT=0）时以下函数仍可编译（buf 为空），但 device 侧
// 不会写任何计数，打印结果全 0；这样 bench.cu 侧零改动即可链接。
// =============================================================================
#include <mcr/mc_runtime.h>

#include <cstdio>
#include <vector>

// 运行期开关：读 env MXKERNEL_COUNT（未设置/为 0 = 关闭）
inline int mx_count_enabled() {
    const char* e = std::getenv("MXKERNEL_COUNT");
    return (e != nullptr && std::atoi(e) != 0) ? 1 : 0;
}

// 计数缓冲尺寸（字节）
inline size_t mx_count_bytes() {
    return (size_t)kExecCtrSlots * sizeof(unsigned long long);
}

// 清零计数缓冲（capture 前、捕获流之外调用）
// 注：kCtrTasksPerBlockMin 用 atomicMin 归约，初值必须为极大值，否则恒 0；
//     故此函数先把整块清零，再把该槽置为 ~0ull。
inline void mx_count_clear(unsigned long long* dev_ptr) {
    if (dev_ptr == nullptr) return;
    if (mcMemset(dev_ptr, 0, mx_count_bytes()) != mcSuccess) {
        std::fprintf(stderr, "[计数] 清零失败\n");
        return;
    }
    const unsigned long long kU64Max = ~0ull;
    if (mcMemcpy(reinterpret_cast<char*>(dev_ptr) +
                     (size_t)kCtrTasksPerBlockMin * sizeof(unsigned long long),
                 &kU64Max, sizeof(kU64Max), mcMemcpyHostToDevice) != mcSuccess)
        std::fprintf(stderr, "[计数] min 槽初始化失败\n");
}

// 回读并按中文标签打印全部槽位
inline void mx_count_print(unsigned long long* dev_ptr, const char* tag = "") {
    std::printf("---- [集71] kernel 内部执行次数统计 %s ----\n", tag);
    if (dev_ptr == nullptr) {
        std::printf("  计数缓冲为空（未分配或运行期开关关闭）\n");
        return;
    }
    std::vector<unsigned long long> host(kExecCtrSlots, 0ull);
    if (mcMemcpy(host.data(), dev_ptr, mx_count_bytes(),
                 mcMemcpyDeviceToHost) != mcSuccess) {
        std::fprintf(stderr, "[计数] 回读失败\n");
        return;
    }
    for (int i = 0; i < kExecCtrSlots; ++i)
        std::printf("  %-24s [%2d] = %llu\n", kExecCtrNames[i], i,
                    (unsigned long long)host[i]);
    if (host[kCtrTasksPerBlockMin] == ~0ull) host[kCtrTasksPerBlockMin] = 0;
    // 派生指标（便于快速定位）
    if (host[kCtrKernelBlocks] > 0)
        std::printf("  >> 平均每 block 任务数 = %.3f\n",
                    (double)host[kCtrTasks] / (double)host[kCtrKernelBlocks]);
    const unsigned long long tmo =
        host[kCtrXbarTimeout] + host[kCtrTaskReadyTimeout] +
        host[kCtrChecksumTimeout] + host[kCtrL1ProgTimeout] +
        host[kCtrL2DoneTimeout] + host[kCtrSliceGateTimeout] +
        host[kCtrReduceTimeout];
    std::printf("  >> 等待超时总次数 = %llu\n", (unsigned long long)tmo);
    std::printf("--------------------------------------------\n");
}
