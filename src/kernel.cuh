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
__global__ void mega_empty_kernel() {}

// ---------------------------------------------------------------- 主 kernel --
// host -> device 参数（按值传结构体；全部为 POD）
struct MegaArgs {
    // ---- 前序 ----
    const int64_t* topk_idx;   // 本 rank 输入路由（对称区内视图）
    const uint8_t* self;       // 本 rank 对称区基址
    uint8_t* const* peer_bases;  // 各 rank 对称区基址表（含 self，设备数组）
    int R, rank;
    int tokens;         // 实际 token 数（<= slot_capacity）
    int slot_capacity;  // 专家 token 容量（= max_tokens）
    int win_cap;        // 窗口容量（窗口划分/staging 行 stride）
    int pool_stride;    // pool 槽地址 stride（= win_cap + win_pad）
    int tbl_cap;        // src_index_table 行容量（= max_tokens）
    int k_in, topk, ring;
    int64_t src_off, rsc_off, recv_off, prefix_off, barrier_off;
    local_flags* lf;               // flag 区（本 rank 私有，普通显存）
    ranks_flags* rf;               // 跨 rank flag 区（FG 显存）
    ranks_flags* const* rf_peers;  // 各 rank ranks_flags 基址表（设备数组，含 self）
    int64_t xrelease_idx, task_ready_idx, task_counter_idx, l2_done_idx;
    int64_t slice_flags_idx, l1_prog_idx;
    int64_t x_off, weights_off, pool_off, w_off, meta_off;
    // 控制/诊断（out_total/out_processed/dbg/ck_dbg/pre_ticks/l1_chk/l2_chk）
    // 已迁入 local_flags（lf->...），此处不再持有独立指针。
    int max_spins;
    unsigned publish_epoch;   // publish 校验槽 epoch（阶段 1 = kPublishEpoch）
    int recv_stride;          // recv 行宽（word）= 12
    int* prefix_ptr;          // 可选独立前缀表缓冲（nullptr = self+prefix_off）    // ---- 计算链 ----
    const uint8_t* w1;        // [EPR][2I][H] lane 序 packed tile（gate/up 配对布局）
    int l1_n_total, l1_block_n, l1_k_out;
    uint8_t* ffn_hidden;      // [EPR][ring][win_cap][I] FP8
    const uint8_t* w2;        // [EPR][H][I] lane 序 packed tile
    uint16_t* l2_out;         // [kGrid][win_cap][H] BF16
    int l2_k_out;
    // ---- 收尾 ----
    int64_t combine_rows_off;  // 目标 rank 的 combine_rows 区偏移（BF16，留在 sym）
    int combine_cap;   // combine_rows 行容量（= max_tokens）
    int sums_stride;   // combine_sums 每行槽数（= 32）
    // ---- [集71] 内部执行次数统计（编译期默认关：MX_DBG_COUNT=0）----
    // 整块条件编译：关闭态 MegaArgs 的布局/大小与改造前**逐字节一致**
    // （已用 .mc_fatbin md5 对比验证，见 test/71.kernel_exec_count/results.md）。
#if MX_DBG_COUNT
    unsigned long long* exec_ctr;  // [kExecCtrSlots] 各计数槽（nullptr = 不统计）
    int dbg_count;                 // 运行期开关（host 读 env MXKERNEL_COUNT 填入）
#endif
};

// --------------------------------------------------- COMBINE 任务尾适配器 --
// 在任务循环 tail 处按 TaskDesc 更新 CombineCtx 并执行本任务 slice 的推送。
struct CombineTail {
    CombineCtx ctx;
    const uint8_t* out_base;  // ffn_out staging 全区基址（BF16）
    int nwarps;
    int sums_stride;
    CombineFuseCtx fuse;      // [R29] 融合推送上下文（capable 由 kernel 置位）
    // [集71] 内部执行次数统计（编译期默认关；运行期 nullptr 关闭）
    //   条件编译：CombineTail 是 mega_kernel 的**按值模板实参**（进 kernel 参数
    //   缓冲），关闭态必须不占位，否则参数缓冲布局变化会改变设备码字节。
#if MX_DBG_COUNT
    unsigned long long* ctr = nullptr;
#endif
    __device__ __forceinline__ void operator()(const TaskDesc& d) const {
        if (ctx.rows_off < 0) {
            // [集71] COMBINE 跳过（combine 关闭）
            if (threadIdx.x == 0) MX_CTR_ADD(ctr, kCtrCombineSkip, 1);
            return;
        }   // [集33] combine 关闭：空操作
        // [集71] COMBINE 实际执行推送次数
        if (threadIdx.x == 0) MX_CTR_ADD(ctr, kCtrCombineExec, 1);
        CombineCtx c = ctx;
        c.slice = d.slice;
        c.epoch = d.epoch;
        c.nslice_eff = d.nslice_eff;   // [R35] 粗化几何
        c.merge = d.merge;
        const int lane = (int)(threadIdx.x % 64);
        const int warp = (int)(threadIdx.x / 64);
        if (fuse.on) {
            // [R29] 数据已在 L2 各列块推完，任务尾只写完成标记
            combine_mark_rows(c, d.count, lane, warp, nwarps, sums_stride);
        } else {
            combine_push_pair(c, out_base, d.count, lane, warp, nwarps,
                              sums_stride);
        }
    }
};

// [R29] 融合上下文获取特化（任务体据此判断是否走融合推送）
template <>
struct R29TailFuse<CombineTail> {
    static __device__ __forceinline__ CombineFuseCtx get(const CombineTail& t) {
        return t.fuse;
    }
};
// [R29] 任务尾回写实际融合状态（供 CombineTail 选择只写标记路径）
template <>
struct R29TailFuseSet<CombineTail> {
    static __device__ __forceinline__ void on(CombineTail& t,
                                              const CombineFuseCtx& f) {
        t.fuse = f;
    }
};

// 计算链任务循环（tail 类型由调用点按运行期开关选择）
template <typename TaskTail, bool kSmallOnly = false>
__device__ __forceinline__ void mega_dispatch_loop(const MegaArgs& a,
                                                   TaskTail tail, int cycle,
                                                   uint8_t* a_smem = nullptr,
                                                   unsigned long long* ctr = nullptr) {
    dispatch_pull_body<TaskTail, kSmallOnly>(
        a.self, a.peer_bases, a.R, a.rank, a.win_cap, a.pool_stride,
        a.tbl_cap, a.k_in, a.topk, a.ring, a.prefix_off, a.src_off, a.x_off,
        a.weights_off, a.pool_off, a.w_off, a.meta_off, a.lf->bar[cycle],
        a.task_counter_idx, a.l2_done_idx, a.task_ready_idx, a.slice_flags_idx,
        a.l1_prog_idx, a.lf->out_processed, a.lf->dbg, a.max_spins, a.w1,
        a.l1_n_total, a.l1_block_n, a.l1_k_out, a.ffn_hidden, a.lf->l1_chk, a.w2,
        a.l2_out, a.l2_k_out, a.lf->l2_chk, /*rep=*/0, /*n_reps=*/1, tail,
        a.prefix_ptr, a.lf->pre_ticks, a_smem, ctr);
}

// 主 kernel（单轮）。
//   kWithCompute=false：只跑前序（M1 验证）；true：完整任务循环（M2/M3）；
//   kWithCompute=true 且 combine_rows_off>=0：任务尾接 COMBINE（M3）
template <bool kWithCompute, typename TaskTail = NoTaskTail,
          bool kSmallOnly = false>
__global__ void __launch_bounds__(512) mega_kernel(MegaArgs a, TaskTail tail) {
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
    // [R24] A-smem 双缓冲（kernel 作用域，两类任务体共享同一块）
    __shared__ __align__(16) uint8_t s_a_smem[2][R24_MT * R24_KS * 512];
    if (kTick) a.lf->pre_ticks[6] = clock64();
    // 0) [M_CYCLE] 预清 ct2=(c+2)%M_CYCLE 槽（本轮 copy c 尚未定，用 current_time
    //   [b][0] 派生；入口全 block 同为 L ⇒ c 一致）。ct1 留给下一轮，本轮不碰。
    mx_preclear_ct2(a.lf, a.rf);
    __syncthreads();
    // 1) PROLOGUE：ROUTE_BUILD + PUBLISH_COUNTS（仅 block0）
    if (blockIdx.x == 0) {
        route_build_body(a.topk_idx, a.peer_bases, a.R, a.rank, a.tokens,
                         a.tbl_cap, a.src_off, a.rsc_off, 0, 0, ctr);
        if (kTick) a.lf->pre_ticks[0] = clock64();
        publish_counts_body(a.self, a.peer_bases, a.R, a.rank, a.rsc_off,
                            a.recv_off, a.publish_epoch, a.recv_stride, 0, ctr);
        if (kTick) a.lf->pre_ticks[1] = clock64();
    }
    // 2) XBAR_ROUTE_READY（全 grid，mode3：转置 + warp 并行发布）
    xbar_route_ready_body(a.rf, a.rf_peers, a.lf, a.R, a.rank,
                          a.xrelease_idx, a.lf->dbg, a.max_spins, 0, ctr);
    if (kTick) a.lf->pre_ticks[2] = clock64();
    // 3) TASK_SETUP（block0 leader：checksum -> prefix -> 窗口表 -> task_ready）
    task_setup_body(a.self, a.R, a.rank, a.slot_capacity, a.recv_off,
                    a.prefix_off, a.lf, a.task_ready_idx,
                    a.task_counter_idx, a.l2_done_idx,
                    a.lf->out_total, a.lf->dbg, a.max_spins,
                    a.recv_stride, /*reset_ring=*/true, /*tid_base=*/0,
                    /*sb_cnt=*/nullptr, /*smem_off=*/0, a.prefix_ptr,
                    a.lf->ck_dbg, ctr);
    if (kTick) a.lf->pre_ticks[3] = clock64();
    // 4) 任务循环（DISPATCH -> L1 -> L2 -> tail(COMBINE)）
    if (kWithCompute) {
        // [M_CYCLE] 任务循环 / COMBINE / REDUCE 用的 copy c：task_setup 结束时
        //   current_time[b][1] 已自增到 L+1，故 (ct-1)%M_CYCLE = L%M_CYCLE，与
        //   xbar/task_setup 所用 c 一致。COMBINE 完成标记与 REDUCE 等待须同片。
        const int cyc = (int)((a.lf->current_time[blockIdx.x][1] - 1) % M_CYCLE);
        // [集33] tail 实例统一：任务循环恒用 CombineTail，combine 关闭时
        //   （combine_rows_off<0）融合能力置 false 且 tail 为空操作。这样
        //   mega_kernel<..., NoTaskTail> 不再实例化第二份重型任务体
        //   （原 NoTaskTail 运行期从不执行，却占整核寄存器/代码布局 collateral；
        //   71 机 TPR=16 −7µs，TPR=256 中性）。
        {
            CombineTail tail;
            if (a.combine_rows_off >= 0) {
                tail.ctx.meta =
                    reinterpret_cast<const int*>(a.self + a.meta_off);
                tail.ctx.weights =
                    reinterpret_cast<const float*>(a.self + a.w_off);
                tail.ctx.peer_bases = a.peer_bases;
                tail.ctx.rf_peers = a.rf_peers;
                tail.ctx.rows_off = a.combine_rows_off;
                tail.ctx.staging_cap = a.win_cap;
                tail.ctx.combine_cap = a.combine_cap;
                tail.ctx.block_idx = (int)blockIdx.x;
                tail.ctx.cycle = cyc;   // [M_CYCLE] combine_sums 片
                tail.out_base = reinterpret_cast<const uint8_t*>(a.l2_out);
                tail.nwarps = 8;
                tail.sums_stride = a.sums_stride;
                // [R29] 融合推送能力 + 上下文（on 由任务体在 r24 L2 路径激活）
                tail.fuse.capable = true;
                tail.fuse.out_base = tail.out_base;
                tail.fuse.meta = tail.ctx.meta;
                tail.fuse.weights = tail.ctx.weights;
                tail.fuse.peer_bases = tail.ctx.peer_bases;
                tail.fuse.rows_off = tail.ctx.rows_off;
                tail.fuse.combine_cap = tail.ctx.combine_cap;
                tail.fuse.staging_cap = tail.ctx.staging_cap;
                tail.fuse.block_idx = tail.ctx.block_idx;
                tail.fuse.sums_stride = a.sums_stride;
            } else {
                // [集33] combine 关闭：同一 tail 实例走空操作（rows_off<0）
                tail.ctx.rows_off = a.combine_rows_off;
                tail.ctx.staging_cap = a.win_cap;
                tail.ctx.combine_cap = a.combine_cap;
                tail.ctx.block_idx = (int)blockIdx.x;
                tail.ctx.cycle = cyc;   // [M_CYCLE]（combine 关闭，仅占位）
                tail.out_base = nullptr;
                tail.nwarps = 8;
                tail.sums_stride = a.sums_stride;
                tail.fuse.capable = false;
            }
            // [集71] 计数缓冲下传（CombineTail 统计 COMBINE 执行/跳过次数）
#if MX_DBG_COUNT
            tail.ctr = ctr;
#endif
            mega_dispatch_loop<CombineTail, kSmallOnly>(
                a, tail, cyc, &s_a_smem[0][0], ctr);
        }
    }
}

// ---------------------------------------------------------------- REDUCE --
// 独立 kernel：每 token 2 个 block 列分半（grid=32 覆盖 TPR=16）
// [集71] 末尾两个参数为带默认值的可选项：老调用点（bench.cu）不传即 nullptr，
//        编译期 MX_DBG_COUNT=0 时统计代码整体消除。
__global__ void mega_reduce_kernel(const uint8_t* self, float* y, int R,
                                   int tokens, int combine_cap,
                                   int64_t rows_off,
                                   int64_t topk_idx_off, int64_t weights_off,
                                   local_flags* lf, ranks_flags* rf,
                                   int64_t task_ready_idx,
                                   int max_spins, int sums_stride
#if MX_DBG_COUNT
                                   ,
                                   unsigned long long* exec_ctr = nullptr,
                                   int dbg_count = 0
#endif
) {
    reduce_body(self, y, R, tokens, combine_cap, rows_off,
                topk_idx_off, weights_off, lf, rf, task_ready_idx,
                max_spins, sums_stride
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
