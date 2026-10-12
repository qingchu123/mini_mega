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

__global__ void __launch_bounds__(1024) mega_empty_kernel();


// 计算链任务循环
template <MegaPhase kPhase = MegaPhase::Both>
__device__ __forceinline__ void mega_dispatch_loop(const MegaArgs* a,
                                                   int cycle,
                                                   uint8_t* a_smem = nullptr,
                                                   uint8_t* b_smem = nullptr,
                                                   unsigned long long* ctr = nullptr) {
    dispatch_pull_body<kPhase>(a, cycle, a_smem, b_smem, ctr);
}

// 主 kernel。kWithCompute=false 只跑前序；kPhase 见 dispatch（Both/L1/L2）。
template <bool kWithCompute, MegaPhase kPhase = MegaPhase::Both>
__global__ void __launch_bounds__(512) mega_kernel(MegaArgs a) {
    const bool kTick = (blockIdx.x == 0) && (threadIdx.x == 0);
    // [计数字段移除] 不再从 MegaArgs 读取 exec_ctr/dbg_count：恒 nullptr，
    //   MX_CTR_ADD 展开为空语句（默认 MX_DBG_COUNT=0）。
    unsigned long long* ctr = nullptr;
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) MX_CTR_ADD(ctr, kCtrKernelLaunch, 1);
        MX_CTR_ADD(ctr, kCtrKernelBlocks, 1);
    }
    // A-smem 双缓冲 + [D1]B-smem 环（kernel 作用域，同一扁平数组）
    struct __align__(16) ABShared {
        uint8_t raw[kABSharedBytes];
    };
    static_assert(kABSharedBytes >= 2 * R24_MT * R24_KS * 512,
                  "AB 共享数组须容纳最大 A 双缓冲");
    __shared__ ABShared s_ab;
    __shared__ MegaArgs s_a;
    if (threadIdx.x == 0) s_a = a;
    if (kTick) a.rwk->lf.pre_ticks[6] = clock64();
    mx_preclear_ct2(&a.rwk->lf, &a.unk->rf, /*skip_xbar=*/a.skip_prologue >= 2);
    __syncthreads();
    if (!a.skip_prologue) {
        // 1) PROLOGUE：ROUTE_BUILD + PUBLISH_COUNTS（仅 block0）
        if (blockIdx.x == 0) {
            const int route_cycle = (int)(a.rwk->lf.current_time[0][0] % M_CYCLE);
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
    // [BF16-IN] 本地 x(bf16) fold → fp8（写 self+prefold_off），供 peer pull。
    if (a.in_bf16)
        x_fold_body(a.self + a.x_off,
                    const_cast<uint8_t*>(a.self) + a.prefold_off,
                    a.tokens * kHidden);   // peer 只读有效 token 行
    // 2) XBAR_ROUTE_READY（每节点：可见性门 + 跨 rank 对齐）
    if (a.skip_prologue >= 2) {
        if constexpr (kPhase != MegaPhase::L2) {
            if (threadIdx.x == 0) a.rwk->lf.current_time[blockIdx.x][0] += 1;
        }
        __syncthreads();
    } else {
        xbar_route_ready_body(&a.unk->rf, a.unk_peers, &a.rwk->lf, a.R, a.rank,
                              kBarXreleaseIdx, a.rwk->lf.dbg, a.max_spins, 0, ctr);
    }
    if (kTick) a.rwk->lf.pre_ticks[2] = clock64();
    // 3) TASK_SETUP（block0 leader 建 prefix/窗口表，其余等 task_ready）
    if constexpr (kPhase == MegaPhase::L2) {
        if (kTick) a.rwk->lf.pre_ticks[3] = clock64();
    } else {
        task_setup_body(a.self, &a.unk->rd, a.R, a.rank, a.slot_capacity,
                        a.prefix_off, &a.rwk->lf, kBarTaskReadyIdx,
                        kBarTaskCounterIdx, kBarL2DoneIdx,
                        a.rwk->lf.out_total, a.rwk->lf.dbg, a.max_spins,
                        kRecvStride, /*reset_ring=*/true, /*tid_base=*/0,
                        /*sb_cnt=*/nullptr, /*smem_off=*/0, a.prefix_ptr,
                        a.rwk->lf.ck_dbg, ctr);
    }
    if (kTick) a.rwk->lf.pre_ticks[3] = clock64();
    // 4) 任务循环（DISPATCH -> L1 -> L2 -> tail(COMBINE)）
    if (kWithCompute) {
        // [M_CYCLE] copy c（与 xbar/task_setup 同片）
        const int cyc = (int)((a.rwk->lf.current_time[blockIdx.x][1] - 1) % M_CYCLE);
        mega_dispatch_loop<kPhase>(
            &s_a, cyc, &s_ab.raw[0], nullptr, ctr);
    }
    if (kTick) a.rwk->lf.pre_ticks[15] = clock64();
}

// [multi_compile] 显式实例化在 src/multi_compile/kernel_*.cu；此处抑制隐式实例化。
extern template __global__ void mega_kernel<false, MegaPhase::Both>(MegaArgs);
extern template __global__ void mega_kernel<true, MegaPhase::Both>(MegaArgs);

// ---------------------------------------------------------------- REDUCE --
// REDUCE：每 token 2 个 block 列分半
__global__ void __launch_bounds__(256) mega_reduce_kernel(
    const uint8_t* self, const ranks_data* remote, float* y, int R, int tokens,
    int64_t rows_off, int64_t topk_idx_off, int64_t weights_off,
    local_flags* lf, ranks_flags* rf, int64_t task_ready_idx, int max_spins
#if MX_DBG_COUNT
    , unsigned long long* exec_ctr, int dbg_count
#endif
);

// =============================================================================
// [集71] host 侧接口：计数缓冲清零 / 打印
//
// 语义（重要，配合 CUDA Graph）：
//   1) mx_count_enabled()：读运行期 env MXKERNEL_COUNT（"1"/非 0 即开）。
//      [计数字段移除] 设备入口恒以 nullptr 运行，计数写入不再生效。
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
