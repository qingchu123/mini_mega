// ============================================================================
// flags_defs.cuh — fp8xfp8 MoE Mega Kernel 同步标志结构体定义
//
// 改造目标（对齐参考实现 /projs/mega/src/kernel/mega_fp8.cuh 的 flag 机制）：
//   把原来一组扁平数组传同步标志，收敛为两个结构体：
//     * local_flags —— 本 rank 私有，普通显存（mcMalloc）；block 间共享
//     * ranks_flags —— 跨 rank 可写，FG 显存（mcExtMallocWithFlags
//                      mcDeviceMallocFinegrained），peer 可写
//
// current_time 机制（取代旧的 epoch_ctr + last_release）：
//   每个 block 一份单调 launch 计数 current_time[b]，由 kernel 自己每次 launch
//   递增 1（CUDA graph replay 也不依赖 host 清零）。用它派生 launch 身份：
//     * current_time[b][0]：xbar_route_ready 相位（release 旗身份）
//     * current_time[b][1]：task_setup 相位（task_ready 旗身份）
//   生产侧（block0/leader）用自己槽位推出本 launch 的 epoch 并发布旗值，
//   消费侧（其余 block）用自己槽位推出期望 epoch 并自旋等待后自增。因为同一
//   次 launch 内所有 block 的 current_time 同步前进一次，两端派生的 epoch 一致。
//
// bar[] 语义与旧的扁平 bar_flags 数组逐索引一致（kBar*Idx），因此所有
// flags.cuh / compute / combine 的读写点只需把指针从「bar_flags」换成「lf->bar」，
// 行为不变。索引常量沿用 include/deep_gemm/layout/mega_moe.cuh 的生产布局
// （kernel 侧不包含 layout 头，故在此重述容量常量）。
// ============================================================================
#pragma once

#include <cstdint>

#include "common.cuh"

// 相邻 block 的 current_time 不共享 cache line（对齐参考实现 NO_CACHE）。
// 每 block 占用 MX_FLAGS_NO_CACHE 个 int64 = 512B，独立于邻块。
constexpr int MX_FLAGS_NO_CACHE = 64;

// 旧扁平 bar_flags 布局容量（与生产布局头同源）：
//   kBarFlagsNumInt32=1024（普通旗区）、kFp8BarFlagsNumInt32=1032（+epoch/done 保留槽）
constexpr int kBarFlagsNumInt32 = 1024;
constexpr int kFp8BarFlagsNumInt32 = 1032;

// ---- 本 rank 私有（普通显存）----
struct alignas(256) local_flags {
    // 每 block 单调 launch 计数：[kGrid][MX_FLAGS_NO_CACHE]
    //   [b][0] xbar 相位；[b][1] task_setup 相位；其余列预留。
    alignas(256) int64_t current_time[kGrid][MX_FLAGS_NO_CACHE];

    // 原 bar_flags 旗区（索引逐一对应原 kBar*Idx，行为不变）：
    //   xrelease/task_ready/task_counter/l2_done/slice_flags/l1_prog/nslice_eff ...
    alignas(256) uint32_t bar[kFp8BarFlagsNumInt32];

    // task_ready 等待水位（原 last_ready）：非 leader 每 block 一份（保留语义备用）
    alignas(256) unsigned last_ready[kGrid];
};

// ---- 跨 rank（FG 显存，peer 可写）----
struct alignas(256) ranks_flags {
    // 跨 rank ROUTE_READY barrier，转置布局 xbar_barrier[src][dst]
    // （与原 sym 内 barrier_slots 的 src*R+dst 语义一致，行 stride 用 kGrid）。
    alignas(256) uint32_t xbar_barrier[kGrid][kGrid];
};
