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

// ---- M_CYCLE 轮转（对齐参考实现 /projs/mega/src/kernel/mega_fp8.cuh）----
//   int64_t ct = lf->current_time[b][k]; int c = (int)(ct % M_CYCLE); // 本轮用 copy c
//   各 per-launch 旗数组带 [M_CYCLE] 旋转维：同一槽位每 M_CYCLE 轮才复用一次，
//   其间无人触碰 ⇒ 开工时干净。完成旗可用常量、无需每轮重置；kernel 入口对
//   ct2=(c+2)%M_CYCLE 槽做一次预清（ct1=(c+1)%M_CYCLE 是下一轮要用的槽，本轮
//   绝不能碰；ct2 是 M_CYCLE-2 轮前用过、早已结束的那一份，清它天然安全）。
//   唯一正确性前提（host 契约）：所有 rank 的 launch 次数必须相同，使各 block /
//   各 rank 的 c 同步前进。M_CYCLE>=3 是「预清 ct2 + 不碰 ct1」成立的前提。
constexpr int M_CYCLE = 4;

// 旧扁平 bar_flags 布局容量（与生产布局头同源）：
//   kBarFlagsNumInt32=1024（普通旗区）、kFp8BarFlagsNumInt32=1032（+epoch/done 保留槽）
constexpr int kBarFlagsNumInt32 = 1024;
constexpr int kFp8BarFlagsNumInt32 = 1032;

// COMBINE 完成标记容量（kernel 侧不 include 布局头，故本地重述）：
//   kCombineSlicesMax 与 include/deep_gemm/layout/mega_moe.cuh 的
//   kMegaMoeMaxCombineSlices=32 一致；kMaxCombineRows = max_tokens 上限 512。
constexpr int kCombineSlicesMax = 32;
constexpr int kMaxCombineRows = 512;

// ---- 本 rank 私有（普通显存）----
struct alignas(256) local_flags {
    // 每 block 单调 launch 计数：[kGrid][MX_FLAGS_NO_CACHE]
    //   [b][0] xbar 相位；[b][1] task_setup 相位；其余列预留。
    alignas(256) int64_t current_time[kGrid][MX_FLAGS_NO_CACHE];

    // 原 bar_flags 旗区（索引逐一对应原 kBar*Idx，行为不变）：
    //   xrelease/task_ready/task_counter/l2_done/slice_flags/l1_prog/nslice_eff ...
    // [M_CYCLE] 旋转维：本轮只用 bar[c]，同一槽位每 M_CYCLE 轮才复用一次。
    alignas(256) uint32_t bar[M_CYCLE][kFp8BarFlagsNumInt32];

    // ---- 控制/诊断状态（原 MegaArgs 独立指针迁入）----
    alignas(256) int out_total[kGrid];       // 每 block 总窗口数（TASK_SETUP 写）
    alignas(256) int out_processed[kGrid];   // 每 block 处理任务数（任务循环写）
    alignas(256) int dbg[8];                 // 诊断码（超时归因）
    alignas(256) uint32_t ck_dbg[kEPR * 16]; // checksum 超时行诊断（可选）
    // 前序/计算链相位时间戳：task_trace 打开时每 block 16 槽，普通 8；统一按 16 分配
    alignas(256) unsigned long long pre_ticks[kGrid * 16];
    alignas(256) float l1_chk[kGrid];        // L1 数值校验累加
    alignas(256) float l2_chk[kGrid];        // L2 数值校验累加
};

// ---- 跨 rank（FG 显存，peer 可写）----
struct alignas(256) ranks_flags {
    // 跨 rank ROUTE_READY barrier，转置布局 xbar_barrier[c][src][dst]
    // （与原 sym 内 barrier_slots 的 src*R+dst 语义一致，行 stride 用 kGrid）。
    // [M_CYCLE] 旋转维：本轮只用 copy c。
    alignas(256) uint32_t xbar_barrier[M_CYCLE][kGrid][kGrid];

    // 跨 rank COMBINE 完成标记：combine_sums[c][k][token][slice] = epoch。
    // 取代原 sym::combine_sums（uint32 区）；行 stride = kMaxCombineRows。
    // combine_rows（BF16 数据）仍留在 sym，不在此。
    // [M_CYCLE] 旋转维：本轮只用 copy c（±2MB/rank，可接受）。
    alignas(256) uint32_t
        combine_sums[M_CYCLE][kTopk][kMaxCombineRows][kCombineSlicesMax];
};
