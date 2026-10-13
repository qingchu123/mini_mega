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
constexpr int M_CYCLE = 8;

// 旗区容量（与生产布局头 include/deep_gemm/layout/mega_moe.cuh 同源，NSLICE=16 扩容后）：
//   kBarFlagsNumInt32=2560（普通旗区），kFp8BarFlagsNumInt32=2568（+epoch/done 保留槽）
constexpr int kBarFlagsNumInt32 = 2560;
constexpr int kFp8BarFlagsNumInt32 = 2568;

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
    // [BALANCE·docs/20] L1 split-K：每 (slot, N片 s) 的 K 段完成计数（atomicAdd 判最后一段）
    //   仅 L1 split-K（L1_SPLITK / R19_BLOCK_KSPLIT）路径用到：默认不分配。
#if (L1_SPLITK || R19_BLOCK_KSPLIT)
    alignas(256) uint32_t l1_part_cnt[kEPR * kRing][kNSlice];
#endif
};

// ---- 跨 rank（FG 显存，peer 可写）----
struct alignas(256) ranks_flags {
    // 跨 rank ROUTE_READY barrier，转置布局 xbar_barrier[c][src][dst]
    // （与原 sym 内 barrier_slots 的 src*R+dst 语义一致，行 stride 用 kGrid）。
    // [M_CYCLE] 旋转维：本轮只用 copy c。
    alignas(256) uint32_t xbar_barrier[M_CYCLE][kGrid][2 * kGrid];

    // 跨 rank COMBINE 完成标记：combine_sums[c][k][token][slice] = epoch。
    // 取代原 sym::combine_sums（uint32 区）；行 stride = kMaxCombineRows。
    // combine_rows（BF16 数据）仍留在 sym，不在此。
    // [M_CYCLE] 旋转维：本轮只用 copy c（±2MB/rank，可接受）。
    alignas(256) uint64_t
        combine_sums[M_CYCLE][kTopk][kMaxCombineRows][kCombineSlicesMax];

    // [MX_GROUP_SCHED] 每个 rank 在本 launch 把本 rank 8 专家的 prefix 发布到 FG，
    //   供 gs 远端 helper 读取（远读→必须 FG）。值与 task_setup 的 checksum 门同源，
    //   owner 写完后置对应 gs_ready[cyc][rank]=epoch，helper 等旗后再读。
    alignas(256) uint32_t gs_ready[M_CYCLE][kRanks];
};

// [WP-G1] 组内权重复制几何：kGroupRanks 个 rank 一组，kGE=kEPR*kGroupRanks 为每组
//   专家数。全局专家映射（连续组，仅 MX_GROUP_WEIGHTS 非 gs 路径）：owner=e/kEPR，
//   组=owner/kGroupRanks，组内专家序 ge=(owner%kGroupRanks)*kEPR+(e%kEPR)∈[0,kGE)。
//   [MX_GROUP_SCHED] 可用 -DMX_GROUP_RANKS=8 扩大组规模（权重复制 2×，均衡更细）；
//   交织组语义见 group_sched.cuh。
#ifndef MX_GROUP_RANKS
#define MX_GROUP_RANKS 4
#endif
constexpr int kGroupRanks = MX_GROUP_RANKS;
constexpr int kGE = kEPR * kGroupRanks;

// ---- 本 rank 私有（普通显存）----
struct alignas(256) local_data {
    // 输入复用
    // alignas(256) uint8_t x[kMaxTPR][kHidden];
    // alignas(256) float topk_weights[kMaxTPR][kTopk];
    // [冗余，暂注释] alignas(256) int64_t topk_idx[kMaxTPR][kTopk];  // 输入走 sym

    // 内部使用
    alignas(256) int32_t route_slot_counts[kRanks * kEPR];
    alignas(256) int32_t prefix_tbl[kEPR][kRanks + 1];
    // [PEERBUF] expert_tokens 迁至 ranks_data（FG，peer 可读，供远端窗口拉取 A 行）。
    // [冗余，暂注释] alignas(256) float token_route_weights[kEPR + kGrid][kMaxTPR];  // 走 scratch 偏移
    // [冗余，暂注释] alignas(256) int token_src_meta[kEPR + kGrid][kMaxTPR][3];  // 走 scratch 偏移


#if MX_GROUP_WEIGHTS
    // [WP-G1] 组内 4× 本地只读权重：ge∈[0,kGE=32) 覆盖本组 32 专家。
    //   ge 块 b∈[0,4) 对应组内第 b 个 owner rank 的 kEPR 个专家（固定种子复现，
    //   组内各 rank 一致）。取代原 w1[kEPR]/w2[kEPR]。
    alignas(256) uint8_t w1g[kGE][2*kIntermediate][kHidden];
    alignas(256) uint8_t w2g[kGE][kHidden][kIntermediate];
#else
    alignas(256) uint8_t w1[kEPR][2*kIntermediate][kHidden];
    alignas(256) uint8_t w2[kEPR][kHidden][kIntermediate];
#endif
    alignas(256) uint8_t ffn_hidden[kEPR * kRing][kMaxTPR][kIntermediate];
    // [BALANCE·docs/20] L1 split-K FP32 部分和（gate/up 原值，未 SwiGLU）：每 (slot, N片 s)
    //   一块 [行][2I/NS]，NK 个 K 段 atomicAdd 累加；最后一段读回→SwiGLU→FP8 写 ffn_hidden。
    //   仅 L1 split-K（L1_SPLITK / R19_BLOCK_KSPLIT）路径用到：默认不分配（~512MB）。
#if (L1_SPLITK || R19_BLOCK_KSPLIT)
    alignas(256) float l1_partial[kEPR * kRing][kNSlice][kMaxTPR][kL1SliceCols];
#endif
    // [PEERBUF] l2_out 迁至 ranks_data（FG，peer 可读，供远端回收 L2 输出）。
    // [FRAG-INT] A/f2 fragment 缓冲，纯 kernel 内部：由本结构体持有，kernel 从
    //   rwk->ld 直接取基址，host/torch 不再传入。尺寸按 kMaxTPR 静态定（容量
    //   上界，运行期实际用 ceil(win_cap/16) 列，只会更小）。
    alignas(256) uint8_t frag_a[kEPR * kRing][int(kHidden / 32) * ((kMaxTPR + 15) / 16) * 512];
    alignas(256) uint8_t frag2[kEPR * kRing][int(kIntermediate / 32) * ((kMaxTPR + 15) / 16) * 512];
    // [DEL-Y] 原 alignas(256) float y[kMaxTPR][kHidden]（12MB）零引用已删除：
    //   真正 y 走 MegaLaunchArgs.y/REDUCE；test 侧另分配 d.y（见 mega_common.cuh）。
    // [冗余，暂注释] alignas(256) float ynode[kMaxTPR][kHidden];  // mega_common 另行分配 d.ynode
};

// ---- 本 rank 私有（普通显存）----
struct alignas(256) rwk_data {
    alignas(256) local_flags lf;
    alignas(256) local_data ld;
};


// ---- 跨 rank（FG 显存，peer 可写）----
struct alignas(256) ranks_data {

    alignas(256) uint32_t src_index_table[M_CYCLE][kEPR][kRanks][kMaxTPR];
    alignas(256) uint32_t expert_recv_counts[M_CYCLE][kRanks][64];
    // [MX_GROUP_SCHED] owner 本 launch 的 prefix（每专家对 src 的前缀），FG 远读用。
    alignas(256) int32_t gs_prefix[M_CYCLE][kEPR][kRanks + 1];
    alignas(256) uint16_t combine_rows[M_CYCLE][kTopk][kMaxTPR][kHidden];
    // [PEERBUF] 原 local_data 私有缓冲迁入 FG（peer 可读）：A pool / L2 块槽输出。
    alignas(256) uint8_t expert_tokens[M_CYCLE][kEPR * kRing][kMaxTPR][kHidden];
    alignas(256) uint16_t l2_out[M_CYCLE][kGrid][kMaxTPR][kHidden];

    // ---- [UNK-MERGE] kernel 自有内部 staging（原 sym scratch 经 offset 访问，
    //   现并入 unk_data：kernel 经 rd 直接访问，peer 亦可读 prefold）。----
    //   维度取最大使用：token 槽按 blockIdx<kGrid、窗口行 stride 用运行期 win_cap
    //   （≤ slot_capacity ≤ kMaxTPR），故扁平维 [kGrid * kMaxTPR] 覆盖全部索引；
    //   prefold = slot_cap*H fp8（[kMaxTPR][kHidden]）。
    alignas(256) float token_route_weights[kGrid * kMaxTPR];
    alignas(256) int token_src_meta[kGrid * kMaxTPR * 3];
    alignas(256) uint8_t prefold[kMaxTPR][kHidden];

};



// ---- 本 rank 私有（普通mcMalloc显存）----
struct alignas(256) unk_data {
    alignas(256)ranks_flags rf;
    alignas(256)ranks_data rd;
};
