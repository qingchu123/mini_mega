// ============================================================================
// megakernel.h — fp8xfp8 MoE Mega Kernel 的 host C++ 接口（公共头）
//
// 目的：bench / 上层调用方只依赖本头 + 链接 `megakernel.so`，**不直接发射 kernel**。
// 实现（src/megakernel.cu）内部 include kernel.cuh 并发射
//   mega_kernel / mega_reduce_kernel / mega_empty_kernel / fill_w_lane_kernel。
//
// 用法：
//   #include "megakernel.h"
//   fp8mega::MegaLaunchArgs a{};  // 见下（字段与设备入参一一对应）
//   fp8mega::launch_full(a, stream);        // 全链：mega_kernel + REDUCE
//   fp8mega::launch_compute(a, stream);     // 仅 mega_kernel（任务循环）
//   fp8mega::launch_prelude(a, stream);     // 仅前序（kWithCompute=false）
//   fp8mega::launch_reduce(a, stream);      // 仅 REDUCE
//   fp8mega::launch_empty(grid, threads, smem, stream);   // 空 kernel 基准
//   fp8mega::fill_weights_lane(w, n_rows, k_cols, seed, stream);  // 设备侧权重生成
// ============================================================================
#pragma once

#include <cstdint>

#include <mcr/mc_runtime.h>

// flag 结构体定义在 src/flags_defs.cuh（设备与 host 共享的 POD 布局）。
// 此处仅前向声明指针类型，避免公共头依赖设备侧头；bench 需要 sizeof 时
// 直接 include "flags_defs.cuh"。
struct local_flags;
struct ranks_flags;

// 设备侧权重生成的值函数（host 参考与 device 生成必须逐字节一致，故同头共享）。
// -DWPROBE=1 切到「非周期 + 小值」诊断模式（见 bench.cu 注释）。
#ifndef WPROBE
#define WPROBE 0
#endif
#if WPROBE
#define WPROBE_VAL(seed, row, k)                                            \
    ((uint8_t)(1 + ((uint32_t)((seed) + (row) * 131 + (k) * 7 +             \
                                ((row) >> 6) * 37 + ((row) >> 11) * 13 +    \
                                ((row) >> 4) * 5) %                          \
                      7u)))
#else
#define WPROBE_VAL(seed, row, k) \
    ((uint8_t)(((seed) + (row) * 131 + (k) * 7) & 0x3f))
#endif

namespace fp8mega {

// 一次 launch 的全部 host 入参（POD）。字段与 kernel 侧 MegaArgs 一一对应；
// 额外带 grid/threads/smem（发射几何，host_layout 计算）与 y（REDUCE 输出）。
struct MegaLaunchArgs {
    // ---- 前序 ----
    const int64_t* topk_idx = nullptr;   // 本 rank 输入路由
    const uint8_t* self = nullptr;       // 本 rank 对称区基址
    uint8_t* const* peer_bases = nullptr;  // 各 rank 对称区基址表（设备数组）
    int R = 0, rank = 0;
    int tokens = 0;          // 实际 token 数
    int slot_capacity = 0;   // 专家 token 容量
    int win_cap = 0;         // 窗口容量
    int pool_stride = 0;     // pool 槽地址 stride
    int tbl_cap = 0;         // src_index_table 行容量
    int k_in = 0, topk = 0, ring = 0;
    int64_t src_off = 0, rsc_off = 0, recv_off = 0, prefix_off = 0,
            barrier_off = 0;
    local_flags* lf = nullptr;             // 本 rank 私有旗区（普通显存）
    ranks_flags* rf = nullptr;             // 跨 rank 旗区（FG 显存）
    ranks_flags* const* rf_peers = nullptr;  // 各 rank ranks_flags 基址表
    int64_t xrelease_idx = 0, task_ready_idx = 0, task_counter_idx = 0,
            l2_done_idx = 0, slice_flags_idx = 0, l1_prog_idx = 0;
    int64_t x_off = 0, weights_off = 0, pool_off = 0, w_off = 0, meta_off = 0;
    // 控制/诊断（out_total/out_processed/dbg/ck_dbg/pre_ticks/l1_chk/l2_chk）
    // 已迁入 local_flags（lf->...）。
    int max_spins = 0;
    unsigned publish_epoch = 0;
    int recv_stride = 0;
    int* prefix_ptr = nullptr;
    // ---- 计算链 ----
    const uint8_t* w1 = nullptr;
    int l1_n_total = 0, l1_block_n = 0, l1_k_out = 0;
    uint8_t* ffn_hidden = nullptr;
    const uint8_t* w2 = nullptr;
    uint16_t* l2_out = nullptr;
    int l2_k_out = 0;
    // ---- 收尾 ----
    int64_t topk_idx_off = 0;    // REDUCE 用：self + topk_idx_off = topk_idx 基址
    int64_t combine_rows_off = -1;  // combine_rows（BF16）区偏移（留在 sym）
    int combine_cap = 0;
    int sums_stride = 0;
    // ---- REDUCE 输出 + 发射几何（host 侧，设备不读）----
    float* y = nullptr;
    int grid = 0;
    int threads = 0;
    int64_t smem = 0;
};

// 仅前序（mega_kernel<false, NoTaskTail>）
void launch_prelude(const MegaLaunchArgs& a, mcStream_t stream);
// 仅计算链（mega_kernel<true, NoTaskTail>，不含 REDUCE）
void launch_compute(const MegaLaunchArgs& a, mcStream_t stream);
// 仅 REDUCE（mega_reduce_kernel，grid=2*tokens）
void launch_reduce(const MegaLaunchArgs& a, mcStream_t stream);
// 全链（同 stream：mega_kernel<true> + REDUCE）
void launch_full(const MegaLaunchArgs& a, mcStream_t stream);
// 空 kernel（graph 链地板基准）
void launch_empty(int grid, int threads, int64_t smem, mcStream_t stream);
// 设备侧生成 lane 序 packed 权重（16B/lane 配对布局）
void fill_weights_lane(uint8_t* w, int n_rows, int k_cols, uint32_t seed,
                       mcStream_t stream);

}  // namespace fp8mega
