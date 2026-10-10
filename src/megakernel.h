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
struct local_data;
struct ranks_data;
struct rwk_data;
struct unk_data;

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
    // ======================================================================
    // [1] grid 规划：发射几何（grid/threads/smem）与任务划分量
    // ======================================================================
    int R = 0, rank = 0;         // rank 数 / 本 rank 号
    int tokens = 0;              // 实际 token 数
    int slot_capacity = 0;       // 专家 token 容量（= max_tokens）
    int win_cap = 0;             // 窗口容量（窗口划分 / pool / staging / ffn / l2 行 stride）
    int grid = 0, threads = 0;   // 发射几何（host 侧，设备不读）
    int64_t smem = 0;

    // ======================================================================
    // [2] 调用入参：结构体/缓冲指针、偏移、flag 索引
    // ======================================================================
    // -- 结构体指针 --
    rwk_data* rwk = nullptr;               // 本 rank 普通显存整块（lf + ld）
    unk_data* unk = nullptr;               // 本 rank FG 整块（rf + rd）
    unk_data* const* unk_peers = nullptr;  // 各 rank unk 基址表（设备数组）
    // -- 对称区 / 输入 --
    const uint8_t* self = nullptr;         // 本 rank 对称区基址
    uint8_t* const* peer_bases = nullptr;  // 各 rank 对称区基址表（设备数组）
    const int64_t* topk_idx = nullptr;     // 本 rank 输入路由
    // -- 可选独立缓冲（nullptr → self 内对应偏移）--
    uint8_t* pool_ptr = nullptr;           // [A-RWK] pool 缓冲
    int* prefix_ptr = nullptr;             // 前缀表缓冲
    // -- 前序 / REDUCE 偏移（self 内）--
    int64_t prefix_off = 0;                // prefix_tbl（prefix_ptr 为空时用）
    int64_t topk_idx_off = 0;              // REDUCE：self + topk_idx_off = topk_idx 基址
    // -- 数据缓冲偏移（self / peer sym 内）--
    int64_t x_off = 0, weights_off = 0, pool_off = 0, w_off = 0, meta_off = 0;
    int64_t combine_rows_off = -1;         // combine_rows（BF16）；<0 关闭 COMBINE
    // -- 计算链缓冲 --
    const uint8_t* w1 = nullptr;
    const uint8_t* w2 = nullptr;
    uint8_t* ffn_hidden = nullptr;         // [FRAG-INT] frag 缓冲为 kernel 内部（rwk->ld）
    uint16_t* l2_out = nullptr;
    int l1_k_out = 0;                      // L1 输出隐藏维 I（l1_n_total/l1_block_n 为常量）
    int l2_k_out = 0;                      // L2 输出维 H
    // -- flag 索引（layout 头）--
    int64_t xrelease_idx = 0, task_ready_idx = 0, task_counter_idx = 0,
            l2_done_idx = 0, slice_flags_idx = 0, l1_prog_idx = 0;

    // ======================================================================
    // [3] 输出
    // ======================================================================
    float* y = nullptr;          // [tokens][kHidden] FP32（REDUCE 写）

    // ======================================================================
    // [4] 性能调优参数
    // ======================================================================
    int max_spins = 0;           // 各同步门的自旋上限
    int skip_prologue = 0;       // [R38] >0=复用前序（同 graph 非首节点；见 kernel.cuh）

    // ======================================================================
    // [5] 调试 / debug
    // ======================================================================
    // （当前无专用字段；运行期计数缓冲经 MegaArgs 的 MX_DBG_COUNT 通道。）
};

// 仅前序（mega_kernel<false, NoTaskTail>）
void launch_prelude(const MegaLaunchArgs& a, mcStream_t stream);
// 仅计算链（mega_kernel<true, NoTaskTail>，不含 REDUCE）
void launch_compute(const MegaLaunchArgs& a, mcStream_t stream);
// [L1L2-SPLIT] 拆分阶段入口：L1 = 前序+DISPATCH+L1；L2 = 轻量前序+L2+COMBINE。
//   正常路径经 launch_compute + env MX_L1L2_SPLIT=1 自动拆分，一般无需直接调用。
void launch_l1(const MegaLaunchArgs& a, mcStream_t stream);
void launch_l2(const MegaLaunchArgs& a, mcStream_t stream);
// 仅 REDUCE（mega_reduce_kernel，grid=2*tokens）
void launch_reduce(const MegaLaunchArgs& a, mcStream_t stream);
// 全链（同 stream：mega_kernel<true> + REDUCE）
void launch_full(const MegaLaunchArgs& a, mcStream_t stream);
// 空 kernel（graph 链地板基准）
void launch_empty(int grid, int threads, int64_t smem, mcStream_t stream);
// 设备侧生成 lane 序 packed 权重（16B/lane 配对布局）
void fill_weights_lane(uint8_t* w, int n_rows, int k_cols, uint32_t seed,
                       mcStream_t stream);
// [诊断] 打印内核寄存器/本地内存占用
void dump_kernel_resources();

}  // namespace fp8mega
