#pragma once

#include <cute/numeric/math.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/exception.cuh>

namespace deep_gemm::layout {

static constexpr int kNumCandidateBlockMs = 7;
static constexpr int kCandidateBlockM[kNumCandidateBlockMs] = {8, 16, 32, 64, 96, 128, 192};
static constexpr int kMaxCandidateBlockM = 192;
static constexpr int kMinCandidateBlockM = 8;
static constexpr int kLCMCandidateBlockM = 384;

// Pool capacity for shared expert token pool: worst-case total tokens + per-expert BLOCK_M alignment padding, among all possible BLOCK_M
template <typename T>
MCTLASS_HOST_DEVICE constexpr T get_num_max_pool_tokens(T num_ranks, T num_max_tokens_per_rank, T num_topk,
                                                        T num_experts_per_rank) {
    const auto num_max_recv_tokens = num_ranks * num_max_tokens_per_rank;
    const auto num_max_experts_per_token = math::constexpr_min(num_topk, num_experts_per_rank);
    return math::constexpr_align(
        num_max_recv_tokens * num_max_experts_per_token + num_experts_per_rank * (static_cast<T>(kMaxCandidateBlockM) - 1),
        static_cast<T>(kLCMCandidateBlockM));
}

// SF pool capacity: all experts share a contiguous SF region, sized by pool blocks × SF_BLOCK_M
template <typename T>
MCTLASS_HOST_DEVICE constexpr T get_num_sf_ring_tokens(T num_ring_tokens, T block_m) {
    return (num_ring_tokens / block_m) * math::constexpr_align(block_m, static_cast<T>(128));
}

// Shared L2 input SF capacity: worst-case aligned SF pages over all candidate BLOCK_M.
template <typename T>
MCTLASS_HOST_DEVICE constexpr T get_num_max_shared_sf_tokens(const T& num_max_tokens_per_rank) {
    return math::constexpr_ceil_div<T>(num_max_tokens_per_rank, kMinCandidateBlockM) * 128;
}

// Per-token source metadata for combine write-back
struct TokenSrcMetadata {
    uint32_t rank_idx;
    uint32_t token_idx;
    uint32_t topk_idx;
};

struct Workspace {
    void* base;
    uint32_t num_ranks, num_experts;
    uint32_t num_experts_per_rank;
    uint32_t num_max_tokens_per_rank;
    uint32_t num_max_recv_tokens_per_expert;

    // Ring-buffer capacity used by reusable token/data buffers
    uint32_t num_ring_tokens;
    uint32_t num_ring_blocks;
    uint32_t num_shared_l2_pool_blocks;

    // Full-pool span used by non-ring token metadata
    uint32_t num_max_pool_tokens;

    // Keep grid/NVLink/schedule counters separated from expert counters.
    // NVIDIA L2 cache lines are 128B, and these counters are hot atomics.
    static constexpr uint64_t kNumBarrierSignalBytes = 128;

    Workspace() = default;

    MCTLASS_HOST_DEVICE
    Workspace(void* base,
              const uint32_t& num_ranks,
              const uint32_t& num_experts,
              const uint32_t& num_max_tokens_per_rank,
              const uint32_t& num_topk,
              const uint32_t& num_ring_tokens):
        base(base),
        num_ranks(num_ranks), num_experts(num_experts),
        num_max_tokens_per_rank(num_max_tokens_per_rank),
        num_ring_tokens(num_ring_tokens) {
        num_experts_per_rank = num_experts / num_ranks;
        num_max_recv_tokens_per_expert = num_ranks * num_max_tokens_per_rank;
        num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank);
        num_ring_blocks = num_ring_tokens / kMinCandidateBlockM;
        num_shared_l2_pool_blocks = math::ceil_div<uint32_t>(num_max_tokens_per_rank, kMinCandidateBlockM);
    }

    MCTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        uint64_t num_bytes = 0;

        // Barrier and in-kernel task scheduling counters
        num_bytes += kNumBarrierSignalBytes;

        // Expert send/recv count
        num_bytes += num_experts * sizeof(uint64_t) * 2;

        // Expert recv count sum
        num_bytes += num_experts_per_rank * sizeof(uint64_t);

        // L1 full token count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L1 empty block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L2 full block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // L2 empty block count (ring)
        num_bytes += num_ring_blocks * sizeof(uint32_t);

        // Shared L2 full block count
        num_bytes += num_shared_l2_pool_blocks * sizeof(uint32_t);

        // Dispatch pulling source token-topk
        num_bytes += num_experts_per_rank * num_ranks * num_max_recv_tokens_per_expert * sizeof(int);

        // Combine push source indices (full)
        num_bytes += num_max_pool_tokens * sizeof(TokenSrcMetadata);

        // Align to TMA descriptor requirements
        num_bytes = math::align<uint64_t>(num_bytes, 16);
        return num_bytes;
    }

    MCTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    // Grid sync counters: `kNumBarrierSignalBytes` layout
    // [ 0..15]: 4 x `uint32_t` grid sync counters
    // [16..20]: `uint32_t` NVLink barrier counter
    // [20..27]: 2 x `int` NVLink barrier signals (phase 0 and 1)
    // [28..31]: `uint32_t` L1 schedule task counter
    // [32..35]: `uint32_t` L2 schedule task counter
    // [36..39]: `uint32_t` shared L1 schedule task counter
    // [40..43]: `uint32_t` shared L2 schedule task counter
    // [44..127]: padding to isolate hot expert counters from barrier/schedule counters
    static constexpr uint32_t kNumMaxGridSyncCounters = 4;

    template <uint32_t kIndex = 0>
    MCTLASS_DEVICE
    uint32_t* get_grid_sync_count_ptr() const {
        DG_STATIC_ASSERT(kIndex < kNumMaxGridSyncCounters, "Grid sync index out of bounds");
        return static_cast<uint32_t*>(base) + kIndex;
    }

    MCTLASS_DEVICE
    uint32_t* get_nvl_barrier_counter_ptr() const {
        return static_cast<uint32_t*>(base) + kNumMaxGridSyncCounters;
    }

    MCTLASS_DEVICE
    int* get_nvl_barrier_signal_ptr(const uint32_t& phase) const {
        // NOTES: the signal is signed, as we may minus
        return math::advance_ptr<int>(base, (kNumMaxGridSyncCounters + 1) * sizeof(uint32_t) + phase * sizeof(int));
    }

    MCTLASS_DEVICE
    uint32_t* get_l1_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 28u);
    }

    MCTLASS_DEVICE
    uint32_t* get_l2_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 32u);
    }

    MCTLASS_DEVICE
    uint32_t* get_shared_l1_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 36u);
    }

    MCTLASS_DEVICE
    uint32_t* get_shared_l2_task_count_ptr() const {
        return math::advance_ptr<uint32_t>(base, 40u);
    }

    MCTLASS_DEVICE
    uint64_t* get_expert_send_count_ptr(const uint32_t& expert_idx = 0) const {
        return math::advance_ptr<uint64_t>(base, kNumBarrierSignalBytes) + expert_idx;
    }

    MCTLASS_DEVICE
    uint64_t* get_expert_recv_count_ptr(
        const uint32_t& rank_idx = 0, const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts) + rank_idx * num_experts_per_rank + expert_idx;
    }

    MCTLASS_DEVICE
    uint64_t* get_expert_recv_count_sum_ptr(const uint32_t& expert_idx = 0) const {
        return get_expert_send_count_ptr(num_experts * 2) + expert_idx;
    }

    MCTLASS_DEVICE
    uint32_t* get_l1_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_expert_recv_count_sum_ptr(num_experts_per_rank);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    MCTLASS_DEVICE
    uint32_t* get_l1_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l1_full_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    MCTLASS_DEVICE
    uint32_t* get_l2_full_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l1_empty_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    MCTLASS_DEVICE
    uint32_t* get_l2_empty_count_ptr(const uint32_t& ring_block_idx = 0) const {
        const auto base = get_l2_full_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + ring_block_idx;
    }

    MCTLASS_DEVICE
    uint32_t* get_shared_l2_full_count_ptr(const uint32_t& block_idx = 0) const {
        const auto base = get_l2_empty_count_ptr(num_ring_blocks);
        return reinterpret_cast<uint32_t*>(base) + block_idx;
    }

    // For dispatch pulling
    MCTLASS_DEVICE
    uint32_t* get_src_token_topk_idx_ptr(
        const uint32_t& expert_idx = 0, const uint32_t& rank_idx = 0, const uint32_t& token_idx = 0) const {
        const auto base = get_shared_l2_full_count_ptr(num_shared_l2_pool_blocks);
        return reinterpret_cast<uint32_t*>(base) +
            expert_idx * (num_ranks * num_max_recv_tokens_per_expert) +
            rank_idx * num_max_recv_tokens_per_expert + token_idx;
    }

    // For combine usages (full)
    MCTLASS_DEVICE
    TokenSrcMetadata* get_token_src_metadata_ptr(const uint32_t& pool_token_idx = 0) const {
        const auto base = reinterpret_cast<TokenSrcMetadata*>(get_src_token_topk_idx_ptr(num_experts_per_rank));
        return base + pool_token_idx;
    }
};

struct Data {
    uint32_t num_bytes;
    bool require_tma_alignment;
    void* base;

    Data() = default;

    MCTLASS_HOST_DEVICE
    constexpr explicit Data(
        const uint32_t& num_bytes,
        const bool& require_tma_alignment = true,
        void* base = nullptr) :
        num_bytes(num_bytes), require_tma_alignment(require_tma_alignment), base(base) {
        DG_UNIFIED_ASSERT(num_bytes % 16 == 0 or not require_tma_alignment);
    }

    template <typename dtype_t = uint32_t>
    MCTLASS_HOST_DEVICE constexpr dtype_t get_num_bytes() const {
        return static_cast<dtype_t>(num_bytes);
    }

    template <typename dtype_t = void>
    MCTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    MCTLASS_HOST_DEVICE void set_base_ptr(void* ptr) {
        base = ptr;
    }
};

struct Buffer {
    Data data_layout;
    uint32_t num_ranks;
    uint32_t num_max_tokens_per_rank;

    void* base;

    Buffer() = default;

    MCTLASS_HOST_DEVICE
    Buffer(const Data& data_layout,
           const uint32_t& num_ranks,
           const uint32_t& num_max_tokens_per_rank,
           void* base = nullptr) :
        data_layout(data_layout),
        num_ranks(num_ranks), num_max_tokens_per_rank(num_max_tokens_per_rank),
        base(base) {}

    MCTLASS_HOST_DEVICE
    uint64_t get_num_bytes_per_rank() const {
        return num_max_tokens_per_rank * data_layout.get_num_bytes<uint64_t>();
    }

    MCTLASS_HOST_DEVICE
    uint64_t get_num_bytes() const {
        return get_num_bytes_per_rank() * num_ranks;
    }

    template <typename dtype_t = void>
    MCTLASS_HOST_DEVICE dtype_t* get_base_ptr() const {
        return static_cast<dtype_t*>(base);
    }

    MCTLASS_HOST_DEVICE
    void* get_end_ptr() const {
        return math::advance_ptr(base, get_num_bytes());
    }

    MCTLASS_HOST_DEVICE
    Buffer get_rank_buffer(const uint32_t& rank_idx) const {
        return {
            data_layout,
            1, num_max_tokens_per_rank,
            math::advance_ptr(base, get_num_bytes_per_rank() * rank_idx)
        };
    }

    MCTLASS_HOST_DEVICE
    Data get_data_buffer(const uint32_t& token_idx, const bool& global = false) const {
        DG_DEVICE_ASSERT(num_ranks == 1 or global);
        return Data(
            data_layout.num_bytes,
            data_layout.require_tma_alignment,
            math::advance_ptr(base, data_layout.get_num_bytes<uint64_t>() * token_idx)
        );
    }
};

struct MegaMoEBuffer {
    Workspace workspace;

    // Input buffers (per-rank)
    Buffer input_token_buffer,
           input_sf_buffer,
           input_topk_idx_buffer,
           input_topk_weights_buffer;

    // Routed expert ring buffers
    // NOTE: shared L1 tokens reuse `input_token_buffer`.
    Buffer shared_l1_token_buffer, shared_l1_sf_buffer,
           shared_l2_token_buffer, shared_l2_sf_buffer;

    // Routed expert ring buffers
    Buffer l1_token_buffer,
           l1_sf_buffer,
           l1_topk_weights_buffer,
           l2_token_buffer,
           l2_sf_buffer,
           combine_token_buffer;

    MCTLASS_HOST_DEVICE
    MegaMoEBuffer(void* base,
                  const uint32_t& hidden,
                  const uint32_t& intermediate_hidden,
                  const uint32_t& num_ranks,
                  const uint32_t& num_experts,
                  const uint32_t& num_max_tokens_per_rank,
                  const uint32_t& num_topk,
                  const uint32_t& num_ring_tokens,
                  const uint32_t& num_sf_ring_tokens,
                  const bool& with_sf,
                  const uint32_t& num_shared_experts = 0) {
        // Workspace
        workspace = Workspace(base, num_ranks, num_experts,
                              num_max_tokens_per_rank, num_topk, num_ring_tokens);

        // Shared
        const auto shared_intermediate_hidden = intermediate_hidden * num_shared_experts;
        const auto num_max_shared_sf_tokens = with_sf ? get_num_max_shared_sf_tokens(num_max_tokens_per_rank) : 0u;

        // Layouts
        const uint32_t num_mma_elem_bytes = with_sf ? 1 : 2;
        const auto input_token_layout = layout::Data(hidden * num_mma_elem_bytes);
        const auto bf16_token_layout = layout::Data(hidden * 2);
        const auto intermediate_token_layout = layout::Data(intermediate_hidden * num_mma_elem_bytes);
        const auto shared_intermediate_token_layout = layout::Data(shared_intermediate_hidden * num_mma_elem_bytes);
        const auto input_sf_layout = layout::Data(with_sf ? hidden / 32 : 0);
        const auto intermediate_sf_layout = layout::Data(with_sf ? intermediate_hidden / 32 : 0);
        const auto shared_intermediate_sf_layout = layout::Data(with_sf ? shared_intermediate_hidden / 32 : 0);
        const auto input_topk_idx_layout = layout::Data(num_topk * sizeof(int64_t), false);
        const auto input_topk_weights_layout = layout::Data(num_topk * sizeof(float), false);
        const auto l1_topk_weights_layout = layout::Data(sizeof(float), false);

        // Input buffers
        input_token_buffer = Buffer(
            input_token_layout, 1, num_max_tokens_per_rank,
            workspace.get_end_ptr());
        input_sf_buffer = Buffer(
            input_sf_layout, 1, num_max_tokens_per_rank,
            input_token_buffer.get_end_ptr());
        input_topk_idx_buffer = Buffer(
            input_topk_idx_layout, 1, num_max_tokens_per_rank,
            with_sf ? input_sf_buffer.get_end_ptr() : input_token_buffer.get_end_ptr());
        input_topk_weights_buffer = Buffer(
            input_topk_weights_layout, 1, num_max_tokens_per_rank,
            input_topk_idx_buffer.get_end_ptr());

        // Shared expert buffers
        shared_l1_token_buffer = input_token_buffer;
        shared_l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_shared_experts > 0 ? num_max_shared_sf_tokens : 0,
            input_topk_weights_buffer.get_end_ptr());
        shared_l2_token_buffer = Buffer(
            shared_intermediate_token_layout, 1, num_shared_experts > 0 ? num_max_tokens_per_rank : 0,
            with_sf ? shared_l1_sf_buffer.get_end_ptr() : input_topk_weights_buffer.get_end_ptr());
        shared_l2_sf_buffer = Buffer(
            shared_intermediate_sf_layout, 1, num_shared_experts > 0 ? num_max_shared_sf_tokens : 0,
            shared_l2_token_buffer.get_end_ptr());

        // Routed expert ring buffers
        l1_token_buffer = Buffer(
            input_token_layout, 1, num_ring_tokens,
            num_shared_experts > 0 ?
                (with_sf ? shared_l2_sf_buffer.get_end_ptr() : shared_l2_token_buffer.get_end_ptr()) :
                input_topk_weights_buffer.get_end_ptr()
        );
        l1_sf_buffer = Buffer(
            input_sf_layout, 1, num_sf_ring_tokens,
            l1_token_buffer.get_end_ptr());
        l1_topk_weights_buffer = Buffer(
            l1_topk_weights_layout, 1, num_ring_tokens,
            with_sf ? l1_sf_buffer.get_end_ptr() : l1_token_buffer.get_end_ptr());

        l2_token_buffer = Buffer(
            intermediate_token_layout, 1, num_ring_tokens,
            l1_topk_weights_buffer.get_end_ptr());
        l2_sf_buffer = Buffer(
            intermediate_sf_layout, 1, num_sf_ring_tokens,
            l2_token_buffer.get_end_ptr());

        combine_token_buffer = Buffer(
            bf16_token_layout, num_topk + (num_shared_experts > 0 ? 1u : 0u), num_max_tokens_per_rank,
            with_sf ? l2_sf_buffer.get_end_ptr() : l2_token_buffer.get_end_ptr());
    }

    MCTLASS_HOST_DEVICE
    int64_t get_num_bytes() const {
        return static_cast<uint8_t*>(combine_token_buffer.get_end_ptr())
               - static_cast<uint8_t*>(workspace.base);
    }
};

//============================================================================
// 融合 MoE FFN kernel 的【对称区 scratch 布局】
//
// 追加在 MegaMoEBuffer 之后（sym_buffer 尾部）。跨 rank 寻址经 SymBuffer::map。
// 只允许 constexpr / 纯函数（host 与 device 共同 include，同源）。
//
// 布局（sym_buffer 尾部追加，跨 rank 寻址经 SymBuffer::map）：
//   [ 控制区·本地页 ]（连续，host 每次 launch 前 cudaMemsetAsync 清零）
//     route_slot_counts[E]        int32   ROUTE_BUILD 本地串行占槽分配器
//     grid_barriers[8]        int32   【已弃用】原 RMW 计数器槽（barrier 已改 flag 版，
//                                    flag 区复用本页 padding）；字段保留以免 offset 变动
//     xrelease[2]             int32   XBAR_ROUTE_READY/XBAR_COMBINE 的 block0→其余 block 放行标志
//     expert_overflow_flag[1] int32   【已废弃·滑动窗口版】超容量不再 clamp 丢弃，
//                                     字段保留防 offset 变动；kernel 不再写 1
//     dbg_report[R][16][2]    int32   有界自旋超时诊断（rank_id*16+slot 索引，随 R 缩放）
//     phase_ticks[8]          uint64  分阶段耗时（block0/tid0 累计 clock64 增量）
//     block_marks[64][10]     uint64  per-block checkpoint（定位 straggler block）
//   [ 控制区·peer 页 ]（4KB 页对齐，与本地控制字段分页，方案 A）
//     barrier_slots[R][R]     uint32 细粒度 push 矩阵：[i][j] 只由 rank j 用 __stwt 写。
//                                    单调 epoch：仅 buffer 创建时清零一次，launch 前
//                                    清零范围【不含】本页（host memset 到 barrier_slots 止）
//   [ 数据区 ]（不清理，每次 launch 全新覆写；expert_tokens 起再 4KB 对齐与 combine_rows 分页）
//     src_index_table[EPR][R][slot_cap]  uint32  ROUTE_BUILD 远端写 / DISPATCH 本地读
//     expert_recv_counts[R][EPR]                uint32  PUBLISH_COUNTS 远端 __stwt / POOL_LAYOUT 本地读
//     combine_rows[TOPK][slot_cap][K_OUT] bf16  COMBINE 远端写 / REDUCE 本地读
//     expert_tokens[EPR][M_PAD][K_IN]           bf16  DISPATCH 拉取 / L1 输入
//     token_route_weights[EPR][M_PAD]                 float  routing weight
//     token_src_meta[EPR][M_PAD]              TokenSrcMetadata
//     ffn_hidden[EPR][M_PAD][K_HIDDEN]            bf16  L1 输出 / L2 输入
//     ffn_out[EPR][M_PAD][K_OUT]             bf16  L2 输出 / combine 源
//
// 语义：
//   slot_cap = num_max_tokens_per_rank（一个源 rank 发往同一 expert 的 token 数 ≤ T）
//   M_PAD = expert_token_cap（每 expert pool 行数，默认 8，DG_MEGA_CAP 可调；
//           count>M_PAD 置 overflow 并 clamp；kernel 内按 count 截断 M 循环）
//   所有 offset 按 256 字节对齐（细粒度跨 rank 写通道与 uint4 对齐要求）。
//============================================================================

// 融合 kernel 固定维度（hidden/intermediate_hidden 可配置）
// kExpertTokenCap == 每专家 token 容量的【默认值】；实际值由 DG_MEGA_CAP 环境变量
// 决定（host 侧 get_mega_moe_expert_token_cap，见 jit_kernels/impls/mx_bf16_mega_moe.hpp），
// 经参数传入 compute_mega_moe_scratch 与 kernel 模板参 kNumMaxTokensPerExpert——
// 契约：同进程内 SymBuffer 尺寸计算与 kernel 实例化必须同值（首读缓存保证）。
static constexpr int kExpertTokenCap = 8;            // 每 expert pool 行数（默认）
static constexpr int kMegaMoeMaxRanks = 72;          // 与 SymBuffer kNumMaxRanks 对齐
static constexpr int kMegaMoeNumGridBarriers = 8;
static constexpr int kMegaMoeNumXRelease = 2;

// bf16 字节数（避免依赖设备头文件）
static constexpr int kMegaMoeBf16Size = 2;
static constexpr int kMegaMoeTokenSrcMetaSize = (int)sizeof(TokenSrcMetadata);  // 12
// 页对齐粒度：peer 可写区与本地控制/数据区分页（方案 A，见 compute_mega_moe_scratch）
static constexpr int kMegaMoePageAlign = 4096;
// per-block checkpoint：grid 上限（grid = EPR*BPE ≤ num_sms，留余量）与打点个数
static constexpr int kMegaMoeMaxGridBlocks = 64;
static constexpr int kMegaMoeNumBlockMarks = 10;

// ---------------- rank 内 barrier 契约常量（host/device 同源，勿另抄） ----------------
// 单调 seq = epoch_base * kBarSeqStride + 固定偏移（epoch_base 每 launch +2，
// 每 launch 有效步进 = 2*kBarSeqStride；launch 内最大累积偏移（窗口数×组数）
// « 每步进，跨 launch 不撞号。滑动窗口版：offset 预算按「窗口数 × 组数(≤16)
// +常数 < 2*stride」规划，host 侧在每次 launch 前按热点最坏路由断言。
// 【2026-09-18】stride 4096→65536：旧值在 32 rank/256 专家/topk8 的压测下，
// 按 host 最坏窗口估算（R×min(T×topk, cap_align)/W，cap 按 384 对齐）越界，
// 触发 DG_HOST_ASSERT(max_sequence_offset < 2*stride)。65536 把可支持的
// 最坏窗口数从 ~255 提到 ~8191（g=16），覆盖 T_max≤16k 的 32-rank 生产档。
static constexpr uint32_t kBarSeqStride = 65536;
static constexpr uint32_t kBarSeqRouteReady = 2;    // 原 BX1：路由信息全 rank 就绪
static constexpr uint32_t kBarSeqDispatch = 3;      // 原 GB_P3（滑动窗口：旗值改为累积式 = 基值+窗口序号）
static constexpr uint32_t kBarSeqL1 = 4;            // 原 GB_L1
// pipe L1→L2 进度旗基（滑动窗口：第 w 窗第 g 组的旗值 = seq_base + kBarSeqL1Progress
//   + w*kGroupsPerBlockL1 + 组内序——跨窗单调累积，消费侧按同公式推导阈值）
static constexpr uint32_t kBarSeqL1Progress = 5;
static constexpr uint32_t kBarSeqCombineRegion = 10;  // 原 P5 region：10+dst
static constexpr uint32_t kBarSeqCombine = 100;     // 原 BX2
// local_bar_flags buffer（普通显存，host 分配）布局，单位 int32：
// [0..63] arrive flags、[64] release、[65..66] cross_rank_release[2]
// 【P0 切片任务化】同专家多窗可被不同 block 并发执行（窗口并发环
// kMegaMoeWinRing），旗子不再按 blockIdx 索引——改按 (expert, ring_slot, slice)：
//   flag_idx = (expert * kMegaMoeWinRing + w % kMegaMoeWinRing) * NSLICE + slice
// 旗值沿用窗口累积平移（同槽位的 w 与 w+ring 相差 ring，旗值严格递增，无假通过）。
// 容量契约（host 断言）：EPR × kMegaMoeWinRing × NSLICE ≤ 256。
static constexpr int kMegaMoeWinRing = 8;   // [RING8] 同专家并发窗上限（超出由 l2_done 门串行化）。
                                            // §5.20 均衡负载下 ring 2/4/8 无差异；倾斜场景
                                            // 2026-09-23 实测（单专家 512 行、WIN=64、nt=16、
                                            // SLICE=4）：2→4 墙钟中位 1.968→1.625ms（−17%，
                                            // 3 轮全对）；均衡 nt=16 +0.6%、nt=1 持平（噪声内）。
                                            // 容量：EPR×ring×NSLICE=128≤256；
                                            // kBarWinReleaseIdx+64=899≤1024。
static constexpr int kBarReleaseIdx = kMegaMoeMaxGridBlocks;       // 64
static constexpr int kBarXreleaseIdx = kMegaMoeMaxGridBlocks + 1;  // 65
static constexpr int kBarSliceFlagsIdx = kMegaMoeMaxGridBlocks + 3;  // 67：DISPATCH 分片就绪旗区
static constexpr int kBarSliceFlagsLen = 256;   // EPR×ring×NSLICE ≤ 256（host 断言）
// 323：L1→L2 进度旗区（同 slice 旗索引方式）
static constexpr int kBarL1ProgressIdx = kBarSliceFlagsIdx + kBarSliceFlagsLen;
static constexpr int kBarL1ProgressLen = 256;
// 579：ring 槽复用门——(expert, ring_slot) 的 L2 完成计数（计数型，XBAR 前由 block0
// 清零；旗值型门放不下「同一槽位被多代窗口复用」的计数语义）
static constexpr int kBarL2DoneIdx = kBarL1ProgressIdx + kBarL1ProgressLen;
// [RING8] 容量只需 EPR×ring ≤ 256（与切片旗区同一契约：EPR×ring×NSLICE ≤ 256
// 在 host 断言；NSLICE≥1 时已覆盖）。不能写成 kMegaMoeMaxGridBlocks×ring——
// ring>4 时会让后续旗区整体后移并越界（RING8 首版即因此编译失败）。
static constexpr int kBarL2DoneLen = 256;   // EPR×ring ≤ 256
// 835：窗口 release 旗区【已弃用：window_expert_barrier 在队列形态下无调用，
// 常量保留防旧代码复活时与切片旗区撞号】
static constexpr int kBarWinReleaseIdx = kBarL2DoneIdx + kBarL2DoneLen;
static constexpr int kBarFlagsNumInt32 = 1024;
static_assert(kBarWinReleaseIdx + kMegaMoeMaxGridBlocks <= kBarFlagsNumInt32,
              "local_bar_flags 容量不足以放下全部旗区");

// ---- FP8 (feature/c600ul-fp8-megamoe) 追加旗区 ----
// REDUCE 前跨 rank combine 完成 arrive 旗（[grid≤64]）。不复用 [0..63] 通用 arrive
// flags：热点专家的多窗口 barrier 会在同槽写更大窗口序号，单调比较会误判提前到达。
// 放在 C600UL 已弃用保留区（kBarWinReleaseIdx=707..770）之后，不与切片/L1/L2 撞号。
static constexpr int kBarFinalCombineArriveIdx =
    kBarWinReleaseIdx + kMegaMoeMaxGridBlocks;      // 771..834
// 动态任务队列控制旗。FP8 任务前缀表已迁入 scratch.prefix_tbl（见 MegaMoeScratch），
// 因此不再占用 bar_flags；这三个旗与 C600UL 内核使用的 1018/1019/1020 同址。
static constexpr int kBarTaskReadyIdx = 1018;
static constexpr int kBarWashSinkIdx = 1019;
static constexpr int kBarTaskCounterIdx = 1020;
static_assert(kBarFinalCombineArriveIdx + kMegaMoeMaxGridBlocks <= kBarTaskReadyIdx,
              "FP8 最终 combine arrive 旗区与动态任务控制旗重叠");
static_assert(kBarTaskCounterIdx < kBarFlagsNumInt32,
              "local_bar_flags 容量不足以放下动态任务控制旗");
// ---- FP8 设备侧 epoch（fp8xfp8；CUDA graph 安全，替代 host 传入的 epoch_base）----
// epoch_slot 持有【本 launch】的 epoch_base：由上一 launch 最后完成的 block
// （finisher）预发布，buffer 创建时清零 → launch0 读得 0，与旧 host
// （launch_count%period)*2 语义逐值一致。launch 内没有任何代码改写 epoch_slot
// （finisher 只在全 block 完成后才写下一 epoch），各 block 入口一次 volatile
// 读即可，无自旋、无新增同步。done_ctr 每 launch 由 block0 在 XBAR 前清零
// （与任务计数器预置同点，可见性随 XBAR 的 fence/release 链传递）；各 block
// 收尾 tid0 atomicAdd，done==gridDim.x-1 的 finisher 推算下一 epoch，并在
// seq 空间回绕点清零普通旗区——此刻全 block 已收尾，清零不与任何读写并发。
// CUDA graph 重放会真实执行本协议 → epoch 逐 launch 递增，eager/graph/混跑
// 语义一致（host 传参在 graph 下被冻结，是本改造的动因）。
static constexpr int kFp8BarEpochSlotIdx = kBarFlagsNumInt32;        // 1024
static constexpr int kFp8BarDoneCtrIdx = kBarFlagsNumInt32 + 1;      // 1025
static constexpr int kFp8BarFlagsNumInt32 = kBarFlagsNumInt32 + 8;   // 1032
// 与 host 侧 _FP8_EPOCH_PERIOD（mxmegakernel/mega/__init__.py）同源：epoch 取
// [0, 2*period) 内偶数，保证 base+1/base+2 不回绕到 0/UINT32_MAX（零初始化态）。
static constexpr uint32_t kFp8EpochPeriod = (1u << 31) - 1u;
// seq_base=epoch*kBarSeqStride 的 uint32 回绕周期（epoch 单位，2^32/65536=2^16；
// epoch 每 launch +2 → 每 2^15=32768 次 launch 回绕一次）。
// 跨回绕点必须清零普通旗区（fp8fp4 由 host 在同点 bar_flags.zero_()，
// fp8xfp8 由 finisher 在设备侧完成）。
static constexpr uint32_t kFp8SeqWrapEpochPeriod =
    (uint32_t)((1ull << 32) / kBarSeqStride);
// combine_sums 每行 slice 校验槽上限：BPE ≤ num_sms/EPR ≤ 32（grid=EPR*BPE ≤ 32 SM 保证）
static constexpr int kMegaMoeMaxCombineSlices = 32;
// 窗口预算：单 launch 内 L1 进度旗最大偏移 = kBarSeqL1Progress + w×g，必须落在
// 2*kBarSeqStride 的 launch 步进内（epoch 每 launch +2）。g = (2I/BN)/BPE 由每个
// 实例的几何决定（生产档 I=2048/BN=256/BPE=1 → g=16；BN=128 或更大 I 会更小/更大），
// host 侧按实际 config 精确断言 ceil(R×min(T×topk, cap_align)/W)×g < 2*stride，
// 先于 device 拒绝越界 launch。kMegaMoeMaxWindows 只是 device 侧上界告警：
// 取 g=16（常见档）的 w 上限 8191+1，实际包线以 host 断言为准；达到该值即已
// 耗尽全部预算，继续跑会撞下一 launch 的 seq 区间。
static constexpr uint32_t kMegaMoeMaxWindows = 8192;
static constexpr uint32_t kMegaMoeWindowBudget =
    (2u * kBarSeqStride - kBarSeqL1Progress) / 16u;   // 8191（g=16 参考值，未使用）

struct MegaMoeScratch {
    int64_t route_slot_counts;          // [E] int32
    int64_t grid_barriers;          // [8] int32【已弃用：barrier 改 flag 版，字段保留防 offset 变动】
    int64_t xrelease;               // [2] int32
    int64_t expert_overflow_flag;   // [1] int32
    int64_t dbg_report;             // [R][16][2] int32（随 num_ranks 缩放）
    int64_t barrier_slots;          // [R][R] uint32
    int64_t phase_ticks;            // [8] uint64（分阶段耗时）
    int64_t block_marks;            // [kMegaMoeMaxGridBlocks][10] uint64（per-block checkpoint）
    int64_t control_end;            // 控制区结束（= 数据区起点，host 清零范围）

    int64_t src_index_table;        // [EPR][R][slot_cap] uint32
    int64_t expert_recv_counts;            // [R][EPR] uint32
    int64_t combine_rows;         // [TOPK][slot_cap][K_OUT] bf16
    int64_t combine_sums;         // [TOPK][slot_cap][kMegaMoeMaxCombineSlices] uint32
                                  // 行级校验槽（§6.4 修复）：行 (k,t) 的 slice b 的
                                  // 推送方写 bitsum(slice)（slice0 折入 epoch）；REDUCE
                                  // 读行校验 Σpartial == bitsum(行)+epoch，未齐自旋重读
    int64_t expert_tokens;                 // [EPR][kMegaMoeWinRing][M_PAD][K_IN] bf16（共享槽）
    int64_t token_route_weights;                 // [32 blocks][M_PAD] float（每块槽）
    int64_t token_src_meta;              // [32 blocks][M_PAD] TokenSrcMetadata（每块槽）
    int64_t ffn_hidden;                      // [EPR][kMegaMoeWinRing][M_PAD][K_HIDDEN] bf16（共享槽）
    int64_t ffn_out;                    // [32 blocks][M_PAD][K_OUT] bf16（每块槽）
    int64_t prefix_tbl;                 // [EPR][R+1] int32（P2 前缀表：task_setup 单一生产者写 /
                                        // 任务解码读；从 bar_flags 迁入 scratch——P0 切片旗区
                                        // 需占用 bar_flags 空间，且大 EPR 时 bar_flags 放不下）
    // ---- FP8 (feature/c600ul-fp8-megamoe) 追加字段 ----
    // 追加在全部 BF16 字段之后，bf16 路径 (fp8_layout=false) 的 offset 与 total_size 不变。
    int64_t expert_token_sf;            // [staging_slots][M_PAD][K_IN/32] uint8（with_sf）
    int64_t ffn_hidden_sf;              // [staging_slots][M_PAD][K_HIDDEN/32] uint8（with_sf）
    int64_t prefolded_input_tokens;     // [slot_cap][K_IN] fp8e4m3（with_sf）
    int64_t total_size;
};

// 256 字节对齐（细粒度跨 rank 写通道与 uint4 对齐要求）
template <typename T>
MCTLASS_HOST_DEVICE constexpr T mega_moe_align_up(T x, T a) {
    return (x + a - 1) / a * a;
}

// 计算 scratch 布局。仅依赖 host 侧已知的维度，全部 constexpr。
//   num_ranks / num_experts（全局）/ num_max_tokens_per_rank / num_topk /
//   hidden（=K_IN=K_OUT）/ intermediate_hidden（=K_HIDDEN）/ expert_window
//   （每专家滑动窗口行数 = DG_MEGA_WIN(bf16)/DG_MEGA_WIN(fp8)，host/device 必须同值）
//   with_sf   ：FP8 是否携带 scale factor 缓冲
//   fp8_layout：false=BF16(C600UL) 共享池布局；true=FP8 staging 每块布局
//   with_prefold：是否为 scale-free 的 bf16→fp8 预转换副本分配槽位
//     （bf16xfp8 变体：with_sf=false 但 dispatch 仍拉预转换后的 fp8 行）
// 契约：同一次 launch 的 host/device 必须以完全相同的 (window,with_sf,fp8_layout,with_prefold) 调用。
template <typename T>
MCTLASS_HOST_DEVICE constexpr MegaMoeScratch compute_mega_moe_scratch(
        T num_ranks, T num_experts, T num_max_tokens_per_rank, T num_topk,
        T hidden, T intermediate_hidden, T expert_window,
        bool with_sf = false, bool fp8_layout = false,
        bool with_prefold = false) {
    constexpr T kAlign = 256;
    const T epr = num_experts / num_ranks;
    const T slot_cap = num_max_tokens_per_rank;
    const T kIn = hidden, kHidden = intermediate_hidden, kOut = hidden;

    MegaMoeScratch o;
    T off = 0;
    o.route_slot_counts = off;
    off = mega_moe_align_up(off + (T)num_experts * (T)sizeof(int), kAlign);
    o.grid_barriers = off;
    off = mega_moe_align_up(off + (T)kMegaMoeNumGridBarriers * (T)sizeof(int), kAlign);
    o.xrelease = off;
    off = mega_moe_align_up(off + (T)kMegaMoeNumXRelease * (T)sizeof(int), kAlign);
    o.expert_overflow_flag = off;
    off = mega_moe_align_up(off + (T)sizeof(int), kAlign);
    o.dbg_report = off;
    off = mega_moe_align_up(off + num_ranks * (T)(16 * 2) * (T)sizeof(int), kAlign);
    o.phase_ticks = off;
    off = mega_moe_align_up(off + (T)(8) * (T)sizeof(uint64_t), kAlign);
    o.block_marks = off;
    off = mega_moe_align_up(off + (T)kMegaMoeMaxGridBlocks * (T)kMegaMoeNumBlockMarks
                                * (T)sizeof(uint64_t), kAlign);
    // 方案 A（多进程 ATU 干扰隔离）：自 barrier_slots 起的【peer 可写区】对齐到独立
    // 4KB 页，与本地自旋/原子控制字段（route_slot_counts..phase_ticks）分页——避免 peer
    // 跨进程写（ATU 路径）干扰本地 grid barrier / xrelease 的原子与 volatile 自旋。
    // 要求 scratch 基址 4KB 对齐（host 侧 get_symm_buffer_size_for_mega_moe 对齐
    // num_required_bytes 保证，bf16_mega_moe 里有绝对地址断言兜底）。
    off = mega_moe_align_up(off, (T)kMegaMoePageAlign);
    o.barrier_slots = off;
    off = mega_moe_align_up(off + (T)num_ranks * (T)num_ranks * (T)sizeof(uint32_t), kAlign);
    o.control_end = off;

    o.src_index_table = off;
    off = mega_moe_align_up(off + (T)epr * (T)num_ranks * slot_cap * (T)sizeof(uint32_t), kAlign);
    o.expert_recv_counts = off;
    // 【drain 竞态修复】行宽 EPR→EPR+1：行尾追加校验槽 = ~(Σ行内计数)，
    // POOL_LAYOUT 读行时自旋校验 Σ==~校验槽，未齐则重读——「读者可验证的不变量」
    // 闭合远端 __stwt 写的可见性（对端写不失效宿主 L2，volatile/ldcv/readback
    // 均无效，2026-08-27 R2 首现；详见 SUMMARY §6.4）
    // 【PC 向量化】EPR+1 再补齐到 4 word（16B 对齐）：publish 可按 uint4 写，
    // 每 dst 从 9 次 4B 散写降为 3 次 16B；校验槽仍在 index EPR，padding 不读。
    const T recv_row_words = ((epr + 1) + 3) & ~(T)3;
    off = mega_moe_align_up(off + (T)num_ranks * recv_row_words * (T)sizeof(uint32_t), kAlign);
    o.combine_rows = off;
    off = mega_moe_align_up(off + (T)num_topk * slot_cap * kOut * (T)kMegaMoeBf16Size, kAlign);
    // 行级校验槽（§6.4 修复，combine 层）：紧随 combine_rows，同处 peer 可写区
    o.combine_sums = off;
    off = mega_moe_align_up(off + (T)num_topk * slot_cap
                                * (T)kMegaMoeMaxCombineSlices * (T)sizeof(uint32_t), kAlign);
    // 本地数据区（expert_tokens 起）同样与 peer 可写的 combine_rows 分页
    off = mega_moe_align_up(off, (T)kMegaMoePageAlign);
    o.expert_tokens = off;
    if (not fp8_layout) {
        // ===== BF16 (C600UL) staging =====
        // 【P0 切片任务化】staging 两类槽位：
        //  - 共享槽（pool/ffn_hidden）：按 (expert, w % kMegaMoeWinRing) 索引——
        //    同 (e,w) 的 NSLICE 个 slice 任务跨块共享一份（L1 列拼接/L2 全 K 读是真依赖），
        //    ring 允许同专家多窗并发；槽复用由 l2_done 门保护。
        //  - 每块槽（weights/meta/ffn_out）：按 blockIdx.x 索引——只被本块自己的
        //    L1/COMBINE 消费（grid ≤ 64，取 max(EPR,32) 保守）。
        const T pool_slots = epr * (T)kMegaMoeWinRing;
        const T block_slots = (T)(epr < 32 ? 32 : epr);
        off = mega_moe_align_up(off + pool_slots * expert_window * kIn * (T)kMegaMoeBf16Size, kAlign);
        o.token_route_weights = off;
        off = mega_moe_align_up(off + block_slots * expert_window * (T)sizeof(float), kAlign);
        o.token_src_meta = off;
        off = mega_moe_align_up(off + block_slots * expert_window * (T)kMegaMoeTokenSrcMetaSize, kAlign);
        o.ffn_hidden = off;
        off = mega_moe_align_up(off + pool_slots * expert_window * kHidden * (T)kMegaMoeBf16Size, kAlign);
        o.ffn_out = off;
        off = mega_moe_align_up(off + block_slots * expert_window * kOut * (T)kMegaMoeBf16Size, kAlign);
        // 前缀表（P2）：task_setup 单一生产者写、任务解码读；从 bar_flags 迁入
        // （P0 切片旗区占用 bar_flags，且 EPR×(R+1) 大时 bar_flags 放不下）
        o.prefix_tbl = off;
        off = mega_moe_align_up(off + epr * (num_ranks + 1) * (T)sizeof(int), kAlign);
        o.expert_token_sf = 0;
        o.ffn_hidden_sf = 0;
        o.prefolded_input_tokens = 0;
    } else {
        // ===== FP8 (feature/c600ul-fp8-megamoe) staging =====
        // 【P0 切片任务化】与 BF16 分支同一套槽位语义：
        //  - 共享槽（expert_tokens/expert_token_sf/ffn_hidden/ffn_hidden_sf）：
        //    按 (expert, w % kMegaMoeWinRing) 索引——同 (e,w) 的 NSLICE 个 slice
        //    任务跨块共享一份（dispatch 各写自己 K 列片、L1 拼整行 / L2 全 K 读是真
        //    依赖）；ring 允许同专家多窗并发，槽复用由 l2_done 门保护。
        //  - 每块槽（weights/meta/ffn_out）：按 blockIdx.x 索引——只被本块自己的
        //    L1/COMBINE 消费（grid ≤ 64，取 max(EPR,32) 保守）。
        const T pool_slots = epr * (T)kMegaMoeWinRing;
        const T block_slots = (T)(epr < 32 ? 32 : epr);
        off = mega_moe_align_up(off + pool_slots * expert_window * kIn * (T)kMegaMoeBf16Size, kAlign);
        o.token_route_weights = off;
        off = mega_moe_align_up(off + block_slots * expert_window * (T)sizeof(float), kAlign);
        o.token_src_meta = off;
        off = mega_moe_align_up(off + block_slots * expert_window * (T)kMegaMoeTokenSrcMetaSize, kAlign);
        o.ffn_hidden = off;
        off = mega_moe_align_up(off + pool_slots * expert_window * kHidden * (T)kMegaMoeBf16Size, kAlign);
        o.ffn_out = off;
        off = mega_moe_align_up(off + block_slots * expert_window * kOut * (T)kMegaMoeBf16Size, kAlign);
        // FP8xFP4-only scale buffers：with_sf=false 时保留 offset 不推进。
        o.expert_token_sf = off;
        if (with_sf)
            off = mega_moe_align_up(
                off + pool_slots * expert_window * (kIn / (T)32), kAlign);
        o.ffn_hidden_sf = off;
        if (with_sf)
            off = mega_moe_align_up(
                off + pool_slots * expert_window * (kHidden / (T)32), kAlign);
        o.prefolded_input_tokens = off;
        if (with_sf || with_prefold)
            off = mega_moe_align_up(off + slot_cap * kIn * (T)sizeof(uint8_t), kAlign);
        // 前缀表迁入 scratch（FP8 任务解码读；原 bar_flags 槽位让给 C600UL 切片/L1/L2 旗区）
        o.prefix_tbl = off;
        off = mega_moe_align_up(off + epr * (num_ranks + 1) * (T)sizeof(int), kAlign);
    }
    o.total_size = off;
    return o;
}

} // namespace deep_gemm::layout
