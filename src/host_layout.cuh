// ============================================================================
// host_layout.cuh — host 侧生产布局（对称区偏移/大小计算）
//
// 与生产一致：MegaMoEBuffer（输入视图）+ compute_mega_moe_scratch(fp8_layout)。
// 由 bench.cu 与 bench_check.cuh 共用。
// ============================================================================
#pragma once

#include <mcr/mc_runtime.h>

#include <deep_gemm/layout/mega_moe.cuh>

#include <algorithm>
#include <cstdint>

#include "common.cuh"

#define CHECK(call)                                                          \
    do {                                                                     \
        mcError_t e_ = (call);                                               \
        if (e_ != mcSuccess) {                                               \
            std::fprintf(stderr, "MACA error %d at %s:%d: %s\n", (int)e_,    \
                         __FILE__, __LINE__, #call);                         \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

struct ProdLayout {
    int64_t mega_bytes = 0;
    int64_t sym_bytes = 0;
    int64_t scratch_offset = 0;
    int64_t scratch_total = 0;
    int64_t x_off = 0;
    int64_t topk_idx_off = 0;
    int64_t topk_weights_off = 0;
    int64_t barrier_slots_off = 0;
    int64_t route_slot_counts_off = 0;
    int64_t src_index_table_off = 0;
    int64_t expert_recv_counts_off = 0;
    int64_t prefix_tbl_off = 0;
    int64_t expert_tokens_off = 0;
    int64_t token_route_weights_off = 0;
    int64_t token_src_meta_off = 0;
    int64_t input_token_bytes = 0;
    int64_t src_index_table_bytes = 0;
    int64_t expert_recv_counts_bytes = 0;
    int64_t prefix_tbl_bytes = 0;
    int64_t slot_capacity = 0;  // 专家 token 容量（= max_tokens）
    int64_t win_capacity = 0;   // 窗口容量（pool/staging/窗口划分）
    int64_t recv_stride = 0;    // recv 行宽（word）：(EPR+1+3)&~3 = 12
};

static inline int64_t align_up_i64(int64_t x, int64_t a) {
    return (x + a - 1) / a * a;
}

static ProdLayout compute_layout(int R, int max_tokens, int topk, int H, int I,
                                 int win_cap) {
    namespace L = deep_gemm::layout;
    const int E = R * kEPR;
    const L::MegaMoEBuffer mb(nullptr, (uint32_t)H, (uint32_t)I, (uint32_t)R,
                              (uint32_t)E, (uint32_t)max_tokens,
                              (uint32_t)topk, /*num_ring_tokens=*/0,
                              /*num_sf_ring_tokens=*/0, /*with_sf=*/false,
                              /*num_shared_experts=*/0);
    const auto scratch = L::compute_mega_moe_scratch<int64_t>(
        R, E, max_tokens, topk, H, I, win_cap, /*with_sf=*/false,
        /*fp8_layout=*/true);

    ProdLayout o;
    o.mega_bytes = mb.get_num_bytes();
    o.scratch_total = scratch.total_size;
    o.sym_bytes =
        align_up_i64(o.mega_bytes, L::kMegaMoePageAlign) + o.scratch_total;
    o.scratch_offset = o.sym_bytes - o.scratch_total;
    o.barrier_slots_off = o.scratch_offset + scratch.barrier_slots;
    o.x_off = reinterpret_cast<int64_t>(mb.input_token_buffer.base);
    o.topk_idx_off = reinterpret_cast<int64_t>(mb.input_topk_idx_buffer.base);
    o.topk_weights_off =
        reinterpret_cast<int64_t>(mb.input_topk_weights_buffer.base);
    o.input_token_bytes = (int64_t)max_tokens * H;
    o.route_slot_counts_off = o.scratch_offset + scratch.route_slot_counts;
    o.src_index_table_off = o.scratch_offset + scratch.src_index_table;
    o.expert_recv_counts_off = o.scratch_offset + scratch.expert_recv_counts;
    o.prefix_tbl_off = o.scratch_offset + scratch.prefix_tbl;
    o.expert_tokens_off = o.scratch_offset + scratch.expert_tokens;
    o.token_route_weights_off = o.scratch_offset + scratch.token_route_weights;
    o.token_src_meta_off = o.scratch_offset + scratch.token_src_meta;
    o.src_index_table_bytes = (int64_t)R * kEPR * max_tokens * 4;
    o.expert_recv_counts_bytes = (int64_t)R * (kEPR + 1) * 4;
    o.prefix_tbl_bytes = (int64_t)kEPR * (R + 1) * 4;
    o.slot_capacity = max_tokens;
    o.win_capacity = win_cap;
    o.recv_stride = (kEPR + 1 + 3) & ~3;
    return o;
}

// N tile 选择（与生产 bn_hint=256 的候选循环一致）
static int prod_block_n(int bn_override) {
    const int64_t l1 = (int64_t)kIntermediate * 2;
    if (bn_override == 512 || bn_override == 256 || bn_override == 128) {
        if (l1 % bn_override == 0 && kHidden % bn_override == 0)
            return bn_override;
    }
    for (const int c : {512, 256, 128})
        if (l1 % c == 0 && kHidden % c == 0) return c;
    return 0;
}

// 生产 smem：数据区 max(window*bn*4, E*4) + 控制区 (EPR*2+R+2)*4
static int64_t prod_smem(int R, int window, int block_n) {
    const int64_t E = (int64_t)R * kEPR;
    const int64_t smem_data = std::max(
        (int64_t)window * block_n * (int64_t)sizeof(float),
        E * (int64_t)sizeof(int));
    const int64_t smem_data_aligned = ((smem_data + 15) / 16) * 16;
    const int64_t control = ((int64_t)kEPR * 2 + R + 2) * (int64_t)sizeof(int);
    return smem_data_aligned + ((control + 127) / 128) * 128;
}
