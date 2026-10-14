// layout.hpp — host 侧对称区布局计算（纯 C++，不依赖 torch/pybind）。
//
// 口径与 mini_mega 的 src/host_layout.cuh::compute_layout **逐字段一致**：
//   MegaMoEBuffer（输入视图，num_ring_tokens=0）+ compute_mega_moe_scratch(fp8)，
//   尾部 scratch 基址按 4KB 页对齐；再在 sym 末端追加 COMBINE 行区（BF16）：
//     barrier_pad / recv_pad（历史 flag 预留）+ combine_rows[topk][cap][H] bf16。
// 只 include mini_mega/include 下的 deep_gemm 布局头（host/device 同源）。
#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <deep_gemm/layout/mega_moe.cuh>

namespace mxk {

inline int64_t align_up_i64(int64_t x, int64_t a) {
  return (x + a - 1) / a * a;
}

// 全部偏移（字节）。字段语义与 fp8mega::MegaLaunchArgs / compute_layout 对齐。
struct SymmLayout {
  int64_t view_bytes = 0;       // MegaMoEBuffer 视图区字节数
  int64_t scratch_offset = 0;   // scratch 在 buffer 内的偏移（4KB 对齐）
  int64_t scratch_bytes = 0;    // scratch.total_size
  int64_t sym_bytes = 0;        // 视图 + scratch
  int64_t total_bytes = 0;      // 实际分配字节数（含 combine 区）
  int64_t x_off = 0;            // 输入激活 [cap, H] fp8
  int64_t x_sf_off = -1;        // scale（fp8xfp8 无，恒 -1）
  int64_t topk_idx_off = 0;     // [cap, topk] int64
  int64_t topk_weights_off = 0; // [cap, topk] float32
  int64_t route_slot_counts_off = 0;
  int64_t src_index_table_off = 0;
  int64_t expert_recv_counts_off = 0;
  int64_t prefix_tbl_off = 0;
  int64_t expert_tokens_off = 0;
  int64_t token_route_weights_off = 0;
  int64_t token_src_meta_off = 0;
  int64_t barrier_slots_off = 0;
  int64_t combine_rows_off = 0;  // COMBINE 行区（BF16，sym 末端）
  int64_t src_index_table_bytes = 0;
  int64_t expert_recv_counts_bytes = 0;
  int64_t prefix_tbl_bytes = 0;
  int64_t recv_stride = 0;       // recv 行宽（word）：(EPR+1+3)&~3 = 12
  int64_t prefold_off = 0;       // [BF16-IN] 本地 fold 的 fp8 x 区（scratch 内）
  int in_bf16 = 0;               // [BF16-IN] 1=输入 x 为 bf16
};

inline int64_t kEPRConst() { return 8; }

inline SymmLayout compute_symm_layout(int64_t num_ranks, int64_t num_experts,
                                      int64_t num_max_tokens_per_rank,
                                      int64_t num_topk, int64_t hidden,
                                      int64_t intermediate_hidden,
                                      int64_t win_cap,
                                      bool bf16_in = false) {
  using namespace deep_gemm::layout;
  if (num_ranks < 1 || num_experts < 1 || num_experts % num_ranks != 0) {
    throw std::runtime_error(
        "compute_symm_layout: num_experts 必须是 num_ranks 的整数倍");
  }
  if (num_max_tokens_per_rank < 1 || num_topk < 1 || hidden < 1 ||
      intermediate_hidden < 1) {
    throw std::runtime_error("compute_symm_layout: 维度必须为正");
  }
  if (win_cap < 1) {
    throw std::runtime_error("compute_symm_layout: win_cap 必须为正");
  }
  const int64_t epr = num_experts / num_ranks;
  if (epr != kEPRConst()) {
    throw std::runtime_error(
        "compute_symm_layout: mini_mega kernel 固定 EPR=8，收到 " +
        std::to_string(epr));
  }
  const int64_t E = num_experts;
  MegaMoEBuffer mb(nullptr, (uint32_t)hidden, (uint32_t)intermediate_hidden,
                   (uint32_t)num_ranks, (uint32_t)E,
                   (uint32_t)num_max_tokens_per_rank, (uint32_t)num_topk,
                   /*num_ring_tokens=*/0, /*num_sf_ring_tokens=*/0,
                   /*with_sf=*/false, /*num_shared_experts=*/0);
  const auto scratch = compute_mega_moe_scratch<int64_t>(
      num_ranks, E, num_max_tokens_per_rank, num_topk, hidden,
      intermediate_hidden, win_cap, /*with_sf=*/false, /*fp8_layout=*/true,
      /*with_prefold=*/bf16_in);

  SymmLayout o;
  o.view_bytes = mb.get_num_bytes();
  o.scratch_bytes = scratch.total_size;
  o.scratch_offset = align_up_i64(o.view_bytes, (int64_t)kMegaMoePageAlign);
  o.sym_bytes = o.scratch_offset + o.scratch_bytes;
  o.x_off = reinterpret_cast<int64_t>(mb.input_token_buffer.base);
  o.topk_idx_off = reinterpret_cast<int64_t>(mb.input_topk_idx_buffer.base);
  o.topk_weights_off =
      reinterpret_cast<int64_t>(mb.input_topk_weights_buffer.base);
  o.route_slot_counts_off = o.scratch_offset + scratch.route_slot_counts;
  o.src_index_table_off = o.scratch_offset + scratch.src_index_table;
  o.expert_recv_counts_off = o.scratch_offset + scratch.expert_recv_counts;
  o.prefix_tbl_off = o.scratch_offset + scratch.prefix_tbl;
  o.expert_tokens_off = o.scratch_offset + scratch.expert_tokens;
  o.token_route_weights_off = o.scratch_offset + scratch.token_route_weights;
  o.token_src_meta_off = o.scratch_offset + scratch.token_src_meta;
  o.barrier_slots_off = o.scratch_offset + scratch.barrier_slots;
  o.src_index_table_bytes = (int64_t)epr * num_ranks * num_max_tokens_per_rank * 4;
  o.expert_recv_counts_bytes =
      (int64_t)num_ranks * (((epr + 1) + 3) & ~(int64_t)3) * 4;
  o.prefix_tbl_bytes = (int64_t)epr * (num_ranks + 1) * 4;
  o.recv_stride = (int)(((epr + 1) + 3) & ~(int64_t)3);
  o.prefold_off = o.scratch_offset + scratch.prefolded_input_tokens;
  o.in_bf16 = bf16_in ? 1 : 0;

  // sym 末端 COMBINE 行区（与 bench run_scene/run_aligned 同式）
  const int64_t barrier_pad_off = align_up_i64(o.sym_bytes, 256);
  const int64_t recv_pad_off =
      align_up_i64(barrier_pad_off + num_ranks * 64 * 4, 256);
  o.combine_rows_off = align_up_i64(recv_pad_off + num_ranks * 64 * 4, 256);
  o.total_bytes = align_up_i64(
      o.combine_rows_off + num_topk * num_max_tokens_per_rank * hidden * 2, 256);
  return o;
}

// 与 python_api 旧接口兼容的包装：mma_type 仅支持 fp8xfp8。
inline SymmLayout get_symm_layout_for_mega_moe(
    int64_t num_ranks, int64_t num_experts, int64_t num_max_tokens_per_rank,
    int64_t num_topk, int64_t hidden, int64_t intermediate_hidden,
    int64_t num_shared_experts, const std::string& mma_type,
    int64_t expert_window) {
  if (num_shared_experts != 0) {
    throw std::runtime_error("get_symm_layout_for_mega_moe: 不支持 shared experts");
  }
  if (mma_type != "fp8xfp8" && mma_type != "bf16xfp8") {
    throw std::runtime_error(
        "get_symm_layout_for_mega_moe: mini_mega 仅支持 fp8xfp8/bf16xfp8，收到 " +
        mma_type);
  }
  return compute_symm_layout(num_ranks, num_experts, num_max_tokens_per_rank,
                             num_topk, hidden, intermediate_hidden,
                             expert_window,
                             /*bf16_in=*/mma_type == "bf16xfp8");
}

inline int64_t get_symm_buffer_size_for_mega_moe(
    int64_t num_ranks, int64_t num_experts, int64_t num_max_tokens_per_rank,
    int64_t num_topk, int64_t hidden, int64_t intermediate_hidden,
    int64_t num_shared_experts, const std::string& mma_type,
    int64_t num_tokens, int64_t expert_window) {
  (void)num_tokens;
  return get_symm_layout_for_mega_moe(
             num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden,
             intermediate_hidden, num_shared_experts, mma_type, expert_window)
      .total_bytes;
}

inline int64_t get_token_alignment_for_mega_moe() {
  return (int64_t)deep_gemm::layout::kLCMCandidateBlockM;
}

inline int64_t get_block_m_for_mega_moe(int64_t, int64_t, int64_t, int64_t,
                                        int64_t, const std::string&) {
  return (int64_t)deep_gemm::layout::kLCMCandidateBlockM;
}

// 发射几何：与 bench prod_block_n / prod_smem 一致。
inline int prod_block_n(int bn_override) {
  const int64_t l1 = 2048 * 2;
  if (bn_override == 512 || bn_override == 256 || bn_override == 128) {
    if (l1 % bn_override == 0 && 6144 % bn_override == 0) return bn_override;
  }
  for (const int c : {512, 256, 128})
    if (l1 % c == 0 && 6144 % c == 0) return c;
  return 0;
}

inline int64_t prod_smem(int R, int window, int block_n) {
  const int64_t E = (int64_t)R * 8;
  const int64_t smem_data =
      std::max((int64_t)window * block_n * (int64_t)sizeof(float),
               E * (int64_t)sizeof(int));
  const int64_t smem_data_aligned = ((smem_data + 15) / 16) * 16;
  const int64_t control = ((int64_t)8 * 2 + R + 2) * (int64_t)sizeof(int);
  const int64_t base = smem_data_aligned + ((control + 127) / 128) * 128;
  // [GS-SMEM-FIX] src/compute.cuh 的 [MX_GROUP_SCHED] smem 布局额外需要
  //   s_count[8]+s_cum[9]+s_prefix[kGE*(R+1)]+s_gscount[kGE]+s_ge/s_gw[2*kGsMemberCap]，
  //   R=32、kGroupRanks=4 时 = 2129*4 = 8516B > 上面 window=8 时的 ~8448B → 溢出
  //   → 垃圾 ge → owner 越界远端 ATU。与 src/host_layout.cuh 的 gs 分支一致取 40KB。
  const int64_t gs_smem = 40960;
  return base > gs_smem ? base : gs_smem;
}

inline std::vector<int64_t> layout_to_vector(const SymmLayout& o) {
  return {o.view_bytes,            o.scratch_offset,
          o.scratch_bytes,         o.sym_bytes,
          o.total_bytes,           o.x_off,
          o.x_sf_off,              o.topk_idx_off,
          o.topk_weights_off,      o.route_slot_counts_off,
          o.src_index_table_off,   o.expert_recv_counts_off,
          o.prefix_tbl_off,        o.expert_tokens_off,
          o.token_route_weights_off, o.token_src_meta_off,
          o.barrier_slots_off,     o.combine_rows_off,
          o.src_index_table_bytes, o.expert_recv_counts_bytes,
          o.prefix_tbl_bytes,      o.recv_stride,
          o.prefold_off,           o.in_bf16};
}

}  // namespace mxk
