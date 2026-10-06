// mega.hpp — mini_mega torch 层的 host 组装层：把 SymmBuffer/张量映射到
// fp8mega::MegaLaunchArgs 并经 fp8mega::launch_full 在当前 stream 发射。
//
// 参考：src/bench.cu 的 make_mega_args / run_aligned（唯一的正确 host 接线口径）。
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <mcr/mc_runtime.h>

#include "layout.hpp"
#include "megakernel.h"  // fp8mega::MegaLaunchArgs / launch_compute / launch_reduce

namespace mxk {

struct SymmBuffer {
  int64_t R = 0, rank = 0, num_experts = 0, epr = 0, cap = 0, topk = 0,
          hidden = 0, intermediate_hidden = 0, win = 0, device = 0;
  SymmLayout L;

  uint8_t* sym = nullptr;
  local_flags* lf = nullptr;
  ranks_flags* rf = nullptr;
  uint8_t** d_base_ptrs = nullptr;
  ranks_flags** d_rf_ptrs = nullptr;

  uint8_t* ffn_hidden = nullptr;  // [EPR][ring][win][I] FP8（普通显存）
  uint16_t* l2_out = nullptr;     // [kGrid][win][H] BF16（普通显存）
  int64_t ffn_hidden_bytes = 0;
  int64_t l2_out_bytes = 0;

  int64_t launch_count = 0;
};

inline int64_t kMegaRing() { return 8; }
inline int64_t kMegaGrid() { return 32; }
inline int64_t kMegaBlockN() { return 256; }

inline SymmBuffer* create_symm_buffer(int64_t sym_ptr, int64_t lf_ptr,
                                      int64_t rf_ptr,
                                      const std::vector<int64_t>& base_ptrs,
                                      const std::vector<int64_t>& rf_ptrs,
                                      int64_t rank, int64_t R,
                                      int64_t num_experts, int64_t cap,
                                      int64_t topk, int64_t hidden,
                                      int64_t intermediate_hidden,
                                      int64_t expert_window, int64_t device) {
  if (sym_ptr == 0 || lf_ptr == 0 || rf_ptr == 0) {
    throw std::runtime_error("create_symm_buffer: 指针为空");
  }
  if ((int64_t)base_ptrs.size() < R || (int64_t)rf_ptrs.size() < R) {
    throw std::runtime_error("create_symm_buffer: 基址表长度不足 R");
  }
  auto* b = new SymmBuffer();
  b->R = R;
  b->rank = rank;
  b->num_experts = num_experts;
  b->epr = num_experts / R;
  b->cap = cap;
  b->topk = topk;
  b->hidden = hidden;
  b->intermediate_hidden = intermediate_hidden;
  b->win = expert_window;
  b->device = device;
  b->sym = reinterpret_cast<uint8_t*>(sym_ptr);
  b->lf = reinterpret_cast<local_flags*>(lf_ptr);
  b->rf = reinterpret_cast<ranks_flags*>(rf_ptr);
  b->L = compute_symm_layout(R, num_experts, cap, topk, hidden,
                             intermediate_hidden, expert_window);

  if (mcSetDevice((int)device) != mcSuccess) {
    throw std::runtime_error("create_symm_buffer: mcSetDevice 失败");
  }
  // 各 rank 对称区基址表（设备数组）
  std::vector<uint8_t*> bases((size_t)R);
  std::vector<ranks_flags*> rfs((size_t)R);
  for (int64_t i = 0; i < R; ++i) {
    bases[i] = reinterpret_cast<uint8_t*>(base_ptrs[i]);
    rfs[i] = reinterpret_cast<ranks_flags*>(rf_ptrs[i]);
  }
  if (mcMalloc((void**)&b->d_base_ptrs, (size_t)R * sizeof(uint8_t*)) !=
          mcSuccess ||
      mcMemcpy(b->d_base_ptrs, bases.data(), (size_t)R * sizeof(uint8_t*),
               mcMemcpyHostToDevice) != mcSuccess) {
    delete b;
    throw std::runtime_error("create_symm_buffer: peer_bases 设备表分配失败");
  }
  if (mcMalloc((void**)&b->d_rf_ptrs, (size_t)R * sizeof(ranks_flags*)) !=
          mcSuccess ||
      mcMemcpy(b->d_rf_ptrs, rfs.data(), (size_t)R * sizeof(ranks_flags*),
               mcMemcpyHostToDevice) != mcSuccess) {
    mcFree(b->d_base_ptrs);
    delete b;
    throw std::runtime_error("create_symm_buffer: rf_peers 设备表分配失败");
  }
  // 计算链 staging（普通显存）：与 bench 同尺寸
  b->ffn_hidden_bytes = b->epr * kMegaRing() * expert_window *
                        intermediate_hidden;
  b->l2_out_bytes = kMegaGrid() * expert_window * hidden * 2;
  if (mcMalloc((void**)&b->ffn_hidden, (size_t)b->ffn_hidden_bytes) !=
          mcSuccess ||
      mcMemset(b->ffn_hidden, 0, (size_t)b->ffn_hidden_bytes) != mcSuccess) {
    mcFree(b->d_base_ptrs);
    mcFree(b->d_rf_ptrs);
    delete b;
    throw std::runtime_error("create_symm_buffer: ffn_hidden 分配失败");
  }
  if (mcMalloc((void**)&b->l2_out, (size_t)b->l2_out_bytes) != mcSuccess ||
      mcMemset(b->l2_out, 0, (size_t)b->l2_out_bytes) != mcSuccess) {
    mcFree(b->ffn_hidden);
    mcFree(b->d_base_ptrs);
    mcFree(b->d_rf_ptrs);
    delete b;
    throw std::runtime_error("create_symm_buffer: l2_out 分配失败");
  }
  return b;
}

inline void destroy_symm_buffer(SymmBuffer* b) {
  if (b == nullptr) return;
  if (b->ffn_hidden) mcFree(b->ffn_hidden);
  if (b->l2_out) mcFree(b->l2_out);
  if (b->d_base_ptrs) mcFree(b->d_base_ptrs);
  if (b->d_rf_ptrs) mcFree(b->d_rf_ptrs);
  delete b;
}

// 用当前 SymmBuffer 构造一次发射入参（逐字段对齐 bench make_mega_args）。
inline fp8mega::MegaLaunchArgs make_mega_args(const SymmBuffer& b,
                                              float* y, int64_t num_tokens,
                                              const uint8_t* w1,
                                              const uint8_t* w2) {
  namespace L = deep_gemm::layout;
  fp8mega::MegaLaunchArgs a{};
  a.topk_idx = reinterpret_cast<const int64_t*>(b.sym + b.L.topk_idx_off);
  a.topk_idx_off = b.L.topk_idx_off;
  a.self = b.sym;
  a.peer_bases = b.d_base_ptrs;
  a.R = (int)b.R;
  a.rank = (int)b.rank;
  a.tokens = (int)num_tokens;
  a.slot_capacity = (int)b.cap;
  a.win_cap = (int)b.win;
  a.pool_stride = (int)b.win;  // win_pad = 0
  a.tbl_cap = (int)b.cap;
  a.k_in = (int)b.hidden;
  a.topk = (int)b.topk;
  a.ring = (int)kMegaRing();
  a.src_off = b.L.src_index_table_off;
  a.rsc_off = b.L.route_slot_counts_off;
  a.recv_off = b.L.expert_recv_counts_off;
  a.prefix_off = b.L.prefix_tbl_off;
  a.barrier_off = b.L.barrier_slots_off;
  a.lf = b.lf;
  a.rf = b.rf;
  a.rf_peers = b.d_rf_ptrs;
  a.xrelease_idx = L::kBarXreleaseIdx;
  a.task_ready_idx = L::kBarTaskReadyIdx;
  a.task_counter_idx = L::kBarTaskCounterIdx;
  a.l2_done_idx = L::kBarL2DoneIdx;
  a.slice_flags_idx = L::kBarSliceFlagsIdx;
  a.l1_prog_idx = L::kBarL1ProgressIdx;
  a.x_off = b.L.x_off;
  a.weights_off = b.L.topk_weights_off;
  a.pool_off = b.L.expert_tokens_off;
  a.w_off = b.L.token_route_weights_off;
  a.meta_off = b.L.token_src_meta_off;
  a.max_spins = 4000000;
  a.publish_epoch = 1u;
  a.recv_stride = (int)b.L.recv_stride;
  a.merge_grid = (int)kMegaGrid();
  a.prefix_ptr = nullptr;
  a.w1 = w1;
  a.l1_n_total = (int)(2 * b.intermediate_hidden);
  a.l1_block_n = (int)kMegaBlockN();
  a.l1_k_out = (int)b.intermediate_hidden;
  a.ffn_hidden = b.ffn_hidden;
  a.frag_a = nullptr;
  a.frag2 = nullptr;
  a.w2 = w2;
  a.l2_out = b.l2_out;
  a.l2_k_out = (int)b.hidden;
  a.combine_rows_off = b.L.combine_rows_off;
  a.combine_cap = (int)b.cap;
  a.sums_stride = (int)L::kMegaMoeMaxCombineSlices;
  a.y = y;
  a.grid = (int)kMegaGrid();
  a.threads = 512;
  a.smem = prod_smem((int)b.R, 8, (int)kMegaBlockN());
  return a;
}

inline void launch_fp8_mega_moe(SymmBuffer* b, float* y, const uint8_t* w1,
                                const uint8_t* w2, int64_t num_tokens,
                                void* stream) {
  if (b == nullptr || y == nullptr || w1 == nullptr || w2 == nullptr) {
    throw std::runtime_error("launch_fp8_mega_moe: 入参为空");
  }
  if (num_tokens < 0 || num_tokens > b->cap) {
    throw std::runtime_error("launch_fp8_mega_moe: num_tokens 越界");
  }
  fp8mega::MegaLaunchArgs a = make_mega_args(*b, y, num_tokens, w1, w2);
  mcStream_t st = reinterpret_cast<mcStream_t>(stream);
  if (num_tokens > 0) {
    fp8mega::launch_full(a, st);
  } else {
    // tokens=0：只跑前序相位（全 rank 一致），REDUCE grid=0 非法故跳过。
    fp8mega::launch_compute(a, st);
  }
  b->launch_count += 1;
}

}  // namespace mxk
