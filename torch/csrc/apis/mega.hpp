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
#include "memory.hpp"
#include "megakernel.h"  // fp8mega::MegaLaunchArgs / launch_compute / launch_reduce

// flags_bridge.cu 导出：rwk_data / unk_data 的字节数（host 头不包含设备定义）
extern "C" int64_t mxk_rwk_bytes();
extern "C" int64_t mxk_unk_bytes();

namespace mxk {

// ---------------------------------------------------------------------------
// SymmBuffer：C++ 独占分配与释放（Python 只持有指针/句柄，不分配）。
//   sym    ：对称区 FG（跨 rank 写），布局见 compute_symm_layout；
//   rwk    ：本 rank 普通显存整块（rwk_data：lf + 内部 ld，kernel 读 lf/ld.route_slot_counts/prefix_tbl）；
//   unk    ：本 rank FG 整块（unk_data：rf + rd，跨 rank 写；FG 失败回退普通显存）；
//   ffn_hidden / l2_out：计算链缓冲，按物理容量 win_alloc 分配（普通显存）。
//   A/f2 fragment 缓冲改为 kernel 内部（rwk->ld），不在此分配。
// 远端：export_ipc_handles() 产出 sym/unk 句柄；import_peers() 打开远端句柄并构造
//   peer_bases（d_base_ptrs）与 unk_peers（d_unk_ptrs）设备表。
// 生命周期：destroy_symm_buffer() 显式释放（不使用 take_ownership，避免 double free）。
// ---------------------------------------------------------------------------
struct SymmBuffer {
  int64_t R = 0, rank = 0, num_experts = 0, epr = 0, cap = 0, topk = 0,
          hidden = 0, intermediate_hidden = 0, win = 0, device = 0;
  SymmLayout L;

  uint8_t* sym = nullptr;        // FG：对称区
  rwk_data* rwk = nullptr;       // 普通显存：lf + ld
  unk_data* unk = nullptr;       // FG（或回退普通）：rf + rd
  uint8_t** d_base_ptrs = nullptr;     // 设备表：各 rank sym 基址（含 self）
  unk_data** d_unk_ptrs = nullptr;     // 设备表：各 rank unk 基址（含 self）
  std::vector<int64_t> remote_ptrs;    // 已打开的远端 IPC 指针（sym 与 unk 各 R-1 个）

  uint8_t* ffn_hidden = nullptr;  // [EPR][ring][win][I] FP8
  uint16_t* l2_out = nullptr;     // [kGrid][win][H] BF16
  int64_t ffn_hidden_bytes = 0;
  int64_t l2_out_bytes = 0;

  // [FRAG-INT] A/f2 fragment 缓冲为 kernel 内部（rwk->ld），torch 不再分配。
  int64_t launch_count = 0;
};

inline int64_t kMegaRing() { return 8; }
inline int64_t kMegaGrid() { return 32; }
inline int64_t kMegaBlockN() { return 256; }

// 释放已分配部分（创建失败回滚共用）。
inline void release_symm_buffer(SymmBuffer* b) {
  if (b == nullptr) return;
  for (int64_t p : b->remote_ptrs) {
    if (p) mcIpcCloseMemHandle(reinterpret_cast<void*>(p));
  }
  b->remote_ptrs.clear();
  if (b->l2_out) mcFree(b->l2_out);
  if (b->ffn_hidden) mcFree(b->ffn_hidden);
  if (b->d_unk_ptrs) mcFree(b->d_unk_ptrs);
  if (b->d_base_ptrs) mcFree(b->d_base_ptrs);
  if (b->unk) mcFree(b->unk);
  if (b->rwk) mcFree(b->rwk);
  if (b->sym) mcFree(b->sym);
  b->l2_out = nullptr;
  b->ffn_hidden = nullptr;
  b->d_unk_ptrs = nullptr;
  b->d_base_ptrs = nullptr;
  b->unk = nullptr;
  b->rwk = nullptr;
  b->sym = nullptr;
}

// C++ 自有分配：sym(FG)、rwk(普通)、unk(FG，回退普通)、计算链缓冲（win_alloc）。
inline SymmBuffer* create_symm_buffer_owned(
    int64_t rank, int64_t R, int64_t num_experts, int64_t cap, int64_t topk,
    int64_t hidden, int64_t intermediate_hidden, int64_t win_alloc,
    int64_t device) {
  if (mcSetDevice((int)device) != mcSuccess) {
    throw std::runtime_error("create_symm_buffer: mcSetDevice 失败");
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
  b->win = win_alloc;
  b->device = device;
  b->L = compute_symm_layout(R, num_experts, cap, topk, hidden,
                             intermediate_hidden, win_alloc);
  try {
    b->sym = reinterpret_cast<uint8_t*>(
        allocate_finegrained(b->L.total_bytes, device));
    b->rwk = reinterpret_cast<rwk_data*>(
        allocate_device(mxk_rwk_bytes(), device));
    // unk：FG 优先，失败回退普通显存（与 mega_common 一致）
    int64_t up = 0;
    try {
      up = allocate_finegrained(mxk_unk_bytes(), device);
    } catch (const std::exception&) {
      up = allocate_device(mxk_unk_bytes(), device);
    }
    b->unk = reinterpret_cast<unk_data*>(up);
    mcMemset(b->sym, 0, (size_t)b->L.total_bytes);
    mcMemset(b->rwk, 0, (size_t)mxk_rwk_bytes());
    mcMemset(b->unk, 0, (size_t)mxk_unk_bytes());

    if (mcMalloc((void**)&b->d_base_ptrs, (size_t)R * sizeof(uint8_t*)) !=
            mcSuccess ||
        mcMalloc((void**)&b->d_unk_ptrs, (size_t)R * sizeof(unk_data*)) !=
            mcSuccess) {
      throw std::runtime_error("create_symm_buffer: 设备表分配失败");
    }
    // 设备表先写入 self 项（远端项由 import_peers 填充）。未初始化的表项会让
    // kernel 经 unk_peers[rank] 访问到随机地址（ATU/Xnack）。
    {
      std::vector<uint8_t*> bases((size_t)R, nullptr);
      std::vector<unk_data*> unks((size_t)R, nullptr);
      bases[(size_t)rank] = b->sym;
      unks[(size_t)rank] = b->unk;
      if (mcMemcpy(b->d_base_ptrs, bases.data(), (size_t)R * sizeof(uint8_t*),
                   mcMemcpyHostToDevice) != mcSuccess ||
          mcMemcpy(b->d_unk_ptrs, unks.data(), (size_t)R * sizeof(unk_data*),
                   mcMemcpyHostToDevice) != mcSuccess) {
        throw std::runtime_error("create_symm_buffer: 设备表 self 项写入失败");
      }
    }
    b->ffn_hidden_bytes = b->epr * kMegaRing() * win_alloc * intermediate_hidden;
    b->l2_out_bytes = kMegaGrid() * win_alloc * hidden * 2;
    if (mcMalloc((void**)&b->ffn_hidden, (size_t)b->ffn_hidden_bytes) !=
            mcSuccess ||
        mcMemset(b->ffn_hidden, 0, (size_t)b->ffn_hidden_bytes) != mcSuccess) {
      throw std::runtime_error("create_symm_buffer: ffn_hidden 分配失败");
    }
    if (mcMalloc((void**)&b->l2_out, (size_t)b->l2_out_bytes) != mcSuccess ||
        mcMemset(b->l2_out, 0, (size_t)b->l2_out_bytes) != mcSuccess) {
      throw std::runtime_error("create_symm_buffer: l2_out 分配失败");
    }
    // [FRAG-INT] A/f2 fragment 缓冲改由 kernel 内部 rwk->ld 持有，torch 不再分配。
  } catch (...) {
    release_symm_buffer(b);
    delete b;
    throw;
  }
  return b;
}

// 导出本 rank 的 IPC 句柄：sym 与 unk（各 mcIpcMemHandle_t 字节）。
inline std::vector<uint8_t> export_sym_handle(const SymmBuffer& b) {
  return get_ipc_handle(reinterpret_cast<int64_t>(b.sym));
}
inline std::vector<uint8_t> export_unk_handle(const SymmBuffer& b) {
  return get_ipc_handle(reinterpret_cast<int64_t>(b.unk));
}

// 导入远端句柄并构造设备表。sym_handles/unk_handles 长度均为 R，第 rank 项忽略（取本地）。
inline void import_peers(SymmBuffer* b,
                         const std::vector<std::vector<uint8_t>>& sym_handles,
                         const std::vector<std::vector<uint8_t>>& unk_handles) {
  const int64_t R = b->R;
  if ((int64_t)sym_handles.size() < R || (int64_t)unk_handles.size() < R) {
    throw std::runtime_error("import_peers: 句柄数量不足 R");
  }
  if (mcSetDevice((int)b->device) != mcSuccess) {
    throw std::runtime_error("import_peers: mcSetDevice 失败");
  }
  std::vector<uint8_t*> bases((size_t)R);
  std::vector<unk_data*> unks((size_t)R);
  for (int64_t i = 0; i < R; ++i) {
    if (i == b->rank) {
      bases[i] = b->sym;
      unks[i] = b->unk;
      continue;
    }
    const int64_t sp = open_ipc_handle(sym_handles[i]);
    const int64_t up = open_ipc_handle(unk_handles[i]);
    b->remote_ptrs.push_back(sp);
    b->remote_ptrs.push_back(up);
    bases[i] = reinterpret_cast<uint8_t*>(sp);
    unks[i] = reinterpret_cast<unk_data*>(up);
    enable_peer_access(b->device, i);
  }
  if (mcMemcpy(b->d_base_ptrs, bases.data(), (size_t)R * sizeof(uint8_t*),
               mcMemcpyHostToDevice) != mcSuccess ||
      mcMemcpy(b->d_unk_ptrs, unks.data(), (size_t)R * sizeof(unk_data*),
               mcMemcpyHostToDevice) != mcSuccess) {
    throw std::runtime_error("import_peers: 设备表拷贝失败");
  }
}

inline void destroy_symm_buffer(SymmBuffer* b) {
  if (b == nullptr) return;
  release_symm_buffer(b);
  delete b;
}


// 用当前 SymmBuffer 构造一次发射入参（逐字段对齐 bench make_mega_args）。
// win_cap/skip_prologue 是逐 launch 决策（Python 侧自适应
// 或调用方显式指定）：win_cap=0 表示用物理容量 b.win；
// 缺省粗化到单波（kMegaGrid）。
inline fp8mega::MegaLaunchArgs make_mega_args(
    const SymmBuffer& b, float* y, int64_t num_tokens, const uint8_t* w1,
    const uint8_t* w2, int64_t win_cap = 0, int64_t skip_prologue = 0) {
  namespace L = deep_gemm::layout;
  const int64_t wc = (win_cap > 0) ? win_cap : b.win;
  if (wc < 8 || wc > b.win || wc > 128) {
    throw std::runtime_error(
        "make_mega_args: win_cap 必须落在 [8, min(物理容量,128)]");
  }
  fp8mega::MegaLaunchArgs a{};
  a.topk_idx = reinterpret_cast<const int64_t*>(b.sym + b.L.topk_idx_off);
  a.topk_idx_off = b.L.topk_idx_off;
  a.self = b.sym;
  a.peer_bases = b.d_base_ptrs;
  a.R = (int)b.R;
  a.rank = (int)b.rank;
  a.tokens = (int)num_tokens;
  a.slot_capacity = (int)b.cap;
  a.win_cap = (int)wc;
  a.prefix_off = b.L.prefix_tbl_off;
  a.rwk = b.rwk;
  a.unk = b.unk;
  a.unk_peers = b.d_unk_ptrs;
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
  a.skip_prologue = (int)skip_prologue;
  a.prefix_ptr = nullptr;
  a.w1 = w1;
  a.l1_k_out = (int)b.intermediate_hidden;
  a.ffn_hidden = b.ffn_hidden;
  a.w2 = w2;
  a.l2_out = b.l2_out;
  a.l2_k_out = (int)b.hidden;
  a.combine_rows_off = b.L.combine_rows_off;
  a.y = y;
  a.grid = (int)kMegaGrid();
  a.threads = 512;
  a.smem = prod_smem((int)b.R, 8, (int)kMegaBlockN());
  return a;
}

inline void launch_fp8_mega_moe(SymmBuffer* b, float* y, const uint8_t* w1,
                                const uint8_t* w2, int64_t num_tokens,
                                void* stream, int64_t win_cap = 0,
                                int64_t skip_prologue = 0) {
  if (b == nullptr || y == nullptr || w1 == nullptr || w2 == nullptr) {
    throw std::runtime_error("launch_fp8_mega_moe: 入参为空");
  }
  if (num_tokens < 0 || num_tokens > b->cap) {
    throw std::runtime_error("launch_fp8_mega_moe: num_tokens 越界");
  }
  fp8mega::MegaLaunchArgs a =
      make_mega_args(*b, y, num_tokens, w1, w2, win_cap, skip_prologue);
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
