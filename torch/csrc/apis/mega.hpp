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

// flags_bridge.cu 导出：rwk_data / unk_data 的字节数与 ld 内 w1/w2 偏移
// （host 头不包含设备定义）
extern "C" int64_t mxk_rwk_bytes();
extern "C" int64_t mxk_unk_bytes();
extern "C" int64_t mxk_rwk_w1_off();
extern "C" int64_t mxk_rwk_w2_off();
extern "C" int mxk_group_ranks();

namespace mxk {

// ---------------------------------------------------------------------------
// SymmBuffer：C++ 独占分配与释放（Python 只持有指针/句柄，不分配）。
//   sym    ：对称区 FG（跨 rank 写），布局见 compute_symm_layout；
//   rwk    ：本 rank 普通显存整块（rwk_data：lf + 内部 ld，kernel 读 lf/ld.*）；
//   unk    ：本 rank FG 整块（unk_data：rf + rd，跨 rank 写；FG 失败回退普通显存）；
//   [RWK-LD] w1/w2/ffn_hidden/l2_out 全部由 kernel 从 rwk->ld（local_data）派生；
//   torch 只在 init 时把打包好的 w1/w2 一次性拷进 rwk->ld.w1/w2（停独立分配
//   ffn_hidden/l2_out，也不再经 MegaLaunchArgs 传）。
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

  // [RWK-LD] 权重已拷进 rwk->ld.w1/w2，记录上次来源指针，只在变化时再拷（静态一次）。
  const void* w1_cached = nullptr;
  const void* w2_cached = nullptr;
  int w1_cached_blocks = 0;   // [GS] 上次拷入的组块数（1=旧路径，gr=组权重）
  int bf16_in = 0;               // [BF16-IN] 1=输入 x 为 bf16

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
  if (b->d_unk_ptrs) mcFree(b->d_unk_ptrs);
  if (b->d_base_ptrs) mcFree(b->d_base_ptrs);
  if (b->unk) mcFree(b->unk);
  if (b->rwk) mcFree(b->rwk);
  if (b->sym) mcFree(b->sym);
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
    int64_t device, int64_t bf16_in = 0) {
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
                             intermediate_hidden, win_alloc, bf16_in != 0);
  b->bf16_in = (int)bf16_in;
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
    // [RWK-LD] ffn_hidden / l2_out / A·f2 fragment 全部为 kernel 内部
    //   （rwk->ld），torch 不再独立分配；w1/w2 在首次 launch 前拷进 ld。
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
    const SymmBuffer& b, float* y, int64_t num_tokens, int64_t win_cap = 0,
    int64_t skip_prologue = 0) {
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
  a.rwk = b.rwk;
  a.unk = b.unk;
  a.unk_peers = b.d_unk_ptrs;
  // flag 索引已内联为设备侧编译期常量 kBar*Idx（src/common.cuh，镜像 layout 头）。
  a.x_off = b.L.x_off;
  a.weights_off = b.L.topk_weights_off;
  a.max_spins = 4000000;
  a.skip_prologue = (int)skip_prologue;
  // [RWK-LD] w1/w2/ffn_hidden/l2_out 由 kernel 从 b.rwk->ld 派生，不再传。
  a.in_bf16 = b.L.in_bf16;
  a.y = y;
  a.grid = (int)kMegaGrid();
  a.threads = 512;
  a.smem = prod_smem((int)b.R, 8, (int)kMegaBlockN());
  return a;
}

inline void ensure_mega_weights(SymmBuffer* b, const uint8_t* w1,
                                const uint8_t* w2, mcStream_t st,
                                int n_group_blocks = 1) {
  if (w1 == b->w1_cached && w2 == b->w2_cached &&
      n_group_blocks == b->w1_cached_blocks) return;
  // [RWK-LD] w1/w2 一次性 D2D 拷进 rwk->ld（仅来源指针变化时）。用
  //   mcMemcpyAsync 走本 stream：capture 合法，且与本 launch 天然排序。
  const int nb = (n_group_blocks > 0) ? n_group_blocks : 1;
  const size_t w1_esz = 2u * (size_t)b->intermediate_hidden * (size_t)b->hidden;
  const size_t w2_esz = (size_t)b->hidden * (size_t)b->intermediate_hidden;
  // [GS] kernel 从 w1g[ge] 读权重，ge 为组内专家序（kGE=gr*epr）。上层应传
  //   组内全部 gr 个 member 的权重（all-gather），本函数逐块拷进 w1g[0..kGE)：
  //   nb=gr 时为完整组权重；nb=1 时只写块 0（base/旧路径）。
  const size_t w1_bytes = (size_t)nb * b->epr * w1_esz;
  const size_t w2_bytes = (size_t)nb * b->epr * w2_esz;
  uint8_t* base = reinterpret_cast<uint8_t*>(b->rwk);
  if (mcMemcpyAsync(base + mxk_rwk_w1_off(), w1, w1_bytes,
                    mcMemcpyDeviceToDevice, st) != mcSuccess ||
      mcMemcpyAsync(base + mxk_rwk_w2_off(), w2, w2_bytes,
                    mcMemcpyDeviceToDevice, st) != mcSuccess) {
    throw std::runtime_error("launch_fp8_mega_moe: w1/w2 拷入 rwk->ld 失败");
  }
  b->w1_cached = w1;
  b->w2_cached = w2;
  b->w1_cached_blocks = nb;
}

inline void launch_fp8_mega_moe(SymmBuffer* b, float* y, const uint8_t* w1,
                                const uint8_t* w2, int64_t num_tokens,
                                void* stream, int64_t win_cap = 0,
                                int64_t skip_prologue = 0,
                                int64_t n_group_blocks = 1) {
  if (b == nullptr || y == nullptr || w1 == nullptr || w2 == nullptr) {
    throw std::runtime_error("launch_fp8_mega_moe: 入参为空");
  }
  if (num_tokens < 0 || num_tokens > b->cap) {
    throw std::runtime_error("launch_fp8_mega_moe: num_tokens 越界");
  }
  mcStream_t st = reinterpret_cast<mcStream_t>(stream);
  ensure_mega_weights(b, w1, w2, st, (int)n_group_blocks);
  fp8mega::MegaLaunchArgs a =
      make_mega_args(*b, y, num_tokens, win_cap, skip_prologue);
  if (num_tokens > 0) {
    fp8mega::launch_full(a, st);
  } else {
    // tokens=0：只跑前序相位（全 rank 一致），REDUCE grid=0 非法故跳过。
    fp8mega::launch_compute(a, st);
  }
  b->launch_count += 1;
}

// ---------------------------------------------------------------------------
// [SPLIT-REDUCE] 拆分发射：compute / reduce 两个独立 op，供框架层在两者之间
//   插入与本 MoE 输出无关的独立 kernel，把 REDUCE 的跨 rank 等待重叠隐藏。
//   - compute：前序 + 任务循环 + COMBINE（不写 y，不需 y）。
//   - reduce ：topk 加权归约 → y；必须在本 rank 的 compute 之后、且等所有 rank
//              的 COMBINE 已发布（内部按 combine_sums 自旋）。
//   两者用同一 SymmBuffer/weights；weights 只在 compute 首发时拷一次。
//   注意：跨 rank 锁步与 current_time/M_CYCLE 契约不变（只有 compute 段
//   task_setup 推进 current_time[1]，中间插入的核不得触碰）。
// ---------------------------------------------------------------------------
inline void launch_fp8_mega_moe_compute(SymmBuffer* b, const uint8_t* w1,
                                        const uint8_t* w2, int64_t num_tokens,
                                        void* stream, int64_t win_cap = 0,
                                        int64_t skip_prologue = 0,
                                        int64_t n_group_blocks = 1) {
  if (b == nullptr || w1 == nullptr || w2 == nullptr) {
    throw std::runtime_error("launch_fp8_mega_moe_compute: 入参为空");
  }
  if (num_tokens < 0 || num_tokens > b->cap) {
    throw std::runtime_error("launch_fp8_mega_moe_compute: num_tokens 越界");
  }
  mcStream_t st = reinterpret_cast<mcStream_t>(stream);
  ensure_mega_weights(b, w1, w2, st, (int)n_group_blocks);
  fp8mega::MegaLaunchArgs a =
      make_mega_args(*b, nullptr, num_tokens, win_cap, skip_prologue);
  fp8mega::launch_compute(a, st);
  b->launch_count += 1;
}

inline void launch_fp8_mega_moe_reduce(SymmBuffer* b, float* y,
                                       int64_t num_tokens, void* stream) {
  if (b == nullptr || y == nullptr) {
    throw std::runtime_error("launch_fp8_mega_moe_reduce: 入参为空");
  }
  if (num_tokens <= 0 || num_tokens > b->cap) {
    throw std::runtime_error("launch_fp8_mega_moe_reduce: num_tokens 越界");
  }
  mcStream_t st = reinterpret_cast<mcStream_t>(stream);
  // reduce 不用 win_cap/skip；make_mega_args(win_cap=0) 取 b.win（合法）。
  fp8mega::MegaLaunchArgs a = make_mega_args(*b, y, num_tokens, 0, 0);
  fp8mega::launch_reduce(a, st);
}

}  // namespace mxk
