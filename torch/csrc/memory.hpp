// memory.hpp — mini_mega torch 层的平台显存/IPC 适配（MACA C600-UL）。
//
// 对称区必须 fine-grained（FG）显存：普通 mcMalloc 跨 rank peer 写会丢写
// （mini_mega bench.cu 的实测结论）。本层提供 FG 分配/IPC 导入导出/peer access。
#pragma once

#include <cstdint>
#include <vector>

namespace mxk {

// 4KB 页对齐（peer 可写 barrier/原子槽要求）
constexpr int64_t kPageAlignBytes = 4096;

// FG 分配（device < 0 表示沿用当前设备）。分配后强制 4KB 页对齐校验。
int64_t allocate_finegrained(int64_t num_bytes, int64_t device);
// 普通（coarse-grained）显存分配：每 rank 私有、无跨 rank 写的工作缓冲。
int64_t allocate_device(int64_t num_bytes, int64_t device);
// 释放（FG / 普通都走 mcFree）。
void free_device(int64_t ptr);

// 导出/导入 IPC handle（handle 字节流）。
std::vector<uint8_t> get_ipc_handle(int64_t ptr);
int64_t open_ipc_handle(const std::vector<uint8_t>& handle);
void close_ipc_handle(int64_t ptr);

// 启用本设备对 peer 的 P2P 直访（不可用时 warn 不抛）。
void enable_peer_access(int64_t local_device, int64_t peer_device);

// 是否 FG（诊断）
bool is_finegrained(int64_t ptr);

}  // namespace mxk
