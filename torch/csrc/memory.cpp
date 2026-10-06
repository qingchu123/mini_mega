// memory.cpp — 平台显存/IPC 适配实现（mcr 原生 API）。
#include "memory.hpp"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

#include <mcr/mc_runtime.h>
#include <mcr/mc_runtime_api.h>
#include <mcr/mc_runtime_types.h>

namespace mxk {

namespace {
[[noreturn]] void fail(const std::string& what) {
  throw std::runtime_error("mxmegakernel memory: " + what);
}
}  // namespace

int64_t allocate_finegrained(int64_t num_bytes, int64_t device) {
  if (num_bytes <= 0) fail("allocate_finegrained: num_bytes 必须为正");
  int prev = -1;
  mcGetDevice(&prev);
  if (device >= 0 && device != prev) {
    if (mcSetDevice((int)device) != mcSuccess) fail("mcSetDevice 失败");
  }
  void* p = nullptr;
  const mcError_t e =
      mcExtMallocWithFlags(&p, (size_t)num_bytes, mcDeviceMallocFinegrained);
  if (device >= 0 && device != prev) mcSetDevice(prev);
  if (e != mcSuccess || p == nullptr) {
    fail("mcExtMallocWithFlags(finegrained) 失败: " +
         std::string(mcGetErrorString(e)));
  }
  if ((reinterpret_cast<uintptr_t>(p) & (kPageAlignBytes - 1)) != 0) {
    mcFree(p);
    fail("finegrained 分配未 4KB 页对齐（跨 rank barrier 会错峰）");
  }
  return reinterpret_cast<int64_t>(p);
}

int64_t allocate_device(int64_t num_bytes, int64_t device) {
  if (num_bytes <= 0) fail("allocate_device: num_bytes 必须为正");
  int prev = -1;
  mcGetDevice(&prev);
  if (device >= 0 && device != prev) {
    if (mcSetDevice((int)device) != mcSuccess) fail("mcSetDevice 失败");
  }
  void* p = nullptr;
  const mcError_t e = mcMalloc(&p, (size_t)num_bytes);
  if (device >= 0 && device != prev) mcSetDevice(prev);
  if (e != mcSuccess || p == nullptr) {
    fail("mcMalloc 失败: " + std::string(mcGetErrorString(e)));
  }
  return reinterpret_cast<int64_t>(p);
}

void free_device(int64_t ptr) {
  if (ptr == 0) return;
  mcFree(reinterpret_cast<void*>(ptr));
}

std::vector<uint8_t> get_ipc_handle(int64_t ptr) {
  if (ptr == 0) fail("get_ipc_handle: ptr 为空");
  mcIpcMemHandle_t handle;
  std::memset(&handle, 0, sizeof(handle));
  const mcError_t e =
      mcIpcGetMemHandle(&handle, reinterpret_cast<void*>(ptr));
  if (e != mcSuccess) {
    fail("mcIpcGetMemHandle 失败: " + std::string(mcGetErrorString(e)));
  }
  std::vector<uint8_t> out(sizeof(mcIpcMemHandle_t));
  std::memcpy(out.data(), &handle, sizeof(mcIpcMemHandle_t));
  return out;
}

int64_t open_ipc_handle(const std::vector<uint8_t>& handle) {
  if (handle.size() < sizeof(mcIpcMemHandle_t)) {
    fail("open_ipc_handle: handle 字节数不足");
  }
  mcIpcMemHandle_t ipc_handle;
  std::memset(&ipc_handle, 0, sizeof(ipc_handle));
  std::memcpy(&ipc_handle, handle.data(), sizeof(mcIpcMemHandle_t));
  void* remote = nullptr;
  const mcError_t e = mcIpcOpenMemHandle(&remote, ipc_handle,
                                         mcIpcMemLazyEnablePeerAccess);
  if (e != mcSuccess || remote == nullptr) {
    fail("mcIpcOpenMemHandle 失败: " + std::string(mcGetErrorString(e)));
  }
  return reinterpret_cast<int64_t>(remote);
}

void close_ipc_handle(int64_t ptr) {
  if (ptr == 0) return;
  mcIpcCloseMemHandle(reinterpret_cast<void*>(ptr));
}

void enable_peer_access(int64_t local_device, int64_t peer_device) {
  if (local_device < 0 || peer_device < 0) return;
  int can = 0;
  if (mcDeviceCanAccessPeer(&can, (int)local_device, (int)peer_device) ==
          mcSuccess &&
      can) {
    const mcError_t e = mcDeviceEnablePeerAccess((int)peer_device, 0);
    if (e != mcSuccess && e != mcErrorPeerAccessAlreadyEnabled) {
      std::fprintf(stderr,
                   "[mxmegakernel] enable_peer_access(%lld->%lld) 失败: %s\n",
                   (long long)local_device, (long long)peer_device,
                   mcGetErrorString(e));
    }
  }
}

bool is_finegrained(int64_t ptr) {
  if (ptr == 0) return false;
  mcPointerAttribute_t attr;
  std::memset(&attr, 0, sizeof(attr));
  if (mcPointerGetAttributes(&attr, reinterpret_cast<void*>(ptr)) !=
      mcSuccess) {
    return false;
  }
  const bool by_flag =
      (attr.allocationFlags & mcDeviceMallocFinegrained) != 0u;
  const bool by_type =
      attr.type == mcMemoryTypeUnified && attr.isManaged;
  return by_flag || by_type;
}

}  // namespace mxk
