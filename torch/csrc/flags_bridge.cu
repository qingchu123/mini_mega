// flags_bridge.cu — 暴露 local_flags / ranks_flags 的字节数与分配（host 侧需要
// sizeof，而定义在设备共享头 flags_defs.cuh；放在 .cu 里由 cucc 编译）。
#include <cstdint>

#include <mcr/mc_runtime.h>

#include "flags_defs.cuh"

extern "C" {

int64_t mxk_local_flags_bytes() { return (int64_t)sizeof(local_flags); }
int64_t mxk_ranks_flags_bytes() { return (int64_t)sizeof(ranks_flags); }

// 普通显存分配 + 清零（本 rank 私有）
int64_t mxk_alloc_local_flags(int device) {
  if (mcSetDevice(device) != mcSuccess) return 0;
  void* p = nullptr;
  if (mcMalloc(&p, sizeof(local_flags)) != mcSuccess || p == nullptr) return 0;
  mcMemset(p, 0, sizeof(local_flags));
  return (int64_t)p;
}

// fine-grained 分配 + 清零（跨 rank peer 可写；失败回退普通显存仅诊断用）
int64_t mxk_alloc_ranks_flags_fg(int device) {
  if (mcSetDevice(device) != mcSuccess) return 0;
  void* p = nullptr;
  const mcError_t e =
      mcExtMallocWithFlags(&p, sizeof(ranks_flags), mcDeviceMallocFinegrained);
  if (e != mcSuccess || p == nullptr) {
    if (mcMalloc(&p, sizeof(ranks_flags)) != mcSuccess || p == nullptr) return 0;
  }
  mcMemset(p, 0, sizeof(ranks_flags));
  return (int64_t)p;
}

}  // extern "C"
