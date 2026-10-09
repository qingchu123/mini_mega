// flags_bridge.cu — 暴露 rwk_data / unk_data 的字节数（host 侧分配需要 sizeof，
// 而定义在设备共享头 flags_defs.cuh；放在 .cu 里由 cucc 编译，避免 host 头依赖设备定义）。
#include <cstdint>

#include "flags_defs.cuh"

extern "C" {

int64_t mxk_rwk_bytes() { return (int64_t)sizeof(rwk_data); }
int64_t mxk_unk_bytes() { return (int64_t)sizeof(unk_data); }

}  // extern "C"
