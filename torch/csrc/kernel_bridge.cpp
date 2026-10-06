// kernel_bridge.cpp — 把 mini_mega 的 host C++ 接口层（src/megakernel.h /
// megakernel.so）接到 torch 扩展里。
//
// 这是本 torch 层与 mini_mega kernel 的**真实绑定点**：
//   * launch_empty     -> fp8mega::launch_empty（空 kernel，验证链接/发射通路）
//   * fill_weights_lane-> fp8mega::fill_weights_lane（设备侧权重生成）
//   * bridge_symbols   -> 扩展实际链接到的 mini_mega 符号清单
//
// 说明：完整的 fp8_mega_moe 需要 SymmBuffer 的跨 rank IPC/FG 通路，mini_mega
// 未提供该平台层，故高层 launch 暂为 not_implemented（见 apis/mega.hpp）。
//
// 编译：随扩展一起编译；megakernel.so 以 extra_objects 链接并写入 rpath。

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "megakernel.h"  // mini_mega/src/megakernel.h（-I ../src）

namespace mxk {

// 当前 CUDA/MACA 流由 torch 传入（python_api.cpp 转换后传入 void*）。
void launch_empty(int grid, int threads, int64_t smem_bytes, void* stream) {
  fp8mega::launch_empty(grid, threads, smem_bytes,
                        reinterpret_cast<mcStream_t>(stream));
}

void fill_weights_lane(uint8_t* w, int n_rows, int k_cols, uint32_t seed,
                       void* stream) {
  fp8mega::fill_weights_lane(w, n_rows, k_cols, seed,
                             reinterpret_cast<mcStream_t>(stream));
}

std::vector<std::string> bridge_symbols() {
  return {
      "fp8mega::launch_prelude",
      "fp8mega::launch_compute",
      "fp8mega::launch_reduce",
      "fp8mega::launch_full",
      "fp8mega::launch_empty",
      "fp8mega::fill_weights_lane",
  };
}

}  // namespace mxk
