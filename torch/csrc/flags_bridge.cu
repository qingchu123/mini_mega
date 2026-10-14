// flags_bridge.cu — 暴露 rwk_data / unk_data 的字节数（host 侧分配需要 sizeof，
// 而定义在设备共享头 flags_defs.cuh；放在 .cu 里由 cucc 编译，避免 host 头依赖设备定义）。
#include <cstddef>
#include <cstdint>

#include "flags_defs.cuh"
#include "win_select.cuh"

extern "C" {

int64_t mxk_rwk_bytes() { return (int64_t)sizeof(rwk_data); }
int64_t mxk_unk_bytes() { return (int64_t)sizeof(unk_data); }

// 组内 rank 数（kGroupRanks，定义在设备头 common.cuh）
int mxk_group_ranks() { return kGroupRanks; }

// [RWK-LD] rwk->ld 内 w1/w2 相对 rwk 基址的字节偏移：torch init 时把打包好的
//   w1/w2 一次性 D2D 拷进这里，kernel 从 a->rwk->ld 读，不再经 args 传权重。
//   组权重开关（MX_GROUP_WEIGHTS=1）下 ld 无 w1/w2，改指向 w1g/w2g（另一 agent 接入）。
#if MX_GROUP_WEIGHTS
int64_t mxk_rwk_w1_off() {
  return (int64_t)offsetof(rwk_data, ld) + (int64_t)offsetof(local_data, w1g);
}
int64_t mxk_rwk_w2_off() {
  return (int64_t)offsetof(rwk_data, ld) + (int64_t)offsetof(local_data, w2g);
}
#else
int64_t mxk_rwk_w1_off() {
  return (int64_t)offsetof(rwk_data, ld) + (int64_t)offsetof(local_data, w1);
}
int64_t mxk_rwk_w2_off() {
  return (int64_t)offsetof(rwk_data, ld) + (int64_t)offsetof(local_data, w2);
}
#endif

// 选窗：与 mybench/check 同源的 C++ 逻辑（src/win_select.cuh::select_win）。
//   counts = 每专家计数（host 侧 uint32 数组）；cap_hint = 窗口上界（通常 slot）；
//   win_min = 窗口下界。返回 win_cap。供 torch 层调用，使选窗策略由 C++ 决定。
int mxk_select_win(const uint32_t* counts, int E, int cap_hint, int win_min) {
  return mxk::select_win(counts, E, cap_hint, win_min);
}

// [TICKS] block0 时间线 pre_ticks 相对 rwk 基址的字节偏移 + 槽数（kGrid*16）。
//   诊断用：host 拷回后按 mybench 同款下标解读（tp[6]=入口，tp[0]=route毕，
//   tp[1]=publish毕，tp[2]=xbar毕，tp[3]=setup毕，tp[4]=dispatch毕，
//   tp[5]=L1毕，tp[7]=L2/计算毕，tp[15]=出口）。kernel 零改动。
int64_t mxk_rwk_ticks_off() {
  return (int64_t)offsetof(rwk_data, lf) +
         (int64_t)offsetof(local_flags, pre_ticks);
}
int64_t mxk_ticks_count() { return (int64_t)kGridBlocks * 16; }

}  // extern "C"
