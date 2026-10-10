// ============================================================================
// win_select.cuh — 设备端/主机端 窗口容量选择（auto window）核心
//
// 方案见 docs/10；标定与实验见 docs/11。host/device 同源，保证逐值一致。
//
// 代价模型（docs/07 §3，merge 已去 m=1）：
//   W(win)  = Σ_e ceil(c_e / win)              // 窗口数（需全部分布）
//   waves   = ceil(W(win)·N / G)               // 波数
//   T̂(win)  = waves · t_task(win)              // 单任务成本 t_task 为【逐候选标定表】
//   win*    = argmin_{win ∈ C, min≤win≤cap_hint} T̂(win)
//
// 关键：t_task(win) 不是线性函数——实测在 count>64（进入 L2 async/MT8 路径）时
//   有台阶，故用逐候选标定表（Q8 定点）而非 T_FIX+A·win。表值来自 top50 标定
//   （docs/11），单位约 µs（越大越慢）；用于【跨工作负载选相对最优】。
// ============================================================================
#pragma once

#include <cstdint>

namespace mxk {

// 候选窗口集合（16 的倍数，守卫区间 32..slot；16 仅在 win==slot 时合法、不含）
static constexpr int kWinCandidates[] = {32, 48, 64, 96, 128};
static constexpr int kWinCandN = (int)(sizeof(kWinCandidates) / sizeof(int));

// 固定常数（与 common.cuh 对齐：N=kNSlice、G=kGrid）
static constexpr int kWinNSlice = 4;
static constexpr int kWinGrid = 32;

// 逐候选单任务成本（Q8 定点；标定自 top50，见 docs/11）。
//   t_task(32)=40.53  48=32.23  64=34.78  96=39.26  128=38.25
// 注：64→96 抬升即 count>64 进入 async 路径的台阶。
static constexpr long long kWinTTaskQ[/*kWinCandN*/] = {
    10375,  // 32  : 40.53*256
    8251,   // 48  : 32.23*256
    8904,   // 64  : 34.78*256
    10051,  // 96  : 39.26*256
    9792,   // 128 : 38.25*256
};

// 单窗口候选的代价 T̂(win)（定点整数，越小越优）。counts = c_e[0..E-1]。
__host__ __device__ inline unsigned long long win_cost(
        const uint32_t* counts, int E, int win) {
    unsigned long long W = 0;
    for (int e = 0; e < E; ++e)
        W += (unsigned long long)((counts[e] + (uint32_t)win - 1u) / (uint32_t)win);
    const unsigned long long waves =
        (W * (unsigned long long)kWinNSlice + (unsigned long long)kWinGrid - 1u) /
        (unsigned long long)kWinGrid;
    long long tq = kWinTTaskQ[kWinCandN - 1];
    for (int i = 0; i < kWinCandN; ++i)
        if (kWinCandidates[i] == win) tq = kWinTTaskQ[i];
    return waves * (unsigned long long)tq;
}

// 选窗：在候选集合 ∩ [win_min, cap_hint] 内取 T̂ 最小者；无合法候选则钳到 cap_hint。
// 设备端不 exit；返回恒 ∈ [win_min, cap_hint]（当 cap_hint≥win_min）或 cap_hint。
__host__ __device__ inline int select_win(const uint32_t* counts, int E,
                                          int cap_hint, int win_min) {
    int best = -1;
    unsigned long long best_c = 0;
    for (int i = 0; i < kWinCandN; ++i) {
        const int w = kWinCandidates[i];
        if (w < win_min || w > cap_hint) continue;
        const unsigned long long c = win_cost(counts, E, w);
        if (best < 0 || c < best_c) {
            best = w;
            best_c = c;
        }
    }
    if (best < 0) best = cap_hint;
    return best;
}

}  // namespace mxk
