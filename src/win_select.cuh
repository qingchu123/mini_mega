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

// [261012 重标定] 专家窗口 ring 槽数（与 common.cuh 的 kRing 对齐）。
//   单专家窗口数 ceil(c_e/win) 超过该值时，后续窗口需在 l2_gate_wait 上等待前一窗
//   的 L2 完成 → 串行台阶。故倾斜负载下抬升 win 使最忙专家窗数 ≤ kWinRing。
static constexpr int kWinRing = 8;

// 倾斜判定阈值：max_e c_e · E > kWinSkewNum · Σc_e（即 max/mean > 3）。
//   均衡负载比值=1，取原模型（大档需 128）；不均衡实录 top50 实测 max/mean≈10~32，
//   走 ring 约束 + 去 128（见 perf_all_win.md / docs/11 的逐 case 扫描）。
static constexpr long long kWinSkewNum = 3;

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

// 旧代价模型选窗（重标定前的行为，供安全回退：MXK_WIN_LEGACY=1）。
// 在候选集合 ∩ [win_min, cap_hint] 内取 T̂ 最小者；无合法候选则钳到 cap_hint。
__host__ __device__ inline int select_win_legacy(const uint32_t* counts, int E,
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

// 选窗 [261012 重标定]：
//   1) 统计 max_exp、Σc_e，判定是否倾斜（max/mean > kWinSkewNum）；
//   2) 倾斜：win 下限抬到 ceil(max_exp/kWinRing)（避免最忙专家的 ring 串行台阶），
//      且上界压到 96（实测 128 在倾斜负载最差：小专家多、宽窗权重/缓冲浪费）；
//      当 ceil(max_exp/kWinRing) > 96 时说明 96 仍不足，放开到 cap_hint（含 128）。
//   3) 非倾斜：原代价模型（含 128，均衡大档实测最优为 128）。
//      在 [floor, cap] 内对候选取 T̂ 最小者。
// 设备端不 exit；返回恒 ∈ [win_min, cap_hint]（当 cap_hint≥win_min）或 cap_hint。
__host__ __device__ inline int select_win(const uint32_t* counts, int E,
                                          int cap_hint, int win_min) {
    uint32_t max_exp = 0;
    unsigned long long total = 0;
    for (int e = 0; e < E; ++e) {
        if (counts[e] > max_exp) max_exp = counts[e];
        total += counts[e];
    }
    int floor_w = win_min;
    int cap = cap_hint;
    const bool skewed =
        (E > 0) && ((unsigned long long)max_exp * (unsigned long long)E >
                    (unsigned long long)kWinSkewNum * total);
    if (skewed) {
        const int need =
            (int)(((unsigned long long)max_exp + (unsigned long long)kWinRing - 1ull) /
                  (unsigned long long)kWinRing);
        if (need > floor_w) floor_w = need;
        if (floor_w <= 96 && cap > 96) cap = 96;
    }
    int best = -1;
    unsigned long long best_c = 0;
    for (int i = 0; i < kWinCandN; ++i) {
        const int w = kWinCandidates[i];
        if (w < floor_w || w > cap) continue;
        const unsigned long long c = win_cost(counts, E, w);
        if (best < 0 || c < best_c) {
            best = w;
            best_c = c;
        }
    }
    if (best < 0) best = cap;
    return best;
}

}  // namespace mxk
