// ============================================================================
// balance_partition.cuh — 确定性 LPT 分组划分（跨 rank 均衡地基）
//
// 场景：R=32 rank / G=8 组 / 每组 4 连续 rank / 每 rank 8 本地专家。
//   group(rank) = rank / 4；全局专家号 e = rank*kEPR + local（0..255）。
//   publish 后 expert_recv_counts[e] 已知；每专家窗口数 = ceil(c_e/win_cap)，
//   每窗口再切 kNSlice 片 → 任务 (e,w,slice)，cost = 窗口行数
//   min(c_e - w*win_cap, win_cap)（N/K 固定，cost∝行数，故仅需相对量）。
//
// 本文件只做划分：纯函数、host+device 同源、无协议/缓冲区副作用；
//   同一输入两次运行结果逐位相同（确定性）。
// ============================================================================
#pragma once

#include <cstdint>

namespace mxk {

constexpr int kBpRanks = 32;                              // 总 rank 数
constexpr int kBpEPR = 8;                                 // 每 rank 本地专家数
constexpr int kBpGroups = 8;                              // 组数
constexpr int kBpGroupSize = 4;                           // 每组 rank 数
constexpr int kBpExpertsPerGroup = kBpEPR * kBpGroupSize; // 每组专家数 = 32
constexpr int kBpNumExperts = kBpRanks * kBpEPR;          // 全局专家数 = 256

// 单个任务：(e,w,slice)。cost = 窗口行数（相对代价，量纲无关）。
struct BpTask {
    int e;          // 全局专家号 [0,256)
    int w;          // 窗口序号 [0, ceil(c_e/win_cap))
    int slice;      // 片号 [0,nslice)
    long long cost; // 窗口行数 = min(c_e - w*win_cap, win_cap)
};

// rank → 组（rank/4）。
__host__ __device__ inline int group_of_rank(int r) {
    return r / kBpGroupSize;
}

// 专家 → 组：先 expert→rank（e/kEPR），再 rank→组。
__host__ __device__ inline int group_of_expert(int e) {
    return group_of_rank(e / kBpEPR);
}

// 该组 4 rank 的 32 个全局专家号：按 rank 递增、rank 内 local 递增写入 out。
// out 需容纳 kBpExpertsPerGroup(=32) 个 int。返回写入个数。
__host__ __device__ inline int group_experts(int g, int* out) {
    int n = 0;
    for (int r = g * kBpGroupSize; r < (g + 1) * kBpGroupSize; ++r)
        for (int l = 0; l < kBpEPR; ++l)
            out[n++] = r * kBpEPR + l;
    return n;
}

// 枚举该组任务 (e,w,slice)，cost = 窗口行数。
//   枚举顺序天然为 (e,w,slice) 升序（确定性 tie-break 的初序）。
// counts 需覆盖全局专家号（长度 ≥ kBpNumExperts）；out_tasks 需足够大。
// 返回写入个数，并写入 *nt。空专家（c_e==0）不产生任务。
__host__ __device__ inline int enumerate_group_tasks(
        const int* counts, int win_cap, int nslice, int group,
        BpTask* out_tasks, int* nt) {
    int exps[kBpExpertsPerGroup];
    group_experts(group, exps);
    int n = 0;
    for (int i = 0; i < kBpExpertsPerGroup; ++i) {
        const int e = exps[i];
        const int c = counts[e];
        const int nwin = (c > 0) ? (c + win_cap - 1) / win_cap : 0;
        for (int w = 0; w < nwin; ++w) {
            int rows = c - w * win_cap;
            if (rows > win_cap) rows = win_cap;
            if (rows < 0) rows = 0;
            const long long cost = rows;
            for (int s = 0; s < nslice; ++s) {
                out_tasks[n].e = e;
                out_tasks[n].w = w;
                out_tasks[n].slice = s;
                out_tasks[n].cost = cost;
                ++n;
            }
        }
    }
    *nt = n;
    return n;
}

// 稳定插入排序：按 cost 降序；cost 相等时保持原 (e,w,slice) 升序（稳定）。
__host__ __device__ inline void bp_sort_lpt(BpTask* t, int n) {
    for (int i = 1; i < n; ++i) {
        BpTask key = t[i];
        int j = i - 1;
        while (j >= 0 && t[j].cost < key.cost) {
            t[j + 1] = t[j];
            --j;
        }
        t[j + 1] = key;
    }
}

// LPT 贪心：任务按 cost 降序（相等按 (e,w,slice) 升序）依次放入当前累计 cost
// 最小的成员；相等取最小成员下标。out_member_of_task[i] 对应排序后第 i 个任务。
// out_load 需容纳 nmembers 个 long long。返回赋值任务数。
__host__ __device__ inline int lpt_assign(
        BpTask* tasks, int nt, int nmembers,
        int* out_member_of_task, long long* out_load) {
    for (int m = 0; m < nmembers; ++m) out_load[m] = 0;
    bp_sort_lpt(tasks, nt);
    for (int i = 0; i < nt; ++i) {
        int best = 0;
        for (int m = 1; m < nmembers; ++m)
            if (out_load[m] < out_load[best]) best = m;  // 严格 < → 相等取最小
        out_member_of_task[i] = best;
        out_load[best] += tasks[i].cost;
    }
    return nt;
}

}  // namespace mxk
