// ============================================================================
// group_sched.cuh — 阶段 3：组内均衡窗口调度（交织组 + 确定性 LPT）
//
// 场景（R=32）：交织组 group = rank % nGroups，nGroups = R / kGroupRanks = 8；
//   组内成员位置 member = rank / nGroups ∈ [0,4)；组 g = {g, g+8, g+16, g+24}。
//   每 rank 8 个本地专家；全局专家 e = owner*kEPR + le，owner = e/kEPR。
//   组内专家序 ge = member(owner)*kEPR + le ∈ [0, kGE=32)。
//
// 目标：把「本组 32 个专家的全部窗口 (ge, w)」按负载（窗口行数，cost =
//   ceil(rows/16)）用确定性 LPT 重分配给组内 4 个 rank；非 owner 的窗口由
//   helper 远端拉取 A、用本组权重计算（见 compute_big.cuh / dispatch）。
//
// 纯函数、host+device 同源、无副作用；同一输入两次运行逐位一致（确定性）：
//   枚举顺序天然 (ge,w) 升序 → 稳定插入排序按 cost 降序 → LPT（平手优先 owner，
//   再取最小 member 序号）。
// ============================================================================
#pragma once

#include "common.cuh"
#include "flags_defs.cuh"   // kGroupRanks / kGE

namespace mxk {

// [调优] 组倾斜判定阈值：max·kGE > MX_GS_SKEW_NUM·Σ 时启用 gs（默认 3，与 win_select
//   同族）。=0 表示恒启用（均衡组也重分配，仅用于实验）。
#ifndef MX_GS_SKEW_NUM
#define MX_GS_SKEW_NUM 2
#endif
// [调优] LPT 代价模型：0=ceil(rows/16)（默认）；1=rows（更细粒度）。
#ifndef MX_GS_COST_MODEL
#define MX_GS_COST_MODEL 0
#endif

__host__ __device__ inline int gs_cost_of(int rows) {
#if MX_GS_COST_MODEL == 1
    return rows;
#elif MX_GS_COST_MODEL == 2
    return (rows + 15) / 16 + 4;   // 每窗固定开销项（实验）
#else
    return (rows + 15) / 16;
#endif
}
__host__ __device__ inline int gs_max_cost(int win_cap) {
#if MX_GS_COST_MODEL == 1
    return win_cap;
#elif MX_GS_COST_MODEL == 2
    return (win_cap + 15) / 16 + 4;
#else
    return (win_cap + 15) / 16;
#endif
}

// 组内窗口任务（window 粒度；4 个 slice 共享同一窗口的 pool/协议）。
struct GsWin {
    int ge;     // 组内专家序 [0, kGE)
    int w;      // 窗口序号
    int cost;   // 代价 = ceil(rows/16)（rows = 本窗真实行数）
};

// 组数 / 组号 / 成员位置 / 成员 rank（交织）。
__host__ __device__ inline int gs_nGroups(int R) { return R / kGroupRanks; }
__host__ __device__ inline int gs_group_of_rank(int R, int r) {
    return r % gs_nGroups(R);
}
__host__ __device__ inline int gs_member_of_rank(int R, int r) {
    return r / gs_nGroups(R);
}
__host__ __device__ inline int gs_rank_of_member(int R, int g, int m) {
    return g + m * gs_nGroups(R);
}

// 全局专家 e → 组内专家序 ge；反查（给定组 g）。
__host__ __device__ inline int gs_ge_of_expert(int R, int e) {
    const int owner = e / kEPR;
    return gs_member_of_rank(R, owner) * kEPR + (e % kEPR);
}
__host__ __device__ inline int gs_expert_of_ge(int R, int g, int ge) {
    const int m = ge / kEPR;
    return gs_rank_of_member(R, g, m) * kEPR + (ge % kEPR);
}

// 每专家窗口数（c<=0 → 0）。
__host__ __device__ inline int gs_win_count(int c, int win_cap) {
    return (c > 0) ? (c + win_cap - 1) / win_cap : 0;
}

// 枚举组内全部窗口到 t[]（ge 升序、w 升序）。只写前 cap 个；返回真实窗口数。
__host__ __device__ inline int gs_enum(const int* cg, int win_cap, GsWin* t,
                                       int cap) {
    int n = 0;
    for (int ge = 0; ge < kGE; ++ge) {
        const int c = cg[ge];
        const int nw = gs_win_count(c, win_cap);
        for (int w = 0; w < nw; ++w) {
            int rows = c - w * win_cap;
            if (rows > win_cap) rows = win_cap;
            if (rows < 0) rows = 0;
            if (n < cap) {
                t[n].ge = ge;
                t[n].w = w;
                t[n].cost = (rows + 15) / 16;
            }
            ++n;
        }
    }
    return n;
}

// 稳定插入排序：cost 降序；相等保持原 (ge,w) 升序。n 小时（≤数百）开销可忽略。
__host__ __device__ inline void gs_sort(GsWin* t, int n) {
    for (int i = 1; i < n; ++i) {
        const GsWin key = t[i];
        int j = i - 1;
        while (j >= 0 && t[j].cost < key.cost) {
            t[j + 1] = t[j];
            --j;
        }
        t[j + 1] = key;
    }
}

// LPT 贪心：按 cost 降序逐个分给当前累计负载最小的 member；平手时优先该窗 owner
// （= ge/kEPR），否则取最小 member 序号。out_member_of[i] 对应排序后第 i 个窗。
__host__ __device__ inline void gs_assign(const GsWin* t, int n, int* member_of,
                                          long long* load) {
    for (int m = 0; m < kGroupRanks; ++m) load[m] = 0;
    for (int i = 0; i < n; ++i) {
        const int owner = t[i].ge / kEPR;
        long long minl = load[0];
        for (int m = 1; m < kGroupRanks; ++m)
            if (load[m] < minl) minl = load[m];
        int chosen = owner;
#ifdef GS_DIAG_OWNER_ONLY
        chosen = owner;
#else
        if (load[owner] != minl) {
            chosen = 0;
            for (int m = 0; m < kGroupRanks; ++m)
                if (load[m] == minl) {
                    chosen = m;
                    break;
                }
        }
#endif
        member_of[i] = chosen;
        load[chosen] += t[i].cost;
    }
}

// 窗口数上界（用于 smem/scratch 下界）：ge 数 × max(1, ceil(kMaxTPR/16))（win 下界 16）。
constexpr int kGsMaxWin = kGE * ((kMaxTPR + 15) / 16);

// [MX_GROUP_SCHED] 单 rank（member）窗口列表容量上界：LPT 均衡后各 member 窗口数
//   ≈ 组总窗口/kGroupRanks + 单个大专家窗口数；兼顾 kGroupRanks=32（全组）时 smem 受限。
#ifndef MX_GS_MEMBER_CAP
#define MX_GS_MEMBER_CAP 512
#endif
constexpr int kGsMemberCap = MX_GS_MEMBER_CAP;

// 桶式无 scratch LPT：直接算出 member 名下的窗口序列（按分配顺序 = cost 降序、
//   同 cost 内 (ge,w) 升序，与 gs_enum+gs_sort+gs_assign 等价）。
//   cost 只取 [1, ceil(win/16)]，逐 cost 降序扫描全部窗口即可，无需暂存全部任务。
//   out_ge/out_w 容量 cap；返回本 member 的窗口数（可能 > cap，仅写前 cap）。
__host__ __device__ inline int gs_schedule_member(
        const int* cg, int win_cap, int member, int* out_ge, int* out_w,
        int cap) {
    long long load[kGroupRanks];
    for (int m = 0; m < kGroupRanks; ++m) load[m] = 0;
    const int maxcost = gs_max_cost(win_cap);
    int nout = 0;
    for (int cost = maxcost; cost >= 1; --cost) {
        for (int ge = 0; ge < kGE; ++ge) {
            const int c = cg[ge];
            const int nw = gs_win_count(c, win_cap);
            for (int w = 0; w < nw; ++w) {
                int rows = c - w * win_cap;
                if (rows > win_cap) rows = win_cap;
                if (rows < 0) rows = 0;
                if (gs_cost_of(rows) != cost) continue;
                const int owner = ge / kEPR;
                long long minl = load[0];
                for (int m = 1; m < kGroupRanks; ++m)
                    if (load[m] < minl) minl = load[m];
                int chosen = owner;
                if (load[owner] != minl) {
                    chosen = 0;
                    for (int m = 0; m < kGroupRanks; ++m)
                        if (load[m] == minl) {
                            chosen = m;
                            break;
                        }
                }
                load[chosen] += cost;
                if (chosen == member) {
                    if (nout < cap) {
                        out_ge[nout] = ge;
                        out_w[nout] = w;
                    }
                    ++nout;
                }
            }
        }
    }
    return nout;
}

}  // namespace mxk
