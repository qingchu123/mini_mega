// ============================================================================
// flags.cuh — fp8xfp8 MoE Mega Kernel 同步/标志库（T1）
//
// 协议、指令序列、可见性顺序（payload→fence→ready）全部照抄骨架
// empty_kernel_bench.cu 的对应实现（行号见各函数注释），仅做参数化适配：
// flag 区索引/偏移一律由调用方传入（common.cuh 不含 kBar* 常量）。
//
// 导出函数签名与语义（冻结接口 v1，docs/04 §2.1；[D*] 为差异登记）：
//   1) void subset_barrier_n(uint32_t* smem_ctr, int nthreads);
//      rank 内子集 barrier：nthreads 个线程（64 的倍数、整 warp）参与，每参与
//      warp 仅 lane0 到达；smem_ctr 初值 0、只增不清零（由累计计数推导 phase）；
//      替代子集的 __syncthreads。
//   2) void subset_barrier_phase(volatile int* cnt, int phase, int tid_base,
//                                int nwarps);
//      骨架原语（bench:428-443）：tid_base 起的 nwarps 个 warp 参与，
//      phase 从 1 递增；WS 分组（带 tid_base）场景用。
//   3) struct SliceGate { flags, nslice, self_slice, k_iters_per_slice, base_slot,
//                         seq, dbg, max_spins };
//      L1 K 循环分片就绪门参数：flags=分片就绪旗区基址（DISPATCH 分片旗），
//      base_slot=pool_slot*nslice，seq=本窗 epoch（= task_ready 旗值）。
//   4) void slice_gate_wait(const SliceGate& g, int k);
//      L1 K 循环按 K 片边界等待：k 落在非本项目片边界时，各 warp lane0
//      volatile 轮询 flags[base_slot + k/k_iters_per_slice] == seq，warp 内广播，
//      真正等过才 __threadfence()（acquire）；本块自己的片无需等待。
//   5) void l2_gate_wait(const uint32_t* bar_flags, int l2_done_idx, int e, int w,
//                        int rep, int ring, int nslice, int self_slice,
//                        int l1_prog_idx = -1, uint32_t l1_epoch = 0,
//                        int* dbg = nullptr, int max_spins = (int)kSpinLimit);
//      l2_done ring 门（单调）：slot=e*ring+w%ring，target=(rep+w/ring)*nslice，
//      w>=ring 或 rep>0 才等，通过后 __threadfence()；可选 l1_prog_idx>=0 时
//      追加等待本 slice 的 l1_prog 旗 == l1_epoch（L2 入口语义）。
//   6) void l1_prog_gate_wait(const uint32_t* bar_flags, int l1_prog_idx,
//                             int pool_slot, int slice, unsigned epoch,
//                             int nslice = kNSlice, int* dbg = nullptr,
//                             int max_spins = (int)kSpinLimit);
//      等 (pool_slot, slice) 的 l1_prog 旗 == epoch，通过后 __threadfence()。
//   7) void l1_progress_publish(uint32_t* bar_flags, int l1_prog_idx,
//                               int pool_slot, int slice, unsigned epoch,
//                               int nslice = kNSlice);
//      L1 完成旗发布：__threadfence() 后 __stwt 写 epoch（每 (pool_slot,slice) 一旗）。
//   8) void xbar_route_ready_body(const uint8_t* self_sym,
//                                 uint8_t* const* peer_bases, int R, int rank,
//                                 int64_t barrier_slots_off, unsigned* epoch_ctr,
//                                 uint32_t* bar_flags, int64_t xrelease_idx,
//                                 unsigned* last_release, int* dbg, int max_spins,
//                                 int tid_base = 0);
//      mode3 跨 rank ROUTE_READY barrier：block0 前 64 lane 发布（lane→peer 并行，
//      barrier_slots[src][dst] 转置，__stwt）+ ballot 收齐（popcll==64）；lane0
//      fence 后写 release=epoch；其余 block tid0 等 release 变化
//      （last_release[blockIdx]），末尾全块 __syncthreads()。
//
// 与骨架的差异登记：
//   [D1] xbar_route_ready_body：冻结签名缺少 release 旗与每 block 等待水位，
//        mode3 语义必需，故在冻结 6 参之后追加 bar_flags/xrelease_idx/
//        last_release/dbg/max_spins（与骨架 xbar_body_v<3> 参数一致，bench:298-385）；
//        骨架诊断开关 skip_wait 移除（恒 false）。tid_base 保留但按骨架恒用
//        block0 的 threadIdx.x<64（WS 子集发布未验证，不启用）。
//   [D2] subset_barrier_n：骨架为 (cnt, phase, tid_base, nwarps)（bench:428-437），
//        冻结签名收敛为 (smem_ctr, nthreads)：phase 由累计计数推导
//        （第 a 次到达的目标 = ((a-1)/strands+1)*strands，strands=nthreads/64）；
//        带 tid_base 的子集用 subset_barrier_phase（骨架原样）。要求 nthreads 为
//        64 的倍数、每参与 warp 的 lane0 调用、smem_ctr 初值 0。
//   [D3] SliceGate：冻结 5 字段之后追加 seq/dbg/max_spins（骨架 L1SliceGate
//        bench:715-726 的到达值/诊断字段）；seq==0（epoch 恒为奇数，从 1 起，
//        0 永不为真值）视为门关闭。冻结文档注释"本 rank l1_prog 旗区"按 §4.3
//        应为 DISPATCH 分片就绪旗；l1_prog 门另见 l1_prog_gate_wait。
//   [D4] l2_gate_wait / l1_progress_publish：冻结签名之后追加可选参数
//        （l1_prog_idx/l1_epoch/dbg/max_spins 与 nslice），前 8 参位置不变。
//   [D5] slice_gate_wait 的 warp 广播：骨架用 __shfl_sync(0xffffffffu,...)
//        （bench:769，32 位掩码），MACA warp=64，此处改用 __shfl_sync(~0ull,...)
//        （与骨架 __shfl 的 UINT64_MAX 语义一致，避免半个 warp 丢 acquire fence）。
//   [D6] xbar_route_ready_body（阶段 9，A/B 见 test/09.prologue_opt/results.md）：
//        删除 lane0 release store 之后的尾 `__threadfence_system()`——release 语义
//        只需 store 之前的 fence；对端观察到 release 后自行 `__threadfence_system()`
//        acquire，再 `__syncthreads()` 放行本块。mode3 barrier/task_ready 协议与
//        可见性顺序（payload→fence→ready）不变。同会话 A/B：R8 graph 链前序
//        20.0→18.4µs，R8 全链 ~411.0→~410.4µs；正确性 PASS。
//   [D7] 阶段 13（test/13.r32_final/results.md，同会话背靠背 A/B）：
//        a) [V4] 发布前的 release 收敛：在 warp0 的 `__threadfence_system()` 与
//           lane0 announce 之间显式 `__syncwarp()`。配合 prologue.cuh [P8] 把
//           route 远端写收敛到 warp0（各 lane 自行 fence 后 warp 内收敛），
//           route 尾 fence 得以删除而 release 语义（payload→fence→ready）不变；
//        b) [V5] 非 leader 观察到 release 后的 acquire 由 `__threadfence_system()`
//           改为 `__threadfence()`（设备域）：release 旗与随后消费的数据均在本地
//           设备显存，设备域 acquire 语义充分，跨 rank 数据不经此路径。
//        同会话 R=32 全链：base 401.0~402.7 → 采用版 396.98~398.23µs（3 次采样
//        全 <400）；R=32 三形态正确性 + 数值三行 PASS。
// ============================================================================
#pragma once

#include "common.cuh"

// ---------------------------------------------------------------- 子集 barrier --
// 冻结接口（[D2]）：整 warp 子集、每 warp lane0 到达、累计计数推导 phase。
__device__ __forceinline__ void subset_barrier_n(uint32_t* smem_ctr,
                                                 int nthreads) {
    const uint32_t strands = (uint32_t)(nthreads >> 6);   // 参与 warp 数
    if (strands == 0u) return;
    __syncwarp();
    if ((threadIdx.x & 63) == 0) {
        const uint32_t arrived = atomicAdd(smem_ctr, 1u) + 1u;
        // 本 barrier 的目标：arrived ∈ ((p-1)*strands, p*strands] → 目标 p*strands
        const uint32_t target =
            ((arrived - 1u) / strands + 1u) * strands;
        volatile uint32_t* p = smem_ctr;
        while (*p < target) { }
    }
    __syncwarp();
    __threadfence_block();
}

// 骨架原语（bench:428-437）：tid_base 起的 nwarps 个 warp 参与，phase 从 1 递增。
__device__ __forceinline__ void subset_barrier_phase(volatile int* cnt,
                                                     int phase, int tid_base,
                                                     int nwarps) {
    __syncwarp();
    if (((threadIdx.x - tid_base) & 63) == 0) {
        atomicAdd(const_cast<int*>(cnt), 1);
        while (*cnt < nwarps * phase) { }
    }
    __syncwarp();
    __threadfence_block();
}

// ------------------------------------------------------- L1 分片就绪门 --
// 冻结接口 + [D3] 扩展字段。flags 指向 DISPATCH 分片就绪旗区（bench:1755-1757）。
struct SliceGate {
    const uint32_t* flags = nullptr;   // 本 rank 分片就绪旗区
    int nslice = kNSlice;              // NSLICE
    int self_slice = 0;                // 本任务负责的 slice
    int k_iters_per_slice = 0;         // (K/32)/NSLICE
    int base_slot = 0;                 // pool_slot * NSLICE
    // --- T1 扩展（骨架 L1SliceGate 语义；冻结字段顺序不变）---
    uint32_t seq = 0;                  // 本窗 epoch；0 = 门关闭
    int* dbg = nullptr;                // 超时计数（dbg 现场，可空）
    int max_spins = (int)kSpinLimit;   // 自旋上限
    // [集71] 内部执行次数统计（编译期关闭时宏展开为空；运行期由 ctr==nullptr 关）
    unsigned long long* ctr = nullptr;  // [kExecCtrSlots] 计数缓冲
};

// L1 K 循环分片门（bench:747-772）：仅非本项目片、K 片边界处逐片等待。
// 轻量协议：lane0 volatile 轮询 + warp 广播；真正等过才 acquire fence。
__device__ __forceinline__ void slice_gate_wait(const SliceGate& g, int k) {
    if (g.flags == nullptr || g.k_iters_per_slice <= 0 || g.seq == 0) return;
    if (k % g.k_iters_per_slice != 0) return;
    const int s = k / g.k_iters_per_slice;
    if (s == g.self_slice) return;
    if (s < 0 || s >= g.nslice) return;
    const int lane = threadIdx.x & 63;
    unsigned waited = 0;
    if (lane == 0) {
        long long spin = 0;
        while (*(volatile const uint32_t*)(g.flags + g.base_slot + s) !=
               g.seq) {
            waited = 1;
            if (++spin > g.max_spins) {
                if (g.dbg) atomicAdd(g.dbg, 1);
                break;
            }
        }
        // [集71] 分片门：等待次数 / 自旋轮数 / 超时次数
        MX_CTR_ADD(g.ctr, kCtrSliceGateWait, 1);
        MX_CTR_ADD(g.ctr, kCtrSliceGateSpin, spin);
    }
    waited = __shfl_sync(~0ull, waited, 0);   // [D5] 全 64-lane 广播
    __syncwarp();
    if (waited) __threadfence();   // acquire：观察到分片旗后再读该片数据
}

// ------------------------------------------------------- L1 进度旗 --
// L2 入口只等本 slice 的 L1 完成旗（bench:1834-1859；骨架用 != epoch 判据）。
__device__ __forceinline__ void l1_prog_gate_wait(
        const uint32_t* bar_flags, int l1_prog_idx, int pool_slot, int slice,
        unsigned epoch, int nslice = kNSlice, int* dbg = nullptr,
        int max_spins = (int)kSpinLimit, unsigned long long* ctr = nullptr) {
    volatile const uint32_t* p =
        bar_flags + l1_prog_idx + pool_slot * nslice + slice;
    long long spin = 0;
    while (*p != epoch) {
        if (++spin > max_spins) {
            if (dbg) atomicAdd(dbg + 4, 1);   // 骨架 l1_prog 等待超时号
            break;
        }
    }
    // [集71] l1_prog 门：等待次数 / 自旋轮数 / 超时次数
    MX_CTR_ADD(ctr, kCtrL1ProgWait, 1);
    MX_CTR_ADD(ctr, kCtrL1ProgSpin, spin);
    if (spin > max_spins) MX_CTR_ADD(ctr, kCtrL1ProgTimeout, 1);
    __threadfence();   // acquire：旗可见后再读该 slice 的 L1 输出
}

// ------------------------------------------------------- l2_done ring 门 --
// 冻结接口 + [D4] 可选 L1 进度等待（骨架 bench:1648-1663 / 1834-1859）。
// 单调计数门：target=(rep+w/ring)*nslice；通过后 __threadfence()。
__device__ __forceinline__ void l2_gate_wait(
        const uint32_t* bar_flags, int l2_done_idx, int e, int w, int rep,
        int ring, int nslice, int self_slice, int l1_prog_idx = -1,
        uint32_t l1_epoch = 0u, int* dbg = nullptr,
        int max_spins = (int)kSpinLimit, unsigned long long* ctr = nullptr) {
    const int pool_slot = e * ring + (w % ring);
    if (w >= ring || rep > 0) {
        const uint32_t target = (uint32_t)(rep + w / ring) * (uint32_t)nslice;
        volatile const uint32_t* p = bar_flags + l2_done_idx + pool_slot;
        long long spin = 0;
        while (*p < target) {
            if (++spin > max_spins) {
                if (dbg) atomicAdd(dbg + 5, 1);   // 骨架 l2_done 门超时号
                break;
            }
        }
        // [集71] l2_done ring 门：等待次数 / 自旋轮数 / 超时次数
        MX_CTR_ADD(ctr, kCtrL2DoneWait, 1);
        MX_CTR_ADD(ctr, kCtrL2DoneSpin, spin);
        if (spin > max_spins) MX_CTR_ADD(ctr, kCtrL2DoneTimeout, 1);
        __threadfence();   // acquire：门通过后复用/读写共享槽
    }
    if (l1_prog_idx >= 0 && self_slice >= 0) {
        l1_prog_gate_wait(bar_flags, l1_prog_idx, pool_slot, self_slice,
                          l1_epoch, nslice, dbg, max_spins, ctr);
    }
}

// L1 完成旗发布（bench:1826-1832）：先 __threadfence 再 __stwt（可见性顺序）。
// 调用方约定：每个 (pool_slot, slice) 只由完成该 slice 的线程（骨架为 tid0）调用。
__device__ __forceinline__ void l1_progress_publish(
        uint32_t* bar_flags, int l1_prog_idx, int pool_slot, int slice,
        unsigned epoch, int nslice = kNSlice) {
    __threadfence();
    // 同设备旗：普通写 + fence（理由同 xbar release）
    *reinterpret_cast<volatile uint32_t*>(bar_flags + l1_prog_idx +
                                          pool_slot * nslice + slice) = epoch;
    __threadfence();
}

// ------------------------------------------------- mode3 跨 rank barrier --
// 与骨架 xbar_body_v<3> 同构（bench:298-385）：转置布局 barrier_slots[src][dst]
// + lane→peer 并行发布 + ballot 收齐 + lane0 release；非 block0 仅 tid0 等 release。
// [D1] 追加 bar_flags/xrelease_idx/last_release/dbg/max_spins；tid_base 保留不启用。
__device__ __forceinline__ void xbar_route_ready_body(
        const uint8_t* self_sym, uint8_t* const* peer_bases, int R, int rank,
        int64_t barrier_slots_off, unsigned* epoch_ctr, uint32_t* bar_flags,
        int64_t xrelease_idx, unsigned* last_release, int* dbg, int max_spins,
        int tid_base = 0, unsigned long long* ctr = nullptr) {
    (void)tid_base;   // 骨架恒用 block0 的 threadIdx.x<64（见 [D1]）
    // [集71] XBAR_ROUTE_READY 参与 block 数（tid0 计一次）
    if (threadIdx.x == 0) MX_CTR_ADD(ctr, kCtrPhaseXbar, 1);
    if (blockIdx.x == 0 && threadIdx.x < 64) {
        const int lane = threadIdx.x;
        __threadfence_system();
        // [D7a][V4] warp 内收敛：确保各 lane 的 release fence 先于 lane0 的
        // announce（route 远端写已收敛到 warp0，见 prologue.cuh [P8]）
        __syncwarp();
        unsigned epoch = 0;
        if (lane == 0) epoch = atomicAdd(epoch_ctr, 2u) + 1u;
        epoch = __shfl(epoch, 0, 64);   // 先广播，否则并行 lane 会发 0
        // 每 lane 发一个 peer（lane==rank 无动作）；mode3 转置 idx = rank*R + lane
        if (lane < R && lane != rank) {
            const size_t idx = (size_t)(rank * R + lane);
            uint32_t* slot = reinterpret_cast<uint32_t*>(
                const_cast<uint8_t*>(peer_bases[lane]) + barrier_slots_off +
                idx * 4);
            __stwt(slot, epoch);
        }
        __threadfence_system();
        // 收齐自己该收的槽：转置槽 (src*R+rank)；lane<R 各自轮询 + ballot 聚合
        int spins = 0;
        while (true) {
            int my_done = 1;
            if (lane < R) {
                const int src = lane;
                if (src != rank) {
                    volatile const uint32_t* mine =
                        reinterpret_cast<volatile const uint32_t*>(
                            self_sym + barrier_slots_off +
                            (size_t)(src * R + rank) * 4);
                    my_done =
                        (static_cast<int32_t>(*mine - epoch) >= 0) ? 1 : 0;
                }
            }
            const unsigned long long mask = __ballot(my_done != 0);
            int ready_total = 0;
            if (lane == 0) ready_total = (int)__popcll(mask);
            ready_total = __shfl(ready_total, 0, 64);
            if (ready_total == 64) break;
            if (++spins > max_spins) {
                if (lane == 0 && dbg) atomicAdd(dbg + 0, 1);
                break;
            }
        }
        // [集71] block0 收齐侧：自旋轮数 + 超时次数
        if (lane == 0) {
            MX_CTR_ADD(ctr, kCtrXbarSpin, spins);
            if (spins > max_spins) MX_CTR_ADD(ctr, kCtrXbarTimeout, 1);
        }
        if (lane == 0) {
            __threadfence_system();
            // 同设备旗用普通写 + fence：__stwt 在普通显存上不更新 L2，
            // volatile 读可能长期命中旧值（实测 release 等待偶发超时）。
            *reinterpret_cast<volatile uint32_t*>(bar_flags + xrelease_idx) =
                epoch;
            // [V1x] 尾 fence_system 删除：release 之后本 lane 不再发布依赖数据
        }
    }
    // 非 block0：tid0 等 release 变化，写回水位 + fence，再全块同步
    if (threadIdx.x == 0 && blockIdx.x != 0) {
        volatile uint32_t* release = bar_flags + xrelease_idx;
        unsigned* last = last_release + blockIdx.x;
        const unsigned prev = *last;
        int spins = 0;
        while (*release == prev) {
            if (++spins > max_spins) {
                if (dbg) {
                    atomicAdd(dbg + 1, 1);
                    dbg[6] = (int)blockIdx.x;   // 诊断：超时 block
                    dbg[7] = (int)*release;     // 诊断：release 读值
                }
                break;
            }
        }
        // [集71] 非 block0 等待侧：等待次数 / 自旋轮数 / 超时次数
        MX_CTR_ADD(ctr, kCtrXbarWait, 1);
        MX_CTR_ADD(ctr, kCtrXbarSpin, spins);
        if (spins > max_spins) MX_CTR_ADD(ctr, kCtrXbarTimeout, 1);
        *last = *release;
        // [D7b][V5] 同设备旗 acquire 用设备域 fence（release 旗与消费数据均在
        // 本地显存；跨 rank 数据不经此路径）
        __threadfence();
    }
    __syncthreads();
}
