// ============================================================================
// prologue.cuh — fp8xfp8 MoE Mega Kernel 前序相位（T1）
//
// 内容：ROUTE_BUILD / PUBLISH_COUNTS / TASK_SETUP 三个设备函数。协议、指令序列、
// flag 索引（由参数传入）、checksum、可见性顺序全部照抄骨架
// empty_kernel_bench.cu 的对应实现（行号见各函数注释），仅做参数化适配。
//
// 导出函数签名与语义：
//   1) void route_build_body(const int64_t* __restrict__ topk_idx, int* rsc_ptr,
//                            unk_data* const* unk_peers, int cycle, int R,
//                            int rank, int tokens, int slot_capacity,
//                            int tid_base = 0, int smem_off = 0);
//      ROUTE_BUILD（仅 block0，bench:190-224）：smem 占槽（atomicAdd）后逐 pair
//      4B __stwt 远端散写 src_index_table[le][rank][slot]（encoded = idx，
//      即 token*topk + k）；非法 ID/超容量丢弃；本地 route_slot_counts[e] =
//      min(slots[e], slot_capacity)。[P8]：生产路径写收敛到 warp0，尾 fence 由
//      xbar 的 release fence 承接（见下）；tid_base≠0 保留原全线程 + 尾 fence。
//   2) void publish_counts_body(const int* rsc_ptr, unk_data* const* unk_peers,
//                               int cycle, int R, int rank, unsigned epoch,
//                               int recv_stride = 12, int tid_base = 0);
//      PUBLISH_COUNTS（仅 block0，bench:229-255）：把本 rank 每专家发送计数
//      __stwt 到目标 rank 的 expert_recv_counts[rank][le]（行=源 rank，宽 12 word）；
//      行尾校验槽 = ~(Σ行内计数 + epoch)。
//   3) int task_setup_body(const uint8_t* self, int R, int rank,
//                          int slot_capacity, int64_t recv_off,
//                          int64_t prefix_off, local_flags* lf,
//                          int64_t task_ready_idx, int64_t task_counter_idx,
//                          int64_t l2_done_idx, int* out_total, int* dbg,
//                          int max_spins, int recv_stride = 12,
//                          bool reset_ring = true, int tid_base = 0,
//                          volatile int* sb_cnt = nullptr, int smem_off = 0,
//                          int* prefix_ptr = nullptr);
//        launch 身份（task_ready 旗值）改由 lf->current_time[b][1] 派生。
//      TASK_SETUP（leader=block0，bench:443-588）：
//        leader：FG 快轮询 checksum（Σ+epoch==~csum；R≤8 warp-per-row 用 __ldcv
//          读 entry/cslot，R>8 lane-per-row 用 __ldcv 读 vals/cslot）→ 成功批写
//          smem → 并行建 prefix_tbl[EPR][R+1] + 并行清零 l2_done ring（EPR×ring，
//          见 [P4c]）→ fence → tid0 建 count/cum 窗口表、重置
//          task_counter=gridDim.x → fence → task_ready=epoch（同设备旗：普通写+
//          fence）→ fence；
//        非 leader：等 task_ready（由 lf->current_time[blockIdx] 派生 launch 身份）
//          后从 prefix_tbl 重建窗口表；
//        返回本 block 总窗口数 Σ_e ceil(count_e/slot_capacity)；checksum 超时 -1。
//
// 与骨架的差异登记：
//   [P1] publish_counts_body：去掉骨架的 tag_mode/fence_mid 诊断开关（V2 fence /
//        V4 tag 均实测否决，bench:259-274 仅为 A/B 用），只保留生产默认路径；
//        recv_stride 由调用方传入（默认 ((EPR+1+3)&~3)=12）；epoch 由调用方传
//        （主控传 kPublishEpoch）。
//   [P2] task_setup_body：
//        a) 去掉 kChecksum/kCkVariant 模板与 tag 变体（checksum 不可省，恒开；
//           骨架 bench:443-601 的 A/B 开关属测试专用）；
//        b) task_counter/l2_done 索引由参数传入（骨架内嵌 layout 常量）；
//        c) 非 leader 增加 task_ready 等待（生产 phase_task_setup 语义，与
//           docs/01 §1.2、results.md §6 一致；骨架 bench:585 缺该等待，单 launch
//           冷启动下非 leader 读 prefix 存在竞态，仅靠 graph 链上一轮残留值通过）；
//           超时计数写 dbg+3；
//        d) out_total 由每个 block 写自己的槽（骨架仅 leader 写 out_total[0]）；
//        e) smem→prefix 构建循环与 checksum 分支判断改用 tid（tid_base 相对），
//           使 tid_base≠0 的 WS 子集调用自洽；tid_base=0 与骨架逐位一致。
//   [P3] 偏移类型统一 int64_t（骨架为 long long，同型）；其余指令序列逐条一致。
//   [P4] O-A1（阶段 2 前序优化，A/B 记录见 test/02.phase2_opt/01.prologue_opt/）：
//        a) publish_counts_body 回退为骨架 __stwt（跨 rank 写、无 fence；普通写+
//           双 __threadfence_system 系诊断实验残留）：R=2 publish 4.5→0.7µs；
//        b) checksum 读回退骨架 V1 的 __ldcv（volatile 读系诊断实验残留）：
//           R=2 setup 15.9→14.5µs（读本身非瓶颈，见 c）；
//        c) l2_done ring 清零由 tid0 串行 volatile 写 64 槽改为全体线程并行
//           （每线程至多 1 槽；消融实测该串行写占 setup ~9µs）→ setup 14.5→5.9µs；
//           写策略不变：普通写 + 释放 fence，与 prefix_tbl 同批 fence 后再放行
//           task_ready（同设备旗策略见 flags.cuh / compute.cuh，均未改）。
//   [P5] O-A2（阶段 9 前序优化，A/B 记录见 test/09.prologue_opt/results.md）：
//        task_setup_body 的同步/串行段收敛，协议语义不变（checksum 恒开、recv 行
//        宽 12、mode3 barrier、task_ready 等待、barrier_slots 清零/epoch 均保持）：
//        a) 删除 `__syncthreads_or` 之后冗余的全块 barrier（前者已是全块 barrier）；
//        b) prefix 构建循环顺带并行写 s_count[e]（每专家一线程），tid0 只做 8 项
//           窗口前缀和（生产路径 warp 内 __syncwarp 同步；kSub 用子集 barrier）；
//        c) out_total 在放行 barrier 前由 tid0 写；leader/非 leader 尾部 2 次
//           barrier 删除（s_count/s_cum 已由放行 barrier 发布，dispatch 会自行
//           从 prefix 重算；返回值对全体线程仍有效）；
//        d) leader 的 epoch 原子自增与 task_counter 发布提前到 checksum 轮询之前
//           （原子/store 延迟与轮询重叠）；task_ready 前的 release fence 保留，
//           store 之后的尾 fence 删除（release 语义只需 store 前 fence）；
//        e) l2_done ring 清零由 volatile 写改为普通写（与其他同设备旗一致：
//           同批 __threadfence + barrier 释放，读者用 volatile/l2 门）。
//        实测（同会话背靠背）：R=2 单 launch setup 5.7→4.7µs；graph 链前序
//        R2 20.3→17.0µs、R8 21.5→18.4µs（含 flags.cuh [D6]）；R8 全链
//        ~411.7→~410.4µs（本 agent 边界内）。
//   [P8] 阶段 13（test/13.r32_final/results.md）：route_build_body 生产路径
//        （tid_base==0）远端 `__stwt` 收敛到 warp0（每 lane 至多 ceil(pairs/64)
//        条），尾 `__threadfence_system()` 删除。release 可见性由 flags.cuh
//        [D7a] 的 warp0 fence + `__syncwarp()` 承接（各写 lane 自行 fence 后
//        warp 内收敛于 announce 之前），payload→fence→ready 顺序不变；槽位分配
//        仍为 smem atomicAdd，slot 顺序本就与线程调度无关，校验不受影响。
//        同会话 R=32 全链 base 401.0~402.7 → 采用版 396.98~398.23µs（3 采样
//        全 <400），R=2/R=8/R=32 三形态正确性与数值三行全 PASS。
// ============================================================================
#pragma once

#include "common.cuh"
#include "flags.cuh"

// [BF16-IN] 4 个 bf16 → 4 个 fp8(E4M3, RNE) 打包为 uint32。
__device__ __forceinline__ uint32_t mx_fold_bf16x4(const uint16_t* p) {
    using F32x4 = __NATIVE_VECTOR__(4, float);
    F32x4 f;
    #pragma unroll
    for (int j = 0; j < 4; ++j) f[j] = __uint_as_float(((uint32_t)p[j]) << 16);
    return __builtin_mxc_cvt_pk4_f32tof8_cfg(
        f, _MACA_FROUND_TO_NEAREST_EVEN, _MACA_DENORMAL_FLUSH_NONE,
        _MACA_SRC_NO_MODIFIER, false, true);
}
// 本地 fold：x(bf16)[n_elems] → fp8（同长度，1B/元素），全 grid 协作；写完
//   __threadfence，供 peer 经 xbar 门后 pull（体积减半）。
__device__ __forceinline__ void x_fold_body(const uint8_t* x, uint8_t* pf,
                                            int n_elems) {
    const int stride = (int)(gridDim.x * blockDim.x) * 4;
    const uint16_t* xh = reinterpret_cast<const uint16_t*>(x);
    for (int i = (int)(blockIdx.x * blockDim.x + threadIdx.x) * 4; i < n_elems;
         i += stride)
        *reinterpret_cast<uint32_t*>(pf + i) = mx_fold_bf16x4(xh + i);
    __threadfence();
}

// ---------------------------------------------------------------- ROUTE_BUILD --
// 骨架 bench:190-224 逐条移植（仅 block0 调用；tid_base/smem_off 供 WS 子集）。
__device__ __forceinline__ void route_build_body(
        const int64_t* __restrict__ topk_idx,
        int* rsc_ptr, unk_data* const* unk_peers,
        int cycle, int R, int rank, int tokens, int slot_capacity,
        int tid_base = 0, int smem_off = 0,
        unsigned long long* ctr = nullptr) {
    extern __shared__ __align__(16) uint8_t smem_raw[];
    int* slots = reinterpret_cast<int*>(smem_raw + smem_off);
    const int E = R * kEPR;
    const int tid = threadIdx.x - tid_base;
    const int nthr = (tid_base != 0) ? 128 : (int)blockDim.x;
    // [集71] ROUTE_BUILD 执行次数（tid0 计一次）+ PROLOGUE 参与 block 数
    if (tid == 0) {
        MX_CTR_ADD(ctr, kCtrPhaseRouteBuild, 1);
        MX_CTR_ADD(ctr, kCtrPhasePrologue, 1);
        MX_CTR_ADD(ctr, kCtrRoutePair, tokens * kTopk);
    }
    for (int e = tid; e < E; e += nthr) slots[e] = 0;
    __syncthreads();
    const int num_pairs = tokens * kTopk;
    // [P8] 原实现把生产路径（tid_base==0）远端写收敛到 warp0 并删尾 fence；
    // [R26] 在 R=8/TPR=128 上全线程并行 + 尾 fence_system 更快（route 相位
    // 18.6→13.6µs），已改回全线程；tid_base≠0（WS 子集，未验证）保持原路径。
    if (tid_base == 0) {
        // [R26] route 并行度自适应（同会话 71 机 R=8）：
        //   pair 多（≥768，如 TPR=128=1024）全 512 线程并行更快：route 相位
        //     18.6→13.6µs，全链 −4~6µs；每线程尾 fence_system 承接自身远端写
        //     的 release 可见性，route 尾 __syncthreads + xbar warp0 release
        //     fence/announce 保持 payload→fence→ready 顺序不变（与骨架全线程
        //     路径同语义）。
        //   pair 少（≤512，如 TPR=16/64）warp0 收敛版更省（fence 成本 > 并行
        //     收益，TPR=64 实测 +3.6µs），保持原 [P8] 路径。
        //   R>8（R=32 等）本轮未在 71 机验证（设备 24-31 按任务约束避开），
        //   保守起见仍走 [P8] warp0 路径，待后续单独验证。
        // [Y1] R>8（R=32/TPR=256=2048 pair）同样受益：warp0 路径需 32 次
        // 远端 __stwt/lane（route 相位 30µs@R32），全 512 线程 4 次/线程，
        // 尾 __threadfence_system 承接各线程自身远端写的 release 可见性；
        // R=32 三形态正确性 + 数值三行已验证（test/28.r32_fabric）。
        if (R > 8 || num_pairs >= 768) {
            for (int idx = tid; idx < num_pairs; idx += (int)blockDim.x) {
                const int64_t raw = topk_idx[idx];
                if (static_cast<uint64_t>(raw) >=
                    static_cast<uint64_t>(E)) continue;
                const int e = static_cast<int>(raw);
                const int slot = atomicAdd(slots + e, 1);
                // [集71] 丢弃计数（超容量）
                if (slot >= slot_capacity) {
                    MX_CTR_ADD(ctr, kCtrRouteDrop, 1);
                    continue;
                }
                const int dst = e / kEPR;
                const int le = e % kEPR;
                uint32_t* dstpt =
                    &unk_peers[dst]->rd.src_index_table[cycle][le][rank][0];
                __stwt(dstpt + slot, (uint32_t)idx);
            }
            // __threadfence_system();
        } else if (tid < 64) {
            for (int idx = tid; idx < num_pairs; idx += 64) {
                const int64_t raw = topk_idx[idx];
                // 非法 ID（负/越界）与生产一致：直接跳过
                if (static_cast<uint64_t>(raw) >=
                    static_cast<uint64_t>(E)) continue;
                const int e = static_cast<int>(raw);
                const int slot = atomicAdd(slots + e, 1);
                // [集71] 丢弃计数（超容量）
                if (slot >= slot_capacity) {
                    MX_CTR_ADD(ctr, kCtrRouteDrop, 1);
                    continue;
                }
                const int dst = e / kEPR;
                const int le = e % kEPR;
                uint32_t* dstpt =
                    &unk_peers[dst]->rd.src_index_table[cycle][le][rank][0];
                __stwt(dstpt + slot, (uint32_t)idx);
            }
        }
    } else {
        for (int idx = tid; idx < num_pairs; idx += nthr) {
            const int64_t raw = topk_idx[idx];
            if (static_cast<uint64_t>(raw) >= static_cast<uint64_t>(E)) continue;
            const int e = static_cast<int>(raw);
            const int slot = atomicAdd(slots + e, 1);
            if (slot >= slot_capacity) continue;   // 超容量丢弃
            const int dst = e / kEPR;
            const int le = e % kEPR;
            uint32_t* dstpt =
                &unk_peers[dst]->rd.src_index_table[cycle][le][rank][0];
            __stwt(dstpt + slot, (uint32_t)idx);
        }
        // __threadfence_system();
    }
    __syncthreads();
    // rsc_ptr 必为非空（调用方传 &rwk->ld.route_slot_counts[0]）
    int* rsc = rsc_ptr;
    for (int e = tid; e < E; e += nthr) {
        const int c = slots[e];
        rsc[e] = c > slot_capacity ? slot_capacity : c;
    }
    __syncthreads();
}

// ------------------------------------------------------------ PUBLISH_COUNTS --
// 骨架 bench:229-255 逐条移植（仅 block0 调用；行宽 recv_stride 由调用方保证）。
__device__ __forceinline__ void publish_counts_body(
        const int* rsc_ptr, unk_data* const* unk_peers, int cycle, int R, int rank,
        unsigned epoch,
        int recv_stride = (int)((kEPR + 1 + 3) & ~3), int tid_base = 0,
        unsigned long long* ctr = nullptr) {
    const int E = R * kEPR;
    const int tid = threadIdx.x - tid_base;
    const int nthr = (tid_base != 0) ? 128 : (int)blockDim.x;
    // [集71] PUBLISH_COUNTS 执行次数（tid0 计一次）
    if (tid == 0) MX_CTR_ADD(ctr, kCtrPhasePublish, 1);
    const int* rsc = rsc_ptr;   // 必为非空
    for (int e = tid; e < E; e += nthr) {
        const int dst = e / kEPR;
        const int le = e % kEPR;
        uint32_t* dstp = &unk_peers[dst]->rd.expert_recv_counts[cycle][rank][0];
        // 跨 rank 写：__stwt（write-through）；peer 可见性由骨架 peer_visibility 验证
        __stwt(dstp + le, (uint32_t)rsc[e]);
    }
    for (int d = tid; d < R; d += nthr) {
        uint32_t csum = 0;
        for (int le = 0; le < kEPR; ++le)
            csum += (uint32_t)rsc[d * kEPR + le];
        uint32_t* dstp = &unk_peers[d]->rd.expert_recv_counts[cycle][rank][0];
        // 行尾校验槽：读者以 Σ+epoch==~csum 闭合远端写可见性
        __stwt(dstp + kEPR, ~(csum + epoch));
    }
}

// -------------------------------------------------------------- TASK_SETUP --
// 64-lane warp 全宽归约（骨架 bench:412-417 逐条一致；仅 task_setup 使用）。
__device__ __forceinline__ uint32_t warp_sum_u32_64(uint32_t value, int lane) {
    #pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1)
        value += __shfl(value, (lane + offset) & 63);
    return value;
}

// leader=block0（bench:443-588）：checksum → prefix_tbl → count/cum 窗口表 →
// task_counter/l2_done 重置 → task_ready=epoch；非 leader 等 task_ready 后重建窗口表。
__device__ __forceinline__ int task_setup_body(
        const uint8_t* self, const ranks_data* remote, int R, int rank,
        int slot_capacity, int64_t prefix_off, local_flags* lf,
        int64_t task_ready_idx, int64_t task_counter_idx, int64_t l2_done_idx,
        int* out_total, int* dbg,
        int max_spins, int recv_stride = (int)((kEPR + 1 + 3) & ~3),
        bool reset_ring = true, int tid_base = 0, volatile int* sb_cnt = nullptr,
        int smem_off = 0, int* prefix_ptr = nullptr,
        uint32_t* ck_dbg = nullptr, unsigned long long* ctr = nullptr) {
    // [M_CYCLE] 本轮 copy c：task_setup 相位用 current_time[b][1]（入口尚未自增，
    //   全 block 同为 L ⇒ c 一致）。bar 取第 c 片，与 xbar 用同一 c（两计数器
    //   每 launch 各同步 +1，模 M_CYCLE 恒相等）。自增仍写回 current_time。
    const int cyc = (int)(lf->current_time[blockIdx.x][1] % M_CYCLE);
    uint32_t* bar_flags = lf->bar[cyc];
    extern __shared__ __align__(16) uint8_t smem_raw[];
    uint32_t* s_recv = reinterpret_cast<uint32_t*>(smem_raw + smem_off);
    const int tid = threadIdx.x - tid_base;
    const int warp = tid / 64;
    const int lane = tid % 64;
    const bool kSub = (tid_base != 0);
    const int nwarps = kSub ? 4 : (int)(blockDim.x / 64);
    const int nthr = kSub ? 256 : (int)blockDim.x;
    int sb_phase = 0;
    auto sb = [&]() {
        if (kSub) subset_barrier_phase(sb_cnt, ++sb_phase, tid_base, 4);
        else __syncthreads();
    };
    const bool leader = (blockIdx.x == 0);
    // [集71] TASK_SETUP 参与 block 数（tid0 计一次）+ leader 次数
    if (threadIdx.x == 0) {
        MX_CTR_ADD(ctr, kCtrPhaseTaskSetup, 1);
        if (leader) MX_CTR_ADD(ctr, kCtrTaskSetupLeader, 1);
    }
    volatile const uint32_t* recv = remote->expert_recv_counts[cyc][0];
    int* prefix = prefix_ptr != nullptr
                      ? prefix_ptr
                      : reinterpret_cast<int*>(const_cast<uint8_t*>(self) +
                                               prefix_off);
    __shared__ int s_count[kEPR];
    __shared__ int s_cum[kEPR + 1];
    bool failed = false;
    unsigned my_epoch = 0u;
    if (leader) {
        // [P5d] epoch 原子自增与任务计数器提前发布：原子延迟与 checksum 轮询重叠；
        // 计数器的可见性由 ② 的 __threadfence + sb 与 task_ready release 覆盖。
        if (tid == 0) {
            *(volatile int*)(bar_flags + task_counter_idx) = (int)gridDim.x;
            // launch 身份由 block0 自己的 current_time[0][1] 派生（取代 epoch_ctr）
            const int64_t c = lf->current_time[0][1];
            my_epoch = (unsigned)(c + 1);
            lf->current_time[0][1] = c + 1;
        }
        // ① FG 快轮询 checksum，成功即把同一批已读值写入 smem（bench:477-548）
        if (R <= 8) {
            // R 小：warp-per-row + 8-lane 并行读 + shuffle 归约
            for (int r = warp; r < R; r += nwarps) {
                const uint32_t* row_g =
                    const_cast<const uint32_t*>(recv) + (size_t)r * recv_stride;
                uint32_t* row_s = s_recv + (size_t)r * recv_stride;
                long long spin = 0;
                while (true) {
                    // [集71] checksum：轮询行数 + 自旋轮数
                    if (lane == 0) MX_CTR_ADD(ctr, kCtrChecksumRow, 1);
                    // 骨架 V1：entry/cslot 用 __ldcv（cache-bypass，读延迟可重叠）
                    uint32_t entry =
                        (lane < kEPR) ? __ldcv(row_g + lane) : 0u;
                    uint32_t sum = warp_sum_u32_64(entry, lane);
                    uint32_t cslot =
                        (lane == 0) ? __ldcv(row_g + kEPR) : 0u;
                    cslot = __shfl(cslot, 0, 64);
                    const bool row_ok =
                        (static_cast<uint32_t>(sum + kPublishEpoch) == ~cslot);
                    if (row_ok) {
                        if (lane < kEPR) row_s[lane] = entry;
                        if (lane == 0) row_s[kEPR] = cslot;
                        break;
                    }
                    if (++spin > max_spins) {
                        failed = true;
                        if (lane == 0 && dbg) atomicAdd(dbg + 2, 1);
                        // [集71] checksum 自旋轮数 / 超时次数
                        if (lane == 0) {
                            MX_CTR_ADD(ctr, kCtrChecksumSpin, spin);
                            MX_CTR_ADD(ctr, kCtrChecksumTimeout, 1);
                        }
                        // 诊断：超时行内容落盘（8 列 + csum + 自旋数）
                        if (ck_dbg != nullptr) {
                            if (lane < kEPR)
                                ck_dbg[(size_t)r * 16 + lane] = entry;
                            if (lane == 0) {
                                ck_dbg[(size_t)r * 16 + 8] = cslot;
                                ck_dbg[(size_t)r * 16 + 9] = (uint32_t)spin;
                            }
                        }
                        break;
                    }
                }
                __syncwarp();
                if (failed) break;
            }
        } else if (tid < 64) {
            // R 大：lane-per-row，每 lane 9 次独立 __ldcv 读整行（避免 volatile 串行）
            const int ln = tid;
            long long spin = 0;
            while (true) {
                bool all_ok = true;
                for (int r = ln; r < R; r += 64) {
                    const uint32_t* row_g =
                        const_cast<const uint32_t*>(recv) +
                        (size_t)r * recv_stride;
                    uint32_t vals[kEPR];
                    uint32_t sum = 0;
                    #pragma unroll
                    for (int e = 0; e < kEPR; ++e) vals[e] = __ldcv(row_g + e);
                    #pragma unroll
                    for (int e = 0; e < kEPR; ++e) sum += vals[e];
                    const uint32_t cslot = __ldcv(row_g + kEPR);
                    const bool ok =
                        static_cast<uint32_t>(sum + kPublishEpoch) == ~cslot;
                    if (!ok) {
                        all_ok = false;   // 行未齐：重读本 lane 的全部行
                        break;
                    }
                    uint32_t* row_s = s_recv + (size_t)r * recv_stride;
                    #pragma unroll
                    for (int e = 0; e < kEPR; ++e) row_s[e] = vals[e];
                    row_s[kEPR] = cslot;
                }
                if (all_ok) break;
                if (++spin > max_spins) {
                    failed = true;
                    if (ln == 0 && dbg) atomicAdd(dbg + 2, 1);
                    // [集71] checksum 自旋轮数 / 超时次数
                    if (ln == 0) {
                        MX_CTR_ADD(ctr, kCtrChecksumSpin, spin);
                        MX_CTR_ADD(ctr, kCtrChecksumTimeout, 1);
                    }
                    break;
                }
            }
        }
        // [集71] TASK_SETUP checksum 失败次数（仅 tid0 计一次：__syncthreads_or
        //   的返回值对全块线程相同，必须收敛为 1）
        if (!kSub && __syncthreads_or(failed)) {
            if (threadIdx.x == 0) MX_CTR_ADD(ctr, kCtrTaskSetupFail, 1);
            return -1;
        }
        if (kSub && failed) {
            if (threadIdx.x == 0) MX_CTR_ADD(ctr, kCtrTaskSetupFail, 1);
            return -1;
        }
        // [P5b] __syncthreads_or 已是全块 barrier，其后的 sb() 冗余，删除
        // ② 从 smem 并行建 prefix_tbl（每专家一个线程）：prefix[e][r]=Σ_{r'<r}cnt
        bool wrote_task_prefix = false;
        for (int e = tid; e < kEPR; e += nthr) {
            int c = 0;
            for (int r = 0; r < R; ++r) {
                prefix[e * (R + 1) + r] = c;
                c += (int)s_recv[(size_t)r * recv_stride + e];
            }
            prefix[e * (R + 1) + R] = c;
            s_count[e] = c;   // [P5c] 计数与 prefix 同批并行写
            wrote_task_prefix = true;
        }
        // l2_done ring 并行清零（每线程至多 1 槽）：tid0 串行 volatile 写 64 槽
        // 实测 ~9µs；写策略仍为普通写 + 释放 fence（与 prefix 同批 fence 后再放行）
        bool wrote_l2_ring = false;
        if (reset_ring) {
            for (int i = tid; i < kEPR * kRing; i += nthr) {
                // [P5h] 普通写（同批 __threadfence 释放；读者用 volatile/门）
                bar_flags[l2_done_idx + i] = 0u;
                wrote_l2_ring = true;
            }
        }
        // [P5c] s_count 由 EPR 个 lane 并行写（生产恒在 warp0），warp 内同步后 tid0
        // 只做 8 项窗口前缀和；kSub 子集可能跨 warp 时退回子集 barrier。
        if (kSub) sb(); else __syncwarp();
        if (tid == 0) {
            int total = 0;
            s_cum[0] = 0;
            for (int e = 0; e < kEPR; ++e) {
                total += (s_count[e] + slot_capacity - 1) / slot_capacity;
                s_cum[e + 1] = total;
            }
            if (out_total) out_total[blockIdx.x] = total;
        }
        if (wrote_task_prefix || wrote_l2_ring) __threadfence();
        sb();
        // ③ 发布 task_ready（ring/prefix/表/任务计数已在上一条 barrier 前放行）
        if (tid == 0) {
            __threadfence();
            // 同设备旗：普通写 + fence（理由同 xbar release）；
            // [P5d] 尾 fence 删除：task_ready 之后 tid0 不再发布依赖它的数据
            *reinterpret_cast<volatile uint32_t*>(bar_flags + task_ready_idx) =
                my_epoch;
        }
    } else {
        // 非 leader（[P2c] 生产语义）：等 task_ready 到本 launch 的 epoch 后再读 prefix_tbl
        if (tid == 0) {
            const int64_t c = lf->current_time[blockIdx.x][1];
            const unsigned expected = (unsigned)(c + 1);
            volatile const uint32_t* ready = bar_flags + task_ready_idx;
            long long spin = 0;
            while (*ready != expected) {
                if (++spin > max_spins) {
                    if (dbg) atomicAdd(dbg + 3, 1);
                    break;
                }
            }
            // [集71] task_ready：等待次数 / 自旋轮数 / 超时次数
            MX_CTR_ADD(ctr, kCtrTaskReadyWait, 1);
            MX_CTR_ADD(ctr, kCtrTaskReadySpin, spin);
            if (spin > max_spins) MX_CTR_ADD(ctr, kCtrTaskReadyTimeout, 1);
            lf->current_time[blockIdx.x][1] = c + 1;
            __threadfence();
            // 等待与建表合并（原两次 barrier -> 一次）
            int total = 0;
            s_cum[0] = 0;
            for (int e = 0; e < kEPR; ++e) {
                s_count[e] = prefix[e * (R + 1) + R];
                total += (s_count[e] + slot_capacity - 1) / slot_capacity;
                s_cum[e + 1] = total;
            }
            if (out_total) out_total[blockIdx.x] = s_cum[kEPR];
        }
        // [P5b] 仅保留一次 barrier：放行本块线程读取 s_count/s_cum
        sb();
    }
    return s_cum[kEPR];
}
