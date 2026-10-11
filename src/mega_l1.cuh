#pragma once
// ============================================================================
// mega_l1.cuh — 顶层计算 kernel（从 mega_kernel 改建，命名 mega_l1）
//   链：launch_compute -> mega_l1(kernel) -> 任务体 -> mega_l1_gemm -> r91。
//   依赖 MegaArgs/NoTaskTail/prologue/combine（由 kernel.cuh 在本文件 include
//   之前定义）。
// ============================================================================
template <bool kWithCompute, typename TaskTail = NoTaskTail,
          bool kSmallOnly = false, int kCap = 0,
          MegaPhase kPhase = MegaPhase::Both>
__global__ void __launch_bounds__(512) mega_l1(MegaArgs a, TaskTail tail) {
    // 相位时间戳（block0 tid0；[6]=入口 [0]=route后 [1]=publish后 [2]=xbar后 [3]=setup后）
    const bool kTick = (blockIdx.x == 0) && (threadIdx.x == 0);
    // [集71] 计数缓冲归一（运行期开关折叠成单指针判空；编译期关闭时恒 nullptr，
    //   该局部量无人读取即被优化器消除，不进入任何 kernel 参数/寄存器）
#if MX_DBG_COUNT
    unsigned long long* ctr = mx_ctr_open(a.dbg_count, a.exec_ctr);
#else
    unsigned long long* ctr = nullptr;
#endif
    // [集71] kernel 启动次数（block0 tid0）+ 参与 block 数（每 block tid0）
    if (threadIdx.x == 0) {
        if (blockIdx.x == 0) MX_CTR_ADD(ctr, kCtrKernelLaunch, 1);
        MX_CTR_ADD(ctr, kCtrKernelBlocks, 1);
    }
    // [R24] A-smem 双缓冲（kernel 作用域）；L1 B-smem 流水已删除。
    // dispatch 参数中的 b_smem 暂保留，供调用边界兼容，但始终为 nullptr。
    union __align__(16) ABShared {
        uint8_t a[2][R24_MT * R24_KS * 512];
    };
    __shared__ ABShared s_ab;
    if (kTick) a.rwk->lf.pre_ticks[6] = clock64();
    // 0) [M_CYCLE] 预清 ct2=(c+2)%M_CYCLE 槽（本轮 copy c 尚未定，用 current_time
    //   [b][0] 派生；入口全 block 同为 L ⇒ c 一致）。ct1 留给下一轮，本轮不碰。
    mx_preclear_ct2(&a.rwk->lf, &a.unk->rf, /*skip_xbar=*/a.skip_prologue >= 2);
    __syncthreads();
    if (!a.skip_prologue) {
        // 1) PROLOGUE：ROUTE_BUILD + PUBLISH_COUNTS（仅 block0）
        if (blockIdx.x == 0) {
            const int route_cycle =
                (int)(a.rwk->lf.current_time[0][0] % M_CYCLE);
            route_build_body(a.topk_idx, &a.rwk->ld.route_slot_counts[0],
                             a.unk_peers, route_cycle, a.R, a.rank,
                             a.tokens, a.slot_capacity, 0, 0, ctr);
            if (kTick) a.rwk->lf.pre_ticks[0] = clock64();
            publish_counts_body(&a.rwk->ld.route_slot_counts[0], a.unk_peers,
                                route_cycle, a.R, a.rank, kPublishEpoch,
                                kRecvStride, 0, ctr);
            if (kTick) a.rwk->lf.pre_ticks[1] = clock64();
        }
    } else {
        // [R38] 复用前序：路由/计数持久，跳过 ROUTE/PUBLISH（数据与 node0 相同）。
        if (kTick) a.rwk->lf.pre_ticks[0] = a.rwk->lf.pre_ticks[1] = clock64();
    }
    // 2) XBAR_ROUTE_READY（全 grid，mode3：转置 + warp 并行发布）：每节点保留，
    //    既是 route/publish 可见性门，也是跨 rank 的每节点对齐（防漂移）。
    if (a.skip_prologue >= 2) {
        // 诊断模式：连 xbar 也跳过，仅推进 xbar 相位计数。
        // [L1L2-SPLIT] L2 阶段：xbar 相位计数不由 L2 推进 —— 每 round 仅 L1 前序
        //   前进一次，使 ct0/ct1 单步同步（combine/REDUCE 的 cyc/epoch 对齐）。
        if constexpr (kPhase != MegaPhase::L2) {
            if (threadIdx.x == 0) a.rwk->lf.current_time[blockIdx.x][0] += 1;
        }
        __syncthreads();
    } else {
        xbar_route_ready_body(&a.unk->rf, a.unk_peers, &a.rwk->lf, a.R, a.rank,
                              a.xrelease_idx, a.rwk->lf.dbg, a.max_spins, 0, ctr);
    }
    if (kTick) a.rwk->lf.pre_ticks[2] = clock64();
    // 3) TASK_SETUP（block0 leader：checksum -> prefix -> 窗口表 -> task_ready）
    // [L1L2-SPLIT] L2 阶段：改用静态任务划分（见 dispatch_pull_body），
    //   不重臂 task_counter/task_ready、不推进 current_time[1]（每 round 仅 L1
    //   前序 +1）；prefix 由本 round L1 的 task_setup 持久化，L2 只读。
    if constexpr (kPhase == MegaPhase::L2) {
        if (kTick) a.rwk->lf.pre_ticks[3] = clock64();
    } else if (a.skip_prologue) {
        // [R38] 复用前序：prefix/路由/计数持久，仅轻量重臂 flag（省 checksum
        //   轮询与 prefix 重建）。
        task_setup_light_body(&a.rwk->lf, a.task_ready_idx, a.task_counter_idx,
                              a.l2_done_idx, a.max_spins, a.rwk->lf.dbg);
    } else {
        task_setup_body(a.self, &a.unk->rd, a.R, a.rank, a.slot_capacity,
                        a.prefix_off, &a.rwk->lf, a.task_ready_idx,
                        a.task_counter_idx, a.l2_done_idx,
                        a.rwk->lf.out_total, a.rwk->lf.dbg, a.max_spins,
                        kRecvStride, /*reset_ring=*/true, /*tid_base=*/0,
                        /*sb_cnt=*/nullptr, /*smem_off=*/0, a.prefix_ptr,
                        a.rwk->lf.ck_dbg, ctr);
    }
    if (kTick) a.rwk->lf.pre_ticks[3] = clock64();
    // 4) 任务循环（DISPATCH -> L1 -> L2 -> tail(COMBINE)）
    if (kWithCompute) {
        // [M_CYCLE] 任务循环 / COMBINE / REDUCE 用的 copy c：task_setup 结束时
        //   current_time[b][1] 已自增到 L+1，故 (ct-1)%M_CYCLE = L%M_CYCLE，与
        //   xbar/task_setup 所用 c 一致。COMBINE 完成标记与 REDUCE 等待须同片。
        const int cyc = (int)((a.rwk->lf.current_time[blockIdx.x][1] - 1) % M_CYCLE);
        // [集33] tail 实例统一：任务循环恒用 CombineTail，combine 关闭时
        //   （combine_rows_off<0）融合能力置 false 且 tail 为空操作。这样
        //   mega_kernel<..., NoTaskTail> 不再实例化第二份重型任务体
        //   （原 NoTaskTail 运行期从不执行，却占整核寄存器/代码布局 collateral；
        //   71 机 TPR=16 −7µs，TPR=256 中性）。
        {
            CombineTail tail;
            if (a.combine_rows_off >= 0) {
                tail.ctx.meta =
                    reinterpret_cast<const int*>(a.self + a.meta_off);
                tail.ctx.weights =
                    reinterpret_cast<const float*>(a.self + a.w_off);
                tail.ctx.peer_bases = a.peer_bases;
                tail.ctx.unk_peers = a.unk_peers;
                tail.ctx.rows_off = a.combine_rows_off;
                tail.ctx.staging_cap = a.win_cap;
                tail.ctx.block_idx = (int)blockIdx.x;
                tail.ctx.cycle = cyc;   // [M_CYCLE] combine_sums 片
                tail.ctx.epoch = (uint64_t)a.rwk->lf.current_time[0][1];
                tail.out_base = reinterpret_cast<const uint8_t*>(a.l2_out);
                tail.nwarps = 8;
                // [R29] 融合推送能力 + 上下文（on 由任务体在 r24 L2 路径激活）
                tail.fuse.capable = true;
                tail.fuse.out_base = tail.out_base;
                tail.fuse.meta = tail.ctx.meta;
                tail.fuse.weights = tail.ctx.weights;
                tail.fuse.peer_bases = tail.ctx.peer_bases;
                tail.fuse.unk_peers = a.unk_peers;
                tail.fuse.cycle = cyc;
                tail.fuse.rows_off = tail.ctx.rows_off;
                tail.fuse.staging_cap = tail.ctx.staging_cap;
                tail.fuse.block_idx = tail.ctx.block_idx;
            } else {
                // [集33] combine 关闭：同一 tail 实例走空操作（rows_off<0）
                tail.ctx.rows_off = a.combine_rows_off;
                tail.ctx.staging_cap = a.win_cap;
                tail.ctx.block_idx = (int)blockIdx.x;
                tail.ctx.cycle = cyc;   // [M_CYCLE]（combine 关闭，仅占位）
                tail.ctx.epoch = (uint64_t)a.rwk->lf.current_time[0][1];
                tail.out_base = nullptr;
                tail.nwarps = 8;
                tail.fuse.capable = false;
            }
            // [集71] 计数缓冲下传（CombineTail 统计 COMBINE 执行/跳过次数）
#if MX_DBG_COUNT
            tail.ctr = ctr;
#endif
            mega_dispatch_loop<CombineTail, kSmallOnly, kCap, kPhase>(
                a, tail, cyc, &s_ab.a[0][0], nullptr,
                ctr);
        }
    }
    if (kTick) a.rwk->lf.pre_ticks[15] = clock64();
}
