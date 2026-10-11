#pragma once

template <bool kWithCompute, typename TaskTail = NoTaskTail,
          bool kSmallOnly = false, int kCap = 0>
__global__ void __launch_bounds__(512) mega_l1(MegaArgs a, TaskTail tail) {
    union __align__(16) ABShared {
        uint8_t a[2][R24_MT * R24_KS * 512];
    };
    __shared__ ABShared s_ab;

    mx_preclear_ct2(&a.rwk->lf, &a.unk->rf, a.skip_prologue >= 2);
    __syncthreads();

    if (!a.skip_prologue) {
        if (blockIdx.x == 0) {
            const int rc = (int)(a.rwk->lf.current_time[0][0] % M_CYCLE);
            route_build_body(a.topk_idx, &a.rwk->ld.route_slot_counts[0],
                             a.unk_peers, rc, a.R, a.rank, a.tokens,
                             a.slot_capacity, 0, 0, nullptr);
            publish_counts_body(&a.rwk->ld.route_slot_counts[0], a.unk_peers,
                                rc, a.R, a.rank, kPublishEpoch, kRecvStride, 0,
                                nullptr);
        }
    }

    if (a.skip_prologue >= 2) {
        if (threadIdx.x == 0) a.rwk->lf.current_time[blockIdx.x][0] += 1;
        __syncthreads();
    } else {
        xbar_route_ready_body(&a.unk->rf, a.unk_peers, &a.rwk->lf, a.R, a.rank,
                              a.xrelease_idx, a.rwk->lf.dbg, a.max_spins, 0,
                              nullptr);
    }

    if (a.skip_prologue)
        task_setup_light_body(&a.rwk->lf, a.task_ready_idx, a.task_counter_idx,
                              a.l2_done_idx, a.max_spins, a.rwk->lf.dbg);
    else
        task_setup_body(a.self, &a.unk->rd, a.R, a.rank, a.slot_capacity,
                        a.prefix_off, &a.rwk->lf, a.task_ready_idx,
                        a.task_counter_idx, a.l2_done_idx, a.rwk->lf.out_total,
                        a.rwk->lf.dbg, a.max_spins, kRecvStride, true, 0,
                        nullptr, 0, a.prefix_ptr, a.rwk->lf.ck_dbg, nullptr);

    if (kWithCompute) {
        const int cyc =
            (int)((a.rwk->lf.current_time[blockIdx.x][1] - 1) % M_CYCLE);
        CombineTail ctail;
        if (a.combine_rows_off >= 0) {
            ctail.ctx.meta = reinterpret_cast<const int*>(a.self + a.meta_off);
            ctail.ctx.weights = reinterpret_cast<const float*>(a.self + a.w_off);
            ctail.ctx.peer_bases = a.peer_bases;
            ctail.ctx.unk_peers = a.unk_peers;
            ctail.ctx.rows_off = a.combine_rows_off;
            ctail.ctx.staging_cap = a.win_cap;
            ctail.ctx.block_idx = (int)blockIdx.x;
            ctail.ctx.cycle = cyc;
            ctail.ctx.epoch = (uint64_t)a.rwk->lf.current_time[0][1];
            ctail.out_base = reinterpret_cast<const uint8_t*>(a.l2_out);
            ctail.nwarps = 8;
            ctail.fuse.capable = true;
            ctail.fuse.out_base = ctail.out_base;
            ctail.fuse.meta = ctail.ctx.meta;
            ctail.fuse.weights = ctail.ctx.weights;
            ctail.fuse.peer_bases = ctail.ctx.peer_bases;
            ctail.fuse.unk_peers = a.unk_peers;
            ctail.fuse.cycle = cyc;
            ctail.fuse.rows_off = ctail.ctx.rows_off;
            ctail.fuse.staging_cap = ctail.ctx.staging_cap;
            ctail.fuse.block_idx = ctail.ctx.block_idx;
        } else {
            ctail.ctx.rows_off = a.combine_rows_off;
            ctail.ctx.staging_cap = a.win_cap;
            ctail.ctx.block_idx = (int)blockIdx.x;
            ctail.ctx.cycle = cyc;
            ctail.ctx.epoch = (uint64_t)a.rwk->lf.current_time[0][1];
            ctail.out_base = nullptr;
            ctail.nwarps = 8;
            ctail.fuse.capable = false;
        }
        mega_dispatch_loop<CombineTail, kSmallOnly, kCap, MegaPhase::Both>(
            a, ctail, cyc, &s_ab.a[0][0], nullptr, nullptr);
    }
}
