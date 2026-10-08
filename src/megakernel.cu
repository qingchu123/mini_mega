// ============================================================================
// megakernel.cu — fp8xfp8 MoE Mega Kernel host 接口实现
//
// 本文件是**唯一**发射 mega_kernel / mega_reduce_kernel / mega_empty_kernel /
// fill_w_lane_kernel 的地方；编译为 `megakernel.so` 供 bench.cu 调用。
// 公共头见 src/megakernel.h（bench 只 include 该头，不感知 kernel.cuh）。
// ============================================================================

#include "megakernel.h"

#include <cstdlib>

#include "common.cuh"
#include "kernel.cuh"

namespace fp8mega {

// ============================================================================
// [MEGA_DEBUG=4] 采集 kernel：在 mega_kernel **之前** 记录本 rank 的
//   ct0 = local_flags::current_time[0][0]（本 launch 的 identity/序号）、
//   ct1 = current_time[0][1]、topk_idx、graph_capture_bool、rank。
// 用途：跑完模型后，从 stdout 日志（设备 printf）重建真实场景的
//   per-rank/per-launch 路由与序号，供离线复现数据集。
// 仅当 host 环境变量 MEGA_DEBUG=4 时由 launch_compute 发射；其余情况不产生任何
//   代码/开销（走同一 stream，graph capture 时随图录制、replay 时执行）。
// ============================================================================
__global__ void mxk_dbg_capture_kernel(
        const int64_t* __restrict__ topk_idx, int tokens, int topk,
        const local_flags* __restrict__ lf, int rank, int capture,
        unsigned long long* __restrict__ sink) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    const long long ct0 =
        (lf != nullptr) ? (long long)lf->current_time[0][0] : -1LL;
    const long long ct1 =
        (lf != nullptr) ? (long long)lf->current_time[0][1] : -1LL;
    // 仅打印元信息（rank/序号/tokens/topk）；topk_idx 由 host 侧另取，
    //   避免日志体积过大。
    printf("[MXK_DBG] rank=%d cap=%d ct0=%lld ct1=%lld tokens=%d topk=%d\n",
           rank, capture, ct0, ct1, tokens, topk);
    if (topk_idx != nullptr && sink != nullptr)
        sink[0] = (unsigned long long)topk_idx[0];   // 防 DCE
}

// 设备侧权重生成（16B/lane supertile 布局；替代 host 生成 + 大体积上传）。
__global__ void __launch_bounds__(256) fill_w_lane_kernel(
                                   uint8_t* __restrict__ w, int n_rows,
                                   int k_cols, uint32_t seed) {
    const int k_iters = k_cols / 32;
    const size_t total = (size_t)(n_rows / 32) * k_iters;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t)gridDim.x * blockDim.x) {
        const int np = (int)(idx / k_iters);
        const int kt = (int)(idx % k_iters);
        uint8_t* tile = w + idx * 1024;
        for (int L = 0; L < 64; ++L) {
            const int row0 = np * 32 + (L % 16);
            const int row1 = row0 + 16;
            const int k0 = kt * 32 + (L / 16) * 8;
            for (int b = 0; b < 8; ++b) {
                tile[L * 16 + b] = WPROBE_VAL(seed, row0, k0 + b);
                tile[L * 16 + 8 + b] = WPROBE_VAL(seed, row1, k0 + b);
            }
        }
    }
}

namespace {

// 公共 POD -> 设备入参（逐字段显式拷贝，避免依赖两份结构体布局一致）。
MegaArgs to_mega_args(const MegaLaunchArgs& a) {
    // [窗口口径守卫] 当前 win_cap < slot_capacity 时 REDUCE/combine 协议仍不正确
    // （实测 804900 win=48 slot=512 时 eager REDUCE/graph 失败）。先响亮失败，
    // 避免静默产出坏数据；待窗口化 combine_rows 协议修好后再放开。
    if (a.slot_capacity != a.win_cap) {
        std::fprintf(stderr,
            "[megakernel] slot_capacity=%d 与 win_cap=%d 不一致：当前组合会产生错误结果。\n"
            "  请令 win_cap == slot_capacity（适配窗 win<slot 已知数值错误）。\n",
            a.slot_capacity, a.win_cap);
        std::exit(2);
    }
    MegaArgs m{};
    m.topk_idx = a.topk_idx;
    m.self = a.self;
    m.peer_bases = a.peer_bases;
    m.R = a.R;
    m.rank = a.rank;
    m.tokens = a.tokens;
    m.slot_capacity = a.slot_capacity;
    m.win_cap = a.win_cap;
    m.pool_stride = a.pool_stride;
    m.tbl_cap = a.tbl_cap;
    m.k_in = a.k_in;
    m.topk = a.topk;
    m.ring = a.ring;
    m.src_off = a.src_off;
    m.rsc_off = a.rsc_off;
    m.recv_off = a.recv_off;
    m.prefix_off = a.prefix_off;
    m.barrier_off = a.barrier_off;
    m.lf = a.lf;
    m.rf = a.rf;
    m.rf_peers = a.rf_peers;
    m.local = a.local;
    m.local_peers = a.local_peers;
    m.remote = a.remote;
    m.remote_peers = a.remote_peers;
    m.xrelease_idx = a.xrelease_idx;
    m.task_ready_idx = a.task_ready_idx;
    m.task_counter_idx = a.task_counter_idx;
    m.l2_done_idx = a.l2_done_idx;
    m.slice_flags_idx = a.slice_flags_idx;
    m.l1_prog_idx = a.l1_prog_idx;
    m.x_off = a.x_off;
    m.weights_off = a.weights_off;
    m.pool_off = a.pool_off;
    m.pool_ptr = a.pool_ptr;
    m.w_off = a.w_off;
    m.meta_off = a.meta_off;
    m.max_spins = a.max_spins;
    m.publish_epoch = a.publish_epoch;
    m.recv_stride = a.recv_stride;
    m.merge_grid = a.merge_grid;
    m.skip_prologue = a.skip_prologue;
    m.prefix_ptr = a.prefix_ptr;
    m.w1 = a.w1;
    m.l1_n_total = a.l1_n_total;
    m.l1_block_n = a.l1_block_n;
    m.l1_k_out = a.l1_k_out;
    m.ffn_hidden = a.ffn_hidden;
    m.frag_a = a.frag_a;
    m.frag2 = a.frag2;
    m.w2 = a.w2;
    m.l2_out = a.l2_out;
    m.l2_k_out = a.l2_k_out;
    m.combine_rows_off = a.combine_rows_off;
    m.combine_cap = a.combine_cap;
    m.sums_stride = a.sums_stride;
    return m;
}

// [MEGA_DEBUG=4] 采集开关 + 当前 stream 是否处于 graph capture。
bool dbg_capture_enabled() {
    const char* s = std::getenv("MEGA_DEBUG");
    return s != nullptr && std::atoi(s) == 4;
}

int stream_is_capturing(mcStream_t stream) {
    mcStreamCaptureStatus st = mcStreamCaptureStatusNone;
    if (mcStreamIsCapturing(stream, &st) != mcSuccess) return -1;
    return (st == mcStreamCaptureStatusActive) ? 1 : 0;
}

// 在 mega_kernel 之前发射采集 kernel（同一 stream → 顺序/图录制天然正确）。
void dbg_capture_launch(const MegaLaunchArgs& a, mcStream_t stream) {
    if (!dbg_capture_enabled()) return;
    mxk_dbg_capture_kernel<<<1, 1, 0, stream>>>(
        a.topk_idx, a.tokens, a.topk, a.lf, a.rank,
        stream_is_capturing(stream), nullptr);
}

}  // namespace

void launch_prelude(const MegaLaunchArgs& a, mcStream_t stream) {
    mega_kernel<false><<<a.grid, a.threads, (size_t)a.smem, stream>>>(
        to_mega_args(a), NoTaskTail{});
}

void launch_compute(const MegaLaunchArgs& a, mcStream_t stream) {
    // [MEGA_DEBUG=4] 采集 kernel（mega 之前；默认关，零开销）
    dbg_capture_launch(a, stream);
    // [OPT-TK-CAP] 仅当宿主提供了 fragment 专用缓冲（不均衡长尾/tk 场景）时，
    //   按 win_cap 选编译期 count 上界实例（kCap=48/64）：不可达的重档
    //   （MT8 异步 / MT4 大 GEMM 兜底）被 if constexpr 删除，缩任务体寄存器/栈。
    //   均衡 test/70（无 frag）与 TPR=256 大档走原实例不变。
    if (a.frag_a != nullptr) {
        if (a.win_cap <= 48)
            mega_kernel<true, NoTaskTail, false, true, 48>
                <<<a.grid, a.threads, (size_t)a.smem, stream>>>(to_mega_args(a),
                                                                NoTaskTail{});
        else if (a.win_cap <= 64)
            mega_kernel<true, NoTaskTail, false, true, 64>
                <<<a.grid, a.threads, (size_t)a.smem, stream>>>(to_mega_args(a),
                                                                NoTaskTail{});
        else
            mega_kernel<true, NoTaskTail, false, true>
                <<<a.grid, a.threads, (size_t)a.smem, stream>>>(to_mega_args(a),
                                                                NoTaskTail{});
    } else {
        mega_kernel<true><<<a.grid, a.threads, (size_t)a.smem, stream>>>(
            to_mega_args(a), NoTaskTail{});
    }
}

void launch_reduce(const MegaLaunchArgs& a, mcStream_t stream) {
    mega_reduce_kernel<<<2 * a.tokens, 256, 0, stream>>>(
        a.self, a.remote, a.y, a.R, a.tokens, a.combine_cap,
        a.combine_rows_off, a.topk_idx_off, a.weights_off, a.lf, a.rf,
        a.task_ready_idx, a.max_spins, a.sums_stride);
}

void launch_full(const MegaLaunchArgs& a, mcStream_t stream) {
    launch_compute(a, stream);
    launch_reduce(a, stream);
}

void launch_empty(int grid, int threads, int64_t smem, mcStream_t stream) {
    mega_empty_kernel<<<grid, threads, (size_t)smem, stream>>>();
}

void fill_weights_lane(uint8_t* w, int n_rows, int k_cols, uint32_t seed,
                       mcStream_t stream) {
    fill_w_lane_kernel<<<256, 256, 0, stream>>>(w, n_rows, k_cols, seed);
}

}  // namespace fp8mega
