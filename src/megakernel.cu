// ============================================================================
// megakernel.cu — fp8xfp8 MoE Mega Kernel host 接口实现
//
// 本文件是**唯一**发射 mega_kernel / mega_reduce_kernel / mega_empty_kernel /
// fill_w_lane_kernel 的地方；编译为 `megakernel.so` 供 bench.cu 调用。
// 公共头见 src/megakernel.h（bench 只 include 该头，不感知 kernel.cuh）。
// ============================================================================

#include "megakernel.h"

#include <cstdlib>
#include <cstdio>

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
    // [窗口口径守卫] 拒绝未验证的 win_cap 组合，避免静默产出坏数据。
    // 放行条件（实验判定，见 .agents/work/guard_sub_w*.log）：
    //   win == slot，或 win 为 16 的倍数且 48 <= win <= slot。
    // 已知失败：win<48（16/32 出现 REDUCE/graph 误差）、win 非 16 倍数（40/56
    // 出现 L1L2 失败）。这些组合仍拒绝。
    const bool win_ok = (a.win_cap == a.slot_capacity) ||
                        (a.win_cap % 16 == 0 && a.win_cap >= 32 &&
                         a.win_cap <= a.slot_capacity);
    if (!win_ok) {
        std::fprintf(stderr,
            "[megakernel] win_cap=%d 不被允许（slot_capacity=%d）：需 win==slot，"
            "或 win 为 16 的倍数且 32<=win<=slot。\n",
            a.win_cap, a.slot_capacity);
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
    m.prefix_off = a.prefix_off;
    // [步骤2b-A] kernel 直接使用结构体指针
    m.rwk = a.rwk;
    m.unk = a.unk;
    m.unk_peers = a.unk_peers;
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
    m.skip_prologue = a.skip_prologue;
    m.prefix_ptr = a.prefix_ptr;
    m.l1_k_out = a.l1_k_out;
    m.l2_k_out = a.l2_k_out;
    m.w1 = a.w1;
    m.w2 = a.w2;
    m.ffn_hidden = a.ffn_hidden;
    m.l2_out = a.l2_out;
    m.combine_rows_off = a.combine_rows_off;
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
        a.topk_idx, a.tokens, kTopk, a.rwk ? &a.rwk->lf : nullptr, a.rank,
        stream_is_capturing(stream), nullptr);
}

// [L1L2-SPLIT] 运行期开关：MX_L1L2_SPLIT=1 时 launch_compute 拆成
//   L1/L2 两个 kernel（同 stream 顺序）。首次调用读 env 定型（graph capture
//   期间只读 env、无设备操作，合法）。
int mx_l1l2_split() {
    static const int v = [] {
        const char* e = std::getenv("MX_L1L2_SPLIT");
        return (e != nullptr && *e != '\0' && *e != '0') ? 1 : 0;
    }();
    return v;
}

// 按 win_cap 分档发射指定 phase 的计算 kernel。
//   skip_override < 0 → 沿用 a.skip_prologue（L1）；>=0 → 强制（L2 用 2：
//   跳过 route/publish/xbar、轻量 task_setup，前序数据由本 round 的 L1 写好）。
template <MegaPhase ph>
void launch_phase_impl(const MegaLaunchArgs& a, mcStream_t stream,
                       int skip_override) {
    MegaLaunchArgs b = a;
    if (skip_override >= 0) b.skip_prologue = skip_override;
    if (a.win_cap <= 48)
        mega_kernel<true, false, 48, ph>
            <<<a.grid, a.threads, (size_t)a.smem, stream>>>(to_mega_args(b));
    else if (a.win_cap <= 64)
        mega_kernel<true, false, 64, ph>
            <<<a.grid, a.threads, (size_t)a.smem, stream>>>(to_mega_args(b));
    else
        mega_kernel<true, false, 0, ph>
            <<<a.grid, a.threads, (size_t)a.smem, stream>>>(to_mega_args(b));
}

}  // namespace

void launch_prelude(const MegaLaunchArgs& a, mcStream_t stream) {
    mega_kernel<false><<<a.grid, a.threads, (size_t)a.smem, stream>>>(
        to_mega_args(a));
}

void launch_l1(const MegaLaunchArgs& a, mcStream_t stream) {
    dbg_capture_launch(a, stream);
    launch_phase_impl<MegaPhase::L1>(a, stream, /*skip_override=*/-1);
}

void launch_l2(const MegaLaunchArgs& a, mcStream_t stream) {
    // L2 前序恒走 skip=2：route/publish/xbar 由本 round 的 L1 完成，
    //   轻量 task_setup 重臂 task_counter/task_ready 并推进 current_time[1]
    //   （combine/REDUCE 的 cyc/epoch 均由此派生，见 combine.cuh:496-508）。
    launch_phase_impl<MegaPhase::L2>(a, stream, /*skip_override=*/2);
}

void launch_compute(const MegaLaunchArgs& a, mcStream_t stream) {
    // [MEGA_DEBUG=4] 采集 kernel（mega 之前；默认关，零开销）
    dbg_capture_launch(a, stream);
    // [L1L2-SPLIT] 拆分模式：L1 → L2 两个 launch（同 stream；graph capture
    //   下两节点依次录入，replay 顺序不变）。
    if (mx_l1l2_split()) {
        launch_phase_impl<MegaPhase::L1>(a, stream, /*skip_override=*/-1);
        launch_phase_impl<MegaPhase::L2>(a, stream, /*skip_override=*/2);
        return;
    }
    // [FRAG-INT] fragment 缓冲为 kernel 内部（rwk->ld），host 不再传入。fragment
    //   分支恒编译（[FRAG-ALWAYS]），运行期按 frag_all/几何自动启用。
    // [OPT-TK-CAP] 按 win_cap 选编译期 count 上界实例（kCap=48/64）：不可达的重档
    //   （MT8 异步 / MT4 大 GEMM 兜底）被 if constexpr 删除，缩任务体寄存器/栈。
    // if (a.win_cap <= 48)
    //     mega_kernel<true, false, 48>
    //         <<<a.grid, a.threads, (size_t)a.smem, stream>>>(to_mega_args(a));
    // else if (a.win_cap <= 64)
    //     mega_kernel<true, false, 64>
    //         <<<a.grid, a.threads, (size_t)a.smem, stream>>>(to_mega_args(a));
    // else
        mega_kernel<true><<<a.grid, a.threads, (size_t)a.smem, stream>>>(
            to_mega_args(a));
}

void launch_reduce(const MegaLaunchArgs& a, mcStream_t stream) {
    mega_reduce_kernel<<<2 * a.tokens, 256, 0, stream>>>(
        a.self, a.unk ? &a.unk->rd : nullptr, a.y, a.R, a.tokens,
        a.combine_rows_off, a.topk_idx_off, a.weights_off,
        a.rwk ? &a.rwk->lf : nullptr, a.unk ? &a.unk->rf : nullptr,
        a.task_ready_idx, a.max_spins);
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

void dump_kernel_resources() {
    auto q = [](const char* n, const void* f) {
        mcFuncAttributes at{};
        if (mcFuncGetAttributes(&at, f) == mcSuccess)
            std::printf("[RES] %-7s regs=%d local=%zu smem=%zu maxThr=%d\n", n,
                        at.numRegs, at.localSizeBytes, at.sharedSizeBytes,
                        at.maxThreadsPerBlock);
        else
            std::printf("[RES] %-7s <fail>\n", n);
    };
    q("kCap48", (const void*)mega_kernel<true, false, 48>);
    q("kCap64", (const void*)mega_kernel<true, false, 64>);
    q("full",   (const void*)mega_kernel<true>);
    q("empty",  (const void*)mega_kernel<false>);
    q("L1full", (const void*)mega_kernel<true, false, 0,
                                        MegaPhase::L1>);
    q("L2full", (const void*)mega_kernel<true, false, 0,
                                        MegaPhase::L2>);
}
}  // namespace fp8mega

