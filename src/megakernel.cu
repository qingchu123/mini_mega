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

// [L1L2-SPLIT] 编开关：默认 0=不编译 L1/L2 拆分路径（launch_l1/l2/phase）。
#ifndef CM_L1L2_SPLIT
#define CM_L1L2_SPLIT 0
#endif

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

// ============================================================================
// [CT-PRINT] 轮数诊断打印 kernel：rank==0 且轮数 %1000==0 时打印当前轮数/token
//   等元信息，便于把 ATU/异常定位到具体轮次。常开（开销仅一次 rank0 比较；
//   graph capture 时随图录制，replay 下每 1000 轮打印一次）。
// ============================================================================
__global__ void mxk_ct_print_kernel(const int64_t* __restrict__ ct, int rank,
                                    int tokens, int slot_capacity, int grid) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (rank != 0) return;
    const int64_t c = (ct != nullptr) ? *ct : -1LL;
    if (c < 0 || (c % 1000) != 0) return;
    printf("[CT-PRINT] round=%lld tokens=%d slot_cap=%d grid=%d rank=%d\n",
           (long long)c, tokens, slot_capacity, grid, rank);
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
#if MX_FP4_W
        // [MX_FP4_W] 每 super-tile 512B：lane L 的 8B = 低 4B(行 row0 的 8×e2m1) +
        //   高 4B(行 row0+16)，与设备 r19_load_b16 的 fp4 分支逐字节对应。
        uint32_t* tile32 = reinterpret_cast<uint32_t*>(w + idx * kBTileBytes);
        for (int L = 0; L < 64; ++L) {
            const int row0 = np * 32 + (L % 16);
            const int row1 = row0 + 16;
            const int k0 = kt * 32 + (L / 16) * 8;
            uint32_t lo = 0, hi = 0;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                lo |= (uint32_t)WPROBE_FP4_CODE(seed, row0, k0 + j) << (4 * j);
                hi |= (uint32_t)WPROBE_FP4_CODE(seed, row1, k0 + j) << (4 * j);
            }
            tile32[L * 2 + 0] = lo;
            tile32[L * 2 + 1] = hi;
        }
#else
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
#endif
    }
}

namespace {

// 公共 POD -> 设备入参（逐字段显式拷贝，避免依赖两份结构体布局一致）。
MegaArgs to_mega_args(const MegaLaunchArgs& a) {
    // [窗口口径] 不再设守卫：win_cap 由调用方/C++ 选窗逻辑决定
    //   （torch 层不干预，见 mxk_select_win / src/win_select.cuh）。
    MegaArgs m{};
    m.topk_idx = a.topk_idx;
    m.w1 = a.w1;
    m.w2 = a.w2;
    m.group_weights_ready = a.group_weights_ready;
    m.self = a.self;
    m.peer_bases = a.peer_bases;
    m.R = a.R;
    m.rank = a.rank;
    m.tokens = a.tokens;
    m.slot_capacity = a.slot_capacity;
    m.win_cap = a.win_cap;
    // [步骤2b-A] kernel 直接使用结构体指针
    m.rwk = a.rwk;
    m.unk = a.unk;
    m.unk_peers = a.unk_peers;
    m.x_off = a.x_off;
    m.weights_off = a.weights_off;
    m.max_spins = a.max_spins;
    m.skip_prologue = a.skip_prologue;
    // [RWK-LD] w1/w2/ffn_hidden/l2_out 由 kernel 从 a.rwk->ld 派生，不再拷贝。
    m.in_bf16 = a.in_bf16;
    m.skip_combine = a.skip_combine;
    // [诊断] MX_SKIP_COMBINE=1：禁用任务尾 COMBINE（skip_combine=1 → 空操作）
    if (const char* s = std::getenv("MX_SKIP_COMBINE");
        s != nullptr && std::atoi(s) != 0)
        m.skip_combine = 1;
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

// [CT-PRINT] 轮数诊断打印：rank0 && 轮数 %1000==0 时打印元信息。
void ct_print_launch(const MegaLaunchArgs& a, mcStream_t stream) {
    // [CT-PRINT] 默认不发射（避免加 graph 节点/扰动）；MX_CT_PRINT=1 才启用。
    static const bool on = [] {
        const char* e = std::getenv("MX_CT_PRINT");
        return e != nullptr && *e != '\0' && *e != '0';
    }();
    if (!on) return;
    mxk_ct_print_kernel<<<1, 1, 0, stream>>>(
        a.rwk ? &a.rwk->lf.current_time[0][1] : nullptr, a.rank, a.tokens,
        a.slot_capacity, a.grid);
}

// [L1L2-SPLIT] 运行期开关：MX_L1L2_SPLIT=1 时 launch_compute 拆成
//   L1/L2 两个 kernel（同 stream 顺序）。首次调用读 env 定型（graph capture
//   期间只读 env、无设备操作，合法）。
#if CM_L1L2_SPLIT
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
    mega_kernel<true, ph>
        <<<a.grid, a.threads, (size_t)a.smem, stream>>>(to_mega_args(b));
}
#endif  // CM_L1L2_SPLIT

}  // namespace

void launch_prelude(const MegaLaunchArgs& a, mcStream_t stream) {
    mega_kernel<false><<<a.grid, a.threads, (size_t)a.smem, stream>>>(
        to_mega_args(a));
}

#if CM_L1L2_SPLIT
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
#endif  // CM_L1L2_SPLIT

void launch_compute(const MegaLaunchArgs& a, mcStream_t stream) {
    // [MEGA_DEBUG=4] 采集 kernel（mega 之前；默认关，零开销）
    dbg_capture_launch(a, stream);
    // [CT-PRINT] 轮数诊断打印（rank0 && 轮数%1000==0；常开，极低开销）
    ct_print_launch(a, stream);
    // [L1L2-SPLIT] 拆分模式：L1 → L2 两个 launch（同 stream；graph capture
    //   下两节点依次录入，replay 顺序不变）。
#if CM_L1L2_SPLIT
    if (mx_l1l2_split()) {
        launch_phase_impl<MegaPhase::L1>(a, stream, /*skip_override=*/-1);
        launch_phase_impl<MegaPhase::L2>(a, stream, /*skip_override=*/2);
        return;
    }
#endif
    // [FRAG-INT] fragment 缓冲为 kernel 内部（rwk->ld），host 不再传入。fragment
    //   分支恒编译（[FRAG-ALWAYS]），运行期按 frag_all/几何自动启用。
    // [OPT-KCAP-REMOVE] 已去掉编译期 count 上界实例（kCap=48/64）：task body 内
    //   按运行期 win_cap 选档，重型 GEMM 档用 R19_GEMM_ATTR=__noinline__ 隔离寄存器。
    mega_kernel<true><<<a.grid, a.threads, (size_t)a.smem, stream>>>(
        to_mega_args(a));
}

void launch_reduce(const MegaLaunchArgs& a, mcStream_t stream) {
    mega_reduce_kernel<<<2 * a.tokens, 256, 0, stream>>>(
        a.self, a.unk ? &a.unk->rd : nullptr, a.y, a.R, a.tokens,
        a.topk_idx_off, a.weights_off,
        a.rwk ? &a.rwk->lf : nullptr, a.unk ? &a.unk->rf : nullptr,
        kBarTaskReadyIdx, a.max_spins);
}

void launch_full(const MegaLaunchArgs& a, mcStream_t stream) {
    launch_compute(a, stream);
    // [诊断] MX_SKIP_REDUCE=1：不发射 REDUCE kernel（仅测 compute 段）
    if (const char* s = std::getenv("MX_SKIP_REDUCE");
        s != nullptr && std::atoi(s) != 0)
        return;
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
    q("full",   (const void*)mega_kernel<true>);
    q("empty",  (const void*)mega_kernel<false>);
}
}  // namespace fp8mega

