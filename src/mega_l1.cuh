#pragma once
// ============================================================================
// mega_l1.cuh — L1 入口（从 0 重建）
//   一律调 r91_small_l1_big_opt（寄存器-A fragment）；MT 按 count 选。
//   依赖：R19Frag/R19_LDB/r24 等类型与 r91_small_l1_big_opt（由 compute_big.cuh
//   在本文件 include 之前定义）。本文件只被 compute_big.cuh include。
// ============================================================================
// ============================================================================
// [MEGA_L1] 新 L1 入口（从 0 重建）：一律调 r91_small_l1_big_opt（寄存器-A fragment）。
//   MT 按 count 选；kMask=true/kMTAct=kMT 保证任意 count 正确（尾块按 row<cc 掩码）。
// ============================================================================
__device__ __forceinline__ void mega_l1(
        uint8_t* s_mem, const uint8_t* fbase, int mt_total, const uint8_t* w1,
        int expert, int count, int k_hidden, int n_total, int block_n,
        int lane, int warp, int nwarps, float* checksum,
        int n_group_begin, int n_group_count, uint8_t* ffn_out, uint8_t* ffn2,
        int out_col_base, int k_out) {
    if (count <= 0) return;
#define MEGA_L1_CALL(MT)                                                       \
    r91_small_l1_big_opt<MT, 2, 2, true, MT>(                                  \
        s_mem, fbase, mt_total, w1, expert, 0, count, k_hidden, n_total,       \
        block_n, lane, warp, nwarps, checksum, n_group_begin, n_group_count,   \
        ffn_out, ffn2, out_col_base, k_out)
    if (count <= 16) MEGA_L1_CALL(1);
    else if (count <= 32) MEGA_L1_CALL(2);
    else if (count <= 48) MEGA_L1_CALL(3);
    else if (count <= 64) MEGA_L1_CALL(4);
    else MEGA_L1_CALL(8);
#undef MEGA_L1_CALL
}
