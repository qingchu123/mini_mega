#include "kernel.cuh"

__global__ void __launch_bounds__(256) mega_reduce_kernel(
                                   const uint8_t* self, const ranks_data* remote,
                                   float* y, int R,
                                   int tokens,
                                   int64_t rows_off,
                                   int64_t topk_idx_off, int64_t weights_off,
                                   local_flags* lf, ranks_flags* rf,
                                   int64_t task_ready_idx,
                                   int max_spins
#if MX_DBG_COUNT
                                   ,
                                   unsigned long long* exec_ctr = nullptr,
                                   int dbg_count = 0
#endif
) {
    reduce_body(self, remote, y, R, tokens, rows_off,
                topk_idx_off, weights_off, lf, rf, task_ready_idx,
                max_spins
#if MX_DBG_COUNT
                , mx_ctr_open(dbg_count, exec_ctr)
#endif
    );
}
