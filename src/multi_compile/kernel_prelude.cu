// [multi_compile] mega_kernel 显式实例化：false, false, 0, MegaPhase::Both
#include "kernel.cuh"

template __global__ void mega_kernel<false, false, 0, MegaPhase::Both>(MegaArgs a);
