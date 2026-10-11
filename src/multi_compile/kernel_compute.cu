// [multi_compile] mega_kernel 显式实例化：true, false, MegaPhase::Both
#include "kernel.cuh"

template __global__ void mega_kernel<true, false, MegaPhase::Both>(MegaArgs a);
