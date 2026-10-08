#!/usr/bin/env bash
# 宿主机编译（cucc + mcr runtime，无需容器、不用 torch）
#
# 产物：
#   megakernel.so   fp8 mega kernel 的 host C++ 接口层（src/megakernel.cu）
#   bench           benchmark：只调用接口，链接 megakernel.so
set -euo pipefail
cd "$(dirname "$0")/.."

# 最小工程自带 deep_gemm 头（include/）；缺失时回退到 MXMOEMega 仓库。
INCLUDE_ROOT=${MEGA_INCLUDE_ROOT:-include}
if [ ! -d "$INCLUDE_ROOT/deep_gemm" ]; then
  REPO=${MXMOEMEGA_REPO:-/home/lchen1/data/projs/MXMOEMega}
  INCLUDE_ROOT="$REPO/mxmegakernel/include"
fi
export MACA_PATH=${MACA_PATH:-/opt/maca}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=${CUDA_HOME:-$MACA_PATH/tools/cu-bridge}
mkdir -p "$HOME/_nvcc"

CUCC="$MACA_PATH/tools/cu-bridge/bin/cucc"
INC=(-I"$MACA_PATH/tools/cu-bridge/include" -I/opt/maca/include
     -I"$INCLUDE_ROOT" -Isrc)
# [OPT] 扫描用额外编译宏（默认空；用于 -D 参数实验）
KERNEL_CFLAGS=${KERNEL_CFLAGS:-}

# 1) 接口层 + kernel -> megakernel.so（-fPIC -shared）
"$CUCC" -O3 -std=c++17 -fPIC -shared "${INC[@]}" $KERNEL_CFLAGS \
  -o megakernel.so src/megakernel.cu

# 2) bench：只 include megakernel.h，链接 megakernel.so（rpath=$ORIGIN 便于原地运行）
#    文件名非 lib*.so，故直接以路径链接（不用 -l）。
"$CUCC" -O3 -std=c++17 "${INC[@]}" -I"$MACA_PATH/ompi/include" \
  -o bench src/bench.cu \
  ./megakernel.so -L"$MACA_PATH/ompi/lib" -lmpi \
  -Wl,-rpath,'$ORIGIN' -Wl,-rpath,"$MACA_PATH/ompi/lib"

# 3) mybench：每进程一张卡的 MPI 基准（test/1.mytest），同样链接 MPI + megakernel.so。
"$CUCC" -O3 -std=c++17 "${INC[@]}" -I"$MACA_PATH/ompi/include" \
  -o mybench src/mybench.cu \
  ./megakernel.so -L"$MACA_PATH/ompi/lib" -lmpi \
  -Wl,-rpath,'$ORIGIN' -Wl,-rpath,"$MACA_PATH/ompi/lib"

echo "built: $PWD/megakernel.so"
echo "built: $PWD/bench"
echo "built: $PWD/mybench"
