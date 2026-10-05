#!/usr/bin/env bash
# 宿主机编译（cucc + mcr runtime，无需容器、不用 torch）
set -euo pipefail
cd "$(dirname "$0")/.."

# 最小工程自带 deep_gemm 头（include/）；缺失时回退到 MXMOEMega 仓库。
INCLUDE_ROOT=${MEGA_INCLUDE_ROOT:-include}
if [ ! -d "$INCLUDE_ROOT/deep_gemm" ]; then
  REPO=${MXMOEMEGA_REPO:-/home/lchen1/data/projs/MXMOEMega}
  INCLUDE_ROOT="$REPO/mxmegakernel/include"
fi
export MACA_PATH=${MACA_PATH:-/opt/maca}
export CUDA_HOME=${CUDA_HOME:-$MACA_PATH/tools/cu-bridge}
mkdir -p "$HOME/_nvcc"

"$MACA_PATH/tools/cu-bridge/bin/cucc" -O3 -std=c++17 \
  -I"$MACA_PATH/tools/cu-bridge/include" \
  -I/opt/maca/include \
  -I"$INCLUDE_ROOT" \
  -Isrc \
  -o bench src/bench.cu

echo "built: $PWD/bench"
