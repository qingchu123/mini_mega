#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MACA_PATH=${MACA_PATH:-/opt/maca}
export CUDA_HOME=${CUDA_HOME:-$MACA_PATH/tools/cu-bridge}
export CUCC_TARGETS=${CUCC_TARGETS:-xcore1600}
mkdir -p "$HOME/_nvcc"
CUCC="$MACA_PATH/tools/cu-bridge/bin/cucc"
INC=(-I"$MACA_PATH/tools/cu-bridge/include" -I/opt/maca/include)
"$CUCC" -O3 -std=c++17 "${INC[@]}" -o l1_mma_micro l1_mma_micro.cu
echo "built: $PWD/l1_mma_micro"
