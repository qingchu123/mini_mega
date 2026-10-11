#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MACA_PATH=${MACA_PATH:-/opt/maca}
export CUDA_HOME=${CUDA_HOME:-$MACA_PATH/tools/cu-bridge}
export CUCC_TARGETS=${CUCC_TARGETS:-xcore1600}
mkdir -p "$HOME/_nvcc"
CUCC="$MACA_PATH/tools/cu-bridge/bin/cucc"
INC=(-I"$MACA_PATH/tools/cu-bridge/include" -I/opt/maca/include -I../../src)
EXTRA=${EXTRA:-}
"$CUCC" -O3 -std=c++17 "${INC[@]}" $EXTRA -o r19_l1_bench r19_l1_bench.cu
echo "built: $PWD/r19_l1_bench"
