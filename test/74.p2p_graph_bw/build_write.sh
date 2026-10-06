#!/usr/bin/env bash
# p2p_gwrite 宿主机编译（cucc + mcr runtime）
set -euo pipefail
cd "$(dirname "$0")"

export MACA_PATH=${MACA_PATH:-/opt/maca}
export CUDA_HOME=${CUDA_HOME:-$MACA_PATH/tools/cu-bridge}
mkdir -p "$HOME/_nvcc"

CUCC="$MACA_PATH/tools/cu-bridge/bin/cucc"
INC=(-I"$MACA_PATH/tools/cu-bridge/include" -I"$MACA_PATH/include")

"$CUCC" -O3 -std=c++17 "${INC[@]}" -o p2p_gwrite p2p_gwrite.cu
echo "built: $PWD/p2p_gwrite"
