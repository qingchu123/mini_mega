#!/usr/bin/env bash
# 编译 p2p_gread（cucc + mcr runtime，无需容器）
set -euo pipefail
cd "$(dirname "$0")"

export MACA_PATH=${MACA_PATH:-/opt/maca}
export CUDA_HOME=${CUDA_HOME:-$MACA_PATH/tools/cu-bridge}
CUCC="$MACA_PATH/tools/cu-bridge/bin/cucc"
INC=(-I"$MACA_PATH/tools/cu-bridge/include" -I/opt/maca/include)

"$CUCC" -O3 -std=c++17 "${INC[@]}" -o p2p_gread p2p_gread.cu -lpthread
echo "built: $PWD/p2p_gread"
