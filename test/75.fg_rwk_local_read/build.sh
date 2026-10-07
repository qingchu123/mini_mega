#!/usr/bin/env bash
# 编译 fg_rwk_read（cucc + mcr runtime）
set -euo pipefail
cd "$(dirname "$0")"

export MACA_PATH=${MACA_PATH:-/opt/maca}
export CUDA_HOME=${CUDA_HOME:-$MACA_PATH/tools/cu-bridge}
# 本机 C600 ISA 报 xcore1610，但 cucc 只认主版本号（minor 置 00），见
# $MACA_PATH/tools/cu-bridge/tools/multi_target.sh 提示。
export CUCC_TARGETS=${CUCC_TARGETS:-xcore1600}
mkdir -p "$HOME/_nvcc"
CUCC="$MACA_PATH/tools/cu-bridge/bin/cucc"
INC=(-I"$MACA_PATH/tools/cu-bridge/include" -I/opt/maca/include)

"$CUCC" -O3 -std=c++17 "${INC[@]}" -o fg_rwk_read fg_rwk_read.cu
echo "built: $PWD/fg_rwk_read"
