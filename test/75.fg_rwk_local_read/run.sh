#!/usr/bin/env bash
# 单卡本地读 FG vs RWK；默认卡 0，FG_BENCH_DEV 覆盖
set -euo pipefail
cd "$(dirname "$0")"

export MACA_PATH=${MACA_PATH:-/opt/maca}
export FG_BENCH_DEV=${FG_BENCH_DEV:-0}
export MACA_VISIBLE_DEVICES=${MACA_VISIBLE_DEVICES:-0}

echo "MACA_VISIBLE_DEVICES=$MACA_VISIBLE_DEVICES  FG_BENCH_DEV=$FG_BENCH_DEV"
timeout 30 ./fg_rwk_read
