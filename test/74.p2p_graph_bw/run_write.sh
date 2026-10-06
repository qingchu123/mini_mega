#!/usr/bin/env bash
# 只用卡 0..15（避开读测试 16..31）
set -euo pipefail
cd "$(dirname "$0")"

export MACA_PATH=${MACA_PATH:-/opt/maca}
export MACA_VISIBLE_DEVICES=${MACA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
PAIRS=${PAIRS:-0:1,0:8,0:15}

echo "MACA_VISIBLE_DEVICES=$MACA_VISIBLE_DEVICES  pairs=$PAIRS"
timeout 60 ./p2p_gwrite --pairs "$PAIRS"
