#!/usr/bin/env bash
# 只用卡 16..31（避开写 agent 0..15）；逻辑 rank 0..15 对应物理 16..31
set -euo pipefail
cd "$(dirname "$0")"

export MACA_PATH=${MACA_PATH:-/opt/maca}
export MACA_VISIBLE_DEVICES=${MACA_VISIBLE_DEVICES:-16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}
export GREAD_READER=${GREAD_READER:-0}
export GREAD_TARGETS=${GREAD_TARGETS:-1,8,15}

echo "MACA_VISIBLE_DEVICES=$MACA_VISIBLE_DEVICES  reader=$GREAD_READER  targets=$GREAD_TARGETS"
timeout 60 ./p2p_gread
