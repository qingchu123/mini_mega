#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MACA_PATH=${MACA_PATH:-/opt/maca}
export MACA_VISIBLE_DEVICES=${MACA_VISIBLE_DEVICES:-0}
timeout 60 ./l1_mma_micro
