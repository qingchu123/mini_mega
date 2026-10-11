#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export MACA_PATH=${MACA_PATH:-/opt/maca}
export MACA_VISIBLE_DEVICES=${MACA_VISIBLE_DEVICES:-0}
ROOT="$(git rev-parse --show-toplevel)"
mkdir -p "$ROOT/.agents/work/lock"
exec 9>"$ROOT/.agents/work/lock/lock_gpu"
flock -w 120 9 || { echo "等待 GPU 锁超时"; exit 1; }
timeout -k 2 90 ./r19_ksplit_bench
