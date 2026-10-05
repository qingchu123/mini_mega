#!/usr/bin/env bash
# test/70.r32_tpr_perf — R=32 TPR=16/64/256 性能测试
# 口径：对齐 mega src/bench/bench_mega_fp8.cu —— 单进程、均衡路由、
#       graph 链 nodes=20、replays=2（取 min）、默认不做正确性校验。
#       ./bench 默认即 R=32 TPR={1,4,16,32,33,64,128,256}；此处限定 3 档。
# 硬约束：单次 timeout 10s（AGENTS 规则）。
set -uo pipefail
cd "$(dirname "$0")/../.."   # -> 工程根

BENCH=${BENCH:-./bench}
DEV=$(seq -s, 0 31)
T=${T:-10}
OUT=test/70.r32_tpr_perf
TPRS=${TPRS:-"16 64 256"}
REPS=${REPS:-2}
TPR_CSV=$(echo "$TPRS" | tr ' ' ',')

for REP in $(seq 1 "$REPS"); do
  log="$OUT/r${REP}.log"
  MACA_VISIBLE_DEVICES="$DEV" timeout -k 3 "$T" stdbuf -oL "$BENCH" \
    --ranks 32 --tprs "$TPR_CSV" --nodes 20 --replays 2 \
    > "$log" 2>&1
  rc=$?
  printf 'rep=%s rc=%s\n' "$REP" "$rc"
  grep -E '^R |^32 ' "$log" | sed 's/^/  /'
done
