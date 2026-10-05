#!/usr/bin/env bash
# test/70.r32_tpr_perf — R=32 下 TPR=16/64/256 性能测试
# 口径：perf.md 现行 R=32 基准（max_tokens=512, win_cap=128, balanced, graph 链 min-of-5）
# 硬约束：单档单次 timeout 10s（AGENTS 规则）
set -uo pipefail
cd "$(dirname "$0")/../.."   # -> 01.fp8xfp8_moe_kernel

BENCH=${BENCH:-./bench}
DEV=$(seq -s, 0 31)
T=${T:-10}
OUT=test/70.r32_tpr_perf
TPRS=${TPRS:-"16 64 256"}
REPS=${REPS:-2}

for TPR in $TPRS; do
  for REP in $(seq 1 "$REPS"); do
    log="$OUT/t${TPR}_r${REP}.log"
    MACA_VISIBLE_DEVICES="$DEV" timeout -k 3 "$T" stdbuf -oL "$BENCH" \
      --ranks 32 --nodes 20 --graph 0 --perf-only 1 --compute 1 \
      --perf-pattern 0 --tokens "$TPR" --max-tokens 512 --win-cap 128 \
      > "$log" 2>&1
    rc=$?
    line=$(grep -m1 "\[全链计时\]" "$log" || echo "(no timing)")
    overall=$(grep -m1 "\[总体\]" "$log" || echo "(no overall)")
    printf 'TPR=%-4s rep=%s rc=%s | %s | %s\n' "$TPR" "$REP" "$rc" "$line" "$overall"
  done
done
