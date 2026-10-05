#!/usr/bin/env bash
# test/71.r32_tk_imbalance — 基于真实 TK_INFO 的不均衡路由性能测试
#
# 数据：test/70.r32_tpr_perf/0928_tk_info.log（MXMOEMega DG_TK_INFO_EVERY 抽样，
#       [TK_INFO] rank=r launch=n : c0..c7 = 该 rank 各物理专家收到的 pair 数）。
# 口径：单进程 run_aligned；均衡参数与不均衡（多个 launch）各建一张 graph
#       （nodes=20、replays=2、取 min），测 full-chain us/kernel。
# 约束：单次 timeout 10s（AGENTS 规则）。
set -uo pipefail
cd "$(dirname "$0")/../.."   # -> 工程根

BENCH=${BENCH:-./bench}
DEV=$(seq -s, 0 31)
T=${T:-10}
OUT=test/71.r32_tk_imbalance
LOG=${LOG:-test/70.r32_tpr_perf/0928_tk_info.log}
LAUNCHES=${LAUNCHES:-"804400 804300 853000"}   # 递增不均衡度（max 176/362/512）

run() {
  MACA_VISIBLE_DEVICES="$DEV" timeout -k 3 "$T" stdbuf -oL "$BENCH" "$@"
}

{
  echo "### 均衡基线 TPR=16（每专家恰好 16 对）"
  run --tprs 16
  for L in $LAUNCHES; do
    echo "### 不均衡 launch=$L"
    run --tk-info "$LOG" --tk-launch "$L"
  done
} 2>&1 | tee "$OUT/run.log"
