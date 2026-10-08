#!/usr/bin/env bash
# ============================================================================
# scripts/run.sh — 用 mpirun 跑 mini_mega bench（GLM-5.2 R=32 口径）
#
# 模型说明：
#   现在 ./bench 支持两种模型：
#     * NP>1（mpirun -np NP）：MPI 模式，**每进程一 rank / 一张卡**（rank=MPI rank，
#       R=MPI size；用 MPI_Allgather 交换 mcIpcMemHandle、MPI_Barrier 做栅栏）。
#     * NP=1（直跑）：单进程多 rank（R 个线程各自 mcSetDevice）。
#   默认 NP=32（每进程一张卡）。
#
# 用法：
#   scripts/run.sh --tprs 1                      # 默认（NP=32，每进程一卡）
#   scripts/run.sh --tprs 16,64,256 --nodes 20
#   scripts/run.sh --tk-info test/70.r32_tpr_perf/0928_tk_info.log --tk-first 50
#   NP=1 scripts/run.sh --tprs 1                 # 退回单进程 32-rank 模型
#   scripts/run.sh --tk-info test/70.r32_tpr_perf/0928_tk_info.log --tk-first 50
#
# 环境变量：
#   NP        进程数（默认 32；>1 走 MPI 每进程一 rank，=1 走单进程 32-rank）
#   DEV       MACA_VISIBLE_DEVICES（默认 0..31）
#   BENCH     bench 可执行（默认 ./bench）
#   TIMEOUT   整体超时秒（默认 600）
#   HOSTS     多机主机列表，传给 mpirun -H（默认空=本机）
#   MACA_PATH 默认 /opt/maca
#   WAIT_FREE 1=等外部 GPU 进程清空后再跑（简单轮询 mx-smi）
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

export MACA_PATH=${MACA_PATH:-/opt/maca}
export PATH="$MACA_PATH/ompi/bin:$MACA_PATH/bin:$MACA_PATH/mxgpu_llvm/bin:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/ompi/lib:$MACA_PATH/lib:$MACA_PATH/tools/cu-bridge/lib:$MACA_PATH/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}"

NP=${NP:-32}
DEV=${DEV:-$(seq -s, 0 31)}
BENCH=${BENCH:-./bench}
TIMEOUT=${TIMEOUT:-600}
HOSTS=${HOSTS:-}
MPIRUN=${MPIRUN:-$MACA_PATH/ompi/bin/mpirun}

ARGS=("$@")
if [ ${#ARGS[@]} -eq 0 ]; then ARGS=(--tprs 1); fi

# 可选：等外部 GPU 进程清空（避免争用导致的 join 等待膨胀）
if [ "${WAIT_FREE:-0}" = "1" ]; then
    me=$(id -u)
    while :; do
        busy=$(mx-smi 2>/dev/null | grep -oE '^\| +[0-9]+ +[0-9]+ ' | awk '{print $3}' | sort -u | \
               while read -r pid; do o=$(stat -c %u "/proc/$pid" 2>/dev/null) || continue; \
                 [ "$o" != "$me" ] && echo "$pid"; done)
        [ -z "$busy" ] && break
        echo "[run.sh] 外部 GPU 进程: $(echo "$busy" | tr '\n' ' ') —— 等待 5s"
        sleep 5
    done
fi

if [ ! -x "$BENCH" ]; then
    echo "[run.sh] $BENCH 不存在，先构建：bash scripts/build.sh" >&2
    exit 2
fi

export MACA_VISIBLE_DEVICES="$DEV"
echo "[run.sh] $MPIRUN -np $NP $BENCH ${ARGS[*]}  (MACA_VISIBLE_DEVICES=$DEV)"
timeout -k 5 "$TIMEOUT" "$MPIRUN" --allow-run-as-root --oversubscribe \
    ${HOSTS:+-H "$HOSTS"} -np "$NP" \
    "$BENCH" "${ARGS[@]}"
rc=$?
echo "[run.sh] rc=$rc"
exit $rc
