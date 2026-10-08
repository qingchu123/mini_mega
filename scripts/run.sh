#!/usr/bin/env bash
# ============================================================================
# scripts/run.sh — 用 mpirun 跑 mini_mega bench（GLM-5.2 R=32 口径）
#
# 模型说明：
#   当前 ./bench 是「单进程多 rank」：一个进程内起 R 个线程，各自 mcSetDevice(rank)，
#   用 mcIpcMemHandle 做对端显存映射（见 src/bench.cu run_aligned 的 rank 初始化）。
#   它**不依赖 MPI**，故 mpirun 只作统一启动器，默认 NP=1（bench 内部管理全部 32 卡），
#   以便与生产侧（mpirun/torchrun 多进程）对齐启动环境。
#   若后续把 bench 改成「每 rank 一进程（MPI 交换 IPC handle + MPI_Barrier）」，
#   可直接 `NP=32 scripts/run.sh ...`，本脚本无需改动。
#
# 用法：
#   scripts/run.sh --tprs 1                      # 默认（NP=1）
#   scripts/run.sh --tprs 16,64,256 --nodes 20
#   scripts/run.sh --tk-info test/70.r32_tpr_perf/0928_tk_info.log --tk-first 50
#   NP=1 DEV=$(seq -s, 0 31) TIMEOUT=900 scripts/run.sh --tk-info ... --tk-all
#
# 环境变量：
#   NP        进程数（默认 1；bench 单进程模型下应保持 1）
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

NP=${NP:-1}
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
