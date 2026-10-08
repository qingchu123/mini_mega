#!/usr/bin/env bash
# ============================================================================
# scripts/run.sh — 统一基准启动脚本（mpirun，每进程一张卡）
#
# 用法：
#   scripts/run.sh -R 32 [mybench 选项...]     # 默认跑 ./mybench（每进程一 rank/一卡）
#   BENCH=./bench scripts/run.sh --tprs 1,4    # 跑 ./bench（单进程多 rank，建议 NP=1）
#
#   -R N   进程数/rank 数（默认 32）= mpirun -np N；对 ./mybench 即 R 个 rank。
#   -NP N  仅 mpirun 进程数（不改 R）；默认 = -R 的值。
#   其余参数透传给 $BENCH，例如 mybench：
#     scripts/run.sh -R 32 --s 0 --tprs 1,4,16,32,64,128 --nodes 20 --replays 2
#
# 环境变量：BENCH（默认 ./mybench）、DEV（默认 0..R-1）、TIMEOUT（默认 600）、
#          NP、MPIRUN、MACA_PATH（默认 /opt/maca）、WAIT_FREE=1（等外部占卡清空）、
#          MPI_MCA（默认见下文，设为空则用 OpenMPI 默认传输）。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

export MACA_PATH=${MACA_PATH:-/opt/maca}
export PATH="$MACA_PATH/ompi/bin:$MACA_PATH/bin:$MACA_PATH/mxgpu_llvm/bin:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/ompi/lib:$MACA_PATH/lib:$MACA_PATH/tools/cu-bridge/lib:$MACA_PATH/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}"

R=${R:-32}
NPROC=${NP:-}
PASSTHRU=()
while [ $# -gt 0 ]; do
    case "$1" in
        -R) R="$2"; shift 2 ;;
        -R*) R="${1#-R}"; shift ;;
        -NP) NPROC="$2"; shift 2 ;;
        -NP*) NPROC="${1#-NP}"; shift ;;
        *) PASSTHRU+=("$1"); shift ;;
    esac
done

[ -z "$NPROC" ] && NPROC="$R"

DEV=${DEV:-$(seq -s, 0 $((R - 1)))}
BENCH=${BENCH:-./mybench}
MPIRUN=${MPIRUN:-$MACA_PATH/ompi/bin/mpirun}
TIMEOUT=${TIMEOUT:-600}
# MPI 传输（单机）：默认 pml=ob1 + btl=self,vader + osc 去 ucx。
# OpenMPI 默认 pml/ucx 选择在本机 R=32 下 MPI_Init 约 6s、MPI_Finalize 约 4s；
# 此组合约 0.2s。mybench 只用 80B handle 的 Allgather/Barrier，无跨机需求。
# 多机运行请设 MPI_MCA="" 恢复 OpenMPI 默认。
MPI_MCA=${MPI_MCA-"--mca pml ob1 --mca btl self,vader --mca osc ^ucx"}
read -ra MCA_ARGS <<< "$MPI_MCA"

if [ ! -x "$BENCH" ]; then
    echo "[run.sh] $BENCH 不存在，先构建：bash scripts/build.sh" >&2
    exit 2
fi

# 可选：等外部 GPU 进程清空，避免争用导致的 join 等待膨胀
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

export MACA_VISIBLE_DEVICES="$DEV"
echo "[run.sh] $MPIRUN -np $NPROC $BENCH ${PASSTHRU[*]:-<default>}  (R=$R, MACA_VISIBLE_DEVICES=$DEV)"
timeout -k 5 "$TIMEOUT" "$MPIRUN" --allow-run-as-root --oversubscribe \
    "${MCA_ARGS[@]}" -np "$NPROC" "$BENCH" "${PASSTHRU[@]}"
rc=$?
echo "[run.sh] rc=$rc"
exit $rc
