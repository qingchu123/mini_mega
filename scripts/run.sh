#!/usr/bin/env bash
# ============================================================================
# scripts/run.sh — 统一启动脚本（mpirun，每进程一张卡）
#
# 用法：
#   scripts/run.sh [程序] [-R N] [程序参数...]
#     程序：mybench（默认）| check
#
# 示例：
#   scripts/run.sh mybench -R 32 -s 1 -tprs 1,2,4,6,8,12,18,24,30,36,42,48,54,60,66,72,78,84,90,128,256   # 均衡
#   scripts/run.sh mybench -R 32 -s 2 -top 50                              # TK 前 50
#   scripts/run.sh mybench -R 32 -s 2 -cases 804900,807500                 # 指定 case
#   scripts/run.sh check --smoke                                           # 冒烟（<10s）
#   scripts/run.sh check --all                                             # 全量（不限时）
#
# 说明：
#   -R N    进程数 = rank 数（默认 32）；mpirun -np N。
#   -NP N   仅覆盖 mpirun 进程数（默认 = R）。bench 为单进程多卡，默认 NP=1。
#   其余参数原样透传给程序。
#
# 环境变量：
#   BENCH       覆盖二进制路径（默认 test/<程序>）
#   MPIRUN      mpirun 路径（默认 $MACA_PATH/ompi/bin/mpirun）
#   MACA_PATH   默认 /opt/maca
#   DEV         MACA_VISIBLE_DEVICES（默认 0..R-1）
#   TIMEOUT     超时秒数（默认 600；check --all 默认 3600）
#   MPI_MCA     OpenMPI 传输参数（默认 "--mca pml ob1 --mca btl self,vader --mca osc ^ucx"，
#               单机快速启动；多机请设为空以使用 OpenMPI 默认）
#   GPU_LOCK    1（默认）= 持有 .agents/work/lock/lock_gpu 再运行，避免多 Agent 争抢 GPU
#   GPU_LOCK_WAIT  等待锁的秒数（默认 3600）
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")/.."

# ---- 程序选择（第一个非横杠参数；缺省 mybench） ----
PROG=mybench
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then
    PROG=$1
    shift
fi
case "$PROG" in
    mybench|check) ;;
    *) echo "[run.sh] 未知程序: $PROG（可选 mybench|check）" >&2; exit 2 ;;
esac

# ---- GPU 临界区（AGENTS.md §4）：可重入保护，GPU_LOCK=0 关闭 ----
LOCKFILE=.agents/work/lock/lock_gpu
if [ "${GPU_LOCK:-1}" = "1" ] && [ "${_RUN_SH_LOCKED:-0}" != "1" ] && command -v flock >/dev/null 2>&1; then
    mkdir -p "$(dirname "$LOCKFILE")"
    export _RUN_SH_LOCKED=1
    exec flock -w "${GPU_LOCK_WAIT:-3600}" "$LOCKFILE" "$0" "$PROG" "$@"
fi

export MACA_PATH=${MACA_PATH:-/opt/maca}
export PATH="$MACA_PATH/ompi/bin:$MACA_PATH/bin:$MACA_PATH/mxgpu_llvm/bin:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/ompi/lib:$MACA_PATH/lib:$MACA_PATH/tools/cu-bridge/lib:$MACA_PATH/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}"

# ---- 解析 -R / -NP，其余透传 ----
R=${R:-32}
NPROC=${NP:-}
NP_SET=0
PASSTHRU=()
while [ $# -gt 0 ]; do
    case "$1" in
        -R) R="$2"; shift 2 ;;
        -R*) R="${1#-R}"; shift ;;
        -NP) NPROC="$2"; NP_SET=1; shift 2 ;;
        -NP*) NPROC="${1#-NP}"; NP_SET=1; shift ;;
        *) PASSTHRU+=("$1"); shift ;;
    esac
done

if [ -z "$NPROC" ]; then
    NPROC="$R"
fi

BENCH=${BENCH:-test/$PROG}
MPIRUN=${MPIRUN:-$MACA_PATH/ompi/bin/mpirun}
DEFAULT_TIMEOUT=600
if [ "$PROG" = check ] && [[ " ${PASSTHRU[*]:-} " == *" --all "* ]]; then
    DEFAULT_TIMEOUT=3600
fi
TIMEOUT=${TIMEOUT:-$DEFAULT_TIMEOUT}
MPI_MCA=${MPI_MCA-"--mca pml ob1 --mca btl self,vader --mca osc ^ucx"}
read -ra MCA_ARGS <<< "$MPI_MCA"

DEV=${DEV:-$(seq -s, 0 $((R - 1)))}

if [ ! -x "$BENCH" ]; then
    echo "[run.sh] $BENCH 不存在，先构建：scripts/build.sh $PROG" >&2
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
START=$(date +%s.%N)
timeout -k 5 "$TIMEOUT" "$MPIRUN" --allow-run-as-root --oversubscribe \
    "${MCA_ARGS[@]}" -np "$NPROC" "$BENCH" "${PASSTHRU[@]}"
rc=$?
END=$(date +%s.%N)
printf '[run.sh] wall=%.2fs\n' "$(echo "$END - $START" | bc 2>/dev/null || echo 0)"
echo "[run.sh] rc=$rc"
exit $rc
