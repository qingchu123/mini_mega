#!/usr/bin/env bash
# 宿主机编译（cucc + mcr runtime，无需容器、不用 torch）
#
# 用法：
#   scripts/build.sh [目标 ...]        （或 ./scripts/build ...）
#
# 目标：
#   so       test/megakernel.so   fp8 mega kernel 的 host C++ 接口层（src/megakernel.cu）
#   mybench  test/mybench         每进程一张卡的 MPI 性能基准（test/mybench.cu）
#   check    test/check           MPI 正确性测试（test/check.cu）
#   bench    test/bench           单进程多卡 benchmark（src/bench.cu，旧口径）
#   whl      torch/dist/*.whl     torch 层 wheel（需 python+torch，PYTHON= 可指定）
#   all      等价于 so bench mybench check（不含 whl：whl 依赖 torch 环境）
#   -h       显示帮助
#
# 不带目标时等价 all。mybench/check/bench/whl 会自动先编 so。
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
用法: scripts/build.sh [目标 ...]

目标:
  so        test/megakernel.so（host C++ 接口层 + kernel）
  mybench   test/mybench（每进程一张卡的 MPI 性能基准）
  check     test/check（MPI 正确性测试）
  bench     test/bench（单进程多卡 benchmark，旧口径）
  whl       torch/dist/*.whl（需 python+torch；PYTHON= 指定解释器）
  all       等价于 so bench mybench check（不含 whl）
  -h, --help 显示本帮助

不带目标时等价 all。mybench/check/bench/whl 会自动先编 so。
环境变量: MACA_PATH（默认 /opt/maca）、PYTHON（whl 目标，默认 python）
EOF
}

TARGETS=()
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    so|mybench|check|bench|whl|all) TARGETS+=("$arg") ;;
    *) echo "未知参数: $arg" >&2; usage >&2; exit 2 ;;
  esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(all)

declare -A DO=()
for t in "${TARGETS[@]}"; do
  if [ "$t" = all ]; then
    DO[so]=1; DO[bench]=1; DO[mybench]=1; DO[check]=1
  else
    DO[$t]=1
  fi
done
# mybench/check/bench/whl 依赖 megakernel.so，自动补齐 so
if [ -n "${DO[bench]:-}${DO[mybench]:-}${DO[check]:-}${DO[whl]:-}" ]; then
  DO[so]=1
fi

OUT=test   # 所有二进制产物输出到此目录（binaries 与 megakernel.so 同目录，rpath=$ORIGIN）

# 最小工程自带 deep_gemm 头（include/）；缺失时回退到 MXMOEMega 仓库。
INCLUDE_ROOT=${MEGA_INCLUDE_ROOT:-include}
if [ ! -d "$INCLUDE_ROOT/deep_gemm" ]; then
  REPO=${MXMOEMEGA_REPO:-/home/lchen1/data/projs/MXMOEMega}
  INCLUDE_ROOT="$REPO/mxmegakernel/include"
fi
export MACA_PATH=${MACA_PATH:-/opt/maca}
export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=${CUDA_HOME:-$MACA_PATH/tools/cu-bridge}
mkdir -p "$HOME/_nvcc"

CUCC="$MACA_PATH/tools/cu-bridge/bin/cucc"
INC=(-I"$MACA_PATH/tools/cu-bridge/include" -I/opt/maca/include
     -I"$INCLUDE_ROOT" -Isrc)
MPI_INC=(-I"$MACA_PATH/ompi/include")
MPI_LIB=(-L"$MACA_PATH/ompi/lib" -lmpi -Wl,-rpath,'$ORIGIN' -Wl,-rpath,"$MACA_PATH/ompi/lib")
# [OPT] 扫描用额外编译宏（默认空；用于 -D 参数实验）
KERNEL_CFLAGS=${KERNEL_CFLAGS:-}

build_so() {
  # 接口层 + kernel -> test/megakernel.so（-fPIC -shared）
  # SONAME=megakernel.so：消费者的 DT_NEEDED 记为裸名，经 RUNPATH=$ORIGIN 解析，
  # 与 cwd 无关（否则链接路径 test/megakernel.so 会被写入 DT_NEEDED）。
  "$CUCC" -O3 -std=c++17 -fPIC -shared "${INC[@]}" $KERNEL_CFLAGS \
    -Wl,-soname,megakernel.so \
    -o "$OUT/megakernel.so" src/megakernel.cu
  echo "built: $PWD/$OUT/megakernel.so"
}

build_bench() {
  "$CUCC" -O3 -std=c++17 "${INC[@]}" "${MPI_INC[@]}" \
    -o "$OUT/bench" src/bench.cu \
    "$OUT/megakernel.so" "${MPI_LIB[@]}"
  echo "built: $PWD/$OUT/bench"
}

build_mybench() {
  "$CUCC" -O3 -std=c++17 "${INC[@]}" "${MPI_INC[@]}" \
    -o "$OUT/mybench" test/mybench.cu \
    "$OUT/megakernel.so" "${MPI_LIB[@]}"
  echo "built: $PWD/$OUT/mybench"
}

build_check() {
  "$CUCC" -O3 -std=c++17 "${INC[@]}" "${MPI_INC[@]}" \
    -o "$OUT/check" test/check.cu \
    "$OUT/megakernel.so" "${MPI_LIB[@]}"
  echo "built: $PWD/$OUT/check"
}

build_whl() {
  # torch 层 wheel：setup.py 链接已构建的 test/megakernel.so 并随包分发。
  local PY=${PYTHON:-python}
  ( cd torch && "$PY" setup.py bdist_wheel )
  local whl
  whl=$(ls -t torch/dist/*.whl 2>/dev/null | head -1 || true)
  [ -n "$whl" ] && echo "built: $PWD/$whl"
}

[ -n "${DO[so]:-}" ] && build_so
[ -n "${DO[bench]:-}" ] && build_bench
[ -n "${DO[mybench]:-}" ] && build_mybench
[ -n "${DO[check]:-}" ] && build_check
[ -n "${DO[whl]:-}" ] && build_whl
exit 0
