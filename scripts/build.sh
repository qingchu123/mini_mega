#!/usr/bin/env bash
# 宿主机编译（cucc + mcr runtime，无需容器、不用 torch）
#
# 用法：
#   scripts/build.sh [目标 ...]
#
# 目标：
#   so       编译 megakernel.so（host C++ 接口层 + kernel）
#   bench    编译 bench（benchmark，只调用接口，链接 megakernel.so）
#   mybench  编译 mybench（每进程一张卡的 MPI 基准）
#   whl      构建 torch wheel（torch/dist/*.whl，需 python+torch）
#   all      等价于 so bench mybench whl
#   -h       显示帮助
#
# 不带目标时等价 all。bench/mybench/whl 会自动先编 so。
#
# 产物：
#   megakernel.so   fp8 mega kernel 的 host C++ 接口层（src/megakernel.cu）
#   bench           benchmark：只调用接口，链接 megakernel.so
#   mybench         每进程一张卡的 MPI 基准（test/1.mytest）
#   torch/dist/*.whl  torch 层可安装 wheel
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
用法: scripts/build.sh [目标 ...]

目标:
  so        编译 megakernel.so（host C++ 接口层 + kernel）
  bench     编译 bench（benchmark，只调接口，链接 megakernel.so）
  mybench   编译 mybench（每进程一张卡的 MPI 基准）
  whl       构建 torch wheel（torch/dist/*.whl，需 python+torch）
  all       等价于 so bench mybench whl
  -h, --help 显示本帮助

不带目标时等价 all。bench/mybench/whl 会自动先编 so。
环境变量: PYTHON（whl 目标，默认 python）
EOF
}

TARGETS=()
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    so|bench|mybench|whl|all) TARGETS+=("$arg") ;;
    *) echo "未知参数: $arg" >&2; usage >&2; exit 2 ;;
  esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(all)

declare -A DO=()
for t in "${TARGETS[@]}"; do
  if [ "$t" = all ]; then
    DO[so]=1; DO[bench]=1; DO[mybench]=1; DO[whl]=1
  else
    DO[$t]=1
  fi
done
# bench/mybench/whl 依赖 megakernel.so，自动补齐 so
if [ -n "${DO[bench]:-}${DO[mybench]:-}${DO[whl]:-}" ]; then
  DO[so]=1
fi

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
# [OPT] 扫描用额外编译宏（默认空；用于 -D 参数实验）
KERNEL_CFLAGS=${KERNEL_CFLAGS:-}

build_so() {
  # 接口层 + kernel -> megakernel.so（-fPIC -shared）
  "$CUCC" -O3 -std=c++17 -fPIC -shared "${INC[@]}" $KERNEL_CFLAGS \
    -o megakernel.so src/megakernel.cu
  echo "built: $PWD/megakernel.so"
}

build_bench() {
  # bench：只 include megakernel.h，链接 megakernel.so（rpath=$ORIGIN 便于原地运行）
  # 文件名非 lib*.so，故直接以路径链接（不用 -l）。
  "$CUCC" -O3 -std=c++17 "${INC[@]}" -I"$MACA_PATH/ompi/include" \
    -o bench src/bench.cu \
    ./megakernel.so -L"$MACA_PATH/ompi/lib" -lmpi \
    -Wl,-rpath,'$ORIGIN' -Wl,-rpath,"$MACA_PATH/ompi/lib"
  echo "built: $PWD/bench"
}

build_mybench() {
  # mybench：每进程一张卡的 MPI 基准（test/1.mytest），同样链接 MPI + megakernel.so。
  "$CUCC" -O3 -std=c++17 "${INC[@]}" -I"$MACA_PATH/ompi/include" \
    -o mybench src/mybench.cu \
    ./megakernel.so -L"$MACA_PATH/ompi/lib" -lmpi \
    -Wl,-rpath,'$ORIGIN' -Wl,-rpath,"$MACA_PATH/ompi/lib"
  echo "built: $PWD/mybench"
}

build_whl() {
  # torch 层 wheel：setup.py 链接已构建的 ../megakernel.so 并随包分发。
  local PY=${PYTHON:-python}
  ( cd torch && "$PY" setup.py bdist_wheel )
  local whl
  whl=$(ls -t torch/dist/*.whl 2>/dev/null | head -1 || true)
  [ -n "$whl" ] && echo "built: $PWD/$whl"
}

[ -n "${DO[so]:-}" ] && build_so
[ -n "${DO[bench]:-}" ] && build_bench
[ -n "${DO[mybench]:-}" ] && build_mybench
[ -n "${DO[whl]:-}" ] && build_whl
exit 0
