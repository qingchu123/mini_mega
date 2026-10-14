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
#   whl      torch/dist/*.whl     torch 层 wheel（需 python+torch，PYTHON= 可指定）
#   all      等价于 so mybench check（不含 whl：whl 依赖 torch 环境）
#   -h       显示帮助
#
# 不带目标时等价 all。mybench/check/whl 会自动先编 so。
set -euo pipefail
cd "$(dirname "$0")/.."

usage() {
  cat <<'EOF'
用法: scripts/build.sh [目标 ...]

目标:
  so        test/megakernel.so（host C++ 接口层 + kernel）
  mybench   test/mybench（每进程一张卡的 MPI 性能基准）
  check     test/check（MPI 正确性测试）
  whl       torch/dist/*.whl（需 python+torch；PYTHON= 指定解释器）
  all       等价于 so mybench check（不含 whl）
  -h, --help 显示本帮助

不带目标时等价 all。mybench/check/bench/whl 会自动先编 so。
环境变量: MACA_PATH（默认 /opt/maca）、PYTHON（whl 目标，默认 python）
EOF
}

TARGETS=()
for arg in "$@"; do
  case "$arg" in
    -h|--help) usage; exit 0 ;;
    so|mybench|check|whl|all) TARGETS+=("$arg") ;;
    *) echo "未知参数: $arg" >&2; usage >&2; exit 2 ;;
  esac
done
[ ${#TARGETS[@]} -eq 0 ] && TARGETS=(all)

declare -A DO=()
for t in "${TARGETS[@]}"; do
  if [ "$t" = all ]; then
    DO[so]=1; DO[mybench]=1; DO[check]=1
  else
    DO[$t]=1
  fi
done
# mybench/check/whl 依赖 megakernel.so，自动补齐 so
if [ -n "${DO[mybench]:-}${DO[check]:-}${DO[whl]:-}" ]; then
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
# kernel grid（= 每卡 block 数）：MXK_KERNEL_GRID 与 torch 侧同一个值（缺省 28，见 src/grid_config.h）
if [ -n "${MXK_KERNEL_GRID:-}" ]; then
  KERNEL_CFLAGS="$KERNEL_CFLAGS -DKGRID_OVERRIDE=$MXK_KERNEL_GRID"
fi

# 依赖文件（-MMD 生成）中是否存在比 target 更新的文件；有则返回 0（需重编）
deps_newer() {
  local d="$1" target="$2"
  [ -e "$d" ] || return 1
  local deps
  deps=$(tr '\\\n' '  ' < "$d")   # 展开续行/反斜杠
  deps=${deps#*:}                 # 去掉 "target:" 前缀
  # shellcheck disable=SC2086
  [ -n "$(find $deps -newer "$target" -print -quit 2>/dev/null)" ]
}

build_so() {
  # 接口层 + kernel -> test/megakernel.so（-fPIC -shared）
  # [multi_compile] 各 kernel 实例在 src/multi_compile/*.cu 单独编译（-j 并行），再链接。
  # [incremental] 依据 -MMD 依赖按 mtime 跳过未变的 TU；仅重链受影响的 .so。
  local objd="$PWD/build/obj_so"
  mkdir -p "$objd"
  local srcs=(src/megakernel.cu)
  for f in src/multi_compile/*.cu; do [ -e "$f" ] && srcs+=("$f"); done

  # 编译参数指纹：-D/包含路径变化时清空对象强制全量重编
  local flagsig="-O3|$KERNEL_CFLAGS|${INC[*]}"
  local stamp="$objd/.buildflags"
  if [ ! -e "$stamp" ] || [ "$(cat "$stamp")" != "$flagsig" ]; then
    rm -f "$objd"/*.o "$objd"/*.d
    printf '%s' "$flagsig" > "$stamp"
  fi

  local -a stale=()
  local s o d
  for s in "${srcs[@]}"; do
    o="$objd/$(basename "$s" .cu).o"
    d="$objd/$(basename "$s" .cu).d"
    if [ ! -e "$o" ] || [ "$s" -nt "$o" ] || deps_newer "$d" "$o"; then
      stale+=("$s")
    fi
  done

  if [ ${#stale[@]} -gt 0 ]; then
    echo "build_so: 重编 ${#stale[@]}/${#srcs[@]} 个 TU"
    export _CUCC="$CUCC" _OBJ="$objd"
    export _INC="${INC[*]}" _KCF="$KERNEL_CFLAGS"
    printf '%s\n' "${stale[@]}" | \
      xargs -P "${BUILD_JOBS:-$(nproc)}" -I{} bash -c \
        '"$_CUCC" -O3 -std=c++17 -fPIC $_INC $_KCF -MMD -MF "$_OBJ/$(basename "{}" .cu).d" -c "{}" -o "$_OBJ/$(basename "{}" .cu).o"'
  else
    echo "build_so: 无 TU 需重编"
  fi

  local so="$OUT/megakernel.so"
  if [ ! -e "$so" ] || [ -n "$(find "$objd" -maxdepth 1 -name '*.o' -newer "$so" -print -quit)" ]; then
    "$CUCC" -O3 -std=c++17 -fPIC -shared -Wl,-soname,megakernel.so "${INC[@]}" \
      -o "$so" "$objd"/*.o
    echo "built: $PWD/$so"
  else
    echo "up-to-date: $PWD/$so"
  fi
}

build_mybench() {
  local out="$OUT/mybench"
  if [ -e "$out" ] && [ "$out" -nt "$OUT/megakernel.so" ] \
     && [ "$out" -nt test/mybench.cu ] && [ "$out" -nt test/mega_common.cuh ] \
     && [ "$out" -nt src/megakernel.h ]; then
    echo "up-to-date: $PWD/$out"; return
  fi
  "$CUCC" -O3 -std=c++17 "${INC[@]}" "${MPI_INC[@]}" $KERNEL_CFLAGS \
    -o "$out" test/mybench.cu \
    "$OUT/megakernel.so" "${MPI_LIB[@]}"
  echo "built: $PWD/$out"
}

build_check() {
  local out="$OUT/check"
  if [ -e "$out" ] && [ "$out" -nt "$OUT/megakernel.so" ] \
     && [ "$out" -nt test/check.cu ] && [ "$out" -nt test/mega_common.cuh ] \
     && [ "$out" -nt src/megakernel.h ]; then
    echo "up-to-date: $PWD/$out"; return
  fi
  "$CUCC" -O3 -std=c++17 "${INC[@]}" "${MPI_INC[@]}" $KERNEL_CFLAGS \
    -o "$out" test/check.cu \
    "$OUT/megakernel.so" "${MPI_LIB[@]}"
  echo "built: $PWD/$out"
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
[ -n "${DO[mybench]:-}" ] && build_mybench
[ -n "${DO[check]:-}" ] && build_check
[ -n "${DO[whl]:-}" ] && build_whl
exit 0
