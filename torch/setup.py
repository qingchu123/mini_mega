"""mxmegakernel torch 层构建脚本（mini_mega）。

产物：`mxmegakernel._C` pybind11 扩展 + Python 包 mxmegakernel。
设计要点：
  - CUDAExtension（MACA cu-bridge 提供 nvcc/gcc 兼容头）。
  - 扩展**不重新编译 kernel**：直接链接 mini_mega 已构建好的
    `megakernel.so`（extra_objects），并由 rpath `$ORIGIN` 指向随包分发的副本。
  - MACA 的 cudart shim 由 megakernel.so 的 RUNPATH(/opt/maca/lib) 负责，
    扩展侧只补 mcruntime/runtime_cu/mccompiler。
"""

import os
import shutil
from datetime import datetime

import torch
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

current_dir = os.path.dirname(os.path.realpath(__file__))
mini_root = os.path.abspath(os.path.join(current_dir, ".."))

public_version = "0.0.0"
build_stamp = os.environ.get("MXK_BUILD_STAMP") or datetime.now().strftime("%Y%m%d%H")
local_version_suffix = os.environ.get("MXK_LOCAL_VERSION_SUFFIX", f"mini_mega.{build_stamp}")
version = f"{public_version}+{local_version_suffix}" if local_version_suffix else public_version

maca_path = os.getenv("MACA_PATH", None)
print(f"MACA directory: {maca_path}", flush=True)
assert maca_path is not None and os.path.exists(maca_path), f"bad MACA_PATH={maca_path}"

cuda_home = os.getenv("CUDA_HOME") or os.path.join(maca_path, "tools", "cu-bridge")
assert os.path.exists(cuda_home), f"Failed to find cu-bridge: {cuda_home}"
os.environ.setdefault("CUDA_HOME", cuda_home)
os.environ["PATH"] = os.path.join(cuda_home, "bin") + os.pathsep + os.environ.get("PATH", "")

cxx_flags = [
    "-std=c++17",
    "-O3",
    "-fPIC",
    "-Wno-psabi",
    "-Wno-deprecated-declarations",
    "-DUSE_MACA",
    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
]

# [GS] 组内 rank 数需与 kernel（scripts/build.sh 的 KERNEL_CFLAGS）一致，否则
#   mxk_group_ranks()/组权重 all-gather 的块数与 kernel 期望不符。用同一 env
#   透传 `-DMX_GROUP_RANKS=<N>`（缺省不定义 → 走 common.cuh 默认 4）。
_grp = os.environ.get("MXK_GROUP_RANKS")
if _grp:
    cxx_flags.append(f"-DMX_GROUP_RANKS={int(_grp)}")
    print(f"[setup] MX_GROUP_RANKS={int(_grp)}", flush=True)

sources = [
    os.path.join("csrc", "python_api.cpp"),
    os.path.join("csrc", "kernel_bridge.cpp"),
    os.path.join("csrc", "memory.cpp"),
    os.path.join("csrc", "flags_bridge.cu"),
]

build_include_dirs = [
    os.path.join(cuda_home, "include"),
    os.path.join(cuda_home, "include", "cccl"),
    os.path.join(maca_path, "include"),
    os.path.join(current_dir, "csrc"),
    os.path.join(mini_root, "include"),
    os.path.join(mini_root, "src"),
]

# 打包并链接 mini_mega 的已构建 kernel 接口层。
kernel_so_src = os.path.join(mini_root, "test", "megakernel.so")
kernel_so_dst = os.path.join(current_dir, "mxmegakernel", "megakernel.so")
if os.path.exists(kernel_so_src):
    shutil.copy2(kernel_so_src, kernel_so_dst)
    print(f"bundled kernel lib: {kernel_so_dst}", flush=True)
else:
    print(f"WARNING: {kernel_so_src} 不存在，扩展将以未链接 kernel 的方式构建", flush=True)

extra_objects = []
build_libraries = ["symbol_cu", "mcruntime", "runtime_cu", "mccompiler", "dl"]
build_library_dirs = [os.path.join(cuda_home, "lib"), os.path.join(maca_path, "lib")]
# 以 -l:megakernel.so 链接，使 DT_NEEDED 记录为裸文件名 "megakernel.so"，
# 配合 rpath $ORIGIN 即可从包目录内加载随包分发的 kernel 库。
extra_link_args = [
    "-Wl,-rpath,$ORIGIN",
    f"-L{os.path.join(current_dir, 'mxmegakernel')}",
    "-l:megakernel.so",
]
if not os.path.exists(kernel_so_dst):
    extra_link_args = ["-Wl,-rpath,$ORIGIN"]

setup(
    name="mxmegakernel",
    version=version,
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    package_data={"mxmegakernel": ["*.so"]},
    ext_modules=[
        CUDAExtension(
            name="mxmegakernel._C",
            sources=sources,
            include_dirs=build_include_dirs,
            extra_objects=extra_objects,
            libraries=build_libraries,
            library_dirs=build_library_dirs,
            extra_compile_args=cxx_flags,
            extra_link_args=extra_link_args,
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
)
