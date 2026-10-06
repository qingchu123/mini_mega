"""layout 辅助（纯 Python）。

mini_mega 最小实现不提供 GEMM 布局的 C++ 符号，故这里给出 mx_test 需要的
对齐查询的纯 Python 口径（128）。后续接入 GEMM 布局时替换为 _C 转发即可。
"""


def get_mk_alignment_for_contiguous_layout() -> int:
    return 128


def set_mk_alignment_for_contiguous_layout(value: int) -> None:
    raise NotImplementedError(
        "mini_mega torch 层未实现 set_mk_alignment_for_contiguous_layout")


def get_theoretical_mk_alignment_for_contiguous_layout() -> int:
    return 128


get_m_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout
get_k_alignment_for_contiguous_layout = get_mk_alignment_for_contiguous_layout


def get_tma_aligned_size(x: int, alignment: int = 128) -> int:
    return (x + alignment - 1) // alignment * alignment
