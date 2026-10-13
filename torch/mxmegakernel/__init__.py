"""mxmegakernel —— mini_mega 的 torch 层（pybind11 扩展 + Python 包）。

导入顺序：先加载 torch 运行库，再加载 `_C` 扩展；子包在导入期持有 `_C` 符号。
"""

import torch  # noqa: F401

from . import _C

transform_sf_into_required_layout = _C.transform_sf_into_required_layout

from . import testing, utils  # noqa: E402
from .mega import (  # noqa: E402
    FROZEN_HIDDEN,
    FROZEN_INTERMEDIATE_HIDDEN,
    FROZEN_NUM_EXPERTS_PER_RANK,
    FROZEN_NUM_TOPK,
    SymmBuffer,
    bf16_fp8_mega_moe,
    bf16_mega_moe,
    fp8_fp4_mega_moe,
    fp8_mega_moe,
    fp8_mega_moe_compute,
    fp8_mega_moe_reduce,
    get_block_m_for_mega_moe,
    get_symm_buffer_for_mega_moe,
    get_symm_buffer_size_for_mega_moe,
    get_token_alignment_for_mega_moe,
    read_block_marks,
    read_dbg_report,
    read_phase_ticks,
    select_fp8_mega_moe_window,
    transform_weights_for_mega_moe,
)

__version__ = "0.0.0"

__all__ = [
    "SymmBuffer",
    "get_symm_buffer_for_mega_moe",
    "get_symm_buffer_size_for_mega_moe",
    "select_fp8_mega_moe_window",
    "get_token_alignment_for_mega_moe",
    "get_block_m_for_mega_moe",
    "transform_weights_for_mega_moe",
    "fp8_mega_moe",
    "fp8_mega_moe_compute",
    "fp8_mega_moe_reduce",
    "bf16_mega_moe",
    "bf16_fp8_mega_moe",
    "fp8_fp4_mega_moe",
    "read_phase_ticks",
    "read_dbg_report",
    "read_block_marks",
    "FROZEN_HIDDEN",
    "FROZEN_INTERMEDIATE_HIDDEN",
    "FROZEN_NUM_TOPK",
    "FROZEN_NUM_EXPERTS_PER_RANK",
    "transform_sf_into_required_layout",
    "_C",
    "utils",
    "testing",
]
