"""mini_mega torch 层冒烟测试（<10s，单卡）。

覆盖：
  1. 包导入与 _C 扩展加载
  2. mini_mega kernel 真实绑定点：mega_launch_empty / fill_weights_lane
  3. 布局/对齐/窗口选型
  4. 权重打包（fp8xfp8 real path，模型形状）
  5. SymmBuffer 单 rank 分配 + 输入视图 + 建档预热
  6. fp8_mega_moe 真实发射（单 rank 数值自洽）
"""
import os
import sys

import torch

import mxmegakernel
from mxmegakernel import (
    get_symm_buffer_for_mega_moe,
    get_token_alignment_for_mega_moe,
    select_fp8_mega_moe_window,
    transform_weights_for_mega_moe,
    fp8_mega_moe,
)

H = 6144
I = 2048
EPR = 8

print("python", sys.version.split()[0], "torch", torch.__version__)
print("mxmegakernel:", mxmegakernel.__file__)
print("_C:", mxmegakernel._C.__file__)
print("bridge symbols:", mxmegakernel._C.mega_bridge_symbols())

# ---- 2. mini_mega kernel 真实绑定点（需要 GPU）----
if torch.cuda.is_available():
    dev = torch.device("cuda")
    mxmegakernel._C.mega_launch_empty(1, 32, 0)
    torch.cuda.synchronize()
    print("[PASS] mega_launch_empty 发射并同步成功")

    n_rows, k_cols = 32, 32
    w = torch.zeros((n_rows // 32) * (k_cols // 32) * 1024, dtype=torch.uint8,
                    device=dev)
    mxmegakernel._C.fill_weights_lane(w, n_rows, k_cols, 7)
    torch.cuda.synchronize()
    print("[PASS] fill_weights_lane 生成权重 tile, nonzero =",
          int((w != 0).sum().item()))
else:
    print("[SKIP] 无 CUDA 设备，跳过 kernel 绑定点测试")

# ---- 3. 对齐 / 窗口 ----
al = get_token_alignment_for_mega_moe()
assert al > 0
win = select_fp8_mega_moe_window(num_tokens=256, num_topk=8, num_ranks=32,
                                 num_experts=256)
print(f"[PASS] token_alignment={al}, window(nt256,R32,E256,topk8)={win}")
assert win in (8, 16, 24, 32, 48, 64, 80)

# ---- 4. 权重打包（模型形状）----
l1 = torch.zeros((EPR, 2 * I, H), dtype=torch.float8_e4m3fn)
l2 = torch.zeros((EPR, H, I), dtype=torch.float8_e4m3fn)
l1_t, l2_t = transform_weights_for_mega_moe(l1, l2)
assert l1_t.shape == l1.shape and l2_t.shape == l2.shape
assert l1_t.dtype == torch.float8_e4m3fn
print("[PASS] transform_weights_for_mega_moe", tuple(l1_t.shape), l1_t.dtype)

# ---- 5/6. SymmBuffer + 真实发射 ----
if torch.cuda.is_available():
    dev = torch.device("cuda")
    buf = get_symm_buffer_for_mega_moe(
        None, num_experts=EPR, num_max_tokens_per_rank=16, num_topk=8,
        hidden=H, intermediate_hidden=I, mma_type="fp8xfp8")
    print("[PASS] SymmBuffer win=", buf.expert_window,
          "x=", tuple(buf.x.shape), buf.x.dtype,
          "topk_idx=", tuple(buf.topk_idx.shape), buf.topk_idx.dtype)
    assert (buf.x.untyped_storage().data_ptr()
            == buf.buffer.untyped_storage().data_ptr())
    T = 8
    buf.x[:T].fill_(1)
    buf.topk_idx[:T].zero_()
    buf.topk_weights[:T].fill_(1.0)
    torch.cuda.synchronize()
    l1_t, l2_t = transform_weights_for_mega_moe(
        torch.zeros((EPR, 2 * I, H), dtype=torch.float8_e4m3fn, device=dev),
        torch.zeros((EPR, H, I), dtype=torch.float8_e4m3fn, device=dev))
    y = torch.empty((T, H), dtype=torch.bfloat16, device=dev)
    fp8_mega_moe(y, l1_t, l2_t, buf, activation="swiglu")
    torch.cuda.synchronize()
    assert torch.isfinite(y.float()).all().item()
    print("[PASS] fp8_mega_moe 真实发射, y", tuple(y.shape), y.dtype,
          "finite")
else:
    print("[SKIP] 无 CUDA 设备，跳过 SymmBuffer/发射测试")

print("SMOKE OK")
