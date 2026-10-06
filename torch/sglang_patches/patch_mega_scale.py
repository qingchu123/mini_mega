import re
import shutil
import sys

path = "/home/m01182/sglang/python/sglang/srt/layers/moe/mega_moe.py"
shutil.copy2(path, path + ".bak_before_minimega_scale")
src = open(path).read()

# ---- 1) weight build branch: raw cast, only W2 pow2 up-scale (+topk compensation)
old_build = """        w13_fp32 = _mega_moe_dequant_blockwise_fp8(w13, w13_sf_fp32)
        w2_fp32 = _mega_moe_dequant_blockwise_fp8(w2, w2_sf_fp32)
        w13_scale = _mega_moe_raw_fp8_scale(w13_fp32.abs().max(), k1)
        w2_scale = _mega_moe_raw_fp8_scale(w2_fp32.abs().max(), k2)
        w13_raw = (w13_fp32 / w13_scale).to(torch.float8_e4m3fn)
        w2_raw = (w2_fp32 / w2_scale).to(torch.float8_e4m3fn)
        _l1_mega, _l2_mega = transform_weights_for_mega_moe(w13_raw, w2_raw)
        del w13_fp32, w2_fp32, w13_raw, w2_raw
"""
new_build = """        import math as _m
        w13_fp32 = _mega_moe_dequant_blockwise_fp8(w13, w13_sf_fp32)
        w2_fp32 = _mega_moe_dequant_blockwise_fp8(w2, w2_sf_fp32)
        # [minimega] SwiGLU 非线性不允许把 gate 缩放后折回输出，故不做 per-tensor
        # 下缩放；W1/x 直接 raw cast。仅 W2 用 2 的幂上缩放，并把 topk_weights
        # 同比例缩小（REDUCE 阶段精确抵消）。
        _amax2 = float(w2_fp32.abs().max())
        w2_scale = (2.0 ** _m.floor(_m.log2(448.0 / _amax2))) if _amax2 > 0 else 1.0
        w13_raw = w13_fp32.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        w2_raw = (w2_fp32 * w2_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        _l1_mega, _l2_mega = transform_weights_for_mega_moe(w13_raw, w2_raw)
        del w13_fp32, w2_fp32, w13_raw, w2_raw
"""
if old_build not in src:
    print("BUILD_ANCHOR_NOT_FOUND")
    sys.exit(1)
src = src.replace(old_build, new_build, 1)

old_scale = """        experts.mega_fp8_scales = (float(w13_scale), float(w2_scale))
"""
new_scale = """        experts.mega_fp8_scales = (1.0, float(w2_scale))
"""
if old_scale not in src:
    print("SCALE_ANCHOR_NOT_FOUND")
    sys.exit(1)
src = src.replace(old_scale, new_scale, 1)

old_print = """        print(
            "[mega-moe] fp8xfp8 raw E4M3 weights built: "
            f"w13_scale={float(w13_scale):.6g} w2_scale={float(w2_scale):.6g}",
            flush=True,
        )
"""
new_print = """        print(
            "[mega-moe] fp8xfp8 raw E4M3 weights built: "
            f"w2_scale={float(w2_scale):.6g}",
            flush=True,
        )
"""
if old_print not in src:
    print("PRINT_ANCHOR_NOT_FOUND")
    sys.exit(1)
src = src.replace(old_print, new_print, 1)

# ---- 2) forward branch: raw x cast + topk/w2_scale, no output rescale
old_fwd = """        act_scale = None
        if num_tokens > 0:
            act_scale = _mega_moe_raw_fp8_scale(
                hidden_states.detach().float().abs().amax(dim=-1, keepdim=True),
                hidden_size,
            )
            buf.x[:num_tokens].copy_(
                (hidden_states.float() / act_scale).to(torch.float8_e4m3fn)
            )
            buf.topk_idx[:num_tokens].copy_(topk_ids_in)
            buf.topk_weights[:num_tokens].copy_(topk_weights_in)
        w13_scale, w2_scale = getattr(moe.experts, "mega_fp8_scales", (1.0, 1.0))
        deep_gemm.fp8_mega_moe(
            y,
            moe.experts.mega_l1_weights,
            moe.experts.mega_l2_weights,
            buf,
            activation="swiglu",
            activation_clamp=swiglu_limit,
        )
        if num_tokens > 0:
            y[:num_tokens].mul_(act_scale * (float(w13_scale) * float(w2_scale)))
"""
new_fwd = """        _, w2_scale = getattr(moe.experts, "mega_fp8_scales", (1.0, 1.0))
        if num_tokens > 0:
            buf.x[:num_tokens].copy_(
                hidden_states.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            )
            buf.topk_idx[:num_tokens].copy_(topk_ids_in)
            buf.topk_weights[:num_tokens].copy_(
                topk_weights_in / float(w2_scale)
            )
        deep_gemm.fp8_mega_moe(
            y,
            moe.experts.mega_l1_weights,
            moe.experts.mega_l2_weights,
            buf,
            activation="swiglu",
            activation_clamp=swiglu_limit,
        )
"""
if old_fwd not in src:
    print("FWD_ANCHOR_NOT_FOUND")
    sys.exit(1)
src = src.replace(old_fwd, new_fwd, 1)

open(path, "w").write(src)
print("PATCH_OK")
