import shutil
import sys

path = "/home/m01182/sglang/python/sglang/srt/layers/moe/mega_moe.py"
shutil.copy2(path, path + ".bak_minimega")
src = open(path).read()

old = """        w13_raw = (w13_fp32 / w13_scale).to(torch.float8_e4m3fn)
        w2_raw = (w2_fp32 / w2_scale).to(torch.float8_e4m3fn)
        experts.mega_l1_weights, experts.mega_l2_weights = (
            transform_weights_for_mega_moe(w13_raw, w2_raw)
        )
        experts.mega_fp8_scales = (float(w13_scale), float(w2_scale))
        experts._mega_moe_weights_built = True
"""

new = """        w13_raw = (w13_fp32 / w13_scale).to(torch.float8_e4m3fn)
        w2_raw = (w2_fp32 / w2_scale).to(torch.float8_e4m3fn)
        _l1_mega, _l2_mega = transform_weights_for_mega_moe(w13_raw, w2_raw)
        del w13_fp32, w2_fp32, w13_raw, w2_raw
        # [minimega] 原位接管：释放 block-quantized 原始权重，避免 2x MoE 权重
        # 常驻（否则 KV cache 无空间）。fallback 路径若被触发将不再可用。
        experts.w13_weight.data = _l1_mega
        experts.w2_weight.data = _l2_mega
        experts.mega_l1_weights = experts.w13_weight.data
        experts.mega_l2_weights = experts.w2_weight.data
        experts.mega_fp8_scales = (float(w13_scale), float(w2_scale))
        experts._mega_moe_weights_built = True
"""

if old not in src:
    print("PATCH_ANCHOR_NOT_FOUND")
    sys.exit(1)
open(path, "w").write(src.replace(old, new, 1))
print("PATCH_OK")
