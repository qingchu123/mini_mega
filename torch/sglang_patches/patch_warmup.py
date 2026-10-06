import sys

path = "/opt/conda/lib/python3.10/site-packages/mxmegakernel/mega/__init__.py"
src = open(path).read()

old = """        T = max(1, min(int(self.num_topk), 16))
        self.topk_idx[:T].fill_(1)
        self.topk_weights[:T].fill_(1.0 / max(1, self.num_topk))
        for _ in range(2):
"""

new = """        T = max(1, min(int(self.num_max_tokens_per_rank), 256))
        idx = (torch.arange(T, device=dev).unsqueeze(1)
               + torch.arange(self.num_topk, device=dev).unsqueeze(0))
        idx = (idx % int(self.num_experts)).to(self.topk_idx.dtype)
        self.topk_idx[:T].copy_(idx)
        self.topk_weights[:T].fill_(1.0 / max(1, self.num_topk))
        for _ in range(2):
"""

if old not in src:
    print("WARMUP_ANCHOR_NOT_FOUND")
    sys.exit(1)
open(path, "w").write(src.replace(old, new, 1))
print("WARMUP_PATCH_OK")
