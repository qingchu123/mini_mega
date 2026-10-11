#!/usr/bin/env python3
# test/94.r19_ksplit_bench/plot.py
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = os.environ.get("TAG", "261011_1845")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG, exist_ok=True)

NK = [1, 2, 4, 8]
full = {
    "M=64":  [402.2, 147.1, 85.9, 56.9],
    "M=128": [544.4, 287.1, 162.2, 106.0],
    "M=256": [1094.1, 572.0, 318.1, 201.6],
}
gemm = {
    "M=64":  [268.6, 140.1, 76.1, 46.5],
    "M=128": [535.8, 275.4, 146.4, 89.1],
    "M=256": [1079.9, 554.5, 291.3, 173.0],
}

# Fig1: time vs NK (full solid, gemm dashed)
plt.figure(figsize=(8.5, 5))
for k in full:
    plt.plot(NK, full[k], "o-", label=f"{k} (GEMM+reduce)")
    plt.plot(NK, gemm[k], "s--", alpha=0.6, label=f"{k} (GEMM only)")
plt.xscale("log", base=2); plt.xticks(NK, [str(n) for n in NK])
plt.xlabel("NK (K-split factor)   [grid = 4 x NK blocks, 4..32]")
plt.ylabel("us / launch")
plt.title("L1 split-K: N=4096(NSLICE=4) K=6144, single window")
plt.grid(alpha=0.3, which="both"); plt.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.savefig(os.path.join(FIG, f"{TAG}_ksplit_time.png"), dpi=130)
plt.close()

# Fig2: speedup vs NK=1 (full)
plt.figure(figsize=(8, 5))
for k, v in full.items():
    plt.plot(NK, [v[0] / x for x in v], "s-", label=f"{k}")
plt.plot(NK, NK, "k--", label="ideal (linear in blocks)")
plt.xscale("log", base=2); plt.yscale("log", base=2)
plt.xticks(NK, [str(n) for n in NK]); plt.yticks([1, 2, 4, 8], ["1", "2", "4", "8"])
plt.xlabel("NK (blocks = 4 x NK)")
plt.ylabel("speedup vs NK=1")
plt.title("L1 split-K speedup (4 -> 32 blocks)")
plt.grid(alpha=0.3, which="both"); plt.legend(fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(FIG, f"{TAG}_ksplit_speedup.png"), dpi=130)
plt.close()
print("figures written to", FIG)
