#!/usr/bin/env python3
# test/93.r19_l2_bench/plot.py — L2 场景图
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = os.environ.get("TAG", "261011_1630")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG, exist_ok=True)

# 单 block, K=I=2048, N=1536(6组), us
Ms = [16, 24, 32, 33, 48, 64, 96, 128]
variants = {
    "2,1,3": [80, 82, 82, 158, 159, 163, 244, 324],
    "3,1,3": [87, 90, 90, 92, 93, 178, 185, 272],
    "4,1,3": [98, 103, 103, 105, 106, 113, 213, 223],
    "4,2,3": [87, 92, 92, 96, 98, 106, 196, 210],
    "6,1,3": [132, 135, 136, 139, 140, 146, 157, 287],
}
Ns = [256, 512, 768, 1536, 3072, 6144]
nus = [32, 44, 56, 112, 222, 440]

# Fig1 M-scan
plt.figure(figsize=(8, 5))
for k, v in variants.items():
    plt.plot(Ms, v, marker="o", label=f"r19_l2<{k}>")
env = [min(variants[k][i] for k in variants) for i in range(len(Ms))]
plt.plot(Ms, env, "k--", lw=2, label="best envelope")
for x, y in zip(Ms, env):
    plt.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 7),
                 ha="center", fontsize=8)
for x in (32, 48, 64, 96):
    plt.axvline(x, color="grey", ls=":", lw=1)
plt.xlabel("M (rows)")
plt.ylabel("us / launch (single block)")
plt.title("r19_l2_big M-scan (1x512, K=I=2048, N=1536)")
plt.grid(alpha=0.3); plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(os.path.join(FIG, f"{TAG}_l2_m_scan.png"), dpi=130); plt.close()

# Fig2 N-scaling (L2 vs L1 normalized per column)
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(Ns, nus, "o-", color="C0", label="L2 us/launch (M=64)")
ax1.set_xscale("log", base=2); ax1.set_xticks(Ns); ax1.set_xticklabels(map(str, Ns))
ax1.set_xlabel("N (columns)"); ax1.set_ylabel("us / launch", color="C0")
ax1.tick_params(axis="y", labelcolor="C0"); ax1.grid(alpha=0.3)
for x, y in zip(Ns, nus):
    ax1.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 7),
                 ha="center", fontsize=8, color="C0")
ax2 = ax1.twinx()
per = [1000.0*u/n for u, n in zip(nus, Ns)]
ax2.plot(Ns, per, "s--", color="C1", label="L2 ns/column")
ax2.set_ylabel("ns / column", color="C1"); ax2.tick_params(axis="y", labelcolor="C1")
ax1.axhline(0, alpha=0)
plt.title("r19_l2_big N-scaling (M=64)  vs L1 ~0.14 us/col")
fig.tight_layout(); plt.savefig(os.path.join(FIG, f"{TAG}_l2_n_scale.png"), dpi=130)
plt.close()

# Fig3 small-M: l2_gemm_body vs r19_l2_big
labels = ["M=16", "M=32"]
body = [46, 45]
r19 = [80, 82]
x = range(len(labels)); w = 0.36
plt.figure(figsize=(7, 5))
plt.bar([i-w/2 for i in x], body, w, label="l2_gemm_body<PF2,MT1,B6>", color="C2")
plt.bar([i+w/2 for i in x], r19, w, label="r19_l2_big<2,1,3>", color="C0")
for i, (b, r) in enumerate(zip(body, r19)):
    plt.text(i-w/2, b+1, str(b), ha="center", fontsize=9)
    plt.text(i+w/2, r+1, str(r), ha="center", fontsize=9)
plt.xticks(list(x), labels); plt.ylabel("us / launch")
plt.title("Small-M L2: single-tile body vs r19_l2_big")
plt.legend(fontsize=9); plt.grid(axis="y", alpha=0.3); plt.tight_layout()
plt.savefig(os.path.join(FIG, f"{TAG}_l2_small_vs_big.png"), dpi=130); plt.close()
print("figures written to", FIG)
