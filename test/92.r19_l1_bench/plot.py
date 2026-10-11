#!/usr/bin/env python3
# test/92.r19_l1_bench/plot.py — 由 92 测得的数据生成图文文档用图
# 依赖 matplotlib；输出 docs/figures/<tag>_r19_*.png
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TAG = os.environ.get("TAG", "261011_1600")
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FIG = os.path.join(ROOT, "docs", "figures")
os.makedirs(FIG, exist_ok=True)

# ---- 数据（单 block 1x512, K=H=6144, N=4096, block_n=256, us/launch）----
Ms = [16, 17, 24, 32, 33, 40, 48, 64]
variants = {
    "2,2,4": [121, 128, 129, 129, 255, 255, 255, 255],
    "2,4,4": [105, 115, 117, 118, 228, 230, 230, 234],
    "3,2,4": [135, 140, 142, 142, 151, 152, 153, 297],
    "3,4,4": [123, 129, 132, 133, 142, 145, 146, 278],
    "4,2,4": [155, 160, 163, 164, 170, 173, 173, 186],
}
Ns = [128, 256, 512, 1024, 2048, 4096]
nus = [69, 106, 120, 147, 290, 575]

# ---- Fig 1: M 扫描 ----
plt.figure(figsize=(8, 5))
for k, v in variants.items():
    plt.plot(Ms, v, marker="o", label=f"r19<{k}>")
env = [min(variants[k][i] for k in variants) for i in range(len(Ms))]
plt.plot(Ms, env, "k--", lw=2, label="best envelope")
for i, m in enumerate(Ms):
    plt.annotate(str(env[i]), (m, env[i]), textcoords="offset points",
                 xytext=(0, 7), ha="center", fontsize=8)
plt.axvline(16, color="grey", ls=":", lw=1)
plt.axvline(32, color="grey", ls=":", lw=1)
plt.text(16, plt.ylim()[1]*0.98, " 16", color="grey", va="top", fontsize=8)
plt.text(32, plt.ylim()[1]*0.98, " 32", color="grey", va="top", fontsize=8)
plt.xlabel("M (rows)")
plt.ylabel("us / launch (single block)")
plt.title("r19_l1_big M-scan (single block, 1x512, K=6144, N=4096)")
plt.grid(alpha=0.3)
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, f"{TAG}_r19_m_scan.png"), dpi=130)
plt.close()

# ---- Fig 2: N 缩放（延迟 + 单位列成本）----
fig, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(Ns, nus, "o-", color="C0", label="us/launch")
ax1.set_xscale("log", base=2)
ax1.set_xticks(Ns)
ax1.set_xticklabels([str(n) for n in Ns])
ax1.set_xlabel("N (columns)")
ax1.set_ylabel("us / launch", color="C0")
ax1.tick_params(axis="y", labelcolor="C0")
ax1.grid(alpha=0.3)
for x, y in zip(Ns, nus):
    ax1.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 7),
                 ha="center", fontsize=8, color="C0")
ax2 = ax1.twinx()
per = [1000.0*u/n for u, n in zip(nus, Ns)]
ax2.plot(Ns, per, "s--", color="C1", label="ns / column")
ax2.set_ylabel("ns / column", color="C1")
ax2.tick_params(axis="y", labelcolor="C1")
plt.title("r19_l1_big N-scaling (single block, M=48, MT3/PF2)")
fig.tight_layout()
plt.savefig(os.path.join(FIG, f"{TAG}_r19_n_scale.png"), dpi=130)
plt.close()

# ---- Fig 3: tiling 结构示意 ----
fig, ax = plt.subplots(figsize=(9, 5.6))
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
ax.text(50, 59, "r19_l1_big<kMT,kPF,kBatch>  GEMM = M x K x N   "
                 "(M=count, K=H=6144, N=ngroups*256)", ha="center", fontsize=10)
ax.annotate("", xy=(88, 12), xytext=(10, 12),
            arrowprops=dict(arrowstyle="->"))
ax.text(90, 12, "K = 6144\n(192 x 32)", va="center", fontsize=9)
ax.annotate("", xy=(10, 52), xytext=(10, 12),
            arrowprops=dict(arrowstyle="->"))
ax.text(10, 54, "M = count", ha="left", fontsize=9)
mt = 3
for m in range(mt):
    y = 14 + m*11
    ax.add_patch(plt.Rectangle((12, y), 70, 9, fill=False, ec="C0"))
    ax.text(11, y+4.5, f"m{m}", ha="right", fontsize=8)
for kk in range(6):
    ax.plot([12+kk*11.5, 12+kk*11.5], [14, 14+mt*11], color="grey", lw=0.4)
ax.text(47, 47.5, "K loop: 192 steps x 32 (kPF-deep prefetch ring)",
        ha="center", fontsize=8)
ax.add_patch(plt.Rectangle((12, 14), 11.5, 11, fc="C0", alpha=0.25))
ax.text(17.7, 19.5, "MMA\n16x16x32", ha="center", va="center", fontsize=7)
ax.add_patch(plt.Rectangle((85, 14), 10, mt*11, fill=False, ec="C2"))
ax.text(90, 14+mt*11+2, "N groups\n(kBatch at a\ntime, 256 each)",
        ha="center", fontsize=8)
ax.text(50, 6, "per g0 iter: M=kMT*16 x N=kBatch*256 x K=32\n"
               "per call   : M=count x N=ngroups*256 x K=6144",
        ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG, f"{TAG}_r19_tiling.png"), dpi=130)
plt.close()
print("figures written to", FIG)
