"""Publication-ready figures for the 2x2 ablation."""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 160,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

with open("results/abl2x2_summary.json") as f:
    data = json.load(f)

s = data["summary"]
cells = list(s.keys())
order = ["C00", "C01-t0.1", "C01-t0.2", "C01-t0.3", "C10", "C11", "neg-ctrl", "oracle"]
labels = ["Binary (baseline)", "Dual t=0.1", "Dual t=0.2", "Dual t=0.3",
          "Soft gate", "Soft+Dual", "Random-bg (control)", "Oracle"]
colors = ["#4C72B0", "#55A868", "#6bce7a", "#88d88f", "#C44E52", "#8172B2", "#7f7f7f", "#CCB974"]

FIGS = "kaggle/figures"
import os
os.makedirs(FIGS, exist_ok=True)

# Fig 1: dice per iteration
fig, ax = plt.subplots(figsize=(8.5, 5))
for c, lab, col in zip(order, labels, colors):
    d = s[c]["dice"]
    ls = "--" if c in ("neg-ctrl", "oracle") else "-"
    ax.plot(range(1, len(d) + 1), d, marker="o", ms=4, label=lab,
            color=col, linestyle=ls, linewidth=1.8 if c != "neg-ctrl" else 1.2)
ax.set_xlabel("Iteration")
ax.set_ylabel("Mean Dice")
ax.set_title("Dual-path feedback vs gating variants (frozen 200ep checkpoint, 40 val images)")
ax.legend(fontsize=8.5, ncol=2)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig_abl_dice_curves.png")
plt.close(fig)

# Fig 2: final-iteration metric comparison (C00 vs C01-t0.1 vs C10)
sel = ["C00", "C01-t0.1", "C10", "oracle"]
x = np.arange(len(sel))
w = 0.22
metrics = [("dice", "Dice"), ("fp_rate", "FP rate"), ("fn_rate", "FN rate")]
fig, ax = plt.subplots(figsize=(8, 4.8))
for j, (k, lab) in enumerate(metrics):
    vals = [s[c][k][-1] * (100 if "rate" in k else 1) for c in sel]
    bars = ax.bar(x + (j - 1) * w, vals, w, label=lab)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.3, f"{v:.1f}",
                ha="center", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(["Baseline", "Dual t=0.1", "Soft gate", "Oracle"], fontsize=9)
ax.set_ylabel("Value (rate = %)")
ax.set_title("Final-iteration comparison: dual-path increases FP (not reduces), soft gate helps most")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig_abl_final_metrics.png")
plt.close(fig)

# Fig 3: oracle gap
gaps = []
for c in order:
    if c == "oracle":
        continue
    gaps.append((c, s["oracle"]["dice"][-1] - s[c]["dice"][-1]))
fig, ax = plt.subplots(figsize=(7.5, 4.5))
names = [g[0] for g in gaps]
vals = [g[1] for g in gaps]
bars = ax.bar(names, vals, color=["#4C72B0", "#55A868", "#6bce7a", "#88d88f",
                                   "#C44E52", "#8172B2", "#7f7f7f"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}",
            ha="center", fontsize=8)
ax.set_ylabel("Dice gap to oracle")
ax.set_title("Oracle gap shrinks from 0.092 (baseline) to 0.055 (soft gate)")
fig.tight_layout()
fig.savefig(f"{FIGS}/fig_abl_oracle_gap.png")
plt.close(fig)

# Fig 4: statistical summary (mean delta with bootstrap CI)
with open("results/abl2x2_stats.json") as f:
    st = json.load(f)
comps = ["C01-t0.1_vs_C00", "C01-t0.2_vs_C00", "C01-t0.3_vs_C00",
         "C10_vs_C00", "C11_vs_C00", "C11_vs_C10", "C11_vs_C01-t0.1"]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
y = np.arange(len(comps))[::-1]
means = [st[c]["mean_delta"] for c in comps]
lo = [st[c]["mean_delta"] - st[c]["ci95"][0] for c in comps]
hi = [st[c]["ci95"][1] - st[c]["mean_delta"] for c in comps]
ax.errorbar(means, y, xerr=[lo, hi], fmt="o", color="#4C72B0",
            capsize=4, label="mean delta (bootstrap 95% CI)")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels([c.replace("_", " ") for c in comps], fontsize=8.5)
ax.set_xlabel("Delta Dice (final iteration)")
ax.set_title("Paired per-image deltas: positive but not significant after Bonferroni")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{FIGS}/fig_abl_stats.png")
plt.close(fig)

print("Saved figures:")
for f in sorted(os.listdir(FIGS)):
    print(" ", f)
