import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIGS = "kaggle/figures"
import os

os.makedirs(FIGS, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.dpi": 150,
})

# ============================================================
# Figure 1: mean dice per iteration per variant
# ============================================================
with open("results/confgated_summary.json") as f:
    s = json.load(f)

variants = ["binary", "conf-weighted", "gated-0.05", "gated-0.1",
            "gated-0.2", "soft", "none", "oracle"]
labels = ["Binary (FANet)", "Conf-weighted", "Gated tau=0.05", "Gated tau=0.10",
          "Gated tau=0.20", "Soft", "None", "Oracle (GT)"]
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
          "#9467bd", "#8c564b", "#7f7f7f", "#e377c2"]
markers = ["o", "s", "^", "v", "<", ">", "D", "*"]

fig, ax = plt.subplots(figsize=(9, 5.5))
for v, lab, c, mk in zip(variants, labels, colors, markers):
    d = s["variant_mean_dice"][v]
    ax.plot(range(1, len(d) + 1), d, marker=mk, label=lab, color=c, linewidth=2)
ax.set_xlabel("Iteration")
ax.set_ylabel("Mean Dice")
ax.set_title("Feedback variants - mean Dice over iterative refinement (40 val images)")
ax.legend(fontsize=9, ncol=2)
ax.set_xticks(range(1, 5))
fig.tight_layout()
fig.savefig(f"{FIGS}/fig1_variant_dice.png")
plt.close(fig)

# ============================================================
# Figure 2: help/hurt bar chart
# ============================================================
hh = s["help_hurt_final"]
names = ["binary", "conf-weighted", "gated-0.05", "gated-0.1",
         "gated-0.2", "soft"]
short = ["Binary", "Conf-w", "G0.05", "G0.10", "G0.20", "Soft"]
helps = [hh[v]["help"] for v in names]
hurts = [hh[v]["hurt"] for v in names]
neutrals = [hh[v]["neutral"] for v in names]

x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x, helps, width=0.6, label="Help", color="#2ca02c")
ax.bar(x, hurts, width=0.6, bottom=helps, label="Hurt", color="#d62728")
ax.bar(x, neutrals, width=0.6, bottom=np.array(helps) + np.array(hurts),
       label="Neutral", color="#bcbd22")
ax.set_xticks(x)
ax.set_xticklabels(short)
ax.set_ylabel("Number of (image, iter) pairs")
ax.set_title("Help / hurt vs no-feedback (final iteration, 40 images)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGS}/fig2_help_hurt.png")
plt.close(fig)

# ============================================================
# Figure 3: uncertainty vs error rate
# ============================================================
bins = pd.read_csv("results/feedback_uncertainty_bins.csv")
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(bins["conf_center"], bins["err_rate"] * 100, "o-", color="#1f77b4",
        linewidth=2, label="Total error rate")
ax.plot(bins["conf_center"], bins["fp_rate"] * 100, "s--", color="#d62728",
        linewidth=2, label="False positive rate")
ax.plot(bins["conf_center"], bins["fn_rate"] * 100, "^--", color="#2ca02c",
        linewidth=2, label="False negative rate")
ax.set_xlabel("Pre-threshold confidence |p - 0.5|")
ax.set_ylabel("Error rate (%)")
ax.set_title("Error rate of binarized prediction vs confidence (48.7% at low conf vs 2.7% at high conf)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGS}/fig3_uncertainty_error.png")
plt.close(fig)

# ============================================================
# Figure 4: 10-iteration refinement baseline
# ============================================================
tr = pd.read_csv("results/test_results.csv")
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(tr["Iteration"], tr["F1"], "o-", color="#1f77b4", linewidth=2, label="F1 (Dice)")
ax.plot(tr["Iteration"], tr["Jaccard"], "s-", color="#ff7f0e", linewidth=2, label="Jaccard (IoU)")
ax.plot(tr["Iteration"], tr["Recall"], "^-", color="#2ca02c", linewidth=2, label="Recall")
ax.plot(tr["Iteration"], tr["Precision"], "v-", color="#d62728", linewidth=2, label="Precision")
ax.set_xlabel("Iteration")
ax.set_ylabel("Score")
ax.set_title("Test-time refinement on 200-epoch checkpoint (40 val images)")
ax.legend()
ax.set_xticks(range(1, 11))
fig.tight_layout()
fig.savefig(f"{FIGS}/fig4_refinement_10iter.png")
plt.close(fig)

# ============================================================
# Figure 5: feedback kept fraction
# ============================================================
kept = {"Gated 0.05": 0.854, "Gated 0.10": 0.715, "Gated 0.20": 0.368,
        "Gated 0.30": 0.028, "Conf-weighted": 0.301, "Binary": 1.000}
fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(kept.keys(), kept.values(), color="#17becf")
for b, v in zip(bars, kept.values()):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
ax.set_ylabel("Fraction of binary foreground kept")
ax.set_title("How much feedback each variant retains")
fig.tight_layout()
fig.savefig(f"{FIGS}/fig5_kept_fraction.png")
plt.close(fig)

print("Saved figures:")
for f in sorted(os.listdir(FIGS)):
    print(" ", f)
