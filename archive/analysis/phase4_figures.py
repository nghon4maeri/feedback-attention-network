"""Phase 4 figures: trained T00 vs T10 (valid cells only)."""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import csv

plt.rcParams.update({
    "font.size": 10, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 160, "axes.spines.top": False, "axes.spines.right": False,
})

FIGS = "kaggle/figures"
import os
os.makedirs(FIGS, exist_ok=True)

# ---- Fig 1: training curves from train_log CSVs ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, metric in [(axes[0], "train_loss"), (axes[1], "val_loss")]:
    for c, lab, col in [("T00", "Binary (baseline)", "#4C72B0"),
                        ("T10", "STE fmask", "#C44E52")]:
        with open(f"kaggle/outputs_phase4/checkpoints/train_log_{c}.csv") as f:
            rows = list(csv.DictReader(f))
        eps = [int(r["epoch"]) for r in rows]
        vals = [float(r[metric]) for r in rows]
        ax.plot(eps, vals, label=lab, color=col, linewidth=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.legend(fontsize=8)
fig.suptitle("Phase 4 training curves (200 epochs, batch=2, seed 42)")
fig.tight_layout()
fig.savefig(f"{FIGS}/fig_p4_train_curves.png")
plt.close(fig)

# ---- Fig 2: final dice per iteration (valid cells) ----
with open("results/phase4_eval.json") as f:
    data = json.load(f)

fig, ax = plt.subplots(figsize=(8, 4.8))
for c, lab, col in [("T00", "Binary trained", "#4C72B0"),
                    ("T10", "STE trained", "#C44E52")]:
    d = data["summary"][c]["dice_mean_iter"]
    ax.plot(range(1, len(d) + 1), d, marker="o", label=lab, color=col, linewidth=2)
ax.set_xlabel("Iteration")
ax.set_ylabel("Mean Dice")
ax.set_title("Trained models - iterative refinement on 40 val images")
ax.legend()
fig.tight_layout()
fig.savefig(f"{FIGS}/fig_p4_dice_iters.png")
plt.close(fig)

# ---- Fig 3: per-image paired delta (binary - ste) ----
dice_b = np.array([r["dice"] for r in data["per_image"]["T00"]])
dice_s = np.array([r["dice"] for r in data["per_image"]["T10"]])
delta = dice_b - dice_s
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(range(len(delta)), delta, color=["#2ca02c" if d > 0 else "#d62728" for d in delta])
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xlabel("Image index (sorted)")
ax.set_ylabel("Delta Dice (binary - STE)")
ax.set_title("Per-image paired delta: binary trained better on 20/40 images")
fig.tight_layout()
fig.savefig(f"{FIGS}/fig_p4_paired_delta.png")
plt.close(fig)

print("Saved figures:", [f for f in os.listdir(FIGS) if f.startswith("fig_p4")])