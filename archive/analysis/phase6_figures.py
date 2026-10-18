"""Phase 6 figures: TC/TD vs baselines + factorial FP/Dice decomposition.

Usage:
    python analysis/phase6_figures.py --eval results/phase6_eval.json \
        --stats results/phase6_stats.json --stats-full results/phase6_stats_full.json \
        --logs checkpoints_phase6

Outputs: kaggle/figures/fig_p6_*.png
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "kaggle", "figures")


def fig_p6_final_metrics(eval_json, stats_json):
    with open(eval_json) as f:
        ev = json.load(f)
    with open(stats_json) as f:
        st = json.load(f)

    s = ev["summary"]
    names = ["TC_base_T00", "TC", "TD_base_T0N", "TD"]
    labels = ["T00\n(FB,orig)", "TC\n(FB,asym)", "T0N\n(noFB,orig)", "TD\n(noFB,asym)"]
    dice = [s[n]["final_dice"] for n in names]
    fp = [s[n]["final_fp_rate"] * 100 for n in names]
    fn = [s[n]["final_fn_rate"] * 100 for n in names]

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#94a3b8", "#2563eb", "#94a3b8", "#16a34a"]
    ax[0].bar(labels, dice, color=colors); ax[0].set_title("Dice (iter4)")
    ax[0].set_ylim(0, 0.5); ax[0].tick_params(axis="x", rotation=15)
    ax[1].bar(labels, fp, color=colors); ax[1].set_title("FP rate %")
    ax[1].tick_params(axis="x", rotation=15)
    ax[2].bar(labels, fn, color=colors); ax[2].set_title("FN rate %")
    ax[2].tick_params(axis="x", rotation=15)
    fig.suptitle("Phase 6: Tversky-asym loss x feedback 2x2 (seed43, matched baselines)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_p6_final_metrics.png")
    fig.savefig(out, dpi=150)
    print("saved", out)

    # paired FP delta per-image for TD vs T0N (the significant finding)
    per = ev["per_image"]
    td_fp = np.array([r["fp_rate"] for r in per["TD"]])
    t0n_fp = np.array([r["fp_rate"] for r in per["TD_base_T0N"]])
    dfp = td_fp - t0n_fp
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(dfp, bins=15, color="#16a34a", alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(dfp.mean(), color="#dc2626", lw=2, label=f"mean {dfp.mean()*100:+.2f}pp")
    ax.set_title("TD vs T0N: per-image FP delta (n_pos_fp=27/40, p=0.005)")
    ax.set_xlabel("FP rate delta"); ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_p6_paired_fp_delta_TD.png")
    fig.savefig(out, dpi=150)
    print("saved", out)

    # paired dice delta per-image for TC vs T00
    tc_d = np.array([r["dice"] for r in per["TC"]])
    t00_d = np.array([r["dice"] for r in per["TC_base_T00"]])
    dd = tc_d - t00_d
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.hist(dd, bins=15, color="#6366f1", alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(dd.mean(), color="#dc2626", lw=2, label=f"mean {dd.mean():+.3f}")
    ax.set_title("TC vs T00: per-image Dice delta (n_pos=21/40, p=0.180)")
    ax.set_xlabel("Dice delta"); ax.legend()
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_p6_paired_dice_delta_TC.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


def fig_p6_factorial(stats_full_json):
    with open(stats_full_json) as f:
        sf = json.load(f)
    fac_fp = sf["factorial_fp"]
    fac_dice = sf["factorial_dice"]

    labels_fp = ["loss", "feedback", "FBxloss"]
    fp_vals = [fac_fp["main_effect_loss_fp_pp"], fac_fp["main_effect_feedback_fp_pp"],
               fac_fp["interaction_loss_x_feedback_fp_pp"]]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(labels_fp, fp_vals, color=["#16a34a", "#94a3b8", "#f59e0b"])
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_title("2x2 main effects + interaction on FP (pp)")
    ax[0].set_ylabel("FP delta (pp)")

    labels_d = ["loss", "feedback", "FBxloss"]
    d_vals = [fac_dice["main_effect_loss_dice"], fac_dice["main_effect_feedback_dice"],
              fac_dice["interaction_loss_x_feedback_dice"]]
    ax[1].bar(labels_d, d_vals, color=["#2563eb", "#94a3b8", "#f59e0b"])
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_title("2x2 main effects + interaction on Dice")
    ax[1].set_ylabel("Dice delta")

    fig.suptitle("Factorial decomposition (per-image): asym loss only cuts FP without feedback")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_p6_factorial_effects.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


def fig_p6_train_curves(log_dir):
    import glob
    import pandas as pd
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, ["bin_val_dice", "bin_val_fpr", "bin_val_prec"]):
        for csvf in sorted(glob.glob(os.path.join(log_dir, "train_log_*.csv"))):
            name = os.path.basename(csvf).replace("train_log_", "").replace(".csv", "")
            df = pd.read_csv(csvf)
            ax.plot(df["epoch"], df[metric], label=name)
        ax.set_xlabel("epoch"); ax.set_ylabel(metric)
        ax.legend()
    axes[0].set_title("Binary val-Dice per epoch (collapse would show here, not loss)")
    axes[1].set_title("Binary val-FPR per epoch")
    axes[2].set_title("Binary val-Precision per epoch")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_p6_train_curves.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=str, default="results/phase6_eval.json")
    parser.add_argument("--stats", type=str, default="results/phase6_stats.json")
    parser.add_argument("--stats-full", type=str, default="results/phase6_stats_full.json")
    parser.add_argument("--logs", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    if os.path.exists(args.eval):
        fig_p6_final_metrics(args.eval, args.stats)
    if os.path.exists(args.stats_full):
        fig_p6_factorial(args.stats_full)
    if args.logs and os.path.isdir(args.logs):
        fig_p6_train_curves(args.logs)


if __name__ == "__main__":
    main()