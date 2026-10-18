"""Phase 5 figures: X4 STE diagnosis + (post-run) Gate 0/1 eval curves.

Run 1 (local, CPU, X4 results only):
    python analysis/phase5_figures.py --x4 results/x4_ste_diagnosis.json

Run 2 (after Kaggle eval):
    python analysis/phase5_figures.py --eval results/phase5_eval.json \
        --stats results/phase5_stats.json --logs checkpoints_phase5

Outputs: kaggle/figures/fig_x4_*.png, fig_p5_*.png
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


def fig_x4_bn_diff(x4_json):
    rows = x4_json.get("Q3_bn_stats_diff_T00_vs_T10", [])
    layers = [r["layer"] for r in rows if "running_mean" in r["layer"]]
    diffs = [r["mean_abs_diff"] for r in rows if "running_mean" in r["layer"]]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(len(layers)), diffs, width=0.8, color="#2563eb")
    ax.set_yscale("log")
    ax.set_xticks(range(len(layers)))
    ax.set_xticklabels(layers, rotation=90, fontsize=6)
    ax.set_ylabel("|running_mean diff| (log)")
    ax.set_title("BN running-mean difference T00 vs T10 (STE)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_x4_bn_diff.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


def fig_x4_grad_variance(x4_json):
    keys = [k for k in x4_json if k.startswith("Q1_grad_")]
    labels = [k.replace("Q1_grad_", "") for k in keys]
    norm = [x4_json[k]["fmask_grad_norm_mean"] for k in keys]
    var = [x4_json[k]["fmask_grad_var_mean"] for k in keys]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    ax[0].bar(labels, norm, color="#dc2626")
    ax[0].set_ylabel("fmask grad norm (mean)")
    ax[0].tick_params(axis="x", rotation=25)
    ax[1].bar(labels, var, color="#16a34a")
    ax[1].set_ylabel("fmask grad variance")
    ax[1].tick_params(axis="x", rotation=25)
    fig.suptitle("Q1 fmask gradient magnitude/variance by gate on frozen weights")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_x4_grad_variance.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


def fig_x4_fmask_corr(x4_json):
    q2 = x4_json.get("Q2_fmask_vs_mfg", {})
    blocks = list(q2["T00"]["corr_with_gt_mfg_mean"].keys())
    c00 = [q2["T00"]["corr_with_gt_mfg_mean"][b] for b in blocks]
    c10 = [q2["T10"]["corr_with_gt_mfg_mean"][b] for b in blocks]
    x = np.arange(len(blocks))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x - 0.2, c00, 0.4, label="T00 (binary, dead fmask)", color="#94a3b8")
    ax.bar(x + 0.2, c10, 0.4, label="T10 (STE, live fmask)", color="#2563eb")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(blocks, rotation=45, fontsize=8)
    ax.set_ylabel("corr(fmask, GT)")
    ax.set_title("Q2 fmask vs GT correlation per MixPool block")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_x4_fmask_corr.png")
    fig.savefig(out, dpi=150)
    print("saved", out)


def fig_p5_eval_curves(eval_json, stats_json, log_dir):
    with open(eval_json) as f:
        ev = json.load(f)
    with open(stats_json) as f:
        st = json.load(f)

    cells = [c for c in ev["summary"]]
    names = list(ev["summary"].keys())
    dice = [ev["summary"][c]["final_dice"] for c in names]
    fp = [ev["summary"][c]["final_fp_rate"] * 100 for c in names]
    fn = [ev["summary"][c]["final_fn_rate"] * 100 for c in names]

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    ax[0].bar(names, dice, color="#2563eb"); ax[0].set_title("Dice (iter4)")
    ax[0].set_ylim(0, 0.5)
    ax[1].bar(names, fp, color="#dc2626"); ax[1].set_title("FP rate %")
    ax[2].bar(names, fn, color="#16a34a"); ax[2].set_title("FN rate %")
    fig.suptitle("Phase 5 Gate 0/1 final metrics vs T00 (disk seed42)")
    fig.tight_layout()
    out = os.path.join(FIG_DIR, "fig_p5_final_metrics.png")
    fig.savefig(out, dpi=150)
    print("saved", out)

    # per-image paired delta dice for each non-T00 cell
    per = ev["per_image"]
    t00_d = np.array([r["dice"] for r in per["T00"]])
    for c in names:
        if c == "T00":
            continue
        dd = np.array([r["dice"] for r in per[c]]) - t00_d
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(dd, bins=15, color="#6366f1", alpha=0.8)
        ax.axvline(0, color="k", lw=1)
        ax.axvline(dd.mean(), color="#dc2626", lw=2, label=f"mean {dd.mean():+.3f}")
        ax.set_title(f"{c} vs T00: per-image Dice delta (n_pos={st[c]['n_pos_dice']}/40)")
        ax.set_xlabel("Dice delta"); ax.legend()
        fig.tight_layout()
        out = os.path.join(FIG_DIR, f"fig_p5_paired_delta_{c}.png")
        fig.savefig(out, dpi=150)
        print("saved", out)

    # training curves from logs
    if log_dir and os.path.isdir(log_dir):
        import glob
        import pandas as pd
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for csvf in sorted(glob.glob(os.path.join(log_dir, "train_log_*.csv"))):
            name = os.path.basename(csvf).replace("train_log_", "").replace(".csv", "")
            df = pd.read_csv(csvf)
            ax.plot(df["epoch"], df["val_dice"], label=name)
        ax.set_xlabel("epoch"); ax.set_ylabel("val soft-Dice")
        ax.set_title("Training val-Dice (loss-scale independent)")
        ax.legend()
        fig.tight_layout()
        out = os.path.join(FIG_DIR, "fig_p5_train_curves.png")
        fig.savefig(out, dpi=150)
        print("saved", out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--x4", type=str, default="results/x4_ste_diagnosis.json")
    parser.add_argument("--eval", type=str, default=None)
    parser.add_argument("--stats", type=str, default=None)
    parser.add_argument("--logs", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(FIG_DIR, exist_ok=True)

    if args.x4 and os.path.exists(args.x4):
        with open(args.x4) as f:
            x4 = json.load(f)
        fig_x4_bn_diff(x4)
        fig_x4_grad_variance(x4)
        fig_x4_fmask_corr(x4)

    if args.eval and os.path.exists(args.eval):
        fig_p5_eval_curves(args.eval, args.stats, args.logs)


if __name__ == "__main__":
    main()