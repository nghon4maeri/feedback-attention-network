"""Analysis 2 (E3): when does hard binary feedback help or hurt?

Compares feedback variants over iterative test-time refinement:
    binary  - current FANet: hard mask (pred > 0.5)
    soft    - raw sigmoid output fed back (keeps confidence, no threshold)
    none    - zero mask (no feedback at all)
    oracle  - ground-truth mask (upper bound)

Then:
  A. help/hurt: per (image, iteration), delta = dice(binary) - dice(none)
     correlated with the previous prediction's quality / uncertainty.
  B. uncertainty-error: error rate of the binarized prediction in bins of
     pre-threshold confidence |p - 0.5|.

Usage:
    python analysis/feedback_analysis.py [--config configs/kvasir_sessile.yaml]
                                         [--num-iter 4]
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from fanet.config import load_config
from fanet.data import load_data
from fanet.models import FANet
from fanet.utils import seeding

VARIANTS = ["binary", "soft", "none", "oracle"]


def dice_np(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = (gt & pred).sum()
    return 2 * inter / (gt.sum() + pred.sum() + 1e-15)


def otsu_bin(img, size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (th / 255.0 > 0.5).astype(np.float32)


def load_pair(x_path, y_path, size):
    img = cv2.imread(x_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, size)
    img_t = torch.from_numpy(np.transpose(img, (2, 0, 1)) / 255.0).float().unsqueeze(0)
    gt = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, size) / 255.0
    gt = (gt > 0.5).astype(np.float32)
    return img, img_t, gt


def next_feedback(variant, p_soft, gt):
    if variant == "binary":
        return (p_soft > 0.5).astype(np.float32)
    if variant == "soft":
        return p_soft
    if variant == "none":
        return np.zeros_like(p_soft)
    if variant == "oracle":
        return gt
    raise ValueError(variant)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kvasir_sessile.yaml")
    parser.add_argument("--num-iter", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    size = tuple(cfg["dataset"]["image_size"])
    num_iter = args.num_iter
    ckpt = args.checkpoint or cfg["paths"]["checkpoint"]
    results_dir = cfg["paths"]["results"]

    seeding(cfg["train"]["seed"])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = FANet()
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model = model.to(device)
    model.eval()

    (train_x, train_y), (test_x, test_y) = load_data(cfg["dataset"]["root"])
    n = len(test_x)

    soft_preds = {v: [[None] * n for _ in range(num_iter)] for v in VARIANTS}
    dice = {v: np.zeros((num_iter, n)) for v in VARIANTS}
    gts = [None] * n

    for i, (x_path, y_path) in tqdm(enumerate(zip(test_x, test_y)), total=n):
        img, img_t, gt = load_pair(x_path, y_path, size)
        gts[i] = gt
        otsu = otsu_bin(img, size)
        fb = {v: otsu.copy() for v in VARIANTS}

        for t in range(num_iter):
            for v in VARIANTS:
                m = torch.from_numpy(fb[v]).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    p = torch.sigmoid(model([img_t.to(device), m]))[0, 0].cpu().numpy()
                soft_preds[v][t][i] = p
                dice[v][t, i] = dice_np(gt, p > 0.5)
                fb[v] = next_feedback(v, p, gt)

    print("\n=== Mean dice per iteration per feedback variant ===")
    print(f"{'iter':>4} " + " ".join(f"{v:>8}" for v in VARIANTS))
    for t in range(num_iter):
        row = " ".join(f"{dice[v][t].mean():>8.4f}" for v in VARIANTS)
        print(f"{t+1:>4} {row}")

    # ---------- A. help/hurt: binary vs none ----------
    print("\n=== A. Help/hurt of hard binary feedback (dice(binary) - dice(none)) ===")
    pair_rows = []
    for t in range(1, num_iter):
        for i in range(n):
            gt = gts[i]
            delta = dice["binary"][t, i] - dice["none"][t, i]
            p_prev = soft_preds["binary"][t - 1][i]
            fb_bin = (p_prev > 0.5).astype(np.float32)
            u = 0.5 - np.abs(p_prev - 0.5)
            prev_dice = dice_np(gt, fb_bin)
            err = (fb_bin != gt)
            pair_rows.append({
                "iter": t, "image": i,
                "delta": delta,
                "prev_dice": prev_dice,
                "prev_err_rate": err.mean(),
                "prev_mean_u_all": u.mean(),
                "prev_mean_u_err": u[err].mean() if err.any() else 0.0,
                "prev_mean_u_ok": u[~err].mean() if (~err).any() else 0.0,
            })

    deltas = np.array([r["delta"] for r in pair_rows])
    helps = (deltas > 1e-4).sum()
    hurts = (deltas < -1e-4).sum()
    print(f"pairs (image, iter): {len(deltas)} | help: {helps} | hurt: {hurts} "
          f"| neutral: {len(deltas) - helps - hurts}")
    print(f"mean delta: {deltas.mean():+.4f} | median: {np.median(deltas):+.4f}")

    feats = ["prev_dice", "prev_err_rate", "prev_mean_u_all", "prev_mean_u_err"]
    print("\nCorrelation of delta with previous-feedback features (Pearson):")
    corrs = {}
    for f in feats:
        x = np.array([r[f] for r in pair_rows])
        corrs[f] = float(np.corrcoef(deltas, x)[0, 1])
        print(f"  delta vs {f:>18s}: {corrs[f]:+.3f}")

    # split by median prev_dice
    med = np.median([r["prev_dice"] for r in pair_rows])
    hi = np.array([r["delta"] for r in pair_rows if r["prev_dice"] >= med])
    lo = np.array([r["delta"] for r in pair_rows if r["prev_dice"] < med])
    print(f"\nGrouped by prev mask quality (median {med:.3f}):")
    print(f"  prev_dice >= median: mean delta {hi.mean():+.4f} (n={len(hi)})")
    print(f"  prev_dice <  median: mean delta {lo.mean():+.4f} (n={len(lo)})")

    # ---------- B. uncertainty vs error of the binarized mask ----------
    print("\n=== B. Error rate of binarized prediction vs pre-threshold confidence ===")
    bins = 10
    edges = np.linspace(0.0, 0.5, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    tot_err = np.zeros(bins)
    tot_n = np.zeros(bins)
    tot_fp = np.zeros(bins)
    tot_fn = np.zeros(bins)

    for t in range(num_iter):
        for i in range(n):
            gt = gts[i]
            p = soft_preds["binary"][t][i]
            c = np.abs(p - 0.5)
            b = (p > 0.5).astype(np.float32)
            err = (b != gt)
            fp = ((b == 1) & (gt == 0))
            fn = ((b == 0) & (gt == 1))
            for k in range(bins):
                mask = (c >= edges[k]) & (c < edges[k + 1])
                if k == bins - 1:
                    mask = (c >= edges[k]) & (c <= edges[k + 1])
                tot_n[k] += mask.sum()
                tot_err[k] += err[mask].sum()
                tot_fp[k] += fp[mask].sum()
                tot_fn[k] += fn[mask].sum()

    print(f"{'conf bin':>8} {'n_pixels':>10} {'err_rate':>9} {'fp_rate':>8} {'fn_rate':>8}")
    for k in range(bins):
        rate = tot_err[k] / max(tot_n[k], 1)
        fp_r = tot_fp[k] / max(tot_n[k], 1)
        fn_r = tot_fn[k] / max(tot_n[k], 1)
        print(f"{centers[k]:>8.3f} {int(tot_n[k]):>10} {rate:>9.3f} {fp_r:>8.3f} {fn_r:>8.3f}")

    low_conf = tot_err[:2].sum() / max(tot_n[:2].sum(), 1)
    high_conf = tot_err[-3:].sum() / max(tot_n[-3:].sum(), 1)
    print(f"\nerr rate conf<0.1: {low_conf:.3f} | conf>0.35: {high_conf:.3f}")

    # ---------- save artifacts ----------
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/feedback_pairs.csv", "w") as f:
        keys = list(pair_rows[0].keys())
        f.write(",".join(keys) + "\n")
        for r in pair_rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    with open(f"{results_dir}/feedback_uncertainty_bins.csv", "w") as f:
        f.write("conf_center,n_pixels,err_rate,fp_rate,fn_rate\n")
        for k in range(bins):
            f.write(f"{centers[k]:.3f},{int(tot_n[k])},"
                    f"{tot_err[k]/max(tot_n[k],1):.4f},"
                    f"{tot_fp[k]/max(tot_n[k],1):.4f},"
                    f"{tot_fn[k]/max(tot_n[k],1):.4f}\n")

    summary = {
        "variant_mean_dice": {v: [float(dice[v][t].mean()) for t in range(num_iter)]
                              for v in VARIANTS},
        "help_count": int(helps),
        "hurt_count": int(hurts),
        "mean_delta": float(deltas.mean()),
        "correlations": corrs,
        "grouped_by_prev_dice": {"hi": float(hi.mean()), "lo": float(lo.mean()),
                                 "median": float(med)},
    }
    with open(f"{results_dir}/feedback_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {results_dir}/feedback_pairs.csv, feedback_uncertainty_bins.csv, feedback_summary.json")


if __name__ == "__main__":
    main()
