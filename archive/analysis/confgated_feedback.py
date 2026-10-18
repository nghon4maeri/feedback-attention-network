"""E4: confidence-gated feedback variants.

Motivation (from E3 findings):
  - Hard threshold feeds back pixels with 48.7% error rate (conf<0.1) at the
    same weight as pixels with 2.7% error rate (conf>0.35).
  - Feedback hurts (46/120 cases) mostly when the previous mask is bad.
  - Hypothesis: gating / weighting feedback by confidence reduces the hurt
    cases while keeping the help.

Variants:
    binary       - baseline FANet: pred > 0.5
    gated-tau    - only feed back pixels with |p-0.5| > tau, else 0
                   (tau swept: 0.05, 0.1, 0.2, 0.3)
    conf-weighted - binary mask scaled by 2*|p-0.5| (soft confidence weight)
    soft         - raw sigmoid (reference from E3)
    none         - no feedback (reference)
    oracle       - ground truth (upper bound)

Metrics: mean dice per iteration, help/hurt vs none, delta distribution.

Usage:
    python analysis/confgated_feedback.py [--config configs/kvasir_sessile.yaml]
                                           [--checkpoint checkpoints/checkpoint.pth]
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

TAUS = [0.05, 0.10, 0.20, 0.30]


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
    c = np.abs(p_soft - 0.5)
    if variant == "binary":
        return (p_soft > 0.5).astype(np.float32)
    if variant == "soft":
        return p_soft
    if variant == "none":
        return np.zeros_like(p_soft)
    if variant == "oracle":
        return gt
    if variant == "conf-weighted":
        return (p_soft > 0.5).astype(np.float32) * (2.0 * c)
    if variant.startswith("gated-"):
        tau = float(variant.split("-")[1])
        fb = (p_soft > 0.5).astype(np.float32)
        fb[c < tau] = 0.0
        return fb
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

    VARIANTS = (["binary", "conf-weighted", "soft", "none", "oracle"]
                + [f"gated-{t}" for t in TAUS])

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

    print("\n=== Mean dice per iteration per variant ===")
    print(f"{'iter':>4} " + " ".join(f"{v:>13}" for v in VARIANTS))
    for t in range(num_iter):
        row = " ".join(f"{dice[v][t].mean():>13.4f}" for v in VARIANTS)
        print(f"{t+1:>4} {row}")

    # ---------- help/hurt vs none at final iteration ----------
    print("\n=== Help/hurt vs none (final iteration) ===")
    rows = {}
    for v in VARIANTS:
        if v in ("none", "oracle"):
            continue
        deltas = dice[v][num_iter - 1] - dice["none"][num_iter - 1]
        helps = (deltas > 1e-4).sum()
        hurts = (deltas < -1e-4).sum()
        rows[v] = {
            "help": int(helps), "hurt": int(hurts),
            "neutral": int(n - helps - hurts),
            "mean_delta": float(deltas.mean()),
            "median_delta": float(np.median(deltas)),
            "final_dice": float(dice[v][num_iter - 1].mean()),
        }
        print(f"  {v:>13}: help {helps:>2} | hurt {hurts:>2} | neutral {n-helps-hurts:>2} "
              f"| mean delta {deltas.mean():+.4f} | dice {dice[v][num_iter-1].mean():.4f}")

    # ---------- gate fraction stats (how much feedback is kept) ----------
    print("\n=== Gated feedback: fraction of foreground pixels kept ===")
    for v in [f"gated-{t}" for t in TAUS] + ["conf-weighted", "binary"]:
        fracs = []
        for t in range(num_iter):
            for i in range(n):
                p = soft_preds[v][t][i]
                fb = next_feedback(v, p, gts[i])
                bin_fb = (p > 0.5).astype(np.float32)
                if bin_fb.sum() > 0:
                    fracs.append(fb.sum() / bin_fb.sum())
        print(f"  {v:>13}: mean kept {np.mean(fracs):.3f} of binary foreground")

    # ---------- save ----------
    os.makedirs(results_dir, exist_ok=True)
    summary = {
        "variant_mean_dice": {v: [float(dice[v][t].mean()) for t in range(num_iter)]
                              for v in VARIANTS},
        "help_hurt_final": rows,
    }
    with open(f"{results_dir}/confgated_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(f"{results_dir}/confgated_variants.csv", "w") as f:
        f.write("variant," + ",".join(f"iter{t+1}" for t in range(num_iter)) + "\n")
        for v in VARIANTS:
            f.write(f"{v}," + ",".join(f"{dice[v][t].mean():.4f}" for t in range(num_iter)) + "\n")

    print(f"\nSaved: {results_dir}/confgated_summary.json, confgated_variants.csv")


if __name__ == "__main__":
    main()
