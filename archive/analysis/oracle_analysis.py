"""Phase 1 (E-B1): what does oracle feedback provide that prediction feedback lacks?

For each (image, iteration) pair, run the model with:
    pred_fb  - feedback = own binarized prediction (current FANet)
    oracle_fb- feedback = ground truth mask (upper bound)

Then compare pixel-wise the two outputs and the two feedback masks:
  A. error decomposition of prediction feedback vs ground truth:
       FP (over-segmentation) vs FN (under-segmentation) - which dominates?
  B. where oracle output differs from pred-feedback output:
       how much of that difference lies in regions where pred feedback had FP?
  C. boundary analysis: error of prediction feedback as function of distance
       to ground-truth boundary (are FP errors near the polyp boundary?)

Hypothesis being tested: prediction feedback over-segments (FP dominates),
so oracle's advantage comes mainly from providing correct BACKGROUND info
-> supports dual-path feedback (fg + bg).

Usage:
    python analysis/oracle_analysis.py [--config configs/kvasir_sessile.yaml]
                                       [--checkpoint checkpoints/checkpoint_200ep_kaggle.pth]
                                       [--num-iter 3]
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

from fanet.config import load_config
from fanet.data import load_data
from fanet.models import FANet
from fanet.utils import seeding


def dice_np(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = (gt & pred).sum()
    return 2 * inter / (gt.sum() + pred.sum() + 1e-15)


def iou_np(gt, pred):
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = (gt & pred).sum()
    union = (gt | pred).sum()
    return inter / (union + 1e-15)


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


def dist_to_boundary(mask):
    """Euclidean distance (px) from each pixel to nearest mask boundary."""
    mask = mask.astype(np.uint8)
    edt_in = ndimage.distance_transform_edt(mask)
    edt_out = ndimage.distance_transform_edt(1 - mask)
    return np.where(mask > 0, edt_in, edt_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kvasir_sessile.yaml")
    parser.add_argument("--num-iter", type=int, default=3)
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

    # per (iter, image) accumulators
    stats = []
    dices = {"pred": np.zeros((num_iter, n)), "oracle": np.zeros((num_iter, n))}
    ious = {"pred": np.zeros((num_iter, n)), "oracle": np.zeros((num_iter, n))}

    for i, (x_path, y_path) in tqdm(enumerate(zip(test_x, test_y)), total=n,
                                    desc="oracle analysis"):
        img, img_t, gt = load_pair(x_path, y_path, size)
        otsu = otsu_bin(img, size)

        fb = {"pred": otsu.copy(), "oracle": otsu.copy()}
        pred_soft = None

        for t in range(num_iter):
            outs = {}
            for k in ("pred", "oracle"):
                m = torch.from_numpy(fb[k]).unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    p = torch.sigmoid(model([img_t.to(device), m]))[0, 0].cpu().numpy()
                outs[k] = p
                dices[k][t, i] = dice_np(gt, p > 0.5)
                ious[k][t, i] = iou_np(gt, p > 0.5)

            pred_soft = outs["pred"]
            pred_bin = (pred_soft > 0.5).astype(np.float32)
            ora_bin = (outs["oracle"] > 0.5).astype(np.float32)

            # ---- A. error decomposition of pred feedback vs gt ----
            fp = ((pred_bin == 1) & (gt == 0))
            fn = ((pred_bin == 0) & (gt == 1))
            n_fp, n_fn = fp.sum(), fn.sum()

            # ---- B. where does oracle output differ from pred output? ----
            diff = (ora_bin != pred_bin)
            n_diff = diff.sum()
            if n_diff > 0:
                diff_at_fp = (diff & fp).sum() / n_diff          # oracle khac pred tai noi pred thua
                diff_at_fn = (diff & fn).sum() / n_diff          # oracle khac pred tai noi pred thieu
                diff_else = 1.0 - diff_at_fp - diff_at_fn
            else:
                diff_at_fp = diff_at_fn = diff_else = 0.0

            # ---- C. boundary distance of FP / FN errors ----
            d = dist_to_boundary(gt)
            fp_d = float(d[fp].mean()) if fp.any() else float('nan')
            fn_d = float(d[fn].mean()) if fn.any() else float('nan')
            err_d = float(d[fp | fn].mean()) if (fp | fn).any() else float('nan')

            stats.append({
                "iter": t + 1, "image": i,
                "dice_pred": dices["pred"][t, i],
                "dice_oracle": dices["oracle"][t, i],
                "delta": dices["oracle"][t, i] - dices["pred"][t, i],
                "fp_rate": n_fp / (gt.size + 1e-15),
                "fn_rate": n_fn / (gt.size + 1e-15),
                "fp_fn_ratio": n_fp / (n_fn + 1e-15) if n_fn > 0 else 99.0,
                "diff_frac": n_diff / (gt.size + 1e-15),
                "diff_at_fp": diff_at_fp,
                "diff_at_fn": diff_at_fn,
                "diff_else": diff_else,
                "fp_boundary_dist": fp_d,
                "fn_boundary_dist": fn_d,
                "err_boundary_dist": err_d,
            })

            # update feedback masks
            fb["pred"] = pred_bin
            fb["oracle"] = gt

    # ================= summaries =================
    print("\n=== A. Error decomposition of prediction feedback (final iteration) ===")
    last = [s for s in stats if s["iter"] == num_iter]
    fp_rate = np.mean([s["fp_rate"] for s in last])
    fn_rate = np.mean([s["fn_rate"] for s in last])
    tot_fp = np.sum([s["fp_rate"] for s in last])
    tot_fn = np.sum([s["fn_rate"] for s in last])
    ratio = tot_fp / (tot_fn + 1e-15)
    print(f"  mean FP rate (over-segmentation): {fp_rate:.4f}")
    print(f"  mean FN rate (under-segmentation): {fn_rate:.4f}")
    print(f"  total FP/FN ratio: {ratio:.2f}  (>1: thua nhieu hon thieu)")

    print("\n=== B. Where oracle output differs from pred-feedback output ===")
    diff_frac = np.mean([s["diff_frac"] for s in last])
    at_fp = np.mean([s["diff_at_fp"] for s in last])
    at_fn = np.mean([s["diff_at_fn"] for s in last])
    at_else = np.mean([s["diff_else"] for s in last])
    print(f"  fraction of pixels where outputs differ: {diff_frac:.4f}")
    print(f"    of those, located at pred-FP regions: {at_fp:.3f}")
    print(f"    located at pred-FN regions:           {at_fn:.3f}")
    print(f"    elsewhere (both agree w/ gt already): {at_else:.3f}")

    print("\n=== C. Boundary distance of errors (px) ===")
    fp_d = np.nanmean([s["fp_boundary_dist"] for s in last])
    fn_d = np.nanmean([s["fn_boundary_dist"] for s in last])
    print(f"  mean dist of FP errors to GT boundary: {fp_d:.2f} px")
    print(f"  mean dist of FN errors to GT boundary: {fn_d:.2f} px")
    print("  (nhỏ = lỗi tập trung ở biên; lớn = lỗi trải xa nền)")

    print("\n=== D. Dice per iteration: pred vs oracle ===")
    for t in range(num_iter):
        print(f"  iter {t+1}: pred {dices['pred'][t].mean():.4f} | "
              f"oracle {dices['oracle'][t].mean():.4f} | "
              f"gap {dices['oracle'][t].mean() - dices['pred'][t].mean():+.4f}")

    # ================= save =================
    os.makedirs(results_dir, exist_ok=True)
    with open(f"{results_dir}/oracle_analysis.csv", "w") as f:
        keys = list(stats[0].keys())
        f.write(",".join(keys) + "\n")
        for r in stats:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    summary = {
        "final_iter_fp_rate": float(fp_rate),
        "final_iter_fn_rate": float(fn_rate),
        "final_iter_fp_fn_ratio": float(ratio),
        "diff_frac": float(diff_frac),
        "diff_at_fp": float(at_fp),
        "diff_at_fn": float(at_fn),
        "diff_else": float(at_else),
        "fp_boundary_dist": float(fp_d),
        "fn_boundary_dist": float(fn_d),
        "mean_dice_pred": [float(dices["pred"][t].mean()) for t in range(num_iter)],
        "mean_dice_oracle": [float(dices["oracle"][t].mean()) for t in range(num_iter)],
    }
    with open(f"{results_dir}/oracle_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {results_dir}/oracle_analysis.csv, oracle_summary.json")


if __name__ == "__main__":
    main()
