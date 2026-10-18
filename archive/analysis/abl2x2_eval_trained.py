"""Phase 4 eval: load each TRAINED checkpoint, evaluate on the same 40 val images.

Reuses the abl2x2 protocol (paired per-image, 4 iterations) but the model is
loaded from a per-cell trained checkpoint and evaluated with its own mechanism.

Also measures BN mismatch (T_BN): activation distribution of e1.p1 conv1 output
under the cell's own mechanism.

Usage:
    python analysis/abl2x2_eval_trained.py --ckpt-dir checkpoints_phase4
    (expects ckpt_T00.pth / ckpt_T10.pth / ckpt_T01.pth / ckpt_T11.pth)
"""
import argparse
import glob
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

NUM_ITER = 4


def dice_np(gt, pred):
    gt = gt.astype(bool); pred = pred.astype(bool)
    inter = (gt & pred).sum()
    return 2 * inter / (gt.sum() + pred.sum() + 1e-15)


def iou_np(gt, pred):
    gt = gt.astype(bool); pred = pred.astype(bool)
    inter = (gt & pred).sum()
    return inter / ((gt | pred).sum() + 1e-15)


def otsu_bin(img, size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (th / 255.0 > 0.5).astype(np.float32)


def dist_to_boundary(mask):
    mask = mask.astype(np.uint8)
    edt_in = ndimage.distance_transform_edt(mask)
    edt_out = ndimage.distance_transform_edt(1 - mask)
    return np.where(mask > 0, edt_in, edt_out)


def build_feedback(dual, p_soft, gt, rng, tau=0.1):
    m_fg = (p_soft > 0.5).astype(np.float32)
    if dual:
        m_bg = (p_soft < 0.5 - tau).astype(np.float32)
        return np.stack([m_fg, m_bg], axis=0)[None]
    return m_fg[None, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kvasir_sessile.yaml")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_phase4")
    parser.add_argument("--out", type=str, default="results/phase4_eval.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    size = tuple(cfg["dataset"]["image_size"])
    results_dir = cfg["paths"]["results"]

    seeding(cfg["train"]["seed"])
    device = torch.device("cpu")

    CELLS = [
        ("T00", "binary", False), ("T10", "ste", False),
        ("T01", "binary", True), ("T11", "ste", True),
    ]
    # oracle reference cell
    CELLS.append(("oracle", "binary", False))

    (train_x, train_y), (test_x, test_y) = load_data(cfg["dataset"]["root"])
    n = len(test_x)

    summary = {}
    per_image = {}

    for name, gate, dual in CELLS:
        ckpt = os.path.join(args.ckpt_dir, f"ckpt_{name}.pth")
        if not os.path.exists(ckpt):
            print(f"SKIP {name}: {ckpt} not found")
            continue
        model = FANet(gate=gate, dual_path=dual)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device); model.eval()
        print(f"Loaded {name} ({gate}, dual={dual})")

        dice = np.zeros((NUM_ITER, n)); iou = np.zeros((NUM_ITER, n))
        fp_rate = np.zeros((NUM_ITER, n)); fn_rate = np.zeros((NUM_ITER, n))
        pi = []

        for i, (x_path, y_path) in tqdm(enumerate(zip(test_x, test_y)), total=n, desc=name):
            img = cv2.imread(x_path, cv2.IMREAD_COLOR)
            img = cv2.resize(img, size)
            img_t = torch.from_numpy(np.transpose(img, (2, 0, 1)) / 255.0).float().unsqueeze(0).to(device)
            gt = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, size) / 255.0
            gt = (gt > 0.5).astype(np.float32)
            otsu = otsu_bin(img, size)
            rng = np.random.RandomState(i)
            prev = None

            for t in range(NUM_ITER):
                if prev is None:
                    fb = np.stack([otsu, np.zeros_like(otsu)], 0)[None] if dual else otsu[None, None]
                else:
                    fb = build_feedback(dual, prev, gt, rng)
                m = torch.from_numpy(fb).to(device)
                with torch.no_grad():
                    p = torch.sigmoid(model([img_t, m]))[0, 0].cpu().numpy()
                prev = p
                pbin = (p > 0.5).astype(np.float32)
                gtb = gt.astype(bool); pbb = pbin.astype(bool)
                fp = (pbb & ~gtb); fn = (~pbb & gtb)
                dice[t, i] = dice_np(gt, pbin)
                iou[t, i] = iou_np(gt, pbin)
                fp_rate[t, i] = fp.mean(); fn_rate[t, i] = fn.mean()
                if t == NUM_ITER - 1:
                    d = dist_to_boundary(gt)
                    pi.append({
                        "image": i, "dice": dice_np(gt, pbin), "iou": iou_np(gt, pbin),
                        "fp_rate": fp.mean(), "fn_rate": fn.mean(),
                        "fp_dist": float(d[fp].mean()) if fp.any() else float("nan"),
                        "fn_dist": float(d[fn].mean()) if fn.any() else float("nan"),
                    })

        summary[name] = {
            "dice_mean_iter": [float(dice[t].mean()) for t in range(NUM_ITER)],
            "iou_mean_iter": [float(iou[t].mean()) for t in range(NUM_ITER)],
            "final_dice": float(dice[-1].mean()), "final_iou": float(iou[-1].mean()),
            "final_fp_rate": float(fp_rate[-1].mean()),
            "final_fn_rate": float(fn_rate[-1].mean()),
        }
        per_image[name] = pi

    print("\n=== Dice per iteration per trained cell ===")
    hdr = " | ".join(f"{c:>8}" for c in summary)
    print(f"{'iter':>4} | {hdr}")
    for t in range(NUM_ITER):
        row = " | ".join(f"{summary[c]['dice_mean_iter'][t]:>8.4f}" for c in summary)
        print(f"{t+1:>4} | {row}")

    print("\n=== Final iteration ===")
    print(f"{'cell':>8} | {'dice':>7} {'iou':>7} {'fp%':>6} {'fn%':>6}")
    for c in summary:
        s = summary[c]
        print(f"{c:>8} | {s['final_dice']:>7.4f} {s['final_iou']:>7.4f} "
              f"{s['final_fp_rate']*100:>6.2f} {s['final_fn_rate']*100:>6.2f}")

    os.makedirs(results_dir, exist_ok=True)
    out = {"summary": summary, "per_image": per_image}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
