"""Phase 3 ablation 2x2: dual-path feedback + gating variants at inference.

Cells (same checkpoint 200ep, frozen weights — tests feedback mechanism only):
    C00        binary gate, single path m_fg            (baseline FANet)
    C01-tau    binary gate, dual path [m_fg, m_bg], bg threshold tau (0.1/0.2/0.3)
    C10        soft gate, single path
    C11        soft gate, dual path (tau=0.1)
    neg-ctrl   binary gate, dual path with RANDOM m_bg (negative control P_B3)
    oracle     ground-truth m_fg (gap reference)

Metrics per (cell, image, iter): dice, iou, precision, recall, fp_rate, fn_rate,
mean FP/FN distance to GT boundary, oracle gap.

Usage:
    python analysis/abl2x2_eval.py [--checkpoint checkpoints/checkpoint_200ep_kaggle.pth]
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

NUM_ITER = 4
TAUS = [0.1, 0.2, 0.3]


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


def dist_to_boundary(mask):
    mask = mask.astype(np.uint8)
    edt_in = ndimage.distance_transform_edt(mask)
    edt_out = ndimage.distance_transform_edt(1 - mask)
    return np.where(mask > 0, edt_in, edt_out)


def build_feedback(mode, p_soft, gt, rng, tau=0.1):
    """Return feedback mask tensor [1, C, H, W] for a mode."""
    m_fg = (p_soft > 0.5).astype(np.float32)

    if mode == "oracle":
        m_fg = gt.astype(np.float32)

    if mode in ("dual", "neg"):
        if mode == "neg":
            # negative control: random background with same density as real m_bg
            real_bg = (p_soft < 0.5 - tau)
            dens = real_bg.mean()
            m_bg = (rng.rand(*p_soft.shape) < dens).astype(np.float32)
        else:
            m_bg = (p_soft < 0.5 - tau).astype(np.float32)
        return np.stack([m_fg, m_bg], axis=0)[None]
    return m_fg[None, None]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kvasir_sessile.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    size = tuple(cfg["dataset"]["image_size"])
    ckpt = args.checkpoint or cfg["paths"]["checkpoint"]
    results_dir = cfg["paths"]["results"]

    seeding(cfg["train"]["seed"])
    device = torch.device("cpu")

    # Build cells: gate mode + dual path
    cells = {}
    cells["C00"] = (FANet(gate="binary"), "single", None)
    for tau in TAUS:
        cells[f"C01-t{tau}"] = (FANet(gate="binary", dual_path=True), "dual", tau)
    cells["C10"] = (FANet(gate="soft"), "single", None)
    cells["C11"] = (FANet(gate="soft", dual_path=True), "dual", 0.1)
    cells["neg-ctrl"] = (FANet(gate="binary", dual_path=True), "neg", 0.1)
    cells["oracle"] = (FANet(gate="binary"), "oracle", None)

    ckpt_state = torch.load(ckpt, map_location=device)
    for name, (model, _, _) in cells.items():
        model.load_state_dict(ckpt_state)
        model.to(device)
        model.eval()

    (train_x, train_y), (test_x, test_y) = load_data(cfg["dataset"]["root"])
    n = len(test_x)

    per_image = {c: [] for c in cells}
    summary = {c: {"dice": np.zeros(NUM_ITER), "iou": np.zeros(NUM_ITER),
                   "fp_rate": np.zeros(NUM_ITER), "fn_rate": np.zeros(NUM_ITER),
                   "precision": np.zeros(NUM_ITER), "recall": np.zeros(NUM_ITER)}
               for c in cells}

    for i, (x_path, y_path) in tqdm(enumerate(zip(test_x, test_y)), total=n,
                                    desc="abl2x2"):
        img = cv2.imread(x_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, size)
        img_t = torch.from_numpy(np.transpose(img, (2, 0, 1)) / 255.0).float().unsqueeze(0).to(device)

        gt = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, size) / 255.0
        gt = (gt > 0.5).astype(np.float32)

        otsu = otsu_bin(img, size)
        rng = np.random.RandomState(i)
        p_prev = {c: None for c in cells}

        for t in range(NUM_ITER):
            for c, (model, mode, tau) in cells.items():
                if p_prev[c] is None:
                    # iteration 1: Otsu feedback (identical start for all cells)
                    if mode in ("dual", "neg"):
                        fb = np.stack([otsu, np.zeros_like(otsu)], axis=0)[None]
                    else:
                        fb = otsu[None, None]
                else:
                    fb = build_feedback(mode, p_prev[c], gt, rng, tau=0.1 if tau is None else tau)

                m = torch.from_numpy(fb).to(device)
                with torch.no_grad():
                    p = torch.sigmoid(model([img_t, m]))[0, 0].cpu().numpy()
                p_prev[c] = p

                pbin = (p > 0.5).astype(np.float32)
                gtb = gt.astype(bool)
                pbb = pbin.astype(bool)
                summary[c]["dice"][t] += dice_np(gt, pbin)
                summary[c]["iou"][t] += iou_np(gt, pbin)
                fp = (pbb & ~gtb)
                fn = (~pbb & gtb)
                summary[c]["fp_rate"][t] += fp.mean()
                summary[c]["fn_rate"][t] += fn.mean()
                tp = (pbb & gtb).sum()
                pred_fg = pbin.sum()
                gt_fg = gtb.sum()
                summary[c]["precision"][t] += tp / (pred_fg + 1e-15)
                summary[c]["recall"][t] += tp / (gt_fg + 1e-15)

                if t == NUM_ITER - 1:
                    d = dist_to_boundary(gt)
                    fp_d = float(d[fp].mean()) if fp.any() else float("nan")
                    fn_d = float(d[fn].mean()) if fn.any() else float("nan")
                    per_image[c].append({
                        "image": i, "dice": dice_np(gt, pbin), "iou": iou_np(gt, pbin),
                        "fp_rate": fp.mean(), "fn_rate": fn.mean(),
                        "fp_dist": fp_d, "fn_dist": fn_d,
                    })

    for c in cells:
        for k in summary[c]:
            summary[c][k] = (summary[c][k] / n).tolist()

    # ================= print =================
    print("\n=== Dice per iteration per cell ===")
    hdr = " | ".join(f"{c:>9}" for c in cells)
    print(f"{'iter':>4} | {hdr}")
    for t in range(NUM_ITER):
        row = " | ".join(f"{summary[c]['dice'][t]:>9.4f}" for c in cells)
        print(f"{t+1:>4} | {row}")

    print("\n=== Final iteration (iter %d) metrics ===" % NUM_ITER)
    print(f"{'cell':>9} | {'dice':>7} {'iou':>7} {'prec':>7} {'recall':>7} "
          f"{'fp%':>6} {'fn%':>6} {'fpDist':>7} {'fnDist':>7}")
    for c in cells:
        s = summary[c]
        pi = per_image[c]
        fp_d = np.nanmean([r["fp_dist"] for r in pi])
        fn_d = np.nanmean([r["fn_dist"] for r in pi])
        print(f"{c:>9} | {s['dice'][-1]:>7.4f} {s['iou'][-1]:>7.4f} "
              f"{s['precision'][-1]:>7.4f} {s['recall'][-1]:>7.4f} "
              f"{s['fp_rate'][-1]*100:>6.2f} {s['fn_rate'][-1]*100:>6.2f} "
              f"{fp_d:>7.1f} {fn_d:>7.1f}")

    print("\n=== Oracle gap (dice_oracle - dice_cell, final iter) ===")
    for c in cells:
        if c == "oracle":
            continue
        gap = summary["oracle"]["dice"][-1] - summary[c]["dice"][-1]
        print(f"  {c:>9}: {gap:+.4f}")

    # ================= save =================
    os.makedirs(results_dir, exist_ok=True)
    out = {"summary": summary,
           "per_image": {c: per_image[c] for c in cells}}
    with open(f"{results_dir}/abl2x2_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {results_dir}/abl2x2_summary.json")


if __name__ == "__main__":
    main()
