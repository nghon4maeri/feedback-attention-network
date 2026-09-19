"""
scripts/evaluate_cross_domain_retinal.py

Cross-Domain Generalization Benchmark: Retinal Vessel Segmentation (DRIVE & CHASE_DB1).
Evaluates whether recurrent feedback with Detached Soft-OR (Feedback Firewall)
generalizes to high-precision tubular vessel delineation outside of colonoscopy.

Datasets:
  1. DRIVE: Digital Retinal Images for Vessel Extraction (20 test frames)
  2. CHASE_DB1: Child Heart and Health Study in England (28 retinal images)
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fanet.models import FANet


def load_model_weights(model, ckpt_path, device):
    raw_sd = torch.load(ckpt_path, map_location=device)
    sd = {}
    for k, v in raw_sd.items():
        k_mapped = k
        for i in range(1, 5):
            if k_mapped.startswith(f"d{i}.up."):
                k_mapped = k_mapped.replace(f"d{i}.up.", f"d{i}.upsample.")
        sd[k_mapped] = v
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def compute_metrics(pred_prob, gt_bin):
    pred_bin = (pred_prob > 0.5).astype(np.uint8)
    gt_bin = (gt_bin > 0.5).astype(np.uint8)

    tp = float(((pred_bin == 1) & (gt_bin == 1)).sum())
    fp = float(((pred_bin == 1) & (gt_bin == 0)).sum())
    fn = float(((pred_bin == 0) & (gt_bin == 1)).sum())
    tn = float(((pred_bin == 0) & (gt_bin == 0)).sum())

    pred_pos = tp + fp
    gt_pos = tp + fn
    total_bg = fp + tn

    dice = (2.0 * tp) / (pred_pos + gt_pos + 1e-12) if (pred_pos + gt_pos) > 0 else 0.0
    iou = tp / (tp + fp + fn + 1e-12) if (tp + fp + fn) > 0 else 0.0
    prec = tp / (pred_pos + 1e-12) if pred_pos > 0 else 0.0
    rec = tp / (gt_pos + 1e-12) if gt_pos > 0 else 0.0
    spec = tn / (total_bg + 1e-12) if total_bg > 0 else 0.0
    fpr = fp / (total_bg + 1e-12) if total_bg > 0 else 0.0

    return {
        "dice": float(dice),
        "iou": float(iou),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "fpr": float(fpr),
    }


def otsu_initialization(img_rgb, size=(256, 256)):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(gray, size)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    # Green channel inversion often used in retinal imaging; Otsu on inverted green
    green = img_rgb[:, :, 1]
    inv_green = 255 - green
    inv_green = cv2.resize(inv_green, size)
    _, th = cv2.threshold(inv_green, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (th / 255.0 > 0.5).astype(np.float32)


def run_recurrent_inference(model, img_rgb, device, size=(256, 256), num_iter=4):
    img_resized = cv2.resize(img_rgb, size)
    img_t = torch.from_numpy(np.transpose(img_resized, (2, 0, 1)) / 255.0).float().unsqueeze(0).to(device)
    otsu_mask = otsu_initialization(img_rgb, size)
    prev = None

    for t in range(num_iter):
        if prev is None:
            m = otsu_mask[None, None]
        else:
            m = (prev > 0.5).astype(np.float32)[None, None]
        m_t = torch.from_numpy(m).to(device)

        with torch.no_grad():
            out = model([img_t, m_t])
            prob = torch.sigmoid(out)[0, 0].cpu().numpy()
        prev = prob

    return prev


def get_drive_pairs(drive_root):
    # Check training or test
    test_img_dir = Path(drive_root) / "test" / "images"
    train_img_dir = Path(drive_root) / "training" / "images"
    train_manual_dir = Path(drive_root) / "training" / "1st_manual"

    pairs = []
    if train_img_dir.exists() and train_manual_dir.exists():
        for img_p in sorted(train_img_dir.glob("*.tif")):
            base = img_p.name.split("_")[0]
            mask_p = train_manual_dir / f"{base}_manual1.gif"
            if mask_p.exists():
                pairs.append((str(img_p), str(mask_p)))

    if not pairs and test_img_dir.exists():
        # Fallback to test
        mask_dir = Path(drive_root) / "test" / "mask"
        for img_p in sorted(test_img_dir.glob("*.tif")):
            base = img_p.name.split("_")[0]
            mask_p = mask_dir / f"{base}_test_mask.gif"
            if mask_p.exists():
                pairs.append((str(img_p), str(mask_p)))

    return pairs


def get_chase_pairs(chase_root):
    img_dir = Path(chase_root) / "images"
    mask_dir = Path(chase_root) / "masks"
    pairs = []
    if img_dir.exists() and mask_dir.exists():
        for img_p in sorted(img_dir.glob("*.jpg")):
            base = img_p.stem
            mask_p = mask_dir / f"{base}_1stHO.png"
            if mask_p.exists():
                pairs.append((str(img_p), str(mask_p)))
    return pairs


def evaluate_dataset(model_m11, model_m12, pairs, dataset_name, device, size=(256, 256)):
    print(f"\n--- Evaluating {dataset_name} ({len(pairs)} samples) ---")
    m11_metrics = []
    m12_metrics = []

    for img_path, mask_path in tqdm(pairs, desc=dataset_name):
        img = np.array(Image.open(img_path).convert("RGB"))
        gt = np.array(Image.open(mask_path).convert("L"))
        gt_resized = cv2.resize(gt, size, interpolation=cv2.INTER_NEAREST)
        gt_bin = (gt_resized > 127).astype(np.float32)

        p_m11 = run_recurrent_inference(model_m11, img, device, size=size)
        p_m12 = run_recurrent_inference(model_m12, img, device, size=size)

        m11_metrics.append(compute_metrics(p_m11, gt_bin))
        m12_metrics.append(compute_metrics(p_m12, gt_bin))

    def aggregate(metrics_list):
        keys = metrics_list[0].keys()
        return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}

    agg_m11 = aggregate(m11_metrics)
    agg_m12 = aggregate(m12_metrics)

    print(f"Results for {dataset_name}:")
    print(f"  M11 (Feedback Trap):  Dice={agg_m11['dice']:.4f}, FPR={agg_m11['fpr']:.4f}, Prec={agg_m11['precision']:.4f}, Rec={agg_m11['recall']:.4f}")
    print(f"  M12 (Detached Soft):  Dice={agg_m12['dice']:.4f}, FPR={agg_m12['fpr']:.4f}, Prec={agg_m12['precision']:.4f}, Rec={agg_m12['recall']:.4f}")
    delta_dice = agg_m12['dice'] - agg_m11['dice']
    delta_fpr = agg_m12['fpr'] - agg_m11['fpr']
    print(f"  Delta (M12 - M11):    Dice: {delta_dice*100:+.2f} pp, FPR: {delta_fpr*100:+.2f} pp")

    return {
        "dataset": dataset_name,
        "n_samples": len(pairs),
        "M11": agg_m11,
        "M12": agg_m12,
        "delta_dice_pp": float(delta_dice * 100),
        "delta_fpr_pp": float(delta_fpr * 100),
    }


def main():
    parser = argparse.ArgumentParser(description="Cross-Domain Retinal Vessel Evaluation")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256])
    args = parser.parse_args()

    device = torch.device(args.device)
    size = tuple(args.size)

    m11_ckpt = REPO_ROOT / "checkpoints_phase7b/M11_seed42.pth"
    m12_ckpt = REPO_ROOT / "checkpoints_phase7b/M12_seed42.pth"

    if not m11_ckpt.exists() or not m12_ckpt.exists():
        print("[ERROR] Checkpoints not found in checkpoints_phase7b/")
        sys.exit(1)

    print("Loading models for cross-domain evaluation...")
    net_m11 = FANet(gating_mode="hard", detach_feedback=False)
    net_m11 = load_model_weights(net_m11, m11_ckpt, device)

    net_m12 = FANet(gating_mode="soft_or", detach_feedback=True)
    net_m12 = load_model_weights(net_m12, m12_ckpt, device)

    results = {}

    # 1. DRIVE
    drive_root = REPO_ROOT / "data/DRIVE"
    if drive_root.exists():
        drive_pairs = get_drive_pairs(drive_root)
        if drive_pairs:
            results["DRIVE"] = evaluate_dataset(net_m11, net_m12, drive_pairs, "DRIVE", device, size)

    # 2. CHASE_DB1
    chase_root = REPO_ROOT / "data/CHASE_DB1"
    if chase_root.exists():
        chase_pairs = get_chase_pairs(chase_root)
        if chase_pairs:
            results["CHASE_DB1"] = evaluate_dataset(net_m11, net_m12, chase_pairs, "CHASE_DB1", device, size)

    out_path = REPO_ROOT / "results/cross_domain_retinal_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved cross-domain evaluation results to {out_path}")


if __name__ == "__main__":
    main()
