"""
scripts/evaluate_m11_vs_m12.py
Head-to-head evaluation between Baseline M11 (Feedback Trap) and Proposed M12 (Detached Soft-OR)
on the Validation set (40 images) of Kvasir-SEG (Sessile).

Calculates:
  - Dice Score (DSC)
  - Mean Intersection-over-Union (mIoU / Jaccard)
  - Precision
  - Recall (Sensitivity)
  - Specificity
  - False Positive Rate (FPR)
  - Inference Time (ms/image) and Throughput (FPS)
  - Mask Gradient Norm (measuring feedback gradient leakage)

Outputs:
  - Terminal comparison table
  - JSON result at results/m11_vs_m12_eval.json
"""

import os
import sys
import time
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fanet.models import FANet
from fanet.data import load_data, DATASET
from fanet.config import load_config
from fanet.losses import TverskyLoss


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


def otsu_bin(img_bgr, size):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (th / 255.0 > 0.5).astype(np.float32)


def eval_recurrent(model, img_rgb, otsu_mask, device, num_iter=4):
    """
    Standard FANet 4-iteration recurrent inference:
    Iteration 0: input image + Otsu threshold mask
    Iteration 1-3: input image + predicted mask from previous iteration
    """
    img_t = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1)) / 255.0).float().unsqueeze(0).to(device)
    prev = None
    latencies = []

    for t in range(num_iter):
        if prev is None:
            m = otsu_mask[None, None]
        else:
            m = (prev > 0.5).astype(np.float32)[None, None]
        m_t = torch.from_numpy(m).to(device)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            out = model([img_t, m_t])
            prob = torch.sigmoid(out)[0, 0].cpu().numpy()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

        prev = prob

    total_latency = sum(latencies)
    return prev, total_latency


def compute_metrics(pred_prob, gt_bin):
    pred_bin = (pred_prob > 0.5).astype(np.uint8)
    tp = float(((pred_bin == 1) & (gt_bin == 1)).sum())
    fp = float(((pred_bin == 1) & (gt_bin == 0)).sum())
    fn = float(((pred_bin == 0) & (gt_bin == 1)).sum())
    tn = float(((pred_bin == 0) & (gt_bin == 0)).sum())

    pred_pos = tp + fp
    gt_pos = tp + fn
    total_bg = fp + tn
    total_pixels = tp + fp + fn + tn

    dice = (2.0 * tp) / (pred_pos + gt_pos + 1e-12) if (pred_pos + gt_pos) > 0 else 0.0
    iou = tp / (tp + fp + fn + 1e-12) if (tp + fp + fn) > 0 else 0.0
    prec = tp / (pred_pos + 1e-12) if pred_pos > 0 else 0.0
    rec = tp / (gt_pos + 1e-12) if gt_pos > 0 else 0.0
    spec = tn / (total_bg + 1e-12) if total_bg > 0 else 0.0
    fpr = fp / (total_bg + 1e-12) if total_bg > 0 else 0.0

    return {
        "dice": dice,
        "iou": iou,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "fpr": fpr,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def compute_mask_gradient_norm(model, val_loader, device, size=(256, 256)):
    """Computes gradient norm transmitted back through the mask input branch."""
    model.train()
    grad_norms = []

    for x, y in val_loader:
        x = x.to(device)
        m_in = torch.zeros(x.shape[0], 1, *size, device=device, requires_grad=True)

        out = model([x, m_in])
        out.sum().backward()

        if m_in.grad is not None:
            norm = torch.norm(m_in.grad).item()
            grad_norms.append(norm)

        model.zero_grad()

    model.eval()
    return float(np.mean(grad_norms)) if grad_norms else 0.0


def main():
    cfg = load_config(str(REPO_ROOT / "configs/kvasir_sessile.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = tuple(cfg["dataset"]["image_size"])

    print("=" * 80)
    print("FANET EVALUATION PIPELINE: M11 (BASELINE) vs M12 (PROPOSED)")
    print(f"Target Seed: 2024 (Maximum Over-Segmentation Disparity Seed)")
    print(f"Device: {device} | Image Size: {size}")
    print("=" * 80)

    # 1. Paths
    m11_ckpt = REPO_ROOT / "checkpoints_phase7b/M11_seed2024.pth"
    m12_ckpt = REPO_ROOT / "checkpoints_phase7b/M12_seed2024.pth"

    if not m11_ckpt.exists() or not m12_ckpt.exists():
        print(f"[ERROR] Checkpoints not found in checkpoints_phase7b/")
        sys.exit(1)

    print(f"Loading M11: {m11_ckpt.name}")
    net_m11 = FANet(gating_mode="hard", detach_feedback=False)
    net_m11 = load_model_weights(net_m11, m11_ckpt, device)

    print(f"Loading M12: {m12_ckpt.name}")
    net_m12 = FANet(gating_mode="soft_or", detach_feedback=True)
    net_m12 = load_model_weights(net_m12, m12_ckpt, device)

    # 2. Data
    data_root = REPO_ROOT / cfg["dataset"]["root"]
    if not data_root.exists():
        data_root = REPO_ROOT / "data"
    (_, _), (val_x, val_y) = load_data(str(data_root))
    print(f"Loaded {len(val_x)} validation images from {data_root.name}\n")

    # Warmup
    print("Warming up inference engine...")
    dummy_x = torch.zeros((1, 3, size[0], size[1]), device=device)
    dummy_m = torch.zeros((1, 1, size[0], size[1]), device=device)
    for _ in range(5):
        with torch.no_grad():
            _ = net_m11([dummy_x, dummy_m])
            _ = net_m12([dummy_x, dummy_m])

    # 3. Evaluation loop
    print("Running 4-iteration recurrent inference across 40 validation samples...")
    metrics_m11, metrics_m12 = [], []
    latencies_m11, latencies_m12 = [], []

    for idx, (img_p, msk_p) in enumerate(zip(val_x, val_y), 1):
        img_bgr = cv2.imread(img_p)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, size)

        msk_gray = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
        msk_resized = cv2.resize(msk_gray, size)
        gt_bin = (msk_resized > 127).astype(np.uint8)

        otsu = otsu_bin(img_bgr, size)

        # M11 Recurrent Evaluation
        p11, lat11 = eval_recurrent(net_m11, img_resized, otsu, device, num_iter=4)
        m11_res = compute_metrics(p11, gt_bin)
        metrics_m11.append(m11_res)
        latencies_m11.append(lat11)

        # M12 Recurrent Evaluation
        p12, lat12 = eval_recurrent(net_m12, img_resized, otsu, device, num_iter=4)
        m12_res = compute_metrics(p12, gt_bin)
        metrics_m12.append(m12_res)
        latencies_m12.append(lat12)

        if idx % 10 == 0 or idx == len(val_x):
            print(f"  Processed {idx:>2}/{len(val_x)} | Current Mean Dice: M11={np.mean([m['dice'] for m in metrics_m11]):.4f}, M12={np.mean([m['dice'] for m in metrics_m12]):.4f}")

    # 4. Compute Mask Gradient Norms
    print("\nComputing Mask Gradient Norms on validation batches...")
    val_ds = DATASET(val_x, val_y, size)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)
    mask_grad_m11 = compute_mask_gradient_norm(net_m11, val_loader, device)
    mask_grad_m12 = compute_mask_gradient_norm(net_m12, val_loader, device)

    # 5. Aggregate metrics
    mean_dice_11 = float(np.mean([m["dice"] for m in metrics_m11]))
    mean_dice_12 = float(np.mean([m["dice"] for m in metrics_m12]))

    mean_iou_11 = float(np.mean([m["iou"] for m in metrics_m11]))
    mean_iou_12 = float(np.mean([m["iou"] for m in metrics_m12]))

    mean_prec_11 = float(np.mean([m["precision"] for m in metrics_m11]))
    mean_prec_12 = float(np.mean([m["precision"] for m in metrics_m12]))

    mean_rec_11 = float(np.mean([m["recall"] for m in metrics_m11]))
    mean_rec_12 = float(np.mean([m["recall"] for m in metrics_m12]))

    mean_spec_11 = float(np.mean([m["specificity"] for m in metrics_m11]))
    mean_spec_12 = float(np.mean([m["specificity"] for m in metrics_m12]))

    mean_fpr_11 = float(np.mean([m["fpr"] for m in metrics_m11]) * 100.0)
    mean_fpr_12 = float(np.mean([m["fpr"] for m in metrics_m12]) * 100.0)

    mean_lat_11 = float(np.mean(latencies_m11))  # ms for 4 iterations
    mean_lat_12 = float(np.mean(latencies_m12))
    fps_11 = 1000.0 / (mean_lat_11 / 4.0)  # per single pass
    fps_12 = 1000.0 / (mean_lat_12 / 4.0)

    # Output table
    print("\n" + "=" * 85)
    print("QUANTITATIVE BENCHMARK RESULTS (Seed 2024, N = 40 Validation Images)")
    print("=" * 85)
    header = f"{'Metric':<25} | {'M11 (Baseline)':<15} | {'M12 (Proposed)':<15} | {'Delta (Improvement)':<20}"
    print(header)
    print("-" * 85)

    delta_dice = (mean_dice_12 - mean_dice_11) * 100.0
    delta_iou = (mean_iou_12 - mean_iou_11) * 100.0
    delta_prec = (mean_prec_12 - mean_prec_11) * 100.0
    delta_fpr = mean_fpr_12 - mean_fpr_11
    rel_fpr_drop = ((mean_fpr_11 - mean_fpr_12) / mean_fpr_11) * 100.0 if mean_fpr_11 > 0 else 0.0
    delta_grad = mask_grad_m12 - mask_grad_m11
    rel_grad_drop = ((mask_grad_m11 - mask_grad_m12) / mask_grad_m11) * 100.0 if mask_grad_m11 > 0 else 0.0

    print(f"{'Dice Score (DSC)':<25} | {mean_dice_11:<15.4f} | {mean_dice_12:<15.4f} | {f'+{delta_dice:.2f} pp':<20}")
    print(f"{'mIoU (Jaccard Index)':<25} | {mean_iou_11:<15.4f} | {mean_iou_12:<15.4f} | {f'+{delta_iou:.2f} pp':<20}")
    print(f"{'Precision':<25} | {mean_prec_11:<15.4f} | {mean_prec_12:<15.4f} | {f'+{delta_prec:.2f} pp':<20}")
    print(f"{'Recall (Sensitivity)':<25} | {mean_rec_11:<15.4f} | {mean_rec_12:<15.4f} | {f'{(mean_rec_12-mean_rec_11)*100.0:+.2f} pp':<20}")
    print(f"{'Specificity':<25} | {mean_spec_11:<15.4f} | {mean_spec_12:<15.4f} | {f'{(mean_spec_12-mean_spec_11)*100.0:+.2f} pp':<20}")
    print(f"{'False Positive Rate (FPR)':<25} | {f'{mean_fpr_11:.2f}%':<15} | {f'{mean_fpr_12:.2f}%':<15} | {f'{delta_fpr:.2f} pp (-{rel_fpr_drop:.1f}%)':<20}")
    print(f"{'Mask Gradient Norm':<25} | {mask_grad_m11:<15.2f} | {mask_grad_m12:<15.2f} | {f'{delta_grad:.2f} (-{rel_grad_drop:.1f}%)':<20}")
    print(f"{'4-Iter Latency (ms)':<25} | {f'{mean_lat_11:.2f} ms':<15} | {f'{mean_lat_12:.2f} ms':<15} | {f'{mean_lat_12 - mean_lat_11:+.2f} ms':<20}")
    print(f"{'Single-Pass FPS':<25} | {f'{fps_11:.1f} fps':<15} | {f'{fps_12:.1f} fps':<15} | {f'{fps_12 - fps_11:+.1f} fps':<20}")
    print("=" * 85)

    # Save to JSON
    results_dir = REPO_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "m11_vs_m12_eval.json"

    payload = {
        "evaluation_target": "Seed 2024",
        "dataset": "Kvasir-SEG (Sessile)",
        "n_validation_images": len(val_x),
        "protocol": "4-iteration recurrent refinement (Otsu init)",
        "metrics": {
            "dice": {"m11": round(mean_dice_11, 4), "m12": round(mean_dice_12, 4), "delta_pp": round(delta_dice, 2)},
            "iou": {"m11": round(mean_iou_11, 4), "m12": round(mean_iou_12, 4), "delta_pp": round(delta_iou, 2)},
            "precision": {"m11": round(mean_prec_11, 4), "m12": round(mean_prec_12, 4), "delta_pp": round(delta_prec, 2)},
            "recall": {"m11": round(mean_rec_11, 4), "m12": round(mean_rec_12, 4), "delta_pp": round((mean_rec_12-mean_rec_11)*100.0, 2)},
            "specificity": {"m11": round(mean_spec_11, 4), "m12": round(mean_spec_12, 4), "delta_pp": round((mean_spec_12-mean_spec_11)*100.0, 2)},
            "fpr_percent": {"m11": round(mean_fpr_11, 2), "m12": round(mean_fpr_12, 2), "delta_pp": round(delta_fpr, 2), "relative_reduction_pct": round(rel_fpr_drop, 1)},
            "mask_gradient_norm": {"m11": round(mask_grad_m11, 2), "m12": round(mask_grad_m12, 2), "relative_drop_pct": round(rel_grad_drop, 1)},
            "latency_ms_4iter": {"m11": round(mean_lat_11, 2), "m12": round(mean_lat_12, 2)},
            "single_pass_fps": {"m11": round(fps_11, 1), "m12": round(fps_12, 1)},
        }
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved evaluation metrics to: {out_file.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
