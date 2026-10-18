"""
scripts/zero_shot_eval.py
Zero-Shot Cross-Center Evaluation on CVC-ClinicDB (612 colonoscopy images).
Evaluates generalization of Baseline M11 (Feedback Trap) vs Proposed M12 (Detached Soft-OR)
trained exclusively on Kvasir-SEG (Sessile).

Metrics:
  - Dice Similarity Coefficient (DSC)
  - Mean Intersection-over-Union (mIoU / Jaccard)
  - Precision
  - Recall (Sensitivity)
  - Specificity
  - False Positive Rate (FPR)
  - Wilcoxon signed-rank test comparing M11 vs M12 distributions

Outputs:
  - Formatted ASCII summary table
  - JSON results saved to results/zero_shot_cvc_clinicdb.json
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import cv2
import numpy as np
import scipy.stats as stats
import torch

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


def otsu_bin(img_bgr, size):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (th / 255.0 > 0.5).astype(np.float32)


def eval_recurrent(model, img_rgb, otsu_mask, device, num_iter=4):
    img_t = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1)) / 255.0).float().unsqueeze(0).to(device)
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


def compute_metrics(pred_prob, gt_bin):
    pred_bin = (pred_prob > 0.5).astype(np.uint8)
    tp = float(((pred_bin == 1) & (gt_bin == 1)).sum())
    fp = float(((pred_bin == 1) & (gt_bin == 0)).sum())
    fn = float(((pred_bin == 0) & (gt_bin == 1)).sum())
    tn = float(((pred_bin == 0) & (gt_bin == 0)).sum())

    pred_pos = tp + fp
    gt_pos = tp + fn
    total_bg = fp + tn

    dice = (2.0 * tp) / (pred_pos + gt_pos + 1e-12) if (pred_pos + gt_pos) > 0 else (1.0 if gt_pos == 0 else 0.0)
    iou = tp / (tp + fp + fn + 1e-12) if (tp + fp + fn) > 0 else (1.0 if gt_pos == 0 else 0.0)
    prec = tp / (pred_pos + 1e-12) if pred_pos > 0 else (1.0 if gt_pos == 0 else 0.0)
    rec = tp / (gt_pos + 1e-12) if gt_pos > 0 else 1.0
    spec = tn / (total_bg + 1e-12) if total_bg > 0 else 1.0
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


def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Cross-Center Evaluation on CVC-ClinicDB")
    parser.add_argument("--data-dir", type=str, default=str(REPO_ROOT / "data/CVC-ClinicDB"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 42, 99, 1337, 2024])
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--num-iter", type=int, default=4)
    parser.add_argument("--output", type=str, default=str(REPO_ROOT / "results/zero_shot_cvc_clinicdb.json"))
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = tuple(args.size)
    data_dir = Path(args.data_dir)
    img_dir = data_dir / "images"
    msk_dir = data_dir / "masks"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ZERO-SHOT CROSS-CENTER VALIDATION: CVC-ClinicDB (Hospital Clinic, Barcelona)")
    print(f"Device: {device} | Target Image Size: {size} | Recurrent Iterations: {args.num_iter}")
    print(f"Evaluating Seeds: {args.seeds}")
    print("=" * 80)

    # 1. Discover data pairs
    img_files = sorted(list(img_dir.glob("*.tif")) + list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg")))
    dataset_items = []
    for p in img_files:
        m = msk_dir / p.name
        if not m.exists():
            # Try other extensions
            stem = p.stem
            candidates = list(msk_dir.glob(f"{stem}.*"))
            if candidates:
                m = candidates[0]
            else:
                continue
        dataset_items.append((p, m))

    print(f"Discovered {len(dataset_items)} matched (image, mask) pairs in {data_dir.name}")
    if len(dataset_items) == 0:
        print("[ERROR] No valid data pairs found.")
        sys.exit(1)

    # 2. Preload and pre-process images + Otsu masks into memory for speed
    print("\nPre-processing and caching images + Otsu masks in memory...")
    t0 = time.time()
    cached_data = []
    for img_p, msk_p in dataset_items:
        img_bgr = cv2.imread(str(img_p))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, size)

        msk_gray = cv2.imread(str(msk_p), cv2.IMREAD_GRAYSCALE)
        msk_resized = cv2.resize(msk_gray, size)
        gt_bin = (msk_resized > 127).astype(np.uint8)

        otsu = otsu_bin(img_bgr, size)
        cached_data.append({
            "name": img_p.name,
            "img_rgb": img_resized,
            "gt_bin": gt_bin,
            "otsu": otsu,
        })
    print(f"Cached {len(cached_data)} samples in {time.time() - t0:.2f}s.\n")

    # 3. Evaluation per seed
    results_per_seed = {}
    all_seed_m11_dice = []
    all_seed_m12_dice = []
    all_seed_m11_fpr = []
    all_seed_m12_fpr = []
    all_seed_m11_iou = []
    all_seed_m12_iou = []
    all_seed_m11_prec = []
    all_seed_m12_prec = []
    all_seed_m11_rec = []
    all_seed_m12_rec = []

    # For sample-level Wilcoxon test, we also collect pooled per-sample values from the median/best seed or pooled
    pooled_m11_dices = []
    pooled_m12_dices = []
    pooled_m11_fprs = []
    pooled_m12_fprs = []

    for s in args.seeds:
        m11_ckpt = REPO_ROOT / f"checkpoints_phase7b/M11_seed{s}.pth"
        m12_ckpt = REPO_ROOT / f"checkpoints_phase7b/M12_seed{s}.pth"

        if not m11_ckpt.exists() or not m12_ckpt.exists():
            print(f"[SKIP] Checkpoints for seed {s} not found.")
            continue

        print(f"--- Evaluating Seed {s} ---")
        net_m11 = FANet(gating_mode="hard", detach_feedback=False)
        net_m11 = load_model_weights(net_m11, m11_ckpt, device)

        net_m12 = FANet(gating_mode="soft_or", detach_feedback=True)
        net_m12 = load_model_weights(net_m12, m12_ckpt, device)

        m11_metrics = []
        m12_metrics = []

        t_eval0 = time.time()
        for sample in cached_data:
            img = sample["img_rgb"]
            gt = sample["gt_bin"]
            otsu = sample["otsu"]

            prob_11 = eval_recurrent(net_m11, img, otsu, device, num_iter=args.num_iter)
            prob_12 = eval_recurrent(net_m12, img, otsu, device, num_iter=args.num_iter)

            m_11 = compute_metrics(prob_11, gt)
            m_12 = compute_metrics(prob_12, gt)

            m11_metrics.append(m_11)
            m12_metrics.append(m_12)

        eval_dur = time.time() - t_eval0

        # Aggregate for this seed
        keys = ["dice", "iou", "precision", "recall", "specificity", "fpr"]
        seed_summary_m11 = {k: float(np.mean([x[k] for x in m11_metrics])) for k in keys}
        seed_summary_m12 = {k: float(np.mean([x[k] for x in m12_metrics])) for k in keys}

        all_seed_m11_dice.append(seed_summary_m11["dice"])
        all_seed_m12_dice.append(seed_summary_m12["dice"])
        all_seed_m11_fpr.append(seed_summary_m11["fpr"])
        all_seed_m12_fpr.append(seed_summary_m12["fpr"])
        all_seed_m11_iou.append(seed_summary_m11["iou"])
        all_seed_m12_iou.append(seed_summary_m12["iou"])
        all_seed_m11_prec.append(seed_summary_m11["precision"])
        all_seed_m12_prec.append(seed_summary_m12["precision"])
        all_seed_m11_rec.append(seed_summary_m11["recall"])
        all_seed_m12_rec.append(seed_summary_m12["recall"])

        # Per sample arrays for wilcoxon
        s_dices_11 = [x["dice"] for x in m11_metrics]
        s_dices_12 = [x["dice"] for x in m12_metrics]
        s_fprs_11 = [x["fpr"] for x in m11_metrics]
        s_fprs_12 = [x["fpr"] for x in m12_metrics]

        try:
            stat_d, p_d = stats.wilcoxon(s_dices_12, s_dices_11, alternative="greater")
        except Exception:
            stat_d, p_d = 0.0, 1.0
        try:
            stat_f, p_f = stats.wilcoxon(s_fprs_11, s_fprs_12, alternative="greater")
        except Exception:
            stat_f, p_f = 0.0, 1.0

        results_per_seed[str(s)] = {
            "M11": seed_summary_m11,
            "M12": seed_summary_m12,
            "wilcoxon_dice_pval": float(p_d),
            "wilcoxon_fpr_pval": float(p_f),
            "eval_duration_sec": eval_dur,
        }

        print(f"  Seed {s} | M11 DSC: {seed_summary_m11['dice']:.4f}, FPR: {seed_summary_m11['fpr']*100:.2f}% | "
              f"M12 DSC: {seed_summary_m12['dice']:.4f}, FPR: {seed_summary_m12['fpr']*100:.2f}% | "
              f"Delta DSC: {(seed_summary_m12['dice'] - seed_summary_m11['dice']):+.4f} (p={p_d:.2e}), "
              f"Delta FPR: {(seed_summary_m12['fpr'] - seed_summary_m11['fpr'])*100:+.2f}% (p={p_f:.2e})")

        # Collect for seed 2024 as focal representative
        if s == 2024:
            pooled_m11_dices = s_dices_11
            pooled_m12_dices = s_dices_12
            pooled_m11_fprs = s_fprs_11
            pooled_m12_fprs = s_fprs_12

    # If seed 2024 was not run, take last
    if not pooled_m11_dices and len(m11_metrics) > 0:
        pooled_m11_dices = [x["dice"] for x in m11_metrics]
        pooled_m12_dices = [x["dice"] for x in m12_metrics]
        pooled_m11_fprs = [x["fpr"] for x in m11_metrics]
        pooled_m12_fprs = [x["fpr"] for x in m12_metrics]

    # 4. Multi-seed grand summary
    grand_summary = {
        "dataset": "CVC-ClinicDB",
        "num_samples": len(cached_data),
        "seeds": args.seeds,
        "metrics": {
            "dice": {
                "M11_mean": float(np.mean(all_seed_m11_dice)),
                "M11_std": float(np.std(all_seed_m11_dice)),
                "M12_mean": float(np.mean(all_seed_m12_dice)),
                "M12_std": float(np.std(all_seed_m12_dice)),
                "delta": float(np.mean(all_seed_m12_dice) - np.mean(all_seed_m11_dice)),
            },
            "iou": {
                "M11_mean": float(np.mean(all_seed_m11_iou)),
                "M11_std": float(np.std(all_seed_m11_iou)),
                "M12_mean": float(np.mean(all_seed_m12_iou)),
                "M12_std": float(np.std(all_seed_m12_iou)),
                "delta": float(np.mean(all_seed_m12_iou) - np.mean(all_seed_m11_iou)),
            },
            "precision": {
                "M11_mean": float(np.mean(all_seed_m11_prec)),
                "M11_std": float(np.std(all_seed_m11_prec)),
                "M12_mean": float(np.mean(all_seed_m12_prec)),
                "M12_std": float(np.std(all_seed_m12_prec)),
                "delta": float(np.mean(all_seed_m12_prec) - np.mean(all_seed_m11_prec)),
            },
            "recall": {
                "M11_mean": float(np.mean(all_seed_m11_rec)),
                "M11_std": float(np.std(all_seed_m11_rec)),
                "M12_mean": float(np.mean(all_seed_m12_rec)),
                "M12_std": float(np.std(all_seed_m12_rec)),
                "delta": float(np.mean(all_seed_m12_rec) - np.mean(all_seed_m11_rec)),
            },
            "fpr": {
                "M11_mean": float(np.mean(all_seed_m11_fpr)),
                "M11_std": float(np.std(all_seed_m11_fpr)),
                "M12_mean": float(np.mean(all_seed_m12_fpr)),
                "M12_std": float(np.std(all_seed_m12_fpr)),
                "delta": float(np.mean(all_seed_m12_fpr) - np.mean(all_seed_m11_fpr)),
            },
        }
    }

    # Statistical significance on pooled sample-level
    try:
        w_stat_dice, p_val_dice = stats.wilcoxon(pooled_m12_dices, pooled_m11_dices, alternative="greater")
    except Exception:
        w_stat_dice, p_val_dice = 0.0, 1.0
    try:
        w_stat_fpr, p_val_fpr = stats.wilcoxon(pooled_m11_fprs, pooled_m12_fprs, alternative="greater")
    except Exception:
        w_stat_fpr, p_val_fpr = 0.0, 1.0

    grand_summary["sample_level_wilcoxon"] = {
        "dice_pval": float(p_val_dice),
        "fpr_pval": float(p_val_fpr),
    }

    # Format and display Markdown / ASCII Table
    print("\n" + "=" * 80)
    print("MULTI-SEED ZERO-SHOT CROSS-CENTER BENCHMARK: CVC-ClinicDB (N=612)")
    print("=" * 80)
    print(f"{'Metric':<18} | {'M11 (Baseline)':<20} | {'M12 (Detached Soft-OR)':<22} | {'Difference (Delta)':<18}")
    print("-" * 84)

    for m_name, label in [("dice", "Dice (DSC)"), ("iou", "mIoU (Jaccard)"),
                          ("precision", "Precision"), ("recall", "Recall"), ("fpr", "FPR (Over-seg)")]:
        m11_str = f"{grand_summary['metrics'][m_name]['M11_mean']:.4f} +- {grand_summary['metrics'][m_name]['M11_std']:.4f}"
        m12_str = f"{grand_summary['metrics'][m_name]['M12_mean']:.4f} +- {grand_summary['metrics'][m_name]['M12_std']:.4f}"
        delta = grand_summary['metrics'][m_name]['delta']
        if m_name == "fpr":
            delta_str = f"{delta*100:+.2f}% (pruned)"
        else:
            delta_str = f"{delta:+.4f}"
        print(f"{label:<18} | {m11_str:<20} | {m12_str:<22} | {delta_str:<18}")

    print("-" * 84)
    print(f"Sample-Level Wilcoxon Signed-Rank Test: DSC p = {p_val_dice:.2e}, FPR p = {p_val_fpr:.2e}")
    print("=" * 80)

    # Save to JSON
    full_output = {
        "grand_summary": grand_summary,
        "per_seed": results_per_seed,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print(f"\n[SAVED] Exported zero-shot benchmark results to: {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
