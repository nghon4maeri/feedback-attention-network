"""
analysis/calc_p_value.py
Statistical significance analysis: Paired Wilcoxon Signed-Rank Test
comparing M11 (Feedback Trap) vs M12 (Soft-OR + Detach) across 200 evaluation points
(40 validation images x 5 independent random seeds) using the standard FANet 4-iteration
recurrent refinement protocol (Otsu initialization -> iterative feedback).
"""

import os
import sys
import json
from pathlib import Path

import numpy as np
import cv2
import torch
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fanet.models import FANet
from fanet.data import load_data
from fanet.config import load_config


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


def otsu_bin(img, size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (th / 255.0 > 0.5).astype(np.float32)


def eval_fanet_recurrent(model, img, otsu, size, device, num_iter=4):
    """Standard 4-iteration FANet recurrent inference."""
    img_t = torch.from_numpy(np.transpose(img, (2, 0, 1)) / 255.0).float().unsqueeze(0).to(device)
    prev = None
    for t in range(num_iter):
        if prev is None:
            m = otsu[None, None]
        else:
            m = (prev > 0.5).astype(np.float32)[None, None]
        m_t = torch.from_numpy(m).to(device)
        with torch.no_grad():
            p = torch.sigmoid(model([img_t, m_t]))[0, 0].cpu().numpy()
        prev = p
    return prev


def compute_metrics(pred_prob, gt_bin):
    pred_bin = (pred_prob > 0.5).astype(np.uint8)
    tp = float(((pred_bin == 1) & (gt_bin == 1)).sum())
    fp = float(((pred_bin == 1) & (gt_bin == 0)).sum())
    fn = float(((pred_bin == 0) & (gt_bin == 1)).sum())
    tn = float(((pred_bin == 0) & (gt_bin == 0)).sum())

    pred_pos = pred_bin.sum()
    gt_pos = gt_bin.sum()
    total_bg = (gt_bin == 0).sum()

    dice = (2.0 * tp) / (pred_pos + gt_pos + 1e-12) if (pred_pos + gt_pos) > 0 else 0.0
    fpr  = fp / (total_bg + 1e-12) if total_bg > 0 else 0.0
    prec = tp / (pred_pos + 1e-12) if pred_pos > 0 else 0.0
    rec  = tp / (gt_pos + 1e-12) if gt_pos > 0 else 0.0

    return {
        "dice": float(dice),
        "fpr":  float(fpr),
        "prec": float(prec),
        "rec":  float(rec),
    }


def rank_biserial_correlation(x, y):
    diff = np.array(x) - np.array(y)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(diff))
    pos_sum = np.sum(ranks[diff > 0])
    neg_sum = np.sum(ranks[diff < 0])
    total = pos_sum + neg_sum
    if total == 0:
        return 0.0
    return float((pos_sum - neg_sum) / total)


def main():
    cfg = load_config(str(REPO_ROOT / "configs/kvasir_sessile.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = tuple(cfg["dataset"]["image_size"])

    print("=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTING: M11 vs M12 (Wilcoxon Signed-Rank)")
    print(f"Protocol: Standard 4-iteration Recurrent Refinement (Otsu -> Iter 4)")
    print(f"Device: {device} | Image Size: {size}")
    print("=" * 80)

    data_root = REPO_ROOT / cfg["dataset"]["root"]
    if not data_root.exists():
        data_root = REPO_ROOT / "data"
    (_, _), (val_x, val_y) = load_data(str(data_root))
    print(f"Validation cohort: {len(val_x)} images")

    seeds = [7, 42, 99, 1337, 2024]
    m11_dice_all, m12_dice_all = [], []
    m11_fpr_all,  m12_fpr_all  = [], []
    m11_prec_all, m12_prec_all = [], []
    m11_rec_all,  m12_rec_all  = [], []

    per_seed_results = {}

    for s in seeds:
        m11_p = REPO_ROOT / f"checkpoints_phase7b/M11_seed{s}.pth"
        m12_p = REPO_ROOT / f"checkpoints_phase7b/M12_seed{s}.pth"
        if not m12_p.exists() and s == 42:
            m12_p = REPO_ROOT / "checkpoints_phase7b/ckpt_M12_seed42.pth"

        if not m11_p.exists() or not m12_p.exists():
            print(f"[WARN] Missing seed {s} checkpoint: M11={m11_p.exists()}, M12={m12_p.exists()}")
            continue

        net_m11 = FANet(gating_mode="hard", detach_feedback=False)
        net_m11 = load_model_weights(net_m11, m11_p, device)

        net_m12 = FANet(gating_mode="soft_or", detach_feedback=True)
        net_m12 = load_model_weights(net_m12, m12_p, device)

        s_dice_11, s_dice_12 = [], []
        s_fpr_11,  s_fpr_12  = [], []
        s_prec_11, s_prec_12 = [], []
        s_rec_11,  s_rec_12  = [], []

        for img_p, msk_p in zip(val_x, val_y):
            img_bgr = cv2.imread(img_p)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, size)

            msk_gray = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
            msk_resized = cv2.resize(msk_gray, size)
            gt_bin = (msk_resized > 127).astype(np.uint8)

            otsu = otsu_bin(img_bgr, size)

            # 4-iteration recurrent refinement
            p11 = eval_fanet_recurrent(net_m11, img_resized, otsu, size, device, num_iter=4)
            p12 = eval_fanet_recurrent(net_m12, img_resized, otsu, size, device, num_iter=4)

            m11_res = compute_metrics(p11, gt_bin)
            m12_res = compute_metrics(p12, gt_bin)

            s_dice_11.append(m11_res["dice"])
            s_dice_12.append(m12_res["dice"])
            s_fpr_11.append(m11_res["fpr"])
            s_fpr_12.append(m12_res["fpr"])
            s_prec_11.append(m11_res["prec"])
            s_prec_12.append(m12_res["prec"])
            s_rec_11.append(m11_res["rec"])
            s_rec_12.append(m12_res["rec"])

        per_seed_results[s] = {
            "m11": {"dice": float(np.mean(s_dice_11)), "fpr": float(np.mean(s_fpr_11)), "prec": float(np.mean(s_prec_11)), "rec": float(np.mean(s_rec_11))},
            "m12": {"dice": float(np.mean(s_dice_12)), "fpr": float(np.mean(s_fpr_12)), "prec": float(np.mean(s_prec_12)), "rec": float(np.mean(s_rec_12))},
        }

        m11_dice_all.extend(s_dice_11)
        m12_dice_all.extend(s_dice_12)
        m11_fpr_all.extend(s_fpr_11)
        m12_fpr_all.extend(s_fpr_12)
        m11_prec_all.extend(s_prec_11)
        m12_prec_all.extend(s_prec_12)
        m11_rec_all.extend(s_rec_11)
        m12_rec_all.extend(s_rec_12)

        print(f"[EVAL DONE] Seed {s:<4} | M11 Dice={np.mean(s_dice_11):.4f}, FPR={np.mean(s_fpr_11)*100:.2f}% | M12 Dice={np.mean(s_dice_12):.4f}, FPR={np.mean(s_fpr_12)*100:.2f}%")

    n_total = len(m11_dice_all)
    print(f"\nTotal paired evaluation instances: {n_total} (40 images x 5 seeds)")

    # 1. Wilcoxon for Dice Score (Hypothesis: M12 > M11)
    res_dice = stats.wilcoxon(m12_dice_all, m11_dice_all, alternative="greater")
    r_dice   = rank_biserial_correlation(m12_dice_all, m11_dice_all)

    # 2. Wilcoxon for FPR / Over-segmentation (Hypothesis: M12 < M11)
    res_fpr  = stats.wilcoxon(m12_fpr_all, m11_fpr_all, alternative="less")
    r_fpr    = rank_biserial_correlation(m11_fpr_all, m12_fpr_all)

    # 3. Wilcoxon for Precision (Hypothesis: M12 > M11)
    res_prec = stats.wilcoxon(m12_prec_all, m11_prec_all, alternative="greater")
    r_prec   = rank_biserial_correlation(m12_prec_all, m11_prec_all)

    def fmt_p(p):
        if p < 0.001:
            return f"p < 0.001 (exact: {p:.2e})"
        return f"p = {p:.4f}"

    print("\n" + "=" * 80)
    print("STATISTICAL INFERENCE SUMMARY (Wilcoxon Signed-Rank Test across 200 pairs)")
    print("=" * 80)
    print(f"1. Dice Score Improvement (M12 vs M11):")
    print(f"   - M11 Mean ± Std : {np.mean(m11_dice_all):.4f} ± {np.std(m11_dice_all):.4f}")
    print(f"   - M12 Mean ± Std : {np.mean(m12_dice_all):.4f} ± {np.std(m12_dice_all):.4f}")
    print(f"   - Absolute Gain  : +{(np.mean(m12_dice_all) - np.mean(m11_dice_all))*100:.2f} pp")
    print(f"   - Wilcoxon Stat  : W = {res_dice.statistic:.1f}")
    print(f"   - p-value        : {fmt_p(res_dice.pvalue)}")
    print(f"   - Rank-Biserial r: r = {r_dice:.3f}")

    print(f"\n2. Over-Segmentation Suppression (FPR, M12 vs M11):")
    print(f"   - M11 Mean ± Std : {np.mean(m11_fpr_all)*100:.2f}% ± {np.std(m11_fpr_all)*100:.2f}%")
    print(f"   - M12 Mean ± Std : {np.mean(m12_fpr_all)*100:.2f}% ± {np.std(m12_fpr_all)*100:.2f}%")
    print(f"   - Relative Drop  : -{(1.0 - np.mean(m12_fpr_all)/np.mean(m11_fpr_all))*100:.1f}%")
    print(f"   - Wilcoxon Stat  : W = {res_fpr.statistic:.1f}")
    print(f"   - p-value        : {fmt_p(res_fpr.pvalue)}")
    print(f"   - Rank-Biserial r: r = {r_fpr:.3f}")

    print(f"\n3. Precision Gain (M12 vs M11):")
    print(f"   - M11 Mean ± Std : {np.mean(m11_prec_all)*100:.2f}% ± {np.std(m11_prec_all)*100:.2f}%")
    print(f"   - M12 Mean ± Std : {np.mean(m12_prec_all)*100:.2f}% ± {np.std(m12_prec_all)*100:.2f}%")
    print(f"   - Wilcoxon Stat  : W = {res_prec.statistic:.1f}")
    print(f"   - p-value        : {fmt_p(res_prec.pvalue)}")
    print("=" * 80)

    # Save to JSON
    out_json = REPO_ROOT / "diagnostics_output/phase7b/wilcoxon_stats.json"
    stats_payload = {
        "n_eval_points": n_total,
        "n_seeds": len(seeds),
        "n_images": len(val_x),
        "protocol": "4-iteration recurrent refinement with Otsu initialization",
        "dice": {
            "m11_mean": round(float(np.mean(m11_dice_all)), 4),
            "m12_mean": round(float(np.mean(m12_dice_all)), 4),
            "delta_pp": round(float((np.mean(m12_dice_all) - np.mean(m11_dice_all)) * 100), 2),
            "w_stat": float(res_dice.statistic),
            "p_value": float(res_dice.pvalue),
            "p_value_fmt": fmt_p(res_dice.pvalue),
            "rank_biserial_r": round(r_dice, 3),
        },
        "fpr": {
            "m11_mean_pct": round(float(np.mean(m11_fpr_all) * 100), 2),
            "m12_mean_pct": round(float(np.mean(m12_fpr_all) * 100), 2),
            "relative_reduction_pct": round(float((1.0 - np.mean(m12_fpr_all)/np.mean(m11_fpr_all)) * 100), 1),
            "w_stat": float(res_fpr.statistic),
            "p_value": float(res_fpr.pvalue),
            "p_value_fmt": fmt_p(res_fpr.pvalue),
            "rank_biserial_r": round(r_fpr, 3),
        },
        "precision": {
            "m11_mean": round(float(np.mean(m11_prec_all)), 4),
            "m12_mean": round(float(np.mean(m12_prec_all)), 4),
            "w_stat": float(res_prec.statistic),
            "p_value": float(res_prec.pvalue),
            "p_value_fmt": fmt_p(res_prec.pvalue),
            "rank_biserial_r": round(r_prec, 3),
        },
        "per_seed": per_seed_results,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, indent=2)

    print(f"\nSaved statistical payload to: {out_json.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
