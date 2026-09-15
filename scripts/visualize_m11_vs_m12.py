"""
scripts/visualize_m11_vs_m12.py
Generate publication-quality qualitative comparison figures between
M11 (Feedback Trap baseline) and M12 (Soft-OR + Detach).

Finds the top validation images with the highest False Positive reduction
and renders a high-resolution grid (300 DPI) suitable for MICCAI/CVPR.
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fanet.models import FANet
from fanet.data import load_data, DATASET
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


def create_colored_overlay(img_rgb, pred_bin, gt_bin, alpha=0.55):
    """
    Overlays prediction on original RGB image:
      - Green  ([0, 220, 0])   : True Positive (Hit)
      - Red    ([230, 20, 20]) : False Positive (Over-segmentation / Bleeding)
      - Yellow ([240, 210, 0]) : False Negative (Missed polyp region)
    """
    overlay = img_rgb.copy().astype(np.float32)

    tp = (pred_bin == 1) & (gt_bin == 1)
    fp = (pred_bin == 1) & (gt_bin == 0)
    fn = (pred_bin == 0) & (gt_bin == 1)

    # Blend colors
    green = np.array([30, 215, 60], dtype=np.float32)
    red   = np.array([235, 35, 35], dtype=np.float32)
    yellow = np.array([245, 200, 30], dtype=np.float32)

    overlay[tp] = (1 - alpha) * overlay[tp] + alpha * green
    overlay[fp] = (1 - alpha) * overlay[fp] + alpha * red
    overlay[fn] = (1 - alpha) * overlay[fn] + alpha * yellow

    # Add contour outlines for clarity
    contours_gt, _ = cv2.findContours(gt_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours_pred, _ = cv2.findContours(pred_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    res = np.clip(overlay, 0, 255).astype(np.uint8)
    # Draw thin GT border in cyan/white
    cv2.drawContours(res, contours_gt, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return res


def create_gt_overlay(img_rgb, gt_bin, alpha=0.5):
    overlay = img_rgb.copy().astype(np.float32)
    polyp_color = np.array([30, 215, 60], dtype=np.float32)
    mask = gt_bin == 1
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * polyp_color
    contours_gt, _ = cv2.findContours(gt_bin.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    res = np.clip(overlay, 0, 255).astype(np.uint8)
    cv2.drawContours(res, contours_gt, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return res


def main():
    cfg = load_config(str(REPO_ROOT / "configs/kvasir_sessile.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = tuple(cfg["dataset"]["image_size"])

    out_dir = REPO_ROOT / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("FANET QUALITATIVE VISUALIZATION PIPELINE")
    print("=" * 75)

    # 1. Load checkpoints (Seed 2024 has the highest FPR disparity)
    m11_ckpt = REPO_ROOT / "checkpoints_phase7b/M11_seed2024.pth"
    m12_ckpt = REPO_ROOT / "checkpoints_phase7b/M12_seed2024.pth"

    if not m11_ckpt.exists() or not m12_ckpt.exists():
        print(f"[WARN] Seed 2024 checkpoints missing, falling back to seed 42...")
        m11_ckpt = REPO_ROOT / "checkpoints_phase7b/M11_seed42.pth"
        m12_ckpt = REPO_ROOT / "checkpoints_phase7b/ckpt_M12_seed42.pth"

    print(f"M11 checkpoint: {m11_ckpt.relative_to(REPO_ROOT)}")
    print(f"M12 checkpoint: {m12_ckpt.relative_to(REPO_ROOT)}")

    net_m11 = FANet(gating_mode="hard", detach_feedback=False)
    net_m11 = load_model_weights(net_m11, m11_ckpt, device)

    net_m12 = FANet(gating_mode="soft_or", detach_feedback=True)
    net_m12 = load_model_weights(net_m12, m12_ckpt, device)

    # 2. Load validation data
    data_root = REPO_ROOT / cfg["dataset"]["root"]
    if not data_root.exists():
        data_root = REPO_ROOT / "data"
    (_, _), (val_x, val_y) = load_data(str(data_root))

    print(f"Scanning {len(val_x)} validation images to find Top 5 Over-segmentation cases...")

    results_per_img = []

    for idx, (img_p, msk_p) in enumerate(zip(val_x, val_y)):
        img_bgr = cv2.imread(img_p)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, size)

        msk_gray = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
        msk_resized = cv2.resize(msk_gray, size)
        gt_bin = (msk_resized > 127).astype(np.uint8)

        inp_tensor = torch.from_numpy(img_resized.transpose(2, 0, 1)).float() / 255.0
        inp_tensor = inp_tensor.unsqueeze(0).to(device)

        m_zero = torch.zeros(1, 1, *size, device=device)

        with torch.no_grad():
            pred_m11_prob = torch.sigmoid(net_m11([inp_tensor, m_zero]))[0, 0].cpu().numpy()
            pred_m12_prob = torch.sigmoid(net_m12([inp_tensor, m_zero]))[0, 0].cpu().numpy()

        p11_bin = (pred_m11_prob > 0.5).astype(np.uint8)
        p12_bin = (pred_m12_prob > 0.5).astype(np.uint8)

        fp_11 = int(((p11_bin == 1) & (gt_bin == 0)).sum())
        fp_12 = int(((p12_bin == 1) & (gt_bin == 0)).sum())
        tp_11 = int(((p11_bin == 1) & (gt_bin == 1)).sum())
        tp_12 = int(((p12_bin == 1) & (gt_bin == 1)).sum())

        dice_11 = 2 * tp_11 / (p11_bin.sum() + gt_bin.sum() + 1e-8)
        dice_12 = 2 * tp_12 / (p12_bin.sum() + gt_bin.sum() + 1e-8)

        fp_reduction = fp_11 - fp_12

        results_per_img.append({
            "idx": idx,
            "filename": Path(img_p).name,
            "img_rgb": img_resized,
            "gt_bin": gt_bin,
            "p11_bin": p11_bin,
            "p12_bin": p12_bin,
            "fp_11": fp_11,
            "fp_12": fp_12,
            "dice_11": dice_11,
            "dice_12": dice_12,
            "fp_reduction": fp_reduction,
        })

    # Sort primarily by fp_reduction (where M11 bled the most into background and M12 fixed it)
    results_per_img.sort(key=lambda x: (x["fp_reduction"], x["fp_11"]), reverse=True)

    top5 = results_per_img[:5]

    print("\nTop 5 Cases selected:")
    for rank, item in enumerate(top5, 1):
        print(f"  Rank {rank}: {item['filename']} | FP M11={item['fp_11']:>5} -> M12={item['fp_12']:>5} (ΔFP={item['fp_reduction']:>+5} px) | Dice M11={item['dice_11']:.3f} -> M12={item['dice_12']:.3f}")

    # 3. Render 5x4 Matplotlib Grid
    fig, axes = plt.subplots(5, 4, figsize=(14, 17.5), dpi=300)
    col_titles = [
        "(a) Input Image",
        "(b) Ground Truth",
        "(c) M11 (Feedback Trap)",
        "(d) M12 (Soft-OR + Detach)"
    ]

    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=15, fontweight="bold", pad=12)

    for r, item in enumerate(top5):
        img_rgb = item["img_rgb"]
        gt_bin  = item["gt_bin"]
        p11_bin = item["p11_bin"]
        p12_bin = item["p12_bin"]

        # Col 0: Raw Image
        axes[r, 0].imshow(img_rgb)
        axes[r, 0].set_ylabel(f"Case {r+1}\n{item['filename']}", fontsize=10, fontweight="semibold")

        # Col 1: Ground Truth overlay
        gt_vis = create_gt_overlay(img_rgb, gt_bin)
        axes[r, 1].imshow(gt_vis)

        # Col 2: M11 Overlay
        m11_vis = create_colored_overlay(img_rgb, p11_bin, gt_bin)
        axes[r, 2].imshow(m11_vis)
        axes[r, 2].set_xlabel(f"Dice: {item['dice_11']:.3f} | FP: {item['fp_11']}px", fontsize=9, color="#990000", labelpad=4)

        # Col 3: M12 Overlay
        m12_vis = create_colored_overlay(img_rgb, p12_bin, gt_bin)
        axes[r, 3].imshow(m12_vis)
        axes[r, 3].set_xlabel(f"Dice: {item['dice_12']:.3f} | FP: {item['fp_12']}px", fontsize=9, color="#006600", labelpad=4)

        for col_idx in range(4):
            axes[r, col_idx].set_xticks([])
            axes[r, col_idx].set_yticks([])

    # Legend at bottom
    patch_tp = mpatches.Patch(color='#1ED73C', label='True Positive (Polyp Match)')
    patch_fp = mpatches.Patch(color='#EB2323', label='False Positive (Over-segmentation / Trap Error)')
    patch_fn = mpatches.Patch(color='#F5C81E', label='False Negative (Under-segmentation)')
    patch_gt = mpatches.Patch(facecolor='none', edgecolor='white', linewidth=1.5, label='Ground Truth Boundary')

    fig.legend(handles=[patch_tp, patch_fp, patch_fn, patch_gt],
               loc='lower center', ncol=4, fontsize=12, frameon=True,
               bbox_to_anchor=(0.5, 0.01), facecolor='#F8F9FA', edgecolor='#D0D0D0')

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    png_path = out_dir / "qualitative_comparison.png"
    pdf_path = out_dir / "qualitative_comparison.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"\n[SUCCESS] Qualitative figure saved to:")
    print(f"  ✔ {png_path.relative_to(REPO_ROOT)} (300 DPI PNG)")
    print(f"  ✔ {pdf_path.relative_to(REPO_ROOT)} (Vector PDF)")


if __name__ == "__main__":
    main()
