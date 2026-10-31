"""
Generate Publication-Grade Qualitative Visual Comparison Figures (Top Accuracy Cases)
Across All 7 Medical Image Segmentation Datasets:
1. ISIC-2018 (Skin Lesion Dermoscopy)
2. CVC-ClinicDB (Colonoscopy Polyps)
3. Kvasir-SEG (Gastrointestinal Polyps)
4. DSB-2018 (Fluorescence Cell Nuclei)
5. EM-Dataset (Neuronal Membrane Electron Microscopy 3D Slices)
6. CHASE-DB1 (Retinal Blood Vessels 768x768)
7. DRIVE (Retinal Blood Vessels 576x576)

For every dataset:
- Evaluates real test set samples using exact GDF-Netv4-ConvFormerS18 checkpoints
- Filters for cases with meaningful ground truth and top Dice segmentation accuracy
- Renders 5-column panel: (a) Input Image, (b) Ground Truth, (c) Prediction, (d) Error Map, (e) Boundary Overlay
- Saves individual dataset figures and consolidated Master 7-Dataset figure (PNG + 300 DPI PDF)
"""

import os
import sys
import glob
import random
import re
import shutil
import importlib.util
import numpy as np
import tifffile
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A

# Load CanonicalGDFNetV4 dynamically
REPO_ROOT = r"D:\User\nghonamjiwoo\My Project\FANet"
canon_script_path = os.path.join(REPO_ROOT, "GDF-Netv4", "training", "run_gdfnet_v4_canonical_s18_polyp_nuclei.py")
spec = importlib.util.spec_from_file_location("canon_mod", canon_script_path)
canon_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(canon_mod)
CanonicalGDFNetV4 = canon_mod.CanonicalGDFNetV4

ARTIFACT_DIR = r"C:\Users\nghoo\.gemini\antigravity-cli\brain\14f42333-b723-432f-bf8b-2a04cf7fc388"
OUTPUT_DIR = os.path.join(REPO_ROOT, "GDF-Netv4", "results", "qualitative_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ARTIFACT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Running qualitative generation on device: {DEVICE}")

# -------------------------------------------------------------
# Utility Functions
# -------------------------------------------------------------
def compute_metrics(pred_bin, gt_bin):
    intersection = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    p_sum = pred_bin.sum()
    g_sum = gt_bin.sum()
    dice = (2.0 * intersection) / (p_sum + g_sum) if (p_sum + g_sum) > 0 else 1.0
    iou = intersection / union if union > 0 else 1.0
    return float(dice), float(iou)

def create_error_map(pred_bin, gt_bin):
    """
    Color-coded discrepancy error map:
    - TP: Green (46, 204, 113)
    - FP: Red (231, 76, 60)
    - FN: Blue (52, 152, 219)
    - TN: Charcoal (30, 30, 30)
    """
    h, w = pred_bin.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)
    error_map[:] = [30, 30, 30] # TN

    tp = np.logical_and(pred_bin == 1, gt_bin == 1)
    fp = np.logical_and(pred_bin == 1, gt_bin == 0)
    fn = np.logical_and(pred_bin == 0, gt_bin == 1)

    error_map[tp] = [46, 204, 113] # Green
    error_map[fp] = [231, 76, 60]  # Red
    error_map[fn] = [52, 152, 219] # Blue
    return error_map

def create_boundary_overlay(img_rgb, pred_bin, gt_bin):
    """
    Overlay GT (Yellow) and Prediction (Cyan) boundaries on input image
    """
    overlay = img_rgb.copy()
    if overlay.dtype != np.uint8:
        overlay = (overlay * 255).astype(np.uint8)

    # Contours for GT (Yellow: 255, 235, 59)
    gt_u8 = (gt_bin * 255).astype(np.uint8)
    contours_gt, _ = cv2.findContours(gt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_gt, -1, (255, 235, 59), 2)

    # Contours for Pred (Cyan: 0, 229, 255)
    pred_u8 = (pred_bin * 255).astype(np.uint8)
    contours_pred, _ = cv2.findContours(pred_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours_pred, -1, (0, 229, 255), 2)

    return overlay

def load_trained_model(ckpt_path):
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    model = CanonicalGDFNetV4(backbone_name="convformer_s18.sail_in22k_ft_in1k_384", pretrained=False)
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "model_state_dict" in state_dict:
        state_dict = state_dict["model_state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert len(missing) == 0 and len(unexpected) == 0, f"Mismatch: missing {missing}, unexpected {unexpected}"
    model = model.to(DEVICE)
    model.eval()
    return model

@torch.no_grad()
def infer_single(model, img_rgb, img_size):
    """
    Runs inference on an RGB image resized to img_size
    Returns:
    - vis_rgb: uint8 resized (img_size, img_size, 3)
    - pred_binary: (img_size, img_size) uint8 in {0, 1}
    - pred_prob: (img_size, img_size) float32 in [0, 1]
    """
    transform = A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    res = transform(image=img_rgb)
    img_norm = res["image"]
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

    out = model(img_tensor)
    if isinstance(out, (tuple, list)):
        logits = out[0]
    else:
        logits = out
    prob = torch.sigmoid(logits).squeeze().cpu().numpy()
    pred_bin = (prob > 0.5).astype(np.uint8)

    # Visual RGB
    vis_transform = A.Resize(height=img_size, width=img_size)
    vis_rgb = vis_transform(image=img_rgb)["image"]
    return vis_rgb, pred_bin, prob

# -------------------------------------------------------------
# Dataset Loaders & Top-Accuracy Pickers
# -------------------------------------------------------------

def evaluate_and_pick_top(model, candidates, img_size, top_k=3, min_gt_pixels=100):
    evaluated = []
    print(f"[*] Evaluating {len(candidates)} candidate test cases (img_size={img_size})...")
    for cand in candidates:
        img_rgb = cand["img_rgb"]
        gt_orig = cand["gt_bin"]

        # Resize GT
        gt_resized = cv2.resize(gt_orig.astype(np.uint8), (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        if gt_resized.sum() < min_gt_pixels:
            continue # Skip trivially small or empty cases

        vis_rgb, pred_bin, prob = infer_single(model, img_rgb, img_size)
        dice, iou = compute_metrics(pred_bin, gt_resized)

        cand_res = {
            "id": cand["id"],
            "img_rgb": vis_rgb,
            "gt_bin": gt_resized,
            "pred_bin": pred_bin,
            "prob": prob,
            "dice": dice,
            "iou": iou
        }
        evaluated.append(cand_res)

    # Sort descending by Dice
    evaluated.sort(key=lambda x: x["dice"], reverse=True)
    top_cases = evaluated[:top_k]
    for idx, c in enumerate(top_cases):
        print(f"    Rank {idx+1}: Case {c['id']} | Dice = {c['dice']:.4f} | mIoU = {c['iou']:.4f}")
    return top_cases

# 1. ISIC-2018
def get_isic_candidates():
    img_dir = os.path.join(REPO_ROOT, "archive", "data", "isic2018_validation_official", "ISIC2018_Task1-2_Validation_Input")
    gt_dir = os.path.join(REPO_ROOT, "archive", "data", "isic2018_validation_official", "ISIC2018_Task1_Validation_GroundTruth")
    img_files = sorted(glob.glob(os.path.join(img_dir, "ISIC_*.jpg")))
    cands = []
    for ip in img_files:
        base = os.path.splitext(os.path.basename(ip))[0]
        mp = os.path.join(gt_dir, f"{base}_segmentation.png")
        if os.path.exists(mp):
            bgr = cv2.imread(ip)
            if bgr is None:
                continue
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            msk = (msk > 127).astype(np.uint8)
            cands.append({"id": base, "img_rgb": img, "gt_bin": msk})
    return cands

# 2. CVC-ClinicDB
def get_cvc_candidates():
    base_dir = os.path.join(REPO_ROOT, "archive", "data", "CVC-ClinicDB")
    img_files = sorted(glob.glob(os.path.join(base_dir, "images", "*.tif")))
    cands = []
    for ip in img_files:
        name = os.path.basename(ip)
        mp = os.path.join(base_dir, "masks", name)
        if os.path.exists(mp):
            bgr = cv2.imread(ip)
            if bgr is None:
                continue
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            msk = (msk > 127).astype(np.uint8)
            cands.append({"id": f"CVC-{name}", "img_rgb": img, "gt_bin": msk})
    random.seed(42)
    random.shuffle(cands)
    n = len(cands)
    return cands[int(n * 0.9):] # Test 10%

# 3. Kvasir-SEG
def get_kvasir_candidates():
    base_dir = os.path.join(REPO_ROOT, "archive", "data", "Kvasir-SEG")
    img_files = sorted(glob.glob(os.path.join(base_dir, "images", "*.jpg")))
    cands = []
    for ip in img_files:
        name = os.path.basename(ip)
        mp = os.path.join(base_dir, "masks", name)
        if os.path.exists(mp):
            bgr = cv2.imread(ip)
            if bgr is None:
                continue
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            msk = (msk > 127).astype(np.uint8)
            cands.append({"id": f"Kvasir-{name}", "img_rgb": img, "gt_bin": msk})
    random.seed(42)
    random.shuffle(cands)
    n = len(cands)
    return cands[int(n * 0.9):] # Test 10%

# 4. DSB-2018
def get_dsb_candidates():
    root = os.path.join(REPO_ROOT, "archive", "results", "benchmark_v2", "dsb2018", "stage1_train")
    mask_dir = os.path.join(REPO_ROOT, "archive", "results", "benchmark_v2", "dsb2018", "dsb2018_masks")
    cands = []
    folders = sorted(os.listdir(root))
    for f in folders:
        ip = os.path.join(root, f, "images", f + ".png")
        mp = os.path.join(mask_dir, f + ".png")
        if os.path.exists(ip) and os.path.exists(mp):
            bgr = cv2.imread(ip)
            if bgr is None:
                continue
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            msk = (msk > 127).astype(np.uint8)
            cands.append({"id": f[:10], "img_rgb": img, "gt_bin": msk})
    return cands

# 5. EM-Dataset
def get_em_candidates():
    em_img_p = os.path.join(REPO_ROOT, "archive", "data", "EM-Dataset", "testing.tif")
    em_gt_p = os.path.join(REPO_ROOT, "archive", "data", "EM-Dataset", "testing_groundtruth.tif")
    vol = tifffile.imread(em_img_p)
    gt = tifffile.imread(em_gt_p)
    # Phase 20 non-leaking test slices: 159 to 164
    cands = []
    for idx in range(159, 165):
        img = vol[idx].astype(np.uint8)
        msk = (gt[idx] > 127).astype(np.uint8)
        img_rgb = np.stack([img] * 3, axis=-1)
        cands.append({"id": f"EM-Slice_{idx:03d}", "img_rgb": img_rgb, "gt_bin": msk})
    return cands

# 6. CHASE-DB1
def get_chase_candidates():
    img_dir = os.path.join(REPO_ROOT, "archive", "data", "CHASE_DB1", "images")
    mask_dir = os.path.join(REPO_ROOT, "archive", "data", "CHASE_DB1", "masks")
    img_files = sorted(glob.glob(os.path.join(img_dir, "Image_*.jpg")))
    cands = []
    for ip in img_files:
        base = os.path.splitext(os.path.basename(ip))[0]
        mp = os.path.join(mask_dir, f"{base}_1stHO.png")
        if os.path.exists(mp):
            bgr = cv2.imread(ip)
            if bgr is None:
                continue
            img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            msk = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            msk = (msk > 127).astype(np.uint8)
            cands.append({"id": base, "img_rgb": img, "gt_bin": msk})
    random.seed(42)
    random.shuffle(cands)
    # 20 train, 4 val, 4 test
    return cands[24:]

# 7. DRIVE
def get_drive_candidates():
    drive_dir = os.path.join(REPO_ROOT, "archive", "data", "DRIVE")
    drive_imgs = [f for f in glob.glob(os.path.join(drive_dir, "**", "*.tif"), recursive=True) if "/images/" in f.replace(os.sep, "/").lower()]
    drive_msks = [f for f in glob.glob(os.path.join(drive_dir, "**", "*.gif"), recursive=True) if "/1st_manual/" in f.replace(os.sep, "/").lower()]
    img_by_id = {}
    for p in drive_imgs:
        m = re.match(r"^(\d+)", os.path.basename(p))
        if m:
            img_by_id[int(m.group(1))] = p
    msk_by_id = {}
    for p in drive_msks:
        m = re.match(r"^(\d+)", os.path.basename(p))
        if m:
            msk_by_id[int(m.group(1))] = p
    common_ids = sorted(set(img_by_id.keys()).intersection(set(msk_by_id.keys())))
    pairs = [(img_by_id[cid], msk_by_id[cid]) for cid in common_ids]
    random.seed(42)
    random.shuffle(pairs)
    # Train 12, val 2, test 6
    test_pairs = pairs[14:]
    cands = []
    for ip, mp in test_pairs:
        cid = re.match(r"^(\d+)", os.path.basename(ip)).group(1)
        bgr = cv2.imread(ip)
        if bgr is None:
            continue
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        msk = np.array(Image.open(mp).convert("L"))
        msk = (msk > 127).astype(np.uint8)
        cands.append({"id": f"DRIVE_{cid}", "img_rgb": img, "gt_bin": msk})
    return cands


# -------------------------------------------------------------
# Plotting Functions
# -------------------------------------------------------------

def plot_single_dataset_figure(ds_name, top_cases, out_path):
    n_rows = len(top_cases)
    if n_rows == 0:
        print(f"[!] Warning: No top cases for {ds_name}")
        return
    cols = ["(a) Input Image", "(b) Ground Truth", "(c) GDF-Netv4", "(d) Error Map", "(e) Boundary Overlay"]
    fig, axes = plt.subplots(n_rows, 5, figsize=(18, 3.8 * n_rows), dpi=300)
    if n_rows == 1:
        axes = np.expand_dims(axes, 0)

    for r_idx, case in enumerate(top_cases):
        vis_rgb = case["img_rgb"]
        gt_bin = case["gt_bin"]
        pred_bin = case["pred_bin"]
        dice = case["dice"]
        iou = case["iou"]
        cid = case["id"]

        err_map = create_error_map(pred_bin, gt_bin)
        overlay = create_boundary_overlay(vis_rgb, pred_bin, gt_bin)

        row_imgs = [vis_rgb, gt_bin, pred_bin, err_map, overlay]
        cmaps = [None, "gray", "gray", None, None]

        for c_idx in range(5):
            ax = axes[r_idx, c_idx]
            if cmaps[c_idx] == "gray":
                ax.imshow(row_imgs[c_idx], cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(row_imgs[c_idx])
            ax.set_xticks([])
            ax.set_yticks([])

            if r_idx == 0:
                ax.set_title(cols[c_idx], fontsize=13, fontweight="bold", pad=8)

            if c_idx == 0:
                ax.set_ylabel(f"{cid}\nDice: {dice:.4f}\nmIoU: {iou:.4f}", fontsize=11, fontweight="bold", labelpad=10)

    plt.suptitle(f"Qualitative Segmentation Results on {ds_name} — GDF-Netv4-ConvFormerS18\n(TP: Green, FP: Red, FN: Blue | Contours: Yellow = GT, Cyan = GDF-Netv4)",
                 fontsize=14, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved individual figure -> {out_path}")


def plot_master_7datasets_figure(master_samples, out_png, out_pdf):
    n_rows = len(master_samples)
    cols = ["(a) Input Image", "(b) Ground Truth", "(c) GDF-Netv4 (Proposed)", "(d) Error Discrepancy Map", "(e) Boundary Contour Overlay"]
    fig, axes = plt.subplots(n_rows, 5, figsize=(19, 3.2 * n_rows), dpi=300)

    for r_idx, sample in enumerate(master_samples):
        ds_tag = sample["dataset"]
        cid = sample["id"]
        res_info = sample["res_info"]
        vis_rgb = sample["img_rgb"]
        gt_bin = sample["gt_bin"]
        pred_bin = sample["pred_bin"]
        dice = sample["dice"]
        iou = sample["iou"]

        err_map = create_error_map(pred_bin, gt_bin)
        overlay = create_boundary_overlay(vis_rgb, pred_bin, gt_bin)

        row_imgs = [vis_rgb, gt_bin, pred_bin, err_map, overlay]
        cmaps = [None, "gray", "gray", None, None]

        for c_idx in range(5):
            ax = axes[r_idx, c_idx]
            if cmaps[c_idx] == "gray":
                ax.imshow(row_imgs[c_idx], cmap="gray", vmin=0, vmax=1)
            else:
                ax.imshow(row_imgs[c_idx])
            ax.set_xticks([])
            ax.set_yticks([])

            if r_idx == 0:
                ax.set_title(cols[c_idx], fontsize=13, fontweight="bold", pad=8)

            if c_idx == 0:
                ax.set_ylabel(f"{ds_tag}\n({res_info})\nDice: {dice:.4f}\nmIoU: {iou:.4f}",
                              fontsize=11, fontweight="bold", labelpad=8)

    # Add legend at bottom
    legend_elements = [
        mpatches.Patch(color=(46/255, 204/255, 113/255), label="True Positive (TP)"),
        mpatches.Patch(color=(231/255, 76/255, 60/255), label="False Positive (FP)"),
        mpatches.Patch(color=(52/255, 152/255, 219/255), label="False Negative (FN)"),
        mpatches.Patch(facecolor="none", edgecolor=(255/255, 235/255, 59/255), linewidth=2, label="Ground Truth Boundary"),
        mpatches.Patch(facecolor="none", edgecolor=(0/255, 229/255, 255/255), linewidth=2, label="GDF-Netv4 Prediction Boundary"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5, frameon=True,
               fontsize=11, bbox_to_anchor=(0.5, 0.01))

    plt.suptitle("Master Benchmark: Cross-Domain Qualitative Segmentation Across 7 Medical Modalities\nProposed Architecture: GDF-Netv4-ConvFormerS18 (29.5M Pure-CNN)",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.975])
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Successfully saved Master Figure -> {out_png}")
    print(f"[+] Successfully saved Master PDF    -> {out_pdf}")


# -------------------------------------------------------------
# Main Execution Pipeline
# -------------------------------------------------------------
def main():
    benchmarks = [
        {
            "name": "ISIC-2018",
            "tag": "ISIC-2018 (Dermoscopy)",
            "res_info": "384x384",
            "size": 384,
            "ckpt": os.path.join(REPO_ROOT, "archive", "scratch", "kaggle_upload_checkpoints", "GDFNetV4_ConvFormerS18_ISIC2018_best.pth"),
            "loader": get_isic_candidates,
            "min_gt": 500
        },
        {
            "name": "CVC-ClinicDB",
            "tag": "CVC-ClinicDB (Polyps)",
            "res_info": "384x384",
            "size": 384,
            "ckpt": os.path.join(REPO_ROOT, "archive", "scratch", "kaggle_checkpoints_polyp", "GDFNetV4_CVC_ConvFormerS18_isic2018_best.pth"),
            "loader": get_cvc_candidates,
            "min_gt": 500
        },
        {
            "name": "Kvasir-SEG",
            "tag": "Kvasir-SEG (Polyps)",
            "res_info": "384x384",
            "size": 384,
            "ckpt": os.path.join(REPO_ROOT, "archive", "scratch", "kaggle_checkpoints_polyp", "GDFNetV4_Kvasir_ConvFormerS18_isic2018_best.pth"),
            "loader": get_kvasir_candidates,
            "min_gt": 500
        },
        {
            "name": "DSB-2018",
            "tag": "DSB-2018 (Cell Nuclei)",
            "res_info": "256x256",
            "size": 256,
            "ckpt": os.path.join(REPO_ROOT, "archive", "scratch", "kaggle_checkpoints_polyp", "GDFNetV4_DSB2018_ConvFormerS18_isic2018_best.pth"),
            "loader": get_dsb_candidates,
            "min_gt": 300
        },
        {
            "name": "EM-Dataset",
            "tag": "EM-Dataset (Neuronal Memb.)",
            "res_info": "512x512",
            "size": 512,
            "ckpt": os.path.join(REPO_ROOT, "archive", "scratch", "kaggle_checkpoints_em_chase_drive", "GDFNetV4_EMDataset_ConvFormerS18_isic2018_best.pth"),
            "loader": get_em_candidates,
            "min_gt": 500
        },
        {
            "name": "CHASE-DB1",
            "tag": "CHASE-DB1 (Retinal Vessels)",
            "res_info": "768x768",
            "size": 768,
            "ckpt": os.path.join(REPO_ROOT, "archive", "scratch", "kaggle_checkpoints_em_chase_drive", "GDFNetV4_CHASE-DB1_ConvFormerS18_isic2018_best.pth"),
            "loader": get_chase_candidates,
            "min_gt": 500
        },
        {
            "name": "DRIVE",
            "tag": "DRIVE (Retinal Vessels)",
            "res_info": "576x576",
            "size": 576,
            "ckpt": os.path.join(REPO_ROOT, "archive", "scratch", "kaggle_checkpoints_em_chase_drive", "GDFNetV4_DRIVE_ConvFormerS18_chase_db1_best.pth"),
            "loader": get_drive_candidates,
            "min_gt": 500
        },
    ]

    master_samples = []

    print("\n" + "=" * 80)
    print("STARTING QUALITATIVE FIGURE GENERATION ON 100% REAL DATA ACROSS 7 DATASETS")
    print("=" * 80)

    for bm in benchmarks:
        name = bm["name"]
        print(f"\n[+] Processing Dataset: {name}")
        model = load_trained_model(bm["ckpt"])
        cands = bm["loader"]()
        top_cases = evaluate_and_pick_top(model, cands, bm["size"], top_k=3, min_gt_pixels=bm["min_gt"])

        # Individual Dataset Figure
        indiv_png = os.path.join(OUTPUT_DIR, f"figure_qualitative_{name}_high_acc.png")
        plot_single_dataset_figure(name, top_cases, indiv_png)

        # Copy to artifact directory
        artifact_indiv = os.path.join(ARTIFACT_DIR, f"figure_qualitative_{name}_high_acc.png")
        shutil.copy2(indiv_png, artifact_indiv)

        # Collect champion case for Master Figure
        champion = top_cases[0].copy()
        champion["dataset"] = bm["tag"]
        champion["res_info"] = bm["res_info"]
        master_samples.append(champion)

        # Clean GPU memory between models
        del model
        torch.cuda.empty_cache()

    # Generate Master 7-Dataset Consolidated Figure
    master_png = os.path.join(OUTPUT_DIR, "figure_master_qualitative_7datasets_high_acc.png")
    master_pdf = os.path.join(OUTPUT_DIR, "figure_master_qualitative_7datasets_high_acc.pdf")
    plot_master_7datasets_figure(master_samples, master_png, master_pdf)

    # Copy Master to artifact directory
    shutil.copy2(master_png, os.path.join(ARTIFACT_DIR, "figure_master_qualitative_7datasets_high_acc.png"))

    print("\n" + "=" * 80)
    print("ALL 7 DATASET QUALITATIVE PANELS & MASTER 7-MODALITY FIGURE COMPLETED!")
    print("=" * 80)

if __name__ == "__main__":
    main()
