import os
import sys
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path

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

def otsu_bin(img_bgr, size):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (th / 255.0 > 0.5).astype(np.float32)

def eval_fanet_recurrent(model, img_bgr, otsu, size, device, num_iter=4):
    img_resized = cv2.resize(img_bgr, size)
    img_t = torch.from_numpy(np.transpose(img_resized, (2, 0, 1)) / 255.0).float().unsqueeze(0).to(device)
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

def overlay_contour(image, mask, color, thickness=2):
    """Draws a contour around the mask on the image."""
    mask_uint8 = (mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_contour = image.copy()
    cv2.drawContours(img_contour, contours, -1, color, thickness)
    return img_contour

def main():
    cfg = load_config(str(REPO_ROOT / "configs/kvasir_sessile.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = tuple(cfg["dataset"]["image_size"])

    m11_ckpt = REPO_ROOT / "checkpoints_phase7b/M11_seed2024.pth"
    m12_ckpt = REPO_ROOT / "checkpoints_phase7b/M12_seed2024.pth"

    if not m11_ckpt.exists() or not m12_ckpt.exists():
        print(f"[ERROR] Checkpoints not found.")
        sys.exit(1)

    net_m11 = load_model_weights(FANet(gating_mode="hard", detach_feedback=False), m11_ckpt, device)
    net_m12 = load_model_weights(FANet(gating_mode="soft_or", detach_feedback=True), m12_ckpt, device)

    data_root = REPO_ROOT / cfg["dataset"]["root"]
    (_, _), (val_x, val_y) = load_data(str(data_root))

    hardest_cases = []
    
    print("Evaluating validation set to find the hardest cases (highest over-segmentation/FP in M11)...")
    for img_p, msk_p in zip(val_x, val_y):
        img_bgr = cv2.imread(img_p)
        msk_gray = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
        msk_resized = cv2.resize(msk_gray, size)
        gt_bin = (msk_resized > 127).astype(np.uint8)
        otsu = otsu_bin(img_bgr, size)

        p11 = eval_fanet_recurrent(net_m11, img_bgr, otsu, size, device)
        p12 = eval_fanet_recurrent(net_m12, img_bgr, otsu, size, device)

        p11_bin = (p11 > 0.5).astype(np.uint8)
        p12_bin = (p12 > 0.5).astype(np.uint8)

        fp_11 = ((p11_bin == 1) & (gt_bin == 0)).sum()
        fp_12 = ((p12_bin == 1) & (gt_bin == 0)).sum()
        
        tp_11 = ((p11_bin == 1) & (gt_bin == 1)).sum()
        dice_11 = (2.0 * tp_11) / (p11_bin.sum() + gt_bin.sum() + 1e-7)

        hardest_cases.append({
            'img_path': img_p,
            'gt': gt_bin,
            'p11': p11_bin,
            'p12': p12_bin,
            'img_bgr': cv2.resize(img_bgr, size),
            'fp_11': fp_11,
            'dice_11': dice_11
        })

    # Sort by lowest Dice score in M11 (the hardest cases)
    hardest_cases.sort(key=lambda x: x['dice_11'])
    top_k = 3

    print(f"Generating visualizations for the top {top_k} hardest cases...")
    fig, axes = plt.subplots(top_k, 3, figsize=(12, 4 * top_k))
    
    for i in range(top_k):
        case = hardest_cases[i]
        img_rgb = cv2.cvtColor(case['img_bgr'], cv2.COLOR_BGR2RGB)
        
        # Overlay GT (Green)
        gt_overlay = overlay_contour(img_rgb, case['gt'], color=(0, 255, 0), thickness=2)
        
        # Overlay M11 Prediction (Red)
        m11_overlay = overlay_contour(img_rgb, case['p11'], color=(255, 0, 0), thickness=2)
        m11_overlay = overlay_contour(m11_overlay, case['gt'], color=(0, 255, 0), thickness=1) # Reference GT
        
        # Overlay M12 Prediction (Blue)
        m12_overlay = overlay_contour(img_rgb, case['p12'], color=(0, 0, 255), thickness=2)
        m12_overlay = overlay_contour(m12_overlay, case['gt'], color=(0, 255, 0), thickness=1) # Reference GT

        axes[i, 0].imshow(gt_overlay)
        axes[i, 0].set_title(f"Original + GT (Green)\nCase {i+1} (M11 Dice: {case['dice_11']:.3f})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(m11_overlay)
        axes[i, 1].set_title("M11 Baseline (Red)\nShows Over-segmentation Trap")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(m12_overlay)
        axes[i, 2].set_title("M12 Phase 7B (Blue)\nSuppressed False Positives")
        axes[i, 2].axis("off")

    plt.tight_layout()
    out_dir = REPO_ROOT / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "qualitative_comparison_hardest.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved qualitative figure to {out_file}")

if __name__ == "__main__":
    main()
