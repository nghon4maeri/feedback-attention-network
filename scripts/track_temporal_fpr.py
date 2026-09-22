import os
import sys
import numpy as np
import torch
import cv2
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fanet.models import FANet
from fanet.data import load_data
from fanet.config import load_config

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = (256, 256)
    
    m12_ckpt = REPO_ROOT / "checkpoints_phase7b/M12_seed2024.pth"
    print(f"Loading M12: {m12_ckpt.name}")
    model = FANet(gating_mode="soft_or", detach_feedback=True).to(device)
    
    # Map weights
    raw_sd = torch.load(m12_ckpt, map_location=device)
    sd = {}
    for k, v in raw_sd.items():
        k_mapped = k.replace(".up.", ".upsample.")
        sd[k_mapped] = v
    model.load_state_dict(sd)
    model.eval()
    
    data_root = REPO_ROOT / "data/sessile-main-Kvasir-SEG"
    (_, _), (val_x, val_y) = load_data(str(data_root))
    
    print("\nTracking Temporal FPR for M12 (Detached Soft-OR) on 40 Validation Images")
    print(f"{'Iteration':<15} | {'Mean Dice':<15} | {'Mean FPR':<15}")
    print("-" * 50)
    
    # Accumulators for each iteration
    metrics_t = [{"dice": [], "fpr": []} for _ in range(4)]
    
    for img_p, msk_p in zip(val_x, val_y):
        img_bgr = cv2.imread(img_p)
        img_resized = cv2.resize(img_bgr, size)
        img_t = torch.from_numpy(np.transpose(img_resized, (2, 0, 1)) / 255.0).float().unsqueeze(0).to(device)
        
        msk_gray = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
        gt_bin = (cv2.resize(msk_gray, size) > 127).astype(np.uint8)
        
        # Otsu init
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        otsu_mask = (th / 255.0 > 0.5).astype(np.float32)
        
        prev = None
        for t in range(4):
            if prev is None:
                m = otsu_mask[None, None]
            else:
                m = (prev > 0.5).astype(np.float32)[None, None]
            m_t = torch.from_numpy(m).to(device)
            
            with torch.no_grad():
                out = model([img_t, m_t])
                prob = torch.sigmoid(out)[0, 0].cpu().numpy()
            
            pred_bin = (prob > 0.5).astype(np.uint8)
            tp = float(((pred_bin == 1) & (gt_bin == 1)).sum())
            fp = float(((pred_bin == 1) & (gt_bin == 0)).sum())
            fn = float(((pred_bin == 0) & (gt_bin == 1)).sum())
            tn = float(((pred_bin == 0) & (gt_bin == 0)).sum())
            
            pred_pos = tp + fp
            gt_pos = tp + fn
            total_bg = fp + tn
            
            dice = (2.0 * tp) / (pred_pos + gt_pos + 1e-12) if (pred_pos + gt_pos) > 0 else 0.0
            fpr = fp / (total_bg + 1e-12) if total_bg > 0 else 0.0
            
            metrics_t[t]["dice"].append(dice)
            metrics_t[t]["fpr"].append(fpr)
            
            prev = prob

    for t in range(4):
        mean_dice = np.mean(metrics_t[t]["dice"])
        mean_fpr = np.mean(metrics_t[t]["fpr"]) * 100
        print(f"t={t:<13} | {mean_dice:<15.4f} | {mean_fpr:.2f}%")

if __name__ == "__main__":
    main()
