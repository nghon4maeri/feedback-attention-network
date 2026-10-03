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

def load_model_weights(model, ckpt_path, device):
    raw_sd = torch.load(ckpt_path, map_location=device)
    sd = {}
    for k, v in raw_sd.items():
        k_mapped = k
        for i in range(1, 5):
            if k_mapped.startswith(f"d{i}.up."):
                k_mapped = k_mapped.replace(f"d{i}.up.", f"d{i}.upsample.")
        sd[k_mapped] = v
    model.load_state_dict(sd, strict=False)
    return model

def compute_metrics(pred_prob, gt_bin):
    pred_bin = (pred_prob > 0.5).astype(np.uint8)
    tp = float(((pred_bin == 1) & (gt_bin == 1)).sum())
    fp = float(((pred_bin == 1) & (gt_bin == 0)).sum())
    fn = float(((pred_bin == 0) & (gt_bin == 1)).sum())
    tn = float(((pred_bin == 0) & (gt_bin == 0)).sum())

    pred_pos = tp + fp
    gt_pos = tp + fn
    total_bg = fp + tn

    dice = (2.0 * tp) / (pred_pos + gt_pos + 1e-12) if (pred_pos + gt_pos) > 0 else 0.0
    prec = tp / (pred_pos + 1e-12) if pred_pos > 0 else 0.0
    rec = tp / (gt_pos + 1e-12) if gt_pos > 0 else 0.0
    fpr = fp / (total_bg + 1e-12) if total_bg > 0 else 0.0

    return {"dice": dice, "precision": prec, "recall": rec, "fpr": fpr}

def otsu_bin(img_bgr, size):
    img_resized = cv2.resize(img_bgr, size)
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (th / 255.0 > 0.5).astype(np.float32)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = (256, 256)

    m12_ckpt = REPO_ROOT / "kaggle/outputs_phase7c_1000/M12_Phase7C_best.pth"

    print("=" * 80)
    print("PHASE 7C EVALUATION: M12 (Trained on 1000 images)")
    print("=" * 80)

    # Load M12 Phase 7C
    net_m12 = FANet(gating_mode="learned_residual", detach_feedback=True)
    net_m12 = load_model_weights(net_m12, m12_ckpt, device)
    net_m12 = net_m12.to(device)
    net_m12.eval()

    # Load Data
    data_root = REPO_ROOT / "data/sessile-main-Kvasir-SEG"
    if not data_root.exists():
        data_root = REPO_ROOT / "data"
    (_, _), (val_x, val_y) = load_data(str(data_root))

    metrics_m12 = []
    
    # Temporal tracking accumulator for Phase 7C
    metrics_t = [{"dice": [], "fpr": [], "recall": []} for _ in range(4)]

    for idx, (img_p, msk_p) in enumerate(zip(val_x, val_y)):
        img_bgr = cv2.imread(img_p)
        img_resized = cv2.resize(img_bgr, size)
        img_t = torch.from_numpy(np.transpose(img_resized, (2, 0, 1)) / 255.0).float().unsqueeze(0).to(device)

        msk_gray = cv2.imread(msk_p, cv2.IMREAD_GRAYSCALE)
        msk_resized = cv2.resize(msk_gray, size, interpolation=cv2.INTER_NEAREST)
        gt_bin = (msk_resized > 127).astype(np.uint8)

        otsu = otsu_bin(img_bgr, size)

        # M12 Phase 7C Recurrent
        prev_12 = None
        for t in range(4):
            m = otsu[None, None] if prev_12 is None else (prev_12 > 0.5).astype(np.float32)[None, None]
            with torch.no_grad():
                out = net_m12([img_t, torch.from_numpy(m).to(device)])
                prob = torch.sigmoid(out)[0, 0].cpu().numpy()
            
            # Record intermediate metrics for tracking Monotonicity Trap
            iter_res = compute_metrics(prob, gt_bin)
            metrics_t[t]["dice"].append(iter_res["dice"])
            metrics_t[t]["fpr"].append(iter_res["fpr"])
            metrics_t[t]["recall"].append(iter_res["recall"])
            
            prev_12 = prob
        
        metrics_m12.append(compute_metrics(prev_12, gt_bin))

    print("\n[PART 1: QUANTITATIVE BENCHMARK (4 Iterations)]")
    m12_dice = np.mean([m['dice'] for m in metrics_m12])
    m12_fpr = np.mean([m['fpr'] for m in metrics_m12]) * 100
    m12_recall = np.mean([m['recall'] for m in metrics_m12]) * 100

    print(f"{'Metric':<15} | {'M12 (1000 images)':<15}")
    print("-" * 50)
    print(f"{'Dice':<15} | {m12_dice:<15.4f}")
    print(f"{'FPR':<15} | {m12_fpr:.2f}%")
    print(f"{'Recall':<15} | {m12_recall:.2f}%")

    print("\n[PART 2: TEMPORAL TRACKING (Phase 7C Pruning Ability)]")
    print("Does FPR shrink over time? (Overcoming Monotonicity Trap)")
    print(f"{'Iteration':<15} | {'Mean Dice':<15} | {'Mean FPR':<15}")
    print("-" * 50)
    for t in range(4):
        mean_dice = np.mean(metrics_t[t]["dice"])
        mean_fpr = np.mean(metrics_t[t]["fpr"]) * 100
        print(f"t={t:<13} | {mean_dice:<15.4f} | {mean_fpr:.2f}%")

if __name__ == "__main__":
    main()
