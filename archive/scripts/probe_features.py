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
def main():
    cfg = load_config(str(REPO_ROOT / "configs/kvasir_sessile.yaml"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    size = tuple(cfg["dataset"]["image_size"])

    m12_ckpt = REPO_ROOT / "checkpoints_phase7b/M12_seed2024.pth"
    if not m12_ckpt.exists():
        print(f"[ERROR] Checkpoint not found: {m12_ckpt}")
        sys.exit(1)

    model = FANet(gating_mode="soft_or", detach_feedback=True)
    model = load_model_weights(model, m12_ckpt, device)
    
    # Enable probing in the 4 MixPool blocks (d1, d2, d3, d4)
    model.d1.p1.probe_heatmaps = True
    model.d2.p1.probe_heatmaps = True
    model.d3.p1.probe_heatmaps = True
    model.d4.p1.probe_heatmaps = True

    data_root = REPO_ROOT / cfg["dataset"]["root"]
    (_, _), (val_x, val_y) = load_data(str(data_root))

    # Pick a random sample image
    img_p, msk_p = val_x[0], val_y[0]
    img_bgr = cv2.imread(img_p)
    img_resized = cv2.resize(img_bgr, size)
    otsu = otsu_bin(img_bgr, size)

    img_t = torch.from_numpy(np.transpose(img_resized, (2, 0, 1)) / 255.0).float().unsqueeze(0).to(device)
    m_t = torch.from_numpy(otsu[None, None]).to(device)

    with torch.no_grad():
        _ = model([img_t, m_t]) # Forward pass
        
    # We will plot the features of the deepest MixPool (d4)
    # The output shapes in d4 are usually 1/2 size
    fmask = model.d4.p1.saved_fmask[0, 0].numpy()
    m_fg = model.d4.p1.saved_m_fg[0, 0].numpy()
    keep = model.d4.p1.saved_keep[0, 0].numpy()

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    # Use 'magma' or 'jet' for cool heatmap vibes
    axes[1].imshow(fmask, cmap='magma', vmin=0, vmax=1)
    axes[1].set_title("fmask (Differentiable Semantic Branch)")
    axes[1].axis('off')

    axes[2].imshow(m_fg, cmap='magma', vmin=0, vmax=1)
    axes[2].set_title("m_fg (Feedback Prior)")
    axes[2].axis('off')

    axes[3].imshow(keep, cmap='magma', vmin=0, vmax=1)
    axes[3].set_title("keep (Phase 7B Soft-OR Output)")
    axes[3].axis('off')

    plt.tight_layout()
    out_dir = REPO_ROOT / "paper_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "probing_heatmaps.pdf"
    plt.savefig(out_file, dpi=300, bbox_inches='tight')
    print(f"Saved feature map probing figure to {out_file}")

if __name__ == "__main__":
    main()
