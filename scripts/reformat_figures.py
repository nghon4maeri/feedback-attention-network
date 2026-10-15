"""
Reformat training convergence curves and qualitative prediction images:
1. Extract exact training history for CVC-ClinicDB, EM-Dataset, ISIC-2018, and CHASE-DB1.
2. Re-plot training curves with large, readable fonts (14pt-16pt), 300 DPI, and GDF-Net branding.
3. Update qualitative comparison images with GDF-Net branding and clean 3-row layout.
"""

import os, re, json, glob
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def extract_history_data():
    histories = {}

    # 1. CVC-ClinicDB
    p_cvc = glob.glob("results/benchmark_v2/cvc-clinicdb/*latest.pth")[0]
    ckpt_cvc = torch.load(p_cvc, map_location="cpu", weights_only=False)
    histories["CVC-ClinicDB"] = {
        "train_loss": ckpt_cvc["history"]["train_loss"],
        "val_dice": ckpt_cvc["history"]["val_dice"],
    }

    # 2. EM-Dataset
    p_em = glob.glob("results/benchmark_v2/em-dataset/*latest.pth")[0]
    ckpt_em = torch.load(p_em, map_location="cpu", weights_only=False)
    histories["EM-Dataset"] = {
        "train_loss": ckpt_em["history"]["train_loss"],
        "val_dice": ckpt_em["history"]["val_dice"],
    }

    # 3. CHASE-DB1
    chase_loss, chase_dice = [], []
    with open("kaggle_logs/fanet-benchmark-chasedb1.log", "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"TrL\s+([0-9\.]+)\s+\|\s+VL\s+[0-9\.]+\s+\|\s+VDice\s+([0-9\.]+)", line)
            if m:
                chase_loss.append(float(m.group(1)))
                chase_dice.append(float(m.group(2)))
    histories["CHASE-DB1"] = {
        "train_loss": chase_loss,
        "val_dice": chase_dice,
    }

    # 4. ISIC-2018
    isic_loss, isic_dice = [], []
    with open("results/isic2018/fanet-benchmark-isic-2018.log", "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"\[FANet_MFAD\]\s+Epoch\s+\d+\s+\|\s+LR\s+[0-9\.e\-]+\s+\|\s+Loss\s+([0-9\.]+)\s+\|\s+Val Dice\s+([0-9\.]+)", line)
            if m:
                isic_loss.append(float(m.group(1)))
                isic_dice.append(float(m.group(2)))
    histories["ISIC-2018"] = {
        "train_loss": isic_loss,
        "val_dice": isic_dice,
    }

    return histories


def plot_large_training_curves(histories):
    out_dir = "papers/overleaf_submission/figures"
    os.makedirs(out_dir, exist_ok=True)

    dataset_titles = {
        "CVC-ClinicDB": "CVC-ClinicDB (Colonoscopy Polyps)",
        "EM-Dataset": "EM-Dataset (Electron Microscopy)",
        "ISIC-2018": "ISIC-2018 (Dermoscopic Melanoma)",
        "CHASE-DB1": "CHASE-DB1 (Retinal Vessel Trees)",
    }

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

    for name, data in histories.items():
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
        
        # Loss plot
        epochs = range(1, len(data["train_loss"]) + 1)
        ax1.plot(epochs, data["train_loss"], label="GDF-Net Train Loss", color="#1A73E8", linewidth=2.5)
        ax1.set_xlabel("Epoch", fontweight="bold")
        ax1.set_ylabel("Loss", fontweight="bold")
        ax1.set_title("Training Loss Convergence", fontweight="bold", pad=10)
        ax1.legend(loc="upper right", framealpha=0.9)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Dice plot
        epochs_val = range(1, len(data["val_dice"]) + 1)
        ax2.plot(epochs_val, data["val_dice"], label="GDF-Net Val Dice", color="#137333", linewidth=2.5)
        ax2.set_xlabel("Epoch", fontweight="bold")
        ax2.set_ylabel("Validation Dice", fontweight="bold")
        ax2.set_title("Validation Dice Score", fontweight="bold", pad=10)
        ax2.legend(loc="lower right", framealpha=0.9)
        ax2.grid(True, linestyle="--", alpha=0.5)

        title = dataset_titles.get(name, name)
        plt.suptitle(f"GDF-Net Convergence — {title}", fontsize=16, fontweight="bold", y=1.02)
        plt.tight_layout()

        out_path = os.path.join(out_dir, f"training_curves_{name}.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close()
        print(f"Saved readable training curve to: {out_path}")


def reformat_qualitative_images():
    out_dir = "papers/overleaf_submission/figures"
    for ds in ["CVC-ClinicDB", "EM-Dataset", "ISIC-2018", "CHASE-DB1"]:
        in_path = os.path.join(out_dir, f"comparison_{ds}.png")
        if not os.path.exists(in_path):
            continue
        im = Image.open(in_path)
        w, h = im.size
        # The image has a header region (~150px) and 5 rows of samples
        # Each row is ~(h - 150) / 5
        header_h = int(h * 0.08)
        row_h = (h - header_h) / 5.0
        # Crop top 3 rows
        crop_h = int(header_h + 3 * row_h)
        cropped = im.crop((0, 0, w, crop_h))
        
        # Save updated 3-row version with better aspect ratio
        cropped.save(in_path, dpi=(300, 300))
        print(f"Re-formatted qualitative image {in_path} to 3 large rows (w={w}, h={crop_h})")


if __name__ == "__main__":
    print("Extracting histories...")
    hist = extract_history_data()
    print("Plotting readable 300 DPI training curves...")
    plot_large_training_curves(hist)
    print("Reformatting qualitative prediction images...")
    reformat_qualitative_images()
    print("Figure reformatting complete.")
