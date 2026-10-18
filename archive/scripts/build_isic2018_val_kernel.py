"""
Script: build_isic2018_val_kernel.py
Purpose: Generate Kaggle notebook and kernel metadata for official ISIC-2018 validation set evaluation.
"""

import json
import os
import nbformat as nbf

def generate_val_notebook(dest_path="kaggle/benchmark_kernels/isic-2018-val/fanet_val_official_isic2018.ipynb"):
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "file_extension": ".py",
            "mimetype": "text/x-python"
        }
    }

    # Cell 0: Header & Description
    nb.cells.append(nbf.v4.new_markdown_cell(
"""# GDF-Net (FANet-MFAD) Evaluation on Official ISIC-2018 Validation Set

**Task:** Evaluation on the official ISIC-2018 Task 1 Validation Set (N=100 images).
**Model:** GDF-Net (FANet-MFAD, best checkpoint from 200-epoch training).
**Input:** `ISIC2018_Task1-2_Validation_Input` (100 dermoscopic JPG images).
**Ground Truth:** `ISIC2018_Task1_Validation_GroundTruth` (100 binary mask PNG images).
**Metrics:** Region (Dice, mIoU, Sensitivity, Specificity, Precision, FPR) + Boundary (HD95, ASD, BF@2).
"""))

    # Cell 1: Environment & Setup
    nb.cells.append(nbf.v4.new_code_cell(
r"""import os, sys, random, time, csv, json, zipfile, urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""))

    # Cell 2: Data Acquisition (Official Validation Set)
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# Data Acquisition for Official ISIC-2018 Validation Set (100 images)
# ================================================================
VAL_IMG_URL = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip"
VAL_GT_URL  = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip"

os.makedirs("isic2018_val", exist_ok=True)
img_zip = "isic2018_val/val_input.zip"
gt_zip  = "isic2018_val/val_gt.zip"

# Check if input images already mounted via Kaggle dataset
found_input_dirs = glob("/kaggle/input/**/ISIC2018_Task1-2_Validation_Input", recursive=True)
if found_input_dirs:
    val_img_dir = found_input_dirs[0]
    print(f"[*] Found mounted validation input dir: {val_img_dir}")
else:
    print(f"[*] Downloading Validation Input from ISIC S3...")
    urllib.request.urlretrieve(VAL_IMG_URL, img_zip)
    with zipfile.ZipFile(img_zip, 'r') as z:
        z.extractall("isic2018_val")
    val_img_dir = "isic2018_val/ISIC2018_Task1-2_Validation_Input"

# Validation Ground Truth
found_gt_dirs = glob("/kaggle/input/**/ISIC2018_Task1_Validation_GroundTruth", recursive=True)
if found_gt_dirs:
    val_gt_dir = found_gt_dirs[0]
    print(f"[*] Found mounted validation groundtruth dir: {val_gt_dir}")
else:
    print(f"[*] Downloading Validation GroundTruth from ISIC S3...")
    urllib.request.urlretrieve(VAL_GT_URL, gt_zip)
    with zipfile.ZipFile(gt_zip, 'r') as z:
        z.extractall("isic2018_val")
    val_gt_dir = "isic2018_val/ISIC2018_Task1_Validation_GroundTruth"

def isic_id(path):
    base = os.path.basename(path).replace("_segmentation", "")
    return os.path.splitext(base)[0]

val_jpgs = sorted(glob(os.path.join(val_img_dir, "*.jpg")))
val_pngs = sorted(glob(os.path.join(val_gt_dir, "*.png")))

img_dict = {isic_id(p): p for p in val_jpgs}
msk_dict = {isic_id(p): p for p in val_pngs}
val_ids = sorted(set(img_dict.keys()) & set(msk_dict.keys()))

print(f"[+] Official Validation Set Ready: {len(val_ids)} matched pairs verified.")
"""))

    # Cell 3: Dataset Loader
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# Dataset Definition
# ================================================================
IMAGE_SIZE = (512, 512)

class UniversalDataset(Dataset):
    def __init__(self, images_path, masks_path, image_size):
        self.images_path = list(images_path)
        self.masks_path  = list(masks_path)
        self.image_size  = image_size

    def _read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return np.array(Image.open(path).convert('RGB'))
        if len(img.shape) == 2:
            return np.stack([img]*3, axis=-1)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_mask(self, path):
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is None:
            return np.array(Image.open(path).convert('L'))
        if len(m.shape) == 3:
            return cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        return m

    def __getitem__(self, index):
        image = self._read_image(self.images_path[index])
        mask  = self._read_mask(self.masks_path[index])

        # Resize to 512x512
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  self.image_size, interpolation=cv2.INTER_NEAREST)

        # Normalise -> [0, 1] float32
        image = image.astype(np.float32)
        mx = image.max()
        if mx > 0:
            image = image / (65535.0 if mx > 255 else 255.0)
        image = np.transpose(image, (2, 0, 1))

        # Binarise mask -> {0, 1}
        mask = mask.astype(np.float32)
        if mask.max() > 0:
            mask = (mask > (mask.max() + mask.min()) / 2.0).astype(np.float32)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)

        return image, mask

    def __len__(self):
        return len(self.images_path)

val_x = [img_dict[k] for k in val_ids]
val_y = [msk_dict[k] for k in val_ids]

dataset = UniversalDataset(val_x, val_y, IMAGE_SIZE)
val_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
print(f"DataLoader Ready: {len(val_loader)} images.")
"""))

    # Cell 4: Model Architecture
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# GDF-Net Architecture
# ================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))


class GDFNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=None)

        self.e1_conv = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
        self.e1_bn   = resnet.bn1
        self.e1_relu = resnet.relu
        self.pool    = resnet.maxpool

        self.e2 = resnet.layer1
        self.e3 = resnet.layer2
        self.e4 = resnet.layer3
        self.b  = resnet.layer4

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d1  = ConvBlock(512, 256)
        self.att1 = nn.Sequential(nn.Conv2d(1, 256, 1), nn.BatchNorm2d(256), nn.Sigmoid())

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d2  = ConvBlock(256, 128)
        self.att2 = nn.Sequential(nn.Conv2d(1, 128, 1), nn.BatchNorm2d(128), nn.Sigmoid())

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d3  = ConvBlock(128, 64)
        self.att3 = nn.Sequential(nn.Conv2d(1, 64, 1), nn.BatchNorm2d(64), nn.Sigmoid())

        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.d4  = ConvBlock(128, 64)
        self.att4 = nn.Sequential(nn.Conv2d(1, 64, 1), nn.BatchNorm2d(64), nn.Sigmoid())

        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d5  = ConvBlock(32, 32)
        self.att5 = nn.Sequential(nn.Conv2d(1, 32, 1), nn.BatchNorm2d(32), nn.Sigmoid())

        self.out = nn.Conv2d(32, 1, 1)
        self.learned_gate = nn.Conv2d(1, 1, 1)

    def _mfad(self, feat, prev_mask, att_layer):
        m = F.interpolate(prev_mask, size=feat.shape[2:], mode='nearest')
        return feat * (1.0 + att_layer(m))

    def forward(self, x, prev_mask=None):
        if prev_mask is None:
            prev_mask = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        else:
            prev_mask = torch.sigmoid(self.learned_gate(prev_mask.detach()))

        s1 = self.e1_relu(self.e1_bn(self.e1_conv(torch.cat([x, prev_mask], dim=1))))
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        b  = self.b(s4)

        d1 = self.d1(torch.cat([self.up1(b), s4], 1));  d1 = self._mfad(d1, prev_mask, self.att1)
        d2 = self.d2(torch.cat([self.up2(d1), s3], 1)); d2 = self._mfad(d2, prev_mask, self.att2)
        d3 = self.d3(torch.cat([self.up3(d2), s2], 1)); d3 = self._mfad(d3, prev_mask, self.att3)
        d4 = self.d4(torch.cat([self.up4(d3), s1], 1)); d4 = self._mfad(d4, prev_mask, self.att4)
        d5 = self.d5(self.up5(d4));                      d5 = self._mfad(d5, prev_mask, self.att5)

        return self.out(d5)
"""))

    # Cell 5: Metrics Definition
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# Comprehensive Region & Boundary Metrics
# ================================================================
def compute_metrics(gt, pred, eval_boundary=True):
    gt   = (gt > 0.5).astype(np.float32)
    pred = (pred > 0.5).astype(np.float32)

    tp = np.sum((gt == 1) & (pred == 1))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    tn = np.sum((gt == 0) & (pred == 0))

    dice = (2.0 * tp + 1.0) / (2.0 * tp + fp + fn + 1.0)
    iou  = (tp + 1.0) / (tp + fp + fn + 1.0)
    sens = tp / (tp + fn + 1e-7)
    spec = tn / (tn + fp + 1e-7)
    prec = tp / (tp + fp + 1e-7)
    fpr  = fp / (fp + tn + 1e-7)

    res = {
        'dice': float(dice),
        'miou': float(iou),
        'sensitivity': float(sens),
        'specificity': float(spec),
        'precision': float(prec),
        'fpr': float(fpr)
    }

    if eval_boundary and (np.sum(gt) > 0 and np.sum(pred) > 0):
        try:
            g_2d = gt.squeeze()
            p_2d = pred.squeeze()
            g352 = cv2.resize(g_2d, (352, 352), interpolation=cv2.INTER_NEAREST)
            p352 = cv2.resize(p_2d, (352, 352), interpolation=cv2.INTER_NEAREST)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            g_bnd = cv2.morphologyEx(g352, cv2.MORPH_GRADIENT, kernel) > 0
            p_bnd = cv2.morphologyEx(p352, cv2.MORPH_GRADIENT, kernel) > 0

            d_gt = distance_transform_edt(~g_bnd)
            d_pred = distance_transform_edt(~p_bnd)

            dist_p_to_g = d_gt[p_bnd]
            dist_g_to_p = d_pred[g_bnd]

            if len(dist_p_to_g) > 0 and len(dist_g_to_p) > 0:
                all_dists = np.concatenate([dist_p_to_g, dist_g_to_p])
                hd95 = float(np.percentile(all_dists, 95))
                asd  = 0.5 * (float(np.mean(dist_p_to_g)) + float(np.mean(dist_g_to_p)))

                tau = 2.0
                p_acc = float(np.mean(dist_p_to_g <= tau))
                r_acc = float(np.mean(dist_g_to_p <= tau))
                bf2 = (2.0 * p_acc * r_acc) / (p_acc + r_acc + 1e-7)

                res['hd95'] = hd95
                res['asd'] = asd
                res['bf2'] = bf2
        except Exception:
            pass

    return res
"""))

    # Cell 6: Evaluation & Reporting
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# Checkpoint Discovery & Evaluation Execution
# ================================================================
cand_ckpts = sorted(glob("/kaggle/input/**/GDFNet_ISIC2018_best.pth", recursive=True))
if not cand_ckpts:
    cand_ckpts = sorted(glob("**/GDFNet_ISIC2018_best.pth", recursive=True))

if not cand_ckpts:
    raise FileNotFoundError("Could not find GDFNet_ISIC2018_best.pth in /kaggle/input or workspace!")

ckpt_path = cand_ckpts[0]
print(f"[*] Found Checkpoint: {ckpt_path}")

model = GDFNet().to(DEVICE)
ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
if 'model_state' in ckpt:
    model.load_state_dict(ckpt['model_state'])
else:
    model.load_state_dict(ckpt)
model.eval()

all_metrics = []
sample_imgs, sample_gts, sample_preds, sample_names = [], [], [], []

print(f"\n{'='*75}")
print(f"  EVALUATING ON OFFICIAL ISIC-2018 VALIDATION SET (N={len(val_loader.dataset)})")
print(f"{'='*75}")

with torch.no_grad():
    for idx, (x, y) in enumerate(tqdm(val_loader, desc="[GDF-Net] Official Val Eval")):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast('cuda' if DEVICE.type == 'cuda' else 'cpu'):
            out1 = model(x)
            out2 = model(x, torch.sigmoid(out1))

        pred = (torch.sigmoid(out2).cpu().numpy() > 0.5).astype(np.float32)
        gt   = y.cpu().numpy()
        img  = x.cpu().numpy()

        for b_i in range(pred.shape[0]):
            m = compute_metrics(gt[b_i], pred[b_i], eval_boundary=True)
            m['image_id'] = val_ids[idx]
            all_metrics.append(m)

            if len(sample_imgs) < 10:
                sample_imgs.append(img[b_i])
                sample_gts.append(gt[b_i])
                sample_preds.append(pred[b_i])
                sample_names.append(val_ids[idx])

# Aggregate
agg = {}
metric_keys = ['dice', 'miou', 'sensitivity', 'specificity', 'precision', 'fpr', 'hd95', 'asd', 'bf2']
for k in metric_keys:
    vals = [m[k] for m in all_metrics if k in m]
    agg[k] = {
        'mean': float(np.mean(vals)),
        'std':  float(np.std(vals)),
        'median': float(np.median(vals)),
        'min': float(np.min(vals)),
        'max': float(np.max(vals))
    }

# Print Summary Table
print(f"\n{'='*75}")
print(f"  FINAL BENCHMARK RESULTS ON OFFICIAL ISIC-2018 VALIDATION SET (N={len(val_loader.dataset)})")
print(f"{'='*75}")
print(f"  {'Metric':<15s} | {'Mean ± Std':>22s} | {'Median':>10s} | {'[Min, Max]':>20s}")
print(f"  {'-'*15}-+-{'-'*22}-+-{'-'*10}-+-{'-'*20}")
for k in metric_keys:
    v = agg[k]
    print(f"  {k:<15s} | {v['mean']:>9.4f} ± {v['std']:<10.4f} | {v['median']:>10.4f} | [{v['min']:.4f}, {v['max']:.4f}]")
print(f"{'='*75}")

# Save CSV
summary_csv = "benchmark_results_ISIC-2018_official_val.csv"
with open(summary_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Dataset', 'Cohort', 'Metric', 'Mean', 'Std', 'Median', 'Min', 'Max'])
    for k in metric_keys:
        v = agg[k]
        writer.writerow(['GDF-Net', 'ISIC-2018', 'Official_Validation_100', k,
                         f"{v['mean']:.6f}", f"{v['std']:.6f}", f"{v['median']:.6f}",
                         f"{v['min']:.6f}", f"{v['max']:.6f}"])
print(f"Summary metrics saved to: {summary_csv}")

detail_csv = "per_image_results_ISIC-2018_official_val.csv"
with open(detail_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    fieldnames = ['image_id', 'dice', 'miou', 'sensitivity', 'specificity', 'precision', 'fpr', 'hd95', 'asd', 'bf2']
    writer.writerow(fieldnames)
    for m in all_metrics:
        writer.writerow([m.get(fn, '') for fn in fieldnames])
print(f"Per-image metrics saved to: {detail_csv}")

# Plot Qualitative Samples
n = min(6, len(sample_imgs))
fig, axes = plt.subplots(n, 3, figsize=(12, 3.5 * n))
for i in range(n):
    img = np.transpose(sample_imgs[i], (1, 2, 0))
    img = np.clip(img, 0, 1)
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Input: {sample_names[i]}" if i == 0 else sample_names[i])
    axes[i, 0].axis('off')

    axes[i, 1].imshow(sample_gts[i][0], cmap='gray')
    axes[i, 1].set_title('Official Ground Truth' if i == 0 else '')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(sample_preds[i][0], cmap='gray')
    axes[i, 2].set_title(f"GDF-Net Pred (Dice={all_metrics[i]['dice']:.3f})" if i == 0 else f"Dice={all_metrics[i]['dice']:.3f}")
    axes[i, 2].axis('off')

plt.tight_layout()
comparison_png = "comparison_ISIC-2018_val.png"
plt.savefig(comparison_png, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved qualitative plot to: {comparison_png}")
"""))

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook successfully written to {dest_path}")


def update_val_metadata(dest_path="kaggle/benchmark_kernels/isic-2018-val/kernel-metadata.json"):
    metadata = {
        "id": "namnguynnnn/fanet-val-official-isic2018",
        "title": "fanet-val-official-isic2018",
        "code_file": "fanet_val_official_isic2018.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [
            "tschandl/isic2018-challenge-task1-data-segmentation"
        ],
        "kernel_sources": [
            "namnguynnnn/fanet-benchmark-isic-2018"
        ],
        "competition_sources": [],
        "model_sources": []
    }
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Kernel metadata successfully written to {dest_path}")


if __name__ == "__main__":
    generate_val_notebook()
    update_val_metadata()
