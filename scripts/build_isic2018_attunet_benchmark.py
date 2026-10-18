"""
Script: build_isic2018_attunet_benchmark.py
Purpose: Build Kaggle kernel for Attention U-Net (Oktay et al., 2018)
         on ISIC-2018 with BGDNet partition and official validation evaluation.
"""

import json
import os
import nbformat as nbf

def generate_attunet_notebook(dest_path="kaggle/benchmark_kernels/isic-2018-attunet/attention_unet_isic2018.ipynb"):
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

    # Cell 0: Header
    nb.cells.append(nbf.v4.new_markdown_cell(
"""# Attention U-Net Baseline Benchmark: ISIC-2018

**Purpose:** Directly verify literature claims for standard Attention U-Net (Oktay et al., 2018)
on the exact same dataset partition (BGDNet protocol: 2,334 Train, 260 Val) and evaluate on the
**Official ISIC-2018 Validation Set (100 images)** to compare against GDF-Net without overclaim bias.

**Architecture:** Standard Attention U-Net (Oktay et al., MIDL 2018) with Attention Gates gating skip connections.
**Loss:** 0.5×BCE + 0.5×Dice | **Resolution:** 512×512 | **Optimizer:** AdamW (lr=1e-4) + ReduceLROnPlateau.
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

try:
    import albumentations as A
except ImportError:
    os.system("pip install -q albumentations")
    import albumentations as A

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""))

    # Cell 2: Dataset Definition
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# Dataset Definition
# ================================================================
IMAGE_SIZE = (512, 512)

class UniversalDataset(Dataset):
    def __init__(self, images_path, masks_path, image_size, is_train=False):
        self.images_path = list(images_path)
        self.masks_path  = list(masks_path)
        self.image_size  = image_size
        self.is_train    = is_train

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.6),
            A.ElasticTransform(p=0.3, alpha=120, sigma=120*0.05),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3, distort_limit=0.05),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
        ])

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

        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  self.image_size, interpolation=cv2.INTER_NEAREST)

        if self.is_train:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']

        image = image.astype(np.float32)
        mx = image.max()
        if mx > 0:
            image = image / (65535.0 if mx > 255 else 255.0)
        image = np.transpose(image, (2, 0, 1))

        mask = mask.astype(np.float32)
        if mask.max() > 0:
            mask = (mask > (mask.max() + mask.min()) / 2.0).astype(np.float32)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)

        return image, mask

    def __len__(self):
        return len(self.images_path)
"""))

    # Cell 3: Data Discovery (Train, Internal Val, Official Val)
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# Data Discovery (2,334 Train, 260 Internal Val, 100 Official Val)
# ================================================================
def isic_id(path):
    base = os.path.basename(path).replace("_segmentation", "")
    return os.path.splitext(base)[0]

def find_dir_by_name(target_name, base_search="/kaggle/input"):
    for root, dirs, files in os.walk(base_search):
        if os.path.basename(root).lower() == target_name.lower():
            return root
    return None

train_img_dir = find_dir_by_name("ISIC2018_Task1-2_Training_Input")
train_gt_dir  = find_dir_by_name("ISIC2018_Task1_Training_GroundTruth")

all_jpgs = sorted(glob(os.path.join(train_img_dir, "*.jpg")))
all_pngs = sorted(glob(os.path.join(train_gt_dir, "*.png")))

img_dict = {isic_id(p): p for p in all_jpgs}
msk_dict = {isic_id(p): p for p in all_pngs}
train_pool_ids = sorted(set(img_dict.keys()) & set(msk_dict.keys()))

random.seed(2026)
random.shuffle(train_pool_ids)

N_TRAIN = 2334
N_VAL   = 260
train_ids = train_pool_ids[:N_TRAIN]
val_ids   = train_pool_ids[N_TRAIN:N_TRAIN + N_VAL]

train_x = [img_dict[k] for k in train_ids]
train_y = [msk_dict[k] for k in train_ids]
val_x   = [img_dict[k] for k in val_ids]
val_y   = [msk_dict[k] for k in val_ids]

print(f"Training Pool Partition -> Train: {len(train_x)} | Internal Val: {len(val_x)}")

# Official Validation Set (100 images)
VAL_IMG_URL = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip"
VAL_GT_URL  = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip"

os.makedirs("isic2018_val", exist_ok=True)
off_val_img_dir = find_dir_by_name("ISIC2018_Task1-2_Validation_Input")
off_val_gt_dir  = find_dir_by_name("ISIC2018_Task1_Validation_GroundTruth")

if not off_val_img_dir:
    print("[*] Downloading Official Val Input from ISIC S3...")
    urllib.request.urlretrieve(VAL_IMG_URL, "isic2018_val/val_in.zip")
    with zipfile.ZipFile("isic2018_val/val_in.zip", 'r') as z:
        z.extractall("isic2018_val")
    off_val_img_dir = "isic2018_val/ISIC2018_Task1-2_Validation_Input"

if not off_val_gt_dir:
    print("[*] Downloading Official Val GroundTruth from ISIC S3...")
    urllib.request.urlretrieve(VAL_GT_URL, "isic2018_val/val_gt.zip")
    with zipfile.ZipFile("isic2018_val/val_gt.zip", 'r') as z:
        z.extractall("isic2018_val")
    off_val_gt_dir = "isic2018_val/ISIC2018_Task1_Validation_GroundTruth"

off_jpgs = sorted(glob(os.path.join(off_val_img_dir, "*.jpg")))
off_pngs = sorted(glob(os.path.join(off_val_gt_dir, "*.png")))

off_img_dict = {isic_id(p): p for p in off_jpgs}
off_msk_dict = {isic_id(p): p for p in off_pngs}
off_val_ids  = sorted(set(off_img_dict.keys()) & set(off_msk_dict.keys()))

off_val_x = [off_img_dict[k] for k in off_val_ids]
off_val_y = [off_msk_dict[k] for k in off_val_ids]

print(f"Official Validation Set Ready -> {len(off_val_x)} matched pairs verified.")

train_loader   = DataLoader(UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True),
                            batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
valid_loader   = DataLoader(UniversalDataset(val_x, val_y, IMAGE_SIZE, is_train=False),
                            batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
off_val_loader = DataLoader(UniversalDataset(off_val_x, off_val_y, IMAGE_SIZE, is_train=False),
                            batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
"""))

    # Cell 4: Attention U-Net Architecture (Oktay et al., 2018)
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# Standard Attention U-Net Architecture (Oktay et al., MIDL 2018)
# ================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)


class AttentionGate(nn.Module):
    '''Attention Gate as proposed in Oktay et al. (MIDL 2018).'''
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1 = ConvBlock(in_channels, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.conv4 = ConvBlock(256, 512)
        self.conv5 = ConvBlock(512, 1024)

        self.up5 = UpConv(1024, 512)
        self.att5 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.up_conv5 = ConvBlock(1024, 512)

        self.up4 = UpConv(512, 256)
        self.att4 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.up_conv4 = ConvBlock(512, 256)

        self.up3 = UpConv(256, 128)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.up_conv3 = ConvBlock(256, 128)

        self.up2 = UpConv(128, 64)
        self.att2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.up_conv2 = ConvBlock(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))
        x5 = self.conv5(self.maxpool(x4))

        # Decoder + Attention Gates
        d5 = self.up5(x5)
        x4_att = self.att5(g=d5, x=x4)
        d5 = torch.cat([x4_att, d5], dim=1)
        d5 = self.up_conv5(d5)

        d4 = self.up4(d5)
        x3_att = self.att4(g=d4, x=x3)
        d4 = torch.cat([x3_att, d4], dim=1)
        d4 = self.up_conv4(d4)

        d3 = self.up3(d4)
        x2_att = self.att3(g=d3, x=x2)
        d3 = torch.cat([x2_att, d3], dim=1)
        d3 = self.up_conv3(d3)

        d2 = self.up2(d3)
        x1_att = self.att2(g=d2, x=x1)
        d2 = torch.cat([x1_att, d2], dim=1)
        d2 = self.up_conv2(d2)

        return self.out_conv(d2)

model = AttentionUNet().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Attention U-Net instantiated: {n_params:,} parameters.")
"""))

    # Cell 5: Loss & Evaluation Metrics
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# Loss & Metrics
# ================================================================
class DiceBCELoss(nn.Module):
    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, inputs, targets, smooth=1e-5):
        inputs, targets = inputs.float(), targets.float()
        inputs_s = torch.sigmoid(inputs).view(-1)
        targets_f = targets.view(-1)
        intersection = (inputs_s * targets_f).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (inputs_s.sum() + targets_f.sum() + smooth)
        bce = F.binary_cross_entropy(inputs_s, targets_f, reduction='mean')
        return 0.5 * bce + 0.5 * dice_loss

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

    # Cell 6: Training Loop & Validation
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# Training & Validation
# ================================================================
EPOCHS = 100
LR = 1e-4

criterion = DiceBCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6)
scaler = torch.amp.GradScaler('cuda')

best_dice = 0.0
ckpt_path = "AttentionUNet_ISIC2018_best.pth"

start_time = time.time()
print(f"Starting Attention U-Net training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{EPOCHS} [Train]", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Internal Validation (260 images)
    model.eval()
    val_dice_sum = 0.0
    n_val = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out = model(x)
            pred = (torch.sigmoid(out).cpu().numpy() > 0.5).astype(np.float32)
            gt   = y.cpu().numpy()
            for b_i in range(pred.shape[0]):
                m = compute_metrics(gt[b_i], pred[b_i], eval_boundary=False)
                val_dice_sum += m['dice']
                n_val += 1

    val_dice = val_dice_sum / max(n_val, 1)
    lr_now = optimizer.param_groups[0]['lr']
    scheduler.step(val_dice)

    print(f"Epoch {epoch+1:03d}/{EPOCHS} | LR {lr_now:.1e} | Loss {train_loss:.4f} | Internal Val Dice {val_dice:.4f}")

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), ckpt_path)
        print(f"  [+] Saved best Attention U-Net model with Val Dice: {best_dice:.4f}")

    if time.time() - start_time > 8.5 * 3600:
        print("[!] Approaching Kaggle timeout. Exiting cleanly.")
        break

print(f"Training Complete. Best Internal Val Dice = {best_dice:.4f}")
"""))

    # Cell 7: Official Validation Evaluation
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# EVALUATION ON OFFICIAL VALIDATION SET (100 IMAGES)
# ================================================================
print(f"\n{'='*75}")
print(f"  EVALUATING ATTENTION U-NET ON OFFICIAL ISIC-2018 VALIDATION SET (N={len(off_val_loader.dataset)})")
print(f"{'='*75}")

model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
model.eval()

all_metrics = []
with torch.no_grad():
    for idx, (x, y) in enumerate(tqdm(off_val_loader, desc="[Attention U-Net] Official Val Eval")):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast('cuda'):
            out = model(x)
        pred = (torch.sigmoid(out).cpu().numpy() > 0.5).astype(np.float32)
        gt   = y.cpu().numpy()
        for b_i in range(pred.shape[0]):
            m = compute_metrics(gt[b_i], pred[b_i], eval_boundary=True)
            m['image_id'] = off_val_ids[idx]
            all_metrics.append(m)

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

print(f"\n{'='*75}")
print(f"  FINAL BENCHMARK: ATTENTION U-NET ON OFFICIAL ISIC-2018 VALIDATION SET (N=100)")
print(f"{'='*75}")
print(f"  {'Metric':<15s} | {'Mean ± Std':>22s} | {'Median':>10s} | {'[Min, Max]':>20s}")
print(f"  {'-'*15}-+-{'-'*22}-+-{'-'*10}-+-{'-'*20}")
for k in metric_keys:
    v = agg[k]
    print(f"  {k:<15s} | {v['mean']:>9.4f} ± {v['std']:<10.4f} | {v['median']:>10.4f} | [{v['min']:.4f}, {v['max']:.4f}]")
print(f"{'='*75}")

# Save CSV
summary_csv = "benchmark_results_ISIC-2018_attention_unet_official_val.csv"
with open(summary_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Dataset', 'Cohort', 'Metric', 'Mean', 'Std', 'Median', 'Min', 'Max'])
    for k in metric_keys:
        v = agg[k]
        writer.writerow(['AttentionUNet', 'ISIC-2018', 'Official_Validation_100', k,
                         f"{v['mean']:.6f}", f"{v['std']:.6f}", f"{v['median']:.6f}",
                         f"{v['min']:.6f}", f"{v['max']:.6f}"])
print(f"Results saved to: {summary_csv}")
"""))

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook successfully written to {dest_path}")


def update_attunet_metadata(dest_path="kaggle/benchmark_kernels/isic-2018-attunet/kernel-metadata.json"):
    metadata = {
        "id": "namnguynnnn/fanet-benchmark-isic2018-attunet",
        "title": "fanet-benchmark-isic2018-attunet",
        "code_file": "attention_unet_isic2018.ipynb",
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
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": []
    }
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Kernel metadata successfully written to {dest_path}")


if __name__ == "__main__":
    generate_attunet_notebook()
    update_attunet_metadata()
