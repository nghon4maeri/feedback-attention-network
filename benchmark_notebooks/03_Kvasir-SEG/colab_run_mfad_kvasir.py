"""
colab_run_mfad_kvasir.py
========================
Self-contained script to train FANet_MFAD on Kvasir-SEG dataset in Google Colab A100 GPU.
Evaluates exclusively the final feedback phase (out2) with paper-grade data augmentation.
"""

import os
import sys
import glob
import json
import time
import random
import subprocess
from collections import OrderedDict

# ------------------------------------------------------------------
# 1. Environment & Dependencies Setup
# ------------------------------------------------------------------
print("=" * 60)
print("  FANet-MFAD Training: Kvasir-SEG Polyp Dataset (Colab A100)")
print("=" * 60)

# Kaggle credentials setup
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
with open(kaggle_json, "w") as f:
    json.dump({"username": "namnguynnnn", "key": "KGAT_dad33ce94fb8b25d8a9ecd4ca26ce03f"}, f)
os.chmod(kaggle_json, 0o600)
print("[+] Kaggle API credentials configured.")

# Install required packages
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "albumentations", "tifffile", "opencv-python-headless"], check=True)
print("[+] Dependencies verified.")

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"[+] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[+] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.backends.cudnn.benchmark = True

NUM_WORKERS = min(4, os.cpu_count() or 2)

# ------------------------------------------------------------------
# 2. Dataset Download & Preparation
# ------------------------------------------------------------------
DATA_DIR = "/content/data_kvasir"
OUT_DIR = "/content/output_kvasir"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(os.path.join(DATA_DIR, "images")):
    print("[*] Downloading Kvasir-SEG dataset from Kaggle...")
    cmd = ["kaggle", "datasets", "download", "-d", "namnguynnnn/kvasir-seg", "-p", DATA_DIR, "--unzip"]
    subprocess.run(cmd, check=True)
    print("[+] Download complete.")

all_files = []
for ext in ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]:
    all_files.extend(glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True))

images_all = sorted([f for f in all_files if "/images/" in f.replace(os.sep, "/").lower()])
masks_all  = sorted([f for f in all_files if "/masks/" in f.replace(os.sep, "/").lower()])

# Match by stem basename
img_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in images_all}
msk_dict = {os.path.splitext(os.path.basename(f))[0]: f for f in masks_all}
common_keys = sorted(list(set(img_dict.keys()) & set(msk_dict.keys())))

matched_imgs = [img_dict[k] for k in common_keys]
matched_msks = [msk_dict[k] for k in common_keys]

print(f"[+] Matched pairs: {len(matched_imgs)}")

# Paper Split: 80% Train (800), 10% Val (100), 10% Test (100)
random.seed(42)
indices = list(range(len(matched_imgs)))
random.shuffle(indices)

n_total = len(indices)
n_train = int(n_total * 0.8)
n_val   = int(n_total * 0.1)

train_idx = indices[:n_train]
valid_idx = indices[n_train:n_train + n_val]
test_idx  = indices[n_train + n_val:]

train_x = [matched_imgs[i] for i in train_idx]
train_y = [matched_msks[i] for i in train_idx]
valid_x = [matched_imgs[i] for i in valid_idx]
valid_y = [matched_msks[i] for i in valid_idx]
test_x  = [matched_imgs[i] for i in test_idx]
test_y  = [matched_msks[i] for i in test_idx]

# If test set is present, use test set as final benchmark evaluation set per paper
eval_x = test_x if len(test_x) > 0 else valid_x
eval_y = test_y if len(test_y) > 0 else valid_y

IMAGE_SIZE = (512, 512)
EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-4
REPEAT_FACTOR = 22

print(f"[+] Kvasir-SEG Dataset: Train={len(train_x)}, Val={len(valid_x)}, Test={len(eval_x)} | Repeat Factor={REPEAT_FACTOR}")

# ------------------------------------------------------------------
# 3. Universal Dataset with Paper Data Augmentation
# ------------------------------------------------------------------
class UniversalDataset(Dataset):
    def __init__(self, images_path, masks_path, image_size, is_train=False, repeat_factor=1):
        self.images_path = list(images_path)
        self.masks_path = list(masks_path)
        self.image_size = image_size
        self.is_train = is_train
        self.repeat_factor = max(1, int(repeat_factor)) if is_train else 1
        self.n_samples = len(self.images_path) * self.repeat_factor

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.6),
            A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3, distort_limit=0.05),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
        ])

    def _read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return np.array(Image.open(path).convert('RGB'))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_mask(self, path):
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return np.array(Image.open(path).convert('L'))
        return m

    def __getitem__(self, index):
        actual_idx = index % len(self.images_path)
        image = self._read_image(self.images_path[actual_idx])
        mask  = self._read_mask(self.masks_path[actual_idx])

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
        uv = np.unique(mask)
        if len(uv) > 2:
            mask = (mask > (mask.max() + mask.min()) / 2.0).astype(np.float32)
        elif mask.max() > 0:
            mask = (mask == mask.max()).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return image, mask

    def __len__(self):
        return self.n_samples

train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True, repeat_factor=REPEAT_FACTOR)
valid_dataset = UniversalDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False, repeat_factor=1)
eval_dataset  = UniversalDataset(eval_x, eval_y, IMAGE_SIZE, is_train=False, repeat_factor=1)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
eval_loader  = DataLoader(eval_dataset, batch_size=1, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f"[+] Dataloaders ready: {len(train_dataset)} augmented train samples per epoch")

# ------------------------------------------------------------------
# 4. Model Architecture: FANet_MFAD
# ------------------------------------------------------------------
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

class FANet_MFAD(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        self.e1_conv = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.e1_conv.weight[:, :3] = resnet.conv1.weight
            self.e1_conv.weight[:, 3:4] = 0.0
        self.e1_bn   = resnet.bn1
        self.e1_relu = resnet.relu
        self.pool    = resnet.maxpool

        self.e2 = resnet.layer1   # 64
        self.e3 = resnet.layer2   # 128
        self.e4 = resnet.layer3   # 256
        self.b  = resnet.layer4   # 512

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
        return feat * (1 + att_layer(m))

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

        d1 = self.d1(torch.cat([self.up1(b), s4], 1))
        d1 = self._mfad(d1, prev_mask, self.att1)

        d2 = self.d2(torch.cat([self.up2(d1), s3], 1))
        d2 = self._mfad(d2, prev_mask, self.att2)

        d3 = self.d3(torch.cat([self.up3(d2), s2], 1))
        d3 = self._mfad(d3, prev_mask, self.att3)

        d4 = self.d4(torch.cat([self.up4(d3), s1], 1))
        d4 = self._mfad(d4, prev_mask, self.att4)

        d5 = self.d5(self.up5(d4))
        d5 = self._mfad(d5, prev_mask, self.att5)

        return self.out(d5)

# ------------------------------------------------------------------
# 5. Loss & Metrics
# ------------------------------------------------------------------
class DiceBCELoss(nn.Module):
    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, inputs, targets, smooth=1):
        inputs, targets = inputs.float(), targets.float()
        probs = torch.sigmoid(inputs).view(-1)
        tgt   = targets.view(-1)
        bce   = F.binary_cross_entropy(probs, tgt, reduction='mean')
        inter = (probs * tgt).sum()
        dice  = 1 - (2. * inter + smooth) / (probs.sum() + tgt.sum() + smooth)
        return 0.5 * bce + 0.5 * dice

def compute_metrics(y_true_np, y_pred_np):
    yt = y_true_np.flatten().astype(np.float32)
    yp = y_pred_np.flatten().astype(np.float32)
    eps = 1e-7
    tp = np.sum((yt == 1) & (yp == 1))
    fp = np.sum((yt == 0) & (yp == 1))
    fn = np.sum((yt == 1) & (yp == 0))
    tn = np.sum((yt == 0) & (yp == 0))

    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou  = tp / (tp + fp + fn + eps)
    sensitivity = tp / (tp + fn + eps)
    specificity = tn / (tn + fp + eps)
    fpr  = fp / (fp + tn + eps)
    precision = tp / (tp + fp + eps)

    return {
        'dice': dice,
        'miou': iou,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'fpr': fpr,
        'precision': precision,
    }

# ------------------------------------------------------------------
# 6. Training Execution (FANet_MFAD Only)
# ------------------------------------------------------------------
model = FANet_MFAD().to(DEVICE)
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6)
scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

best_ckpt = os.path.join(OUT_DIR, "FANet_MFAD_Kvasir-SEG_best.pth")
best_dice = 0.0
history = {'train_loss': [], 'val_dice': []}

print(f"[*] Starting training for {EPOCHS} epochs on A100 GPU...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            out1 = model(x)
            out2 = model(x, torch.sigmoid(out1))
            loss = criterion(out2, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_dice_sum, n_val = 0.0, 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                out1 = model(x)
                out2 = model(x, torch.sigmoid(out1))
            pred = (torch.sigmoid(out2).cpu().numpy() > 0.5).astype(np.float32)
            gt   = y.cpu().numpy()
            for b_i in range(pred.shape[0]):
                m = compute_metrics(gt[b_i], pred[b_i])
                val_dice_sum += m['dice']
                n_val += 1

    val_dice = val_dice_sum / max(n_val, 1)
    lr_now = optimizer.param_groups[0]['lr']
    scheduler.step(val_dice)

    history['train_loss'].append(train_loss)
    history['val_dice'].append(val_dice)

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), best_ckpt)
        star = " *"
    else:
        star = ""

    print(f"  [Epoch {epoch+1:03d}/{EPOCHS}] Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | Best: {best_dice:.4f} | LR: {lr_now:.1e}{star}", flush=True)

print(f"\n[+] Training Complete. Best Val Dice = {best_dice:.4f}")

# ------------------------------------------------------------------
# 7. Final Evaluation: Final Feedback (Pass-2) on Benchmark Test Set
# ------------------------------------------------------------------
print("\n[*] Evaluating Best Model on Test Set (Final Feedback Pass-2)...")
model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE, weights_only=True))
model.eval()

all_metrics = []
with torch.no_grad():
    for x, y in eval_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out1 = model(x)
        out2 = model(x, torch.sigmoid(out1))
        pred = (torch.sigmoid(out2).cpu().numpy() > 0.5).astype(np.float32)
        gt = y.cpu().numpy()
        for b_i in range(pred.shape[0]):
            m = compute_metrics(gt[b_i], pred[b_i])
            all_metrics.append(m)

agg = {}
for k in all_metrics[0]:
    vals = [m[k] for m in all_metrics]
    agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

print("=" * 60)
print("  FANet-MFAD Final Benchmark Results: Kvasir-SEG")
print("=" * 60)
for k, v in agg.items():
    print(f"  {k:>15s}: {v['mean']:.4f} ± {v['std']:.4f}")
print("=" * 60)

# Save CSV
import csv
csv_path = os.path.join(OUT_DIR, "benchmark_results_Kvasir-SEG.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Metric", "Mean", "Std"])
    for k, v in agg.items():
        writer.writerow(["FANet_MFAD", k, f"{v['mean']:.6f}", f"{v['std']:.6f}"])

# Save JSON
json_path = os.path.join(OUT_DIR, "benchmark_summary_Kvasir-SEG.json")
with open(json_path, "w") as f:
    json.dump({"dataset": "Kvasir-SEG", "model": "FANet_MFAD", "results": agg}, f, indent=2)

with open(os.path.join(OUT_DIR, "COMPLETED"), "w") as f:
    f.write(f"Done at {time.ctime()}, best_dice={best_dice:.6f}\n")

print(f"[+] All results saved to {OUT_DIR}")
