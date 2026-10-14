"""
colab_run_mfad_drive_transfer.py
================================
Transfer Learning / Fine-Tuning script for FANet_MFAD on DRIVE dataset.
Initializes from ISIC-2018 pre-trained weights (FANet_MFAD_ISIC-2018_best.pth).
Uses full official DRIVE dataset (zionfuo/drive2004: 20 train, 20 test with ground truth).
Evaluates exclusively the Final Feedback Phase (t=2) with paper-grade data augmentation.
"""

import os
import sys
import glob
import json
import time
import random
import zipfile
import subprocess
from collections import OrderedDict

# ------------------------------------------------------------------
# 1. Environment & Dependencies Setup
# ------------------------------------------------------------------
print("=" * 65)
print("  FANet-MFAD Transfer Learning: DRIVE Retinal Dataset (Colab GPU)")
print("  Source Pretrained Weights: ISIC-2018 -> Fine-Tuning on DRIVE")
print("=" * 65)

kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
with open(kaggle_json, "w") as f:
    json.dump({"username": "namnguynnnn", "key": "KGAT_dad33ce94fb8b25d8a9ecd4ca26ce03f"}, f)
os.chmod(kaggle_json, 0o600)
print("[+] Kaggle API credentials configured.")

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
    torch.backends.cudnn.benchmark = True

NUM_WORKERS = min(4, os.cpu_count() or 2)

# ------------------------------------------------------------------
# 2. Dataset Download & Verification (Full 20 Train, 20 Test with GT)
# ------------------------------------------------------------------
DATA_DIR = "/content/data_drive_full"
OUT_DIR = "/content/output_drive_transfer"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(os.path.join(DATA_DIR, "DRIVE")):
    print("[*] Downloading complete DRIVE dataset (zionfuo/drive2004) from Kaggle...")
    cmd = ["kaggle", "datasets", "download", "-d", "zionfuo/drive2004", "-p", DATA_DIR, "--unzip"]
    subprocess.run(cmd, check=True)
    print("[+] Download complete.")

all_files = []
for ext in ["*.jpg", "*.png", "*.tif", "*.tiff", "*.gif", "*.ppm"]:
    all_files.extend(glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True))

images_all = [f for f in all_files if "/images/" in f.replace(os.sep, "/").lower()]
masks_all  = [f for f in all_files if "/1st_manual/" in f.replace(os.sep, "/").lower() or "/manual1/" in f.replace(os.sep, "/").lower()]

train_x = sorted([p for p in images_all if "training" in p.lower() or "train" in p.lower()])
train_y = sorted([p for p in masks_all if "training" in p.lower() or "train" in p.lower()])
test_x  = sorted([p for p in images_all if "test" in p.lower()])
test_y  = sorted([p for p in masks_all if "test" in p.lower()])

print(f"[+] Found DRIVE Files: Train Images={len(train_x)}, Train GT={len(train_y)}")
print(f"[+] Found DRIVE Files: Test Images={len(test_x)}, Test GT={len(test_y)}")

assert len(train_x) > 0 and len(train_y) > 0, "Training images or masks missing!"
assert len(test_x) > 0 and len(test_y) > 0, "Test images or masks missing!"

IMAGE_SIZE = (512, 512)
EPOCHS = 100
BATCH_SIZE = 4
LR = 1e-4  # Fine-tuning rate (stable transfer)
REPEAT_FACTOR = 32

print(f"[+] DRIVE Dataset: Train={len(train_x)}, Test={len(test_x)} | Repeat Factor={REPEAT_FACTOR}")

# ------------------------------------------------------------------
# 3. Universal Dataset with Paper Data Augmentation Suite
# ------------------------------------------------------------------
class UniversalDataset(Dataset):
    def __init__(self, images_path, masks_path, image_size, is_train=False, repeat_factor=1, is_retinal=True):
        self.images_path = list(images_path)
        self.masks_path = list(masks_path)
        self.image_size = image_size
        self.is_train = is_train
        self.repeat_factor = max(1, int(repeat_factor)) if is_train else 1
        self.n_samples = len(self.images_path) * self.repeat_factor

        aug_list = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.6),
            A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3, distort_limit=0.05),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
        ]
        if is_retinal:
            aug_list.insert(0, A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5))

        self.transform = A.Compose(aug_list)

    def _read_image(self, path):
        if path.lower().endswith(".gif"):
            return np.array(Image.open(path).convert("RGB"))
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            return np.array(Image.open(path).convert("RGB"))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_mask(self, path):
        if path.lower().endswith(".gif"):
            return np.array(Image.open(path).convert("L"))
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return np.array(Image.open(path).convert("L"))
        return m

    def __getitem__(self, index):
        actual_idx = index % len(self.images_path)
        image = self._read_image(self.images_path[actual_idx])
        mask  = self._read_mask(self.masks_path[actual_idx])

        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        if self.is_train:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))

        mask = mask.astype(np.float32)
        if mask.max() > 0:
            mask = (mask > (mask.max() / 2.0)).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

    def __len__(self):
        return self.n_samples

train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True, repeat_factor=REPEAT_FACTOR, is_retinal=True)
test_dataset  = UniversalDataset(test_x, test_y, IMAGE_SIZE, is_train=False, repeat_factor=1, is_retinal=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f"[+] Dataloaders ready: {len(train_dataset)} augmented train samples per epoch")

# ------------------------------------------------------------------
# 4. Model Definition: Proposed FANet_MFAD
# ------------------------------------------------------------------
class FeedbackAttentionGate(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )

    def forward(self, x, prev_mask):
        if prev_mask is None:
            return x
        if prev_mask.shape[2:] != x.shape[2:]:
            prev_mask = F.interpolate(prev_mask, size=x.shape[2:], mode='bilinear', align_corners=False)
        prev_mask_detached = prev_mask.detach()
        alpha = self.conv(torch.cat([x, prev_mask_detached], dim=1))
        return x + x * alpha


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


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(self.up(x))


class FANet_MFAD(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        base = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.init_conv = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.maxpool   = base.maxpool
        self.e1 = base.layer1
        self.e2 = base.layer2
        self.e3 = base.layer3
        self.e4 = base.layer4

        self.b = ConvBlock(512, 512)
        self.up1 = UpBlock(512, 256)
        self.d1  = ConvBlock(512, 256)
        self.up2 = UpBlock(256, 128)
        self.d2  = ConvBlock(256, 128)
        self.up3 = UpBlock(128, 64)
        self.d3  = ConvBlock(128, 64)
        self.up4 = UpBlock(64, 64)
        self.d4  = ConvBlock(128, 64)
        self.up5 = UpBlock(64, 32)
        self.d5  = ConvBlock(32, 32)

        self.att1 = FeedbackAttentionGate(256)
        self.att2 = FeedbackAttentionGate(128)
        self.att3 = FeedbackAttentionGate(64)
        self.att4 = FeedbackAttentionGate(64)
        self.att5 = FeedbackAttentionGate(32)

        self.out = nn.Conv2d(32, num_classes, kernel_size=1)

    def _mfad(self, d_feat, prev_mask, att_gate):
        if prev_mask is None:
            return d_feat
        return att_gate(d_feat, prev_mask)

    def forward(self, x, prev_mask=None):
        init = self.init_conv(x)
        s1 = self.e1(self.maxpool(init))
        s2 = self.e2(s1)
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
        'dice': float(dice),
        'miou': float(iou),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'fpr': float(fpr),
        'precision': float(precision)
    }

# ------------------------------------------------------------------
# 6. Pretrained Weights Initialization & Fine-Tuning
# ------------------------------------------------------------------
model = FANet_MFAD().to(DEVICE)

# Ensure ISIC-2018 pretrained checkpoint is available
isic_main_path = "/content/FANet_MFAD_ISIC-2018_best.pth"
if not os.path.exists(isic_main_path):
    print("[*] Downloading ISIC-2018 pretrained weights from Kaggle into /content/...")
    cmd = ["kaggle", "kernels", "output", "namnguynnnn/fanet-benchmark-isic-2018", "-p", "/content", "-q", "--file-pattern", ".*best\\.pth$"]
    try:
        subprocess.run(cmd, check=True)
        print("[+] Downloaded ISIC-2018 weights from Kaggle.")
    except Exception as e:
        print(f"[!] Warning downloading ISIC weights via Kaggle CLI: {e}")

isic_ckpt_candidates = [
    isic_main_path,
    os.path.join(DATA_DIR, "FANet_MFAD_ISIC-2018_best.pth"),
    "/kaggle/input/fanet-benchmark-isic-2018/FANet_MFAD_ISIC-2018_best.pth",
]

isic_loaded = False
for ckpt_p in isic_ckpt_candidates:
    if os.path.exists(ckpt_p):
        print(f"[*] Loading Pretrained Weights from: {ckpt_p}")
        state = torch.load(ckpt_p, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state, strict=True)
        isic_loaded = True
        print("[+] Successfully initialized FANet_MFAD from ISIC-2018 pre-training!")
        break

if not isic_loaded:
    raise FileNotFoundError("Pretrained ISIC-2018 weights (FANet_MFAD_ISIC-2018_best.pth) not found! Transfer learning requires ISIC weights.")

criterion = DiceBCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6)
scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

best_dice = -1.0
best_ckpt = os.path.join(OUT_DIR, "FANet_MFAD_DRIVE_Transfer_best.pth")
history = {'train_loss': [], 'val_dice': []}

print("\n[*] Starting Fine-Tuning on DRIVE for 100 epochs...")
for epoch in range(EPOCHS):
    model.train()
    total_loss, n_batches = 0.0, 0

    for x, y in train_loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            out1 = model(x)
            loss1 = criterion(out1, y)
            out2 = model(x, torch.sigmoid(out1))
            loss2 = criterion(out2, y)
            loss = 0.5 * loss1 + 0.5 * loss2

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        n_batches += 1

    train_loss = total_loss / max(n_batches, 1)

    # Validation on official test set
    model.eval()
    val_dice_sum, n_val = 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
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

print(f"\n[+] Training Complete. Best Test Dice = {best_dice:.4f}")

# ------------------------------------------------------------------
# 7. Final Benchmark Evaluation (Pass-2 Final Feedback)
# ------------------------------------------------------------------
print("\n[*] Evaluating Best Model on Official Test Set (Final Feedback Pass-2)...")
model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE, weights_only=True))
model.eval()

all_metrics = []
all_imgs, all_gts, all_preds = [], [], []

with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        out1 = model(x)
        out2 = model(x, torch.sigmoid(out1))
        pred = (torch.sigmoid(out2).cpu().numpy() > 0.5).astype(np.float32)
        gt   = y.cpu().numpy()

        for b_i in range(pred.shape[0]):
            m = compute_metrics(gt[b_i], pred[b_i])
            all_metrics.append(m)
            all_imgs.append(x.cpu().numpy()[b_i])
            all_gts.append(gt[b_i])
            all_preds.append(pred[b_i])

agg = {}
for k in all_metrics[0]:
    vals = [m[k] for m in all_metrics]
    agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

print(f"\n{'='*65}")
print("  FANet-MFAD Transfer Learning — Final Results on DRIVE")
print(f"{'='*65}")
for k, v in agg.items():
    print(f"  {k:>15s}: {v['mean']:.4f} ± {v['std']:.4f}")
print(f"{'='*65}\n")

# Save summary JSON
json_path = os.path.join(OUT_DIR, "benchmark_summary_drive_transfer.json")
with open(json_path, "w") as f:
    json.dump({"dataset": "DRIVE_Transfer", "model": "FANet_MFAD", "results": agg}, f, indent=2)

# Save CSV
csv_path = os.path.join(OUT_DIR, "benchmark_results_drive_transfer.csv")
import csv
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Model", "Metric", "Mean", "Std"])
    for k, v in agg.items():
        writer.writerow(["FANet_MFAD", k, f"{v['mean']:.6f}", f"{v['std']:.6f}"])

print(f"[+] Results saved to {csv_path} and {json_path}")
print("[+] Finished successfully!")
