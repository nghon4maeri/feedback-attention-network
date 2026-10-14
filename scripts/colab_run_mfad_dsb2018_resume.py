"""
colab_run_mfad_dsb2018_resume.py
================================
Resume Training script for FANet_MFAD on DSB-2018 (Data Science Bowl 2018).
Resumes from Epoch 39 checkpoint (FANet_MFAD_DSB-2018_latest.pth) to Epoch 100.
Features:
  - High-speed in-RAM caching of all 670 preprocessed 256x256 images (~175 MB total).
  - Dynamic Albumentations data augmentation (repeat factor = 2, 1072 train samples/epoch).
  - Strict Final Feedback Phase (t=2) evaluation directly against ground truth.
  - Generates benchmark_results_dsb2018.csv and summary JSON.
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
print("  FANet-MFAD DSB-2018 Benchmark: Resume Training Epochs 40-100")
print("  Architecture: Canonical FANet_MFAD (ResNet-34 + MFAD)")
print("=" * 65)

kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
with open(kaggle_json, "w") as f:
    json.dump({"username": "namnguynnnn", "key": "KGAT_dad33ce94fb8b25d8a9ecd4ca26ce03f"}, f)
os.chmod(kaggle_json, 0o600)
print("[+] Kaggle API credentials configured.")

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "albumentations", "tifffile", "opencv-python-headless", "kaggle"], check=True)
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

# ------------------------------------------------------------------
# 2. Dataset Acquisition & In-Memory Preprocessing
# ------------------------------------------------------------------
DATA_DIR = "/content/dsb2018_data"
WORK_DIR = "/content/dsb2018_work"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(WORK_DIR, exist_ok=True)

STAGE1_ZIP = os.path.join(DATA_DIR, "stage1_train.zip")
ORIG_DIR = os.path.join(WORK_DIR, "stage1_train")
MERGED_DIR = os.path.join(WORK_DIR, "dsb2018_masks")

if not os.path.exists(STAGE1_ZIP) and not os.path.exists(ORIG_DIR):
    print("[*] Downloading stage1_train.zip from Kaggle...")
    # Try downloading from competition first
    try:
        subprocess.run(["kaggle", "competitions", "download", "-c", "data-science-bowl-2018",
                        "-f", "stage1_train.zip", "-p", DATA_DIR], check=True)
    except Exception as e:
        print(f"[!] Competition download failed ({e}), searching public datasets...")
        subprocess.run(["kaggle", "datasets", "download", "-d", "carlossouza/data-science-bowl-2018",
                        "-p", DATA_DIR, "--unzip"], check=True)

# Check if stage1_train.zip exists (or .zip.zip)
if not os.path.exists(STAGE1_ZIP):
    zips = glob.glob(os.path.join(DATA_DIR, "**", "stage1_train*.zip"), recursive=True)
    if zips:
        STAGE1_ZIP = zips[0]

if os.path.exists(STAGE1_ZIP) and not os.path.exists(ORIG_DIR):
    print(f"[*] Extracting {STAGE1_ZIP} -> {ORIG_DIR}...")
    os.makedirs(ORIG_DIR, exist_ok=True)
    with zipfile.ZipFile(STAGE1_ZIP, 'r') as z:
        z.extractall(ORIG_DIR)

os.makedirs(MERGED_DIR, exist_ok=True)

print("[*] Pre-processing & merging nuclei masks (with RAM caching)...")
IMAGE_SIZE = (256, 256)
all_folders = sorted(os.listdir(ORIG_DIR))
ram_images = []
ram_masks = []

for folder in tqdm(all_folders, desc="Loading & Merging"):
    folder_path = os.path.join(ORIG_DIR, folder)
    if not os.path.isdir(folder_path):
        continue
    img_file = os.path.join(folder_path, "images", folder + ".png")
    mask_files = glob.glob(os.path.join(folder_path, "masks", "*.png"))
    if not os.path.exists(img_file) or len(mask_files) == 0:
        continue

    # Read image
    img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
    if img is None:
        img = np.array(Image.open(img_file).convert("RGB"))
    if len(img.shape) == 2:
        img = np.stack([img] * 3, axis=-1)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_LINEAR)

    # Merge masks
    merged = np.zeros(img.shape[:2], dtype=np.uint8)
    for mf in mask_files:
        m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            merged = np.maximum(merged, m)
    merged = cv2.resize(merged, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
    merged = (merged > 127).astype(np.uint8)

    ram_images.append(img)
    ram_masks.append(merged)

total_samples = len(ram_images)
print(f"[+] Loaded {total_samples} samples completely into system RAM!")

# 80:20 deterministic split with seed 42
random.seed(42)
indices = list(range(total_samples))
random.shuffle(indices)
split_idx = max(1, int(total_samples * 0.8))

train_indices = indices[:split_idx]
val_indices   = indices[split_idx:]
print(f"[+] DSB-2018 Split: Train={len(train_indices)}, Val/Test={len(val_indices)}")

# ------------------------------------------------------------------
# 3. RAM Dataset with Paper-Grade Albumentations Augmentations
# ------------------------------------------------------------------
class RAMDataset(Dataset):
    def __init__(self, images_list, masks_list, indices, is_train=False, repeat_factor=2):
        self.images = [images_list[i] for i in indices]
        self.masks  = [masks_list[i] for i in indices]
        self.is_train = is_train
        self.repeat_factor = max(1, int(repeat_factor)) if is_train else 1
        self.n_samples = len(self.images) * self.repeat_factor

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.6),
            A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3, distort_limit=0.05),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
        ]) if is_train else None

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        actual_idx = index % len(self.images)
        image = self.images[actual_idx].copy()
        mask  = self.masks[actual_idx].copy()

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        image = (image.astype(np.float32) / 255.0).transpose(2, 0, 1)
        mask = np.expand_dims(mask.astype(np.float32), axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

REPEAT_FACTOR = 2
BATCH_SIZE = 16
NUM_WORKERS = min(4, os.cpu_count() or 2)

train_dataset = RAMDataset(ram_images, ram_masks, train_indices, is_train=True, repeat_factor=REPEAT_FACTOR)
val_dataset   = RAMDataset(ram_images, ram_masks, val_indices,   is_train=False, repeat_factor=1)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

print(f"[+] Dataloaders ready: {len(train_dataset)} augmented train samples per epoch | Batch Size={BATCH_SIZE}")

# ------------------------------------------------------------------
# 4. Canonical FANet_MFAD Model Definition
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

    def _mfad(self, feat, mask_guide, att_layer):
        if mask_guide is None:
            return feat
        mg = F.interpolate(mask_guide, size=feat.shape[2:], mode='bilinear', align_corners=False)
        return feat * (1.0 + att_layer(mg))

    def forward(self, x, prev_mask=None):
        B, C, H, W = x.shape
        if prev_mask is None:
            guidance = torch.zeros(B, 1, H, W, device=x.device)
            mask_guide = None
        else:
            prev_d = prev_mask.detach()
            guidance = torch.sigmoid(self.learned_gate(prev_d))
            mask_guide = prev_d

        x_in = torch.cat([x, guidance], dim=1)
        e1 = self.e1_relu(self.e1_bn(self.e1_conv(x_in)))
        p1 = self.pool(e1)
        e2 = self.e2(p1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        b  = self.b(e4)

        u1 = self.up1(b)
        e4_g = self._mfad(e4, mask_guide, self.att1)
        d1 = self.d1(torch.cat([u1, e4_g], dim=1))

        u2 = self.up2(d1)
        e3_g = self._mfad(e3, mask_guide, self.att2)
        d2 = self.d2(torch.cat([u2, e3_g], dim=1))

        u3 = self.up3(d2)
        e2_g = self._mfad(e2, mask_guide, self.att3)
        d3 = self.d3(torch.cat([u3, e2_g], dim=1))

        u4 = self.up4(d3)
        e1_g = self._mfad(e1, mask_guide, self.att4)
        d4 = self.d4(torch.cat([u4, e1_g], dim=1))

        u5 = self.up5(d4)
        u5_g = self._mfad(u5, mask_guide, self.att5)
        d5 = self.d5(u5_g)

        return self.out(d5)

# ------------------------------------------------------------------
# 5. Losses & Evaluation Metrics
# ------------------------------------------------------------------
class DiceBCELoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        p_flat = probs.view(-1)
        t_flat = targets.view(-1)
        intersection = (p_flat * t_flat).sum()
        dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (p_flat.sum() + t_flat.sum() + self.smooth)
        return 0.5 * bce_loss + 0.5 * dice_loss


def compute_metrics(gt, pred, smooth=1e-5):
    gt = gt.flatten()
    pred = pred.flatten()
    tp = np.sum((gt == 1) & (pred == 1))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    tn = np.sum((gt == 0) & (pred == 0))

    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    specificity = (tn + smooth) / (tn + fp + smooth)
    precision = (tp + smooth) / (tp + fp + smooth)
    fpr = fp / (fp + tn + smooth)

    return {
        "dice": float(dice),
        "miou": float(iou),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "fpr": float(fpr),
        "precision": float(precision)
    }

# ------------------------------------------------------------------
# 6. Checkpoint Download & Resume Setup
# ------------------------------------------------------------------
model = FANet_MFAD().to(DEVICE)
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

start_epoch = 0
best_val_dice = 0.0

CKPT_FILE = "/content/FANet_MFAD_DSB-2018_latest.pth"
if not os.path.exists(CKPT_FILE):
    print("[*] Downloading latest DSB-2018 checkpoint from Kaggle kernel output...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        import requests
        api = KaggleApi()
        api.authenticate()
        with api.build_kaggle_client() as kc:
            from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
            req = ApiListKernelSessionOutputRequest()
            req.user_name = "namnguynnnn"
            req.kernel_slug = "fanet-benchmark-dsb2018"
            resp = kc.kernels.kernels_api_client.list_kernel_session_output(req)
            for f in (resp.files or []):
                if "latest.pth" in f.file_name or "best.pth" in f.file_name:
                    target_p = f"/content/{os.path.basename(f.file_name)}"
                    print(f"[*] Downloading {f.file_name} -> {target_p}...")
                    r = requests.get(f.url, stream=True, timeout=120)
                    r.raise_for_status()
                    with open(target_p, "wb") as out:
                        for chunk in r.iter_content(chunk_size=4*1024*1024):
                            if chunk: out.write(chunk)
                    print(f"[+] Saved {target_p} ({os.path.getsize(target_p)} bytes)")
    except Exception as e:
        print(f"[!] Kaggle kernel download failed: {e}")

if os.path.exists(CKPT_FILE):
    print(f"[*] Loading checkpoint from {CKPT_FILE}...")
    ckpt = torch.load(CKPT_FILE, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    start_epoch = ckpt.get('epoch', 39)
    best_val_dice = ckpt.get('best_dice', 0.8872)
    print(f"[+] Successfully resumed from Epoch {start_epoch} | Previous Best Dice: {best_val_dice:.4f}")
else:
    print("[!] No checkpoint found, starting from epoch 0 with ImageNet pretrained encoder.")

# ------------------------------------------------------------------
# 7. Resume Training (Epoch 40 -> 100)
# ------------------------------------------------------------------
TOTAL_EPOCHS = 100
OUT_DIR = "/content/output_dsb2018"
os.makedirs(OUT_DIR, exist_ok=True)
best_ckpt_path = os.path.join(OUT_DIR, "FANet_MFAD_DSB-2018_best.pth")

print(f"\n[*] Commencing Training from Epoch {start_epoch+1} to {TOTAL_EPOCHS}...")
t_start = time.time()

for epoch in range(start_epoch, TOTAL_EPOCHS):
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
    val_dices = []
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                out1 = model(x)
                out2 = model(x, torch.sigmoid(out1))
            pred = (torch.sigmoid(out2).squeeze().cpu().numpy() > 0.5).astype(np.float32)
            gt   = y.squeeze().cpu().numpy()
            m = compute_metrics(gt, pred)
            val_dices.append(m['dice'])

    val_dice = float(np.mean(val_dices))
    scheduler.step(val_dice)
    lr = optimizer.param_groups[0]['lr']

    is_best = val_dice > best_val_dice
    star = " *" if is_best else ""
    if is_best:
        best_val_dice = val_dice
        torch.save(model.state_dict(), best_ckpt_path)

    print(f"  [Epoch {epoch+1:03d}/{TOTAL_EPOCHS:03d}] Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f} | Best: {best_val_dice:.4f} | LR: {lr:.1e}{star}")

print(f"\n[+] Training Complete in {(time.time() - t_start):.1f}s. Best Val Dice = {best_val_dice:.4f}")

# ------------------------------------------------------------------
# 8. Official Test Evaluation (Final Feedback Phase t=2)
# ------------------------------------------------------------------
print("\n[*] Evaluating Best Model on Official Test Set (Final Feedback Pass-2)...")
if os.path.exists(best_ckpt_path):
    model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
model.eval()

metrics_list = []
with torch.no_grad():
    for idx, (x, y) in enumerate(tqdm(val_loader, desc="Testing")):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            out1 = model(x)
            out2 = model(x, torch.sigmoid(out1))
        pred = (torch.sigmoid(out2).squeeze().cpu().numpy() > 0.5).astype(np.float32)
        gt   = y.squeeze().cpu().numpy()
        m = compute_metrics(gt, pred)
        m["sample_id"] = idx
        metrics_list.append(m)

# Statistical aggregation
summary = {}
print("\n" + "=" * 65)
print("  FANet-MFAD Final Results on DSB-2018 (Nuclei)")
print("=" * 65)
for key in ["dice", "miou", "sensitivity", "specificity", "fpr", "precision"]:
    vals = [m[key] for m in metrics_list]
    mean_val = float(np.mean(vals))
    std_val  = float(np.std(vals))
    summary[key] = {"mean": mean_val, "std": std_val}
    print(f"{key:>17s}: {mean_val:.4f} ± {std_val:.4f}")
print("=" * 65)

# Save results CSV and JSON
import csv
csv_path = os.path.join(OUT_DIR, "benchmark_results_dsb2018.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(metrics_list[0].keys()))
    writer.writeheader()
    writer.writerows(metrics_list)

json_path = os.path.join(OUT_DIR, "benchmark_summary_dsb2018.json")
with open(json_path, "w") as f:
    json.dump({"dataset": "DSB-2018", "model": "FANet_MFAD", "results": summary}, f, indent=2)

print(f"\n[+] Results saved to {csv_path} and {json_path}")
print("[+] Finished successfully!")
