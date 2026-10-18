"""
Script: build_isic2018_retrain_official_val.py
Purpose: Generate Kaggle kernel to retrain GDF-Net on all 2,594 images,
         using the Official Validation Set (100 images) for checkpoint selection,
         and evaluating on the Official Test Set (1,000 images).
"""

import json
import os
import nbformat as nbf

def generate_retrain_notebook(dest_path="kaggle/benchmark_kernels/isic-2018-retrain-official-val/fanet_isic2018_retrain_official_val.ipynb"):
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
"""# GDF-Net Retraining: Full 2,594 Train + Official 100 Val + Official 1,000 Test

**Experimental Design:**
- **Training Set (N=2,594 pairs):** The entire ISIC-2018 training pool (`ISIC2018_Task1-2_Training_Input` + `ISIC2018_Task1_Training_GroundTruth`), with full Albumentations data augmentation.
- **Validation Set (N=100 pairs):** The **Official ISIC-2018 Validation Set** (`ISIC2018_Task1-2_Validation_Input` + `ISIC2018_Task1_Validation_GroundTruth`). Used to drive the ReduceLROnPlateau scheduler and select the best checkpoint (`GDFNet_ISIC2018_best.pth`).
- **Official Test Set (N=1,000 pairs):** Final evaluation on the unseen official test set (`ISIC2018_Task1-2_Test_Input` + `ISIC2018_Task1_Test_GroundTruth`) using the official-val-selected checkpoint.

**Architecture:** GDF-Net (ResNet-34 ImageNet pretrained + Feedback Firewall + MFAD Decoder).
**Loss:** 0.5×BCE + 0.5×Dice | **Horizon:** 200 Epochs | **Resolution:** 512×512
"""))

    # Cell 1: Imports & Setup
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

    # Cell 3: Data Discovery
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# Data Discovery: 2,594 Train | 100 Official Val | 1,000 Official Test
# ================================================================
def isic_id(path):
    base = os.path.basename(path).replace("_segmentation", "")
    return os.path.splitext(base)[0]

def find_dir_by_name(target_name, base_search="/kaggle/input"):
    for root, dirs, files in os.walk(base_search):
        if os.path.basename(root).lower() == target_name.lower():
            return root
    return None

# 1. Training Set: All 2,594 images
train_img_dir = find_dir_by_name("ISIC2018_Task1-2_Training_Input")
train_gt_dir  = find_dir_by_name("ISIC2018_Task1_Training_GroundTruth")

train_jpgs = sorted(glob(os.path.join(train_img_dir, "*.jpg")))
train_pngs = sorted(glob(os.path.join(train_gt_dir, "*.png")))

train_img_dict = {isic_id(p): p for p in train_jpgs}
train_msk_dict = {isic_id(p): p for p in train_pngs}
train_ids = sorted(set(train_img_dict.keys()) & set(train_msk_dict.keys()))

train_x = [train_img_dict[k] for k in train_ids]
train_y = [train_msk_dict[k] for k in train_ids]
print(f"Full Training Pool: {len(train_x)} matched pairs verified.")

# 2. Official Validation Set: 100 images
VAL_IMG_URL = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip"
VAL_GT_URL  = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip"

os.makedirs("isic2018_val", exist_ok=True)
val_img_dir = find_dir_by_name("ISIC2018_Task1-2_Validation_Input")
val_gt_dir  = find_dir_by_name("ISIC2018_Task1_Validation_GroundTruth")

if not val_img_dir:
    print("[*] Downloading Official Val Input from ISIC S3...")
    urllib.request.urlretrieve(VAL_IMG_URL, "isic2018_val/val_in.zip")
    with zipfile.ZipFile("isic2018_val/val_in.zip", 'r') as z:
        z.extractall("isic2018_val")
    val_img_dir = "isic2018_val/ISIC2018_Task1-2_Validation_Input"

if not val_gt_dir:
    print("[*] Downloading Official Val GroundTruth from ISIC S3...")
    urllib.request.urlretrieve(VAL_GT_URL, "isic2018_val/val_gt.zip")
    with zipfile.ZipFile("isic2018_val/val_gt.zip", 'r') as z:
        z.extractall("isic2018_val")
    val_gt_dir = "isic2018_val/ISIC2018_Task1_Validation_GroundTruth"

val_jpgs = sorted(glob(os.path.join(val_img_dir, "*.jpg")))
val_pngs = sorted(glob(os.path.join(val_gt_dir, "*.png")))

val_img_dict = {isic_id(p): p for p in val_jpgs}
val_msk_dict = {isic_id(p): p for p in val_pngs}
val_ids  = sorted(set(val_img_dict.keys()) & set(val_msk_dict.keys()))

val_x = [val_img_dict[k] for k in val_ids]
val_y = [val_msk_dict[k] for k in val_ids]
print(f"Official Validation Set: {len(val_x)} matched pairs verified.")

# 3. Official Test Set: 1,000 images
test_img_dir = find_dir_by_name("ISIC2018_Task1-2_Test_Input")
test_gt_dir  = find_dir_by_name("ISIC2018_Task1_Test_GroundTruth")

test_jpgs = sorted(glob(os.path.join(test_img_dir, "*.jpg")))
test_pngs = sorted(glob(os.path.join(test_gt_dir, "*.png")))

test_img_dict = {isic_id(p): p for p in test_jpgs}
test_msk_dict = {isic_id(p): p for p in test_pngs}
test_ids  = sorted(set(test_img_dict.keys()) & set(test_msk_dict.keys()))

test_x = [test_img_dict[k] for k in test_ids]
test_y = [test_msk_dict[k] for k in test_ids]
print(f"Official Test Set: {len(test_x)} matched pairs verified.")

train_loader = DataLoader(UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True),
                          batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(UniversalDataset(val_x, val_y, IMAGE_SIZE, is_train=False),
                          batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(UniversalDataset(test_x, test_y, IMAGE_SIZE, is_train=False),
                          batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

print(f"DataLoaders Ready -> Train: {len(train_loader)} batches | Val: {len(val_loader)} | Test: {len(test_loader)}")
"""))

    # Cell 4: Model Architecture (GDF-Net)
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# GDF-Net Architecture (ResNet-34 + Feedback Firewall + MFAD)
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
# Loss & Comprehensive Metrics
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

    # Cell 6: Training Loop (Driven by Official 100 Validation Images)
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# Training Loop (Checkpoint Selection driven by Official 100 Val Images)
# ================================================================
EPOCHS = 200
LR = 1e-4

model = GDFNet().to(DEVICE)
criterion = DiceBCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6
)
scaler = torch.amp.GradScaler('cuda')

best_dice = 0.0
history = {'train_loss': [], 'val_dice': []}
best_ckpt_path = "GDFNet_ISIC2018_best.pth"
latest_ckpt_path = "GDFNet_ISIC2018_latest.pth"
start_epoch = 0

# Check for auto-resume if rerun
if os.path.exists(latest_ckpt_path):
    print(f"[*] Resuming from latest checkpoint: {latest_ckpt_path}")
    ckpt = torch.load(latest_ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    best_dice = ckpt.get('best_dice', 0.0)
    history = ckpt.get('history', {'train_loss': [], 'val_dice': []})
    start_epoch = ckpt.get('epoch', 0)
    print(f"  [+] Resumed at Epoch {start_epoch+1}/{EPOCHS} | Historical Best Official Val Dice: {best_dice:.4f}")

start_time = time.time()
print(f"Starting GDF-Net training on {len(train_loader.dataset)} images with official 100 val guidance...")

for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0.0
    for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{EPOCHS} [Train]", leave=False):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out1 = model(x)
            out2 = model(x, torch.sigmoid(out1))
            loss = criterion(out2, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation on Official 100 images
    model.eval()
    val_dice_sum = 0.0
    n_val = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out1 = model(x)
                out2 = model(x, torch.sigmoid(out1))
            pred = (torch.sigmoid(out2).cpu().numpy() > 0.5).astype(np.float32)
            gt   = y.cpu().numpy()
            for b_i in range(pred.shape[0]):
                m = compute_metrics(gt[b_i], pred[b_i], eval_boundary=False)
                val_dice_sum += m['dice']
                n_val += 1

    val_dice = val_dice_sum / max(n_val, 1)
    lr_now = optimizer.param_groups[0]['lr']
    scheduler.step(val_dice)

    history['train_loss'].append(train_loss)
    history['val_dice'].append(val_dice)

    print(f"Epoch {epoch+1:03d}/{EPOCHS} | LR {lr_now:.1e} | Loss {train_loss:.4f} | Official Val Dice {val_dice:.4f}")

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), best_ckpt_path)
        print(f"  [+] Best model saved with Official Val Dice: {best_dice:.4f}")

    torch.save({
        'epoch': epoch + 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_dice': best_dice,
        'history': history
    }, latest_ckpt_path)

    if time.time() - start_time > 8.5 * 3600:
        print("[!] Approaching Kaggle timeout. Exiting cleanly.")
        break

print(f"Training Complete. Peak Official Val Dice = {best_dice:.4f}")
"""))

    # Cell 7: Official Test Set Evaluation (1,000 images)
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# EVALUATION ON OFFICIAL TEST SET (1,000 IMAGES)
# ================================================================
print(f"\n{'='*75}")
print(f"  EVALUATING ON OFFICIAL ISIC-2018 TEST SET (N={len(test_loader.dataset)})")
print(f"  Model Selected By Peak Official Val Dice ({best_dice:.4f})")
print(f"{'='*75}")

model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE, weights_only=True))
model.eval()

all_metrics = []
sample_imgs, sample_gts, sample_preds, sample_names = [], [], [], []

with torch.no_grad():
    for idx, (x, y) in enumerate(tqdm(test_loader, desc="[GDF-Net] Official Test Set Eval")):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast('cuda'):
            out1 = model(x)
            out2 = model(x, torch.sigmoid(out1))

        pred = (torch.sigmoid(out2).cpu().numpy() > 0.5).astype(np.float32)
        gt   = y.cpu().numpy()
        img  = x.cpu().numpy()

        for b_i in range(pred.shape[0]):
            m = compute_metrics(gt[b_i], pred[b_i], eval_boundary=True)
            m['image_id'] = test_ids[idx]
            all_metrics.append(m)

            if len(sample_imgs) < 10:
                sample_imgs.append(img[b_i])
                sample_gts.append(gt[b_i])
                sample_preds.append(pred[b_i])
                sample_names.append(test_ids[idx])

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
print(f"  FINAL BENCHMARK: OFFICIAL TEST SET (N={len(test_loader.dataset)})")
print(f"  Training Split: Full 2594 Train | 100 Official Val Guided")
print(f"{'='*75}")
print(f"  {'Metric':<15s} | {'Mean ± Std':>22s} | {'Median':>10s} | {'[Min, Max]':>20s}")
print(f"  {'-'*15}-+-{'-'*22}-+-{'-'*10}-+-{'-'*20}")
for k in metric_keys:
    v = agg[k]
    print(f"  {k:<15s} | {v['mean']:>9.4f} ± {v['std']:<10.4f} | {v['median']:>10.4f} | [{v['min']:.4f}, {v['max']:.4f}]")
print(f"{'='*75}")

# Save CSV
summary_csv = "benchmark_results_ISIC-2018_test_official_val_split.csv"
with open(summary_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Dataset', 'Cohort', 'Metric', 'Mean', 'Std', 'Median', 'Min', 'Max'])
    for k in metric_keys:
        v = agg[k]
        writer.writerow(['GDF-Net', 'ISIC-2018', 'Official_Test_1000_OfficialValGuided', k,
                         f"{v['mean']:.6f}", f"{v['std']:.6f}", f"{v['median']:.6f}",
                         f"{v['min']:.6f}", f"{v['max']:.6f}"])
print(f"Results saved to: {summary_csv}")

detail_csv = "per_image_results_ISIC-2018_test_official_val_split.csv"
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
    axes[i, 0].set_title(f"Test Input: {sample_names[i]}" if i == 0 else sample_names[i])
    axes[i, 0].axis('off')

    axes[i, 1].imshow(sample_gts[i][0], cmap='gray')
    axes[i, 1].set_title('Official Ground Truth' if i == 0 else '')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(sample_preds[i][0], cmap='gray')
    axes[i, 2].set_title(f"GDF-Net Pred (Dice={all_metrics[i]['dice']:.3f})" if i == 0 else f"Dice={all_metrics[i]['dice']:.3f}")
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig("comparison_ISIC-2018_test.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history['train_loss'], label='Train Loss (2,594 pairs)', color='#1f77b4', linewidth=2)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Training Convergence'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history['val_dice'], label='Official Val Dice (100 pairs)', color='#2ca02c', linewidth=2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice')
ax2.set_title('Official Validation Convergence'); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves_ISIC-2018.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved plots: comparison_ISIC-2018_test.png and training_curves_ISIC-2018.png")
"""))

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook successfully written to {dest_path}")


def update_retrain_metadata(dest_path="kaggle/benchmark_kernels/isic-2018-retrain-official-val/kernel-metadata.json"):
    metadata = {
        "id": "namnguynnnn/fanet-isic2018-retrain-official-val",
        "title": "fanet-isic2018-retrain-official-val",
        "code_file": "fanet_isic2018_retrain_official_val.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": ["gpu"],
        "dataset_sources": [
            "tschandl/isic2018-challenge-task1-data-segmentation",
            "mfahimhossen/isic2018-test-groundtruth"
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
    generate_retrain_notebook()
    update_retrain_metadata()
