"""
Script: build_isic2018_official_benchmark.py
Purpose: Generate the updated ISIC-2018 benchmark notebook and kernel-metadata.json
         following the BGDNet (AIMS Bioengineering 2026) data partitioning:
         - Training pool (2594 images): 2334 Train, 260 Val (for model selection)
         - Official Validation (100 images): NOT used as val
         - Official Test (1000 images): Evaluated on ISIC2018_Task1-2_Test_Input + ISIC2018_Task1_Test_GroundTruth
"""

import json
import os
import nbformat as nbf

def generate_notebook(dest_path="kaggle/benchmark_kernels/isic-2018/fanet_vs_mfad_isic_2018.ipynb"):
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

    # Cell 0: Header & Documentation
    nb.cells.append(nbf.v4.new_markdown_cell(
"""# GDF-Net (FANet-MFAD) Official Benchmark: ISIC-2018

**Evaluation Protocol:** Strictly follows BGDNet (*AIMS Bioengineering* 2026, DOI: 10.3934/bioeng.2026019):
- **Training Pool (2,594 pairs):** Divided into **2,334 training images** and **260 validation images** (seed=2026 / 42).
- **Validation Rule:** Official 100-image validation pool is NOT used as validation; checkpoint selection is driven strictly by highest validation Dice on the 260 hold-out training-pool split.
- **Official Test Set (1,000 images):** Final generalization is evaluated on the complete **official 1,000 test images** (`ISIC2018_Task1-2_Test_Input` + `ISIC2018_Task1_Test_GroundTruth`).

**Architecture:** GDF-Net (ResNet-34 pre-trained ImageNet Backbone + 4-channel Guidance Stem with Feedback Firewall `detach()` + 5-stage Multi-scale Feedback Attention Decoder (MFAD)).
**Loss:** 0.5×BCE + 0.5×Dice | **Scheduler:** ReduceLROnPlateau
"""))

    # Cell 1: Imports & Setup
    nb.cells.append(nbf.v4.new_code_cell(
r"""import os, sys, random, time, csv, json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict
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
# UniversalDataset for ISIC-2018
# ================================================================
class UniversalDataset(Dataset):
    def __init__(self, images_path, masks_path, image_size, is_train=False, repeat_factor=1):
        self.images_path = list(images_path)
        self.masks_path  = list(masks_path)
        self.image_size  = image_size
        self.is_train    = is_train
        self.repeat_factor = max(1, int(repeat_factor)) if is_train else 1
        self.n_samples   = len(self.images_path) * self.repeat_factor

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
        actual_idx = index % len(self.images_path)
        image = self._read_image(self.images_path[actual_idx])
        mask  = self._read_mask(self.masks_path[actual_idx])

        # Resize
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  self.image_size, interpolation=cv2.INTER_NEAREST)

        # Augmentation (train only)
        if self.is_train:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']

        # Normalise image -> [0,1] float32
        image = image.astype(np.float32)
        mx = image.max()
        if mx > 0:
            image = image / (65535.0 if mx > 255 else 255.0)
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        # Binarise mask -> {0, 1}
        mask = mask.astype(np.float32)
        if mask.max() > 0:
            mask = (mask > (mask.max() + mask.min()) / 2.0).astype(np.float32)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)

        return image, mask

    def __len__(self):
        return self.n_samples
"""))

    # Cell 3: Data Discovery & BGDNet Splitting Protocol
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# ISIC-2018 PROTOCOL & PARTITIONING (BGDNet Paper Protocol)
# Train: 2,334 | Val: 260 | Test: 1,000 (Official Test Set)
# ================================================================
DATASET_NAME = "ISIC-2018"
IMAGE_SIZE = (512, 512)
EPOCHS = 60
BATCH_SIZE = 8
LR = 1e-4

def find_dir_by_name(target_name, base_search="/kaggle/input"):
    for root, dirs, files in os.walk(base_search):
        if os.path.basename(root).lower() == target_name.lower():
            return root
    return None

def isic_id(path):
    base = os.path.basename(path).replace("_segmentation", "")
    return os.path.splitext(base)[0]

# 1. Locate Training Pool Directories
train_img_dir = find_dir_by_name("ISIC2018_Task1-2_Training_Input")
train_gt_dir  = find_dir_by_name("ISIC2018_Task1_Training_GroundTruth")

# 2. Locate Official Test Directories
test_img_dir  = find_dir_by_name("ISIC2018_Task1-2_Test_Input")
test_gt_dir   = find_dir_by_name("ISIC2018_Task1_Test_GroundTruth")

# Fallback: scan any subdirectories in /kaggle/input if folder names differ
if not train_img_dir or not train_gt_dir:
    all_jpgs = sorted(glob("/kaggle/input/**/*.jpg", recursive=True))
    all_pngs = sorted(glob("/kaggle/input/**/*_segmentation.png", recursive=True))
else:
    all_jpgs = sorted(glob(os.path.join(train_img_dir, "*.jpg")))
    all_pngs = sorted(glob(os.path.join(train_gt_dir, "*.png")))

img_dict = {isic_id(p): p for p in all_jpgs}
msk_dict = {isic_id(p): p for p in all_pngs}
train_pool_ids = sorted(set(img_dict.keys()) & set(msk_dict.keys()))

print(f"Training Pool: Found {len(train_pool_ids)} matched image-mask pairs.")

# Deterministic BGDNet Partition: 2,334 Train, 260 Val
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

print(f"BGDNet Training Partition -> Train: {len(train_x)} | Val: {len(val_x)}")

# 3. Discover Official Test Set (1,000 images)
if test_img_dir and test_gt_dir:
    test_jpgs = sorted(glob(os.path.join(test_img_dir, "*.jpg")))
    test_pngs = sorted(glob(os.path.join(test_gt_dir, "*.png")))
    test_img_dict = {isic_id(p): p for p in test_jpgs}
    test_msk_dict = {isic_id(p): p for p in test_pngs}
    test_ids = sorted(set(test_img_dict.keys()) & set(test_msk_dict.keys()))
    test_x = [test_img_dict[k] for k in test_ids]
    test_y = [test_msk_dict[k] for k in test_ids]
    print(f"Official Test Set: Found {len(test_x)} matched image-mask pairs.")
else:
    # If test ground truth not mounted, fall back to val_x for safety
    print("Notice: Official test directories not found. Using hold-out val for testing.")
    test_x, test_y = val_x, val_y

train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True, repeat_factor=1)
valid_dataset = UniversalDataset(val_x, val_y, IMAGE_SIZE, is_train=False, repeat_factor=1)
test_dataset  = UniversalDataset(test_x, test_y, IMAGE_SIZE, is_train=False, repeat_factor=1)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                          num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"DataLoaders Ready -> Train Batches: {len(train_loader)} | Val: {len(valid_loader)} | Test: {len(test_loader)}")
"""))

    # Cell 4: Model Architecture (GDF-Net)
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# GDF-Net Architecture (ResNet-34 + Decoupled MFAD Decoder)
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
    '''Gradient-Decoupled Feedback Network.
    
    1. Pre-trained ResNet-34 ImageNet encoder
    2. Feedback Firewall: .detach() at recurrent input boundary
    3. Learned spatial gate on detached prior
    4. 4-channel input stem with ImageNet warm-up
    5. Multi-scale Feedback Attention Decoder (MFAD)
    '''
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # 4-Channel input stem (RGB + spatial prior)
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

        # 5-Stage Decoder with Skip Connections
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
            # Feedback Firewall: detach + learned modulation
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

    # Cell 5: Loss & Evaluation Metrics
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# Loss Function & Comprehensive Metrics (Region + Boundary)
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

    # BGDNet Boundary Metrics (HD95, ASD, BF@2 at 352x352)
    if eval_boundary and (np.sum(gt) > 0 and np.sum(pred) > 0):
        try:
            # Resize binary masks to 352x352 for standard boundary contour evaluation
            g_2d = gt.squeeze()
            p_2d = pred.squeeze()
            g352 = cv2.resize(g_2d, (352, 352), interpolation=cv2.INTER_NEAREST)
            p352 = cv2.resize(p_2d, (352, 352), interpolation=cv2.INTER_NEAREST)

            # Contours via Morphological Gradient
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            g_bnd = cv2.morphologyEx(g352, cv2.MORPH_GRADIENT, kernel) > 0
            p_bnd = cv2.morphologyEx(p352, cv2.MORPH_GRADIENT, kernel) > 0

            # Distance transforms
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

    # Cell 6: Training Loop with Auto-Resume
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# Training Function (Driven by Validation Set of 260 images)
# ================================================================
def train_model(model, train_loader, valid_loader, epochs=EPOCHS, lr=LR):
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    scaler = torch.amp.GradScaler('cuda')

    best_dice = 0.0
    history = {'train_loss': [], 'val_dice': []}
    ckpt_path = "GDFNet_ISIC2018_best.pth"
    latest_path = "GDFNet_ISIC2018_latest.pth"
    start_epoch = 0

    # Auto-resume logic
    if os.path.exists(latest_path):
        print(f"[*] Resuming from local latest checkpoint: {latest_path}")
        ckpt = torch.load(latest_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        best_dice = ckpt['best_dice']
        history = ckpt['history']
        start_epoch = ckpt['epoch']

    start_time = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{epochs} [Train]", leave=False):
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

        # Validation on 260 Hold-out images
        model.eval()
        val_dice_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in valid_loader:
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

        print(f"Epoch {epoch+1:03d}/{epochs} | LR {lr_now:.1e} | Loss {train_loss:.4f} | Val Dice (260 split) {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [+] Best model saved with Val Dice: {best_dice:.4f}")

        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_dice': best_dice,
            'history': history
        }, latest_path)

        if time.time() - start_time > 11.5 * 3600:
            print("[!] Approaching Kaggle time limit. Exiting cleanly.")
            break

    print(f"Training Complete. Peak Val Dice = {best_dice:.4f}")
    return ckpt_path, history
"""))

    # Cell 7: Official Test Set Evaluation & Reporting
    nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# EVALUATION ON OFFICIAL TEST SET (1,000 IMAGES)
# ================================================================
def evaluate_official_test(model, ckpt_path, test_loader):
    print(f"\n{'='*70}")
    print(f"  EVALUATING ON OFFICIAL ISIC-2018 TEST SET (N={len(test_loader.dataset)})")
    print(f"{'='*70}")

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    model.eval()

    all_metrics = []
    all_imgs, all_gts, all_preds = [], [], []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="[GDF-Net] Official Test Set Eval"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out1 = model(x)
                out2 = model(x, torch.sigmoid(out1))

            pred = (torch.sigmoid(out2).cpu().numpy() > 0.5).astype(np.float32)
            gt   = y.cpu().numpy()
            imgs = x.cpu().numpy()

            for b_i in range(pred.shape[0]):
                m = compute_metrics(gt[b_i], pred[b_i], eval_boundary=True)
                all_metrics.append(m)
                if len(all_imgs) < 10:
                    all_imgs.append(imgs[b_i])
                    all_gts.append(gt[b_i])
                    all_preds.append(pred[b_i])

    # Aggregation
    agg = {}
    keys = all_metrics[0].keys()
    for k in keys:
        vals = [m[k] for m in all_metrics if k in m]
        agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    return agg, all_imgs, all_gts, all_preds

# Instantiate and Train
model = GDFNet().to(DEVICE)
best_ckpt, history = train_model(model, train_loader, valid_loader, epochs=EPOCHS, lr=LR)

# Evaluate on Official 1,000 Test Images
agg, sample_imgs, sample_gts, sample_preds = evaluate_official_test(model, best_ckpt, test_loader)

# Print Final Benchmark Table
print(f"\n{'='*70}")
print(f"  FINAL BENCHMARK RESULTS ON OFFICIAL ISIC-2018 TEST SET (N={len(test_loader.dataset)})")
print(f"{'='*70}")
print(f"  {'Metric':<25s} | {'Mean ± Std':>25s}")
print(f"  {'-'*25}-+-{'-'*25}")
for k, v in agg.items():
    print(f"  {k:<25s} | {v['mean']:>10.4f} ± {v['std']:<10.4f}")
print(f"{'='*70}")

# Save CSV
csv_path = "benchmark_results_ISIC-2018_official_test.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Dataset', 'Cohort', 'Metric', 'Mean', 'Std'])
    for k, v in agg.items():
        writer.writerow(['GDF-Net', 'ISIC-2018', 'Official_Test_1000', k, f"{v['mean']:.6f}", f"{v['std']:.6f}"])
print(f"Results saved to: {csv_path}")

# Plot Qualitative Samples
n = min(5, len(sample_imgs))
fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
for i in range(n):
    img = np.transpose(sample_imgs[i], (1, 2, 0))
    img = np.clip(img, 0, 1)
    axes[i, 0].imshow(img)
    axes[i, 0].set_title('Test Input Image' if i == 0 else '')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(sample_gts[i][0], cmap='gray')
    axes[i, 1].set_title('Official Ground Truth' if i == 0 else '')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(sample_preds[i][0], cmap='gray')
    axes[i, 2].set_title('GDF-Net Prediction' if i == 0 else '')
    axes[i, 2].axis('off')

plt.tight_layout()
plt.savefig("comparison_ISIC-2018.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(history['train_loss'], label='GDF-Net Loss', color='#1f77b4', linewidth=2)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
ax1.set_title('Training Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(history['val_dice'], label='Val Dice (260 Split)', color='#2ca02c', linewidth=2)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Dice')
ax2.set_title('Validation Convergence'); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves_ISIC-2018.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved figures: comparison_ISIC-2018.png and training_curves_ISIC-2018.png")
"""))

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook successfully written to {dest_path}")


def update_metadata(dest_path="kaggle/benchmark_kernels/isic-2018/kernel-metadata.json"):
    metadata = {
        "id": "namnguynnnn/fanet-benchmark-isic-2018",
        "title": "fanet-benchmark-isic-2018",
        "code_file": "fanet_vs_mfad_isic_2018.ipynb",
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
    with open(dest_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Kernel metadata successfully written to {dest_path}")


if __name__ == "__main__":
    generate_notebook()
    update_metadata()
