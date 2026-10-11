"""
build_mfad_only_benchmarks.py
=============================
Generates 7 Kaggle benchmark notebooks and 7 Google Colab notebooks
training ONLY the proposed FANet_MFAD model with enhanced paper-grade
data augmentation (including 32x repeat factor for DRIVE & CHASE-DB1,
16x repeat factor for EM-Dataset, CLAHE, and full Albumentations suite).

Datasets:
  1. Kvasir-SEG
  2. CHASE-DB1
  3. DRIVE
  4. DSB-2018
  5. ISIC-2018
  6. CVC-ClinicDB
  7. EM-Dataset

Usage:
  python scripts/build_mfad_only_benchmarks.py
"""

import os
import shutil
import nbformat as nbf

# ════════════════════════════════════════════════════════════════
# SHARED CODE CELLS
# ════════════════════════════════════════════════════════════════

CELL_IMPORTS = r"""import os, sys, random, time, csv, json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

try:
    import albumentations as A
    HAS_ALBUM = True
except ImportError:
    os.system("pip install -q albumentations")
    import albumentations as A
    HAS_ALBUM = True

try:
    import tifffile as tiff
except ImportError:
    os.system("pip install -q tifffile")
    import tifffile as tiff

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if DEVICE.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
"""

CELL_DATASET = r"""# ================================================================
# UniversalDataset — handles jpg/png/tif/gif, auto-binarises masks,
# dynamic Albumentations data augmentation & repeat_factor.
# ================================================================
class UniversalDataset(Dataset):
    def __init__(self, images_path, masks_path, image_size, is_train=False, repeat_factor=1, is_retinal=False):
        self.images_path = list(images_path)
        self.masks_path  = list(masks_path)
        self.image_size  = image_size
        self.is_train    = is_train
        self.repeat_factor = max(1, int(repeat_factor)) if is_train else 1
        self.n_samples   = len(self.images_path) * self.repeat_factor

        # Paper augmentation suite
        aug_list = [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.6),
            A.ElasticTransform(p=0.3, alpha=120, sigma=120*0.05, alpha_affine=120*0.03),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3, distort_limit=0.05, shift_limit=0.05),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
        ]
        if is_retinal:
            # Contrast Limited Adaptive Histogram Equalization for micro-vessels
            aug_list.insert(0, A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5))
        else:
            aug_list.append(A.ToGray(p=0.2))

        self.transform = A.Compose(aug_list)

    def _read_image(self, path):
        if path.lower().endswith('.gif'):
            return np.array(Image.open(path).convert('RGB'))
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return np.array(Image.open(path).convert('RGB'))
        if len(img.shape) == 2:
            return np.stack([img]*3, axis=-1)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_mask(self, path):
        if path.lower().endswith('.gif'):
            return np.array(Image.open(path).convert('L'))
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
        uv = np.unique(mask)
        if len(uv) > 2:
            mask = (mask > (mask.max() + mask.min()) / 2.0).astype(np.float32)
        elif mask.max() > 0:
            mask = (mask == mask.max()).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return image, mask

    def __len__(self):
        return self.n_samples


# ================================================================
# EM Volume Dataset (for Electron Microscopy 3D TIFF)
# ================================================================
class EMVolumeDataset(Dataset):
    def __init__(self, volume, labels, indices, image_size, is_train=False, repeat_factor=1):
        self.volume  = volume
        self.labels  = labels
        self.indices = list(indices)
        self.image_size = image_size
        self.is_train = is_train
        self.repeat_factor = max(1, int(repeat_factor)) if is_train else 1
        self.n_samples = len(self.indices) * self.repeat_factor
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=45, p=0.6),
            A.ElasticTransform(p=0.3, alpha=120, sigma=120*0.05, alpha_affine=120*0.03),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3, distort_limit=0.05, shift_limit=0.05),
            A.RandomBrightnessContrast(p=0.3),
        ])

    def __getitem__(self, idx):
        actual_idx = self.indices[idx % len(self.indices)]
        img = self.volume[actual_idx]
        msk = self.labels[actual_idx]

        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)

        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, self.image_size, interpolation=cv2.INTER_NEAREST)

        if self.is_train:
            aug = self.transform(image=img, mask=msk)
            img, msk = aug['image'], aug['mask']

        img = img.astype(np.float32)
        mx = img.max()
        if mx > 0:
            img = img / (65535.0 if mx > 255 else 255.0)
        img = np.transpose(img, (2, 0, 1))

        msk = msk.astype(np.float32)
        if msk.max() > 0:
            msk = (msk / msk.max()).astype(np.float32)
        msk = np.expand_dims(msk, 0)

        return img, msk

    def __len__(self):
        return self.n_samples
"""

CELL_MFAD = r"""# ================================================================
# FANet-MFAD (Proposed Architecture)
# ResNet-34 Encoder + Multi-scale Feedback Attention Decoder
# with Gradient Decoupling (.detach()) Feedback Firewall
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


class FANet_MFAD(nn.Module):
    '''Multi-scale Feedback Attention Decoder with Gradient Decoupling.
    
    Key innovations:
      1. ResNet-34 pre-trained encoder (transfer learning)
      2. 4-channel input (RGB + detached prev_mask) for spatial guidance
      3. Learned gate on prev_mask with .detach() — Feedback Firewall
      4. Multi-scale attention: feat * (1 + att(mask)) at every decoder level
    '''
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Encoder stem: accept 4-ch input (RGB + prev_mask)
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

        # Decoder with skip connections
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d1  = ConvBlock(512, 256)
        self.att1 = nn.Sequential(nn.Conv2d(1, 256, 1),
                                  nn.BatchNorm2d(256), nn.Sigmoid())

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d2  = ConvBlock(256, 128)
        self.att2 = nn.Sequential(nn.Conv2d(1, 128, 1),
                                  nn.BatchNorm2d(128), nn.Sigmoid())

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d3  = ConvBlock(128, 64)
        self.att3 = nn.Sequential(nn.Conv2d(1, 64, 1),
                                  nn.BatchNorm2d(64), nn.Sigmoid())

        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.d4  = ConvBlock(128, 64)
        self.att4 = nn.Sequential(nn.Conv2d(1, 64, 1),
                                  nn.BatchNorm2d(64), nn.Sigmoid())

        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d5  = ConvBlock(32, 32)
        self.att5 = nn.Sequential(nn.Conv2d(1, 32, 1),
                                  nn.BatchNorm2d(32), nn.Sigmoid())

        self.out = nn.Conv2d(32, 1, 1)
        self.learned_gate = nn.Conv2d(1, 1, 1)

    def _mfad(self, feat, prev_mask, att_layer):
        '''Multi-scale Feedback Attention: feat * (1 + att(mask))'''
        m = F.interpolate(prev_mask, size=feat.shape[2:], mode='nearest')
        return feat * (1 + att_layer(m))

    def forward(self, x, prev_mask=None):
        if prev_mask is None:
            prev_mask = torch.zeros(x.size(0), 1, x.size(2), x.size(3),
                                    device=x.device)
        else:
            # GRADIENT DECOUPLING: .detach() severs the Feedback Trap
            prev_mask = torch.sigmoid(self.learned_gate(prev_mask.detach()))

        # Encoder
        s1 = self.e1_relu(self.e1_bn(self.e1_conv(
            torch.cat([x, prev_mask], dim=1))))
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        b  = self.b(s4)

        # Decoder with multi-scale feedback attention
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
"""

CELL_LOSS_METRICS = r"""# ================================================================
# Loss: 0.5*BCE + 0.5*Dice (matching original FANet paper)
# ================================================================
class DiceBCELoss(nn.Module):
    @torch.autocast(device_type='cuda', enabled=False)  # BCE is unsafe under autocast -> fp32
    def forward(self, inputs, targets, smooth=1):
        inputs, targets = inputs.float(), targets.float()
        probs = torch.sigmoid(inputs).view(-1)
        tgt   = targets.view(-1)
        bce   = F.binary_cross_entropy(probs, tgt, reduction='mean')
        inter = (probs * tgt).sum()
        dice  = 1 - (2. * inter + smooth) / (probs.sum() + tgt.sum() + smooth)
        return 0.5 * bce + 0.5 * dice


# ================================================================
# Metrics — per-image, computed on binarised predictions
# ================================================================
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
    sensitivity = tp / (tp + fn + eps)          # = Recall
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
"""

CELL_TRAIN_FN = r"""# ================================================================
# Training function — trains FANet_MFAD with 2-iter feedback training
# ================================================================
def train_model(model, train_loader, valid_loader, model_name,
                epochs=EPOCHS, lr=LR, device=DEVICE):
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    use_amp = (device.type == 'cuda')

    out_prefix = OUT_DIR if 'OUT_DIR' in globals() else '.'
    ckpt_path = os.path.join(out_prefix, f"{model_name}_{DATASET_NAME.replace(' ', '_')}_best.pth")
    latest_path = os.path.join(out_prefix, f"{model_name}_{DATASET_NAME.replace(' ', '_')}_latest.pth")
    
    best_dice = 0.0
    history = {'train_loss': [], 'val_dice': []}
    start_epoch = 0

    # Auto-resume logic
    resume_candidates = [
        latest_path,
        f"/kaggle/input/fanet-benchmark-{DATASET_NAME.lower().replace('_', '-')}/{os.path.basename(latest_path)}",
        f"/kaggle/input/fanet-benchmark-{DATASET_NAME.lower()}/{os.path.basename(latest_path)}"
    ]
    for p in resume_candidates:
        if os.path.exists(p):
            print(f"[*] Found checkpoint: {p}. Resuming...")
            ckpt = torch.load(p, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            best_dice = ckpt['best_dice']
            history = ckpt['history']
            start_epoch = ckpt['epoch']
            break

    if start_epoch >= epochs:
        print(f"[*] {model_name} already completed {epochs} epochs.")
        return ckpt_path, history

    if 'GLOBAL_START' not in globals():
        global GLOBAL_START
        GLOBAL_START = time.time()

    for epoch in range(start_epoch, epochs):
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader,
                         desc=f'[{model_name}] E{epoch+1}/{epochs} Train',
                         leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                out1 = model(x)                       # iter 1 (no feedback)
                out2 = model(x, torch.sigmoid(out1))  # iter 2 (with feedback)
                loss = criterion(out2, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- VALID ----
        model.eval()
        val_dice_sum = 0.0
        n_val = 0
        with torch.no_grad():
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                with torch.amp.autocast('cuda', enabled=use_amp):
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

        print(f"  [{model_name}] Epoch {epoch+1:03d} | "
              f"LR {lr_now:.1e} | Loss {train_loss:.4f} | "
              f"Val Dice {val_dice:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), ckpt_path)
            
        torch.save({
            'epoch': epoch + 1,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'best_dice': best_dice,
            'history': history
        }, latest_path)

        if time.time() - GLOBAL_START > 11.5 * 3600:
            print(f"[!] TIME LIMIT REACHED (11.5h). Exiting gracefully.")
            sys.exit(0)

    print(f"  [{model_name}] Training Complete. Best Val Dice = {best_dice:.4f}")
    return ckpt_path, history
"""

CELL_EVAL_VIS = r"""# ================================================================
# Evaluation: FANet-MFAD with Final Feedback
# ================================================================
def evaluate_model(model, ckpt_path, valid_loader, device=DEVICE):
    '''Load best checkpoint and compute per-image metrics on val set with final feedback.'''
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.eval()

    all_metrics = []
    all_imgs, all_gts, all_preds = [], [], []

    with torch.no_grad():
        for x, y in tqdm(valid_loader, desc='[MFAD] Evaluating Val Set', leave=False):
            x, y = x.to(device), y.to(device)
            out1 = model(x)
            out2 = model(x, torch.sigmoid(out1))  # final feedback
            
            pred = (torch.sigmoid(out2).cpu().numpy() > 0.5).astype(np.float32)
            gt   = y.cpu().numpy()
            imgs = x.cpu().numpy()

            for b_i in range(pred.shape[0]):
                m = compute_metrics(gt[b_i], pred[b_i])
                all_metrics.append(m)
                all_imgs.append(imgs[b_i])
                all_gts.append(gt[b_i])
                all_preds.append(pred[b_i])

    agg = {}
    for k in all_metrics[0]:
        vals = [m[k] for m in all_metrics]
        agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

    return agg, all_metrics, all_imgs, all_gts, all_preds


def print_results(name, agg):
    print(f"\n{'='*60}")
    print(f"  {name} — Final Results on {DATASET_NAME}")
    print(f"{'='*60}")
    for k, v in agg.items():
        print(f"  {k:>15s}: {v['mean']:.4f} ± {v['std']:.4f}")
    print(f"{'='*60}\n")


def save_benchmark_csv(agg, dataset_name):
    out_prefix = OUT_DIR if 'OUT_DIR' in globals() else '.'
    csv_path = os.path.join(out_prefix, f'benchmark_results_{dataset_name.replace(" ", "_")}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Metric', 'Mean', 'Std'])
        for k, v in agg.items():
            writer.writerow(['FANet_MFAD', k, f"{v['mean']:.6f}", f"{v['std']:.6f}"])
    print(f"Results saved to {csv_path}")
    return csv_path


def plot_comparison(images, gts, preds, n_samples=5):
    '''Plot: Input Image | Ground Truth | MFAD Final Prediction.'''
    out_prefix = OUT_DIR if 'OUT_DIR' in globals() else '.'
    n = min(n_samples, len(images))
    fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img = np.transpose(images[i], (1, 2, 0))
        img = np.clip(img, 0, 1)
        gt  = gts[i][0]
        p   = preds[i][0]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image' if i == 0 else '')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].set_title('Ground Truth' if i == 0 else '')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(p, cmap='gray')
        axes[i, 2].set_title('MFAD Prediction' if i == 0 else '')
        axes[i, 2].axis('off')

    plt.suptitle(f'Qualitative Results — {DATASET_NAME}', fontsize=16, y=1.01)
    plt.tight_layout()
    png_path = os.path.join(out_prefix, f'comparison_{DATASET_NAME.replace(" ", "_")}.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved qualitative figure -> {png_path}")


def plot_training_curves(history, dataset_name):
    out_prefix = OUT_DIR if 'OUT_DIR' in globals() else '.'
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history['train_loss'], label='MFAD Train Loss', color='#1f77b4', linewidth=2)
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(history['val_dice'], label='MFAD Val Dice', color='#2ca02c', linewidth=2)
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Val Dice')
    ax2.set_title('Validation Dice Score'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.suptitle(f'MFAD Training Curves — {dataset_name}', fontsize=14)
    plt.tight_layout()
    png_path = os.path.join(out_prefix, f'training_curves_{dataset_name.replace(" ", "_")}.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved training curves -> {png_path}")
"""

CELL_RUN_BENCHMARK = r"""# ================================================================
# RUN MFAD BENCHMARK: Train MFAD (Proposed) + Evaluate Final Feedback
# ================================================================
print(f"\n{'#'*70}")
print(f"#  MFAD TRAINING & BENCHMARK: {DATASET_NAME}")
print(f"#  Image Size: {IMAGE_SIZE} | Epochs: {EPOCHS} | LR: {LR}")
print(f"#  Train Samples (with aug): {len(train_loader.dataset)} | Val: {len(valid_loader.dataset)}")
print(f"{'#'*70}\n")

# Instantiate MFAD model
model_mfad = FANet_MFAD().to(DEVICE)
mfad_params = sum(p.numel() for p in model_mfad.parameters())
print(f"FANet MFAD (Proposed): {mfad_params/1e6:.2f}M parameters\n")

# Train MFAD
print("=" * 60)
print(f"  TRAINING: FANet MFAD (Proposed) on {DATASET_NAME}")
print("=" * 60)
ckpt_mfad, hist_mfad = train_model(model_mfad, train_loader, valid_loader,
                                    "FANet_MFAD", epochs=EPOCHS, lr=LR)

# Final Evaluation with Final Feedback
print("\n" + "=" * 60)
print(f"  FINAL EVALUATION on {DATASET_NAME}")
print("=" * 60)

model_eval = FANet_MFAD().to(DEVICE)
agg, all_metrics, imgs, gts, preds = evaluate_model(model_eval, ckpt_mfad, valid_loader)

print_results("FANet MFAD (Final Feedback)", agg)

# Save results CSV
csv_path = save_benchmark_csv(agg, DATASET_NAME)

# Qualitative Comparison
plot_comparison(imgs, gts, preds, n_samples=5)

# Training Curves
plot_training_curves(hist_mfad, DATASET_NAME)

# Print Summary Table
print(f"\n{'='*70}")
print(f"  BENCHMARK SUMMARY — {DATASET_NAME}")
print(f"{'='*70}")
print(f"  {'Metric':<20s} | {'Mean ± Std':>20s}")
print(f"  {'-'*20}-+-{'-'*20}")
for k in ['dice', 'miou', 'sensitivity', 'specificity', 'fpr', 'precision']:
    v = agg[k]
    print(f"  {k:<20s} | {v['mean']:>8.4f} ± {v['std']:<8.4f}")
print(f"{'='*70}")
"""

# ════════════════════════════════════════════════════════════════
# RAW CONFIG TEMPLATES
# ════════════════════════════════════════════════════════════════

TEMPLATE_DRIVE = r"""# ================================================================
# DRIVE — Retinal vessel segmentation
# Paper Table I: 20 train -> 640 after aug (32x factor!), 20 test
# Resolution: 512x512 | LR: 1e-3 | Aug: CLAHE + elastic/grid + repeat_factor=32
# ================================================================
DATASET_NAME = "DRIVE"
IMAGE_SIZE = (512, 512)
EPOCHS = 150
BATCH_SIZE = 4
LR = 1e-3
REPEAT_FACTOR = 32

DATA_DIR = "__DATA_DIR__"

all_files = []
for ext in ['*.jpg', '*.png', '*.tif', '*.tiff', '*.gif', '*.ppm']:
    all_files.extend(glob(os.path.join(DATA_DIR, '**', ext), recursive=True))

images_all = [f for f in all_files if '/images/' in f.replace(os.sep, '/').lower()]
masks_all  = [f for f in all_files if '/1st_manual/' in f.replace(os.sep, '/').lower()
              or '/manual1/' in f.replace(os.sep, '/').lower()]

train_x = sorted([p for p in images_all if 'train' in p.lower()])
train_y = sorted([p for p in masks_all if 'train' in p.lower()])
valid_x = sorted([p for p in images_all if 'test' in p.lower()])
valid_y = sorted([p for p in masks_all if 'test' in p.lower()])

if len(valid_y) == 0:
    print("No test GT found. Splitting training set 16:4.")
    all_imgs = sorted(images_all)
    all_msks = sorted(masks_all)
    n = min(len(all_imgs), len(all_msks))
    all_imgs, all_msks = all_imgs[:n], all_msks[:n]
    random.seed(42)
    indices = list(range(n))
    random.shuffle(indices)
    split_pt = max(1, int(n * 0.8))
    train_x = [all_imgs[i] for i in indices[:split_pt]]
    train_y = [all_msks[i] for i in indices[:split_pt]]
    valid_x = [all_imgs[i] for i in indices[split_pt:]]
    valid_y = [all_msks[i] for i in indices[split_pt:]]

n_tr = min(len(train_x), len(train_y))
train_x, train_y = train_x[:n_tr], train_y[:n_tr]
n_va = min(len(valid_x), len(valid_y))
valid_x, valid_y = valid_x[:n_va], valid_y[:n_va]

print(f"DRIVE: Raw Train={n_tr}, Val={n_va} | Repeat Factor={REPEAT_FACTOR}")
assert n_tr > 0, "No training images found!"

train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True,
                                 repeat_factor=REPEAT_FACTOR, is_retinal=True)
valid_dataset = UniversalDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False,
                                 repeat_factor=1, is_retinal=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=__WORKERS__, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                          num_workers=__WORKERS__, pin_memory=True)

print(f"DRIVE Dataloader ready: {len(train_dataset)} augmented train samples per epoch")
"""

TEMPLATE_CHASE = r"""# ================================================================
# CHASE-DB1 — Retinal vessel segmentation
# Paper Table I: 20 train -> 640 after aug (32x factor!), 8 test
# Resolution: 512x512 | LR: 1e-3 | Aug: CLAHE + elastic/grid + repeat_factor=32
# ================================================================
DATASET_NAME = "CHASE-DB1"
IMAGE_SIZE = (512, 512)
EPOCHS = 150
BATCH_SIZE = 4
LR = 1e-3
REPEAT_FACTOR = 32

DATA_DIR = "__DATA_DIR__"

N_TRAIN = 20
N_TEST = 8

exts = ['*.jpg', '*.png', '*.tif', '*.tiff', '*.bmp']
all_files = []
for ext in exts:
    all_files.extend(glob(os.path.join(DATA_DIR, '**', ext), recursive=True))

images = sorted([f for f in all_files
                 if '1stho' not in os.path.basename(f).lower()
                 and '2ndho' not in os.path.basename(f).lower()
                 and 'mask' not in os.path.basename(f).lower()])
masks = sorted([f for f in all_files
                if '1stho' in os.path.basename(f).lower()])

if len(images) == 0 or len(masks) == 0:
    for ext in exts:
        images.extend(glob(os.path.join(DATA_DIR, 'images', '**', ext), recursive=True))
        masks.extend(glob(os.path.join(DATA_DIR, 'masks', '**', ext), recursive=True))
    images = sorted(images)
    masks = sorted(masks)

n = min(len(images), len(masks))
images, masks = images[:n], masks[:n]
assert n > 0, f"No images found in {DATA_DIR}!"

random.seed(42)
indices = list(range(n))
random.shuffle(indices)
split = min(N_TRAIN, n - N_TEST)
train_idx = indices[:split]
valid_idx = indices[split:split + N_TEST]

train_x = [images[i] for i in train_idx]
train_y = [masks[i] for i in train_idx]
valid_x = [images[i] for i in valid_idx]
valid_y = [masks[i] for i in valid_idx]

print(f"CHASE-DB1: Raw Train={len(train_x)}, Val={len(valid_x)} | Repeat Factor={REPEAT_FACTOR}")

train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True,
                                 repeat_factor=REPEAT_FACTOR, is_retinal=True)
valid_dataset = UniversalDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False,
                                 repeat_factor=1, is_retinal=True)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=__WORKERS__, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                          num_workers=__WORKERS__, pin_memory=True)

print(f"CHASE-DB1 Dataloader ready: {len(train_dataset)} augmented train samples per epoch")
"""

TEMPLATE_EM = r"""# ================================================================
# EM Dataset — Electron microscopy 3D segmentation
# Paper Table I: 24 slices -> 384 after aug (16x factor!), 6 test
# Resolution: 512x512 | LR: 1e-4 | Aug: repeat_factor=16
# ================================================================
DATASET_NAME = "EM-Dataset"
IMAGE_SIZE = (512, 512)
EPOCHS = 100
BATCH_SIZE = 4
LR = 1e-4
REPEAT_FACTOR = 16

DATA_DIR = "__DATA_DIR__"

train_vol_p = glob(os.path.join(DATA_DIR, "**", "training.tif"), recursive=True)
train_gt_p  = glob(os.path.join(DATA_DIR, "**", "training_groundtruth.tif"), recursive=True)

assert len(train_vol_p) > 0 and len(train_gt_p) > 0, f"EM TIFF volumes not found in {DATA_DIR}!"

train_vol = tiff.imread(train_vol_p[0])
train_gt  = tiff.imread(train_gt_p[0])

print(f"EM volume: {train_vol.shape}, dtype={train_vol.dtype}")
print(f"EM GT:     {train_gt.shape}, unique={np.unique(train_gt)}")

N_TOTAL = train_vol.shape[0]
N_TEST = 6
N_TRAIN = N_TOTAL - N_TEST

random.seed(42)
all_idx = list(range(N_TOTAL))
random.shuffle(all_idx)
train_idx = all_idx[:N_TRAIN]
valid_idx = all_idx[N_TRAIN:]

train_dataset = EMVolumeDataset(train_vol, train_gt, train_idx,
                                IMAGE_SIZE, is_train=True, repeat_factor=REPEAT_FACTOR)
valid_dataset = EMVolumeDataset(train_vol, train_gt, valid_idx,
                                IMAGE_SIZE, is_train=False, repeat_factor=1)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=__WORKERS__, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                          num_workers=__WORKERS__, pin_memory=True)

print(f"EM-Dataset: Train={len(train_dataset)} (augmented), Val={len(valid_dataset)}")
"""

TEMPLATE_CVC = r"""# ================================================================
# CVC-ClinicDB — Polyp segmentation
# Paper Table I: 612 images, Train 490, Test 61
# Resolution: 384x288 (Paper Table I) | LR: 1e-4 | Aug: repeat_factor=2
# ================================================================
DATASET_NAME = "CVC-ClinicDB"
IMAGE_SIZE = (384, 288)
EPOCHS = 150
BATCH_SIZE = 8
LR = 1e-4
REPEAT_FACTOR = 2

DATA_DIR = "__DATA_DIR__"
IMG_SUBDIR = "Original"
MASK_SUBDIR = "Ground Truth"

N_TRAIN = 490
N_TEST = 61

exts = ['*.jpg', '*.png', '*.tif', '*.tiff', '*.jpeg', '*.bmp']
images, masks = [], []
for ext in exts:
    images.extend(glob(os.path.join(DATA_DIR, IMG_SUBDIR, '**', ext), recursive=True))
    masks.extend(glob(os.path.join(DATA_DIR, MASK_SUBDIR, '**', ext), recursive=True))

if len(images) == 0:
    for root, dirs, files in os.walk(DATA_DIR):
        d_lower = [d.lower() for d in dirs]
        if any(x in d_lower for x in ['original', 'images']):
            for ext in exts:
                images.extend(glob(os.path.join(root,
                    dirs[d_lower.index(next(x for x in ['original', 'images'] if x in d_lower))], '**', ext), recursive=True))
        if any(x in d_lower for x in ['ground truth', 'masks']):
            for ext in exts:
                masks.extend(glob(os.path.join(root,
                    dirs[d_lower.index(next(x for x in ['ground truth', 'masks'] if x in d_lower))], '**', ext), recursive=True))

images = sorted(images)
masks = sorted(masks)
assert len(images) > 0, f"No images found in {DATA_DIR}!"

if len(images) != len(masks):
    n = min(len(images), len(masks))
    images, masks = images[:n], masks[:n]

random.seed(42)
combined = list(zip(images, masks))
random.shuffle(combined)
images, masks = zip(*combined)

total = len(images)
if total >= N_TRAIN + N_TEST:
    train_x, train_y = list(images[:N_TRAIN]), list(masks[:N_TRAIN])
    valid_x, valid_y = list(images[-N_TEST:]), list(masks[-N_TEST:])
else:
    split = max(1, int(total * 0.8))
    train_x, train_y = list(images[:split]), list(masks[:split])
    valid_x, valid_y = list(images[split:]), list(masks[split:])

train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True, repeat_factor=REPEAT_FACTOR)
valid_dataset = UniversalDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False, repeat_factor=1)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=__WORKERS__, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                          num_workers=__WORKERS__, pin_memory=True)

print(f"CVC-ClinicDB: Train={len(train_dataset)} (augmented), Val={len(valid_dataset)}")
"""

TEMPLATE_KVASIR = r"""# ================================================================
# Kvasir-SEG — Polyp segmentation
# Paper Table I: 1000 images, Train 880, Test 120
# Resolution: 256x256 | LR: 1e-4 | Aug: dynamic Albumentations
# ================================================================
DATASET_NAME = "Kvasir-SEG"
IMAGE_SIZE = (256, 256)
EPOCHS = 150
BATCH_SIZE = 8
LR = 1e-4
REPEAT_FACTOR = 1

DATA_DIR = "__DATA_DIR__"

N_TRAIN = 880
N_TEST = 120

exts = ['*.jpg', '*.png', '*.jpeg']
images, masks = [], []
for ext in exts:
    images.extend(glob(os.path.join(DATA_DIR, 'images', '**', ext), recursive=True))
    masks.extend(glob(os.path.join(DATA_DIR, 'masks', '**', ext), recursive=True))

if len(images) == 0:
    for root, dirs, files in os.walk(DATA_DIR):
        d_lower = [d.lower() for d in dirs]
        if 'images' in d_lower:
            img_dir = os.path.join(root, dirs[d_lower.index('images')])
            for ext in exts:
                images.extend(glob(os.path.join(img_dir, '**', ext), recursive=True))
        if 'masks' in d_lower:
            msk_dir = os.path.join(root, dirs[d_lower.index('masks')])
            for ext in exts:
                masks.extend(glob(os.path.join(msk_dir, '**', ext), recursive=True))

images = sorted(images)
masks = sorted(masks)
assert len(images) > 0, f"No images found in {DATA_DIR}!"

if len(images) != len(masks):
    n = min(len(images), len(masks))
    images, masks = images[:n], masks[:n]

random.seed(42)
combined = list(zip(images, masks))
random.shuffle(combined)
images, masks = zip(*combined)

total = len(images)
if total >= N_TRAIN + N_TEST:
    train_x = list(images[:N_TRAIN])
    train_y = list(masks[:N_TRAIN])
    valid_x = list(images[N_TRAIN:N_TRAIN+N_TEST])
    valid_y = list(masks[N_TRAIN:N_TRAIN+N_TEST])
else:
    split = max(1, int(total * 0.88))
    train_x, train_y = list(images[:split]), list(masks[:split])
    valid_x, valid_y = list(images[split:]), list(masks[split:])

train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True, repeat_factor=REPEAT_FACTOR)
valid_dataset = UniversalDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False, repeat_factor=1)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=__WORKERS__, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                          num_workers=__WORKERS__, pin_memory=True)

print(f"Kvasir-SEG: Train={len(train_dataset)}, Val={len(valid_dataset)}")
"""

TEMPLATE_DSB = r"""# ================================================================
# Data Science Bowl 2018 (DSB-2018) — Nuclei Segmentation
# Paper Table I: 670 images, ~80:20 split | Resolution: 256x256
# Masks merged dynamically from individual nuclei instances
# ================================================================
import zipfile

DATASET_NAME = "DSB-2018"
IMAGE_SIZE = (256, 256)
EPOCHS = 100
BATCH_SIZE = 16
LR = 1e-4
REPEAT_FACTOR = 2

ZIP_PATH = "__ZIP_PATH__"
if not os.path.exists(ZIP_PATH):
    _zips = glob("/kaggle/input/**/stage1_train.zip", recursive=True) + glob("/content/**/stage1_train.zip", recursive=True)
    if _zips: ZIP_PATH = _zips[0]

ORIG_DIR = "__ORIG_DIR__"
MERGED_MASK_DIR = "__MERGED_DIR__"

if os.path.exists(ZIP_PATH) and not os.path.exists(ORIG_DIR):
    print(f"Extracting {ZIP_PATH}...")
    os.makedirs(ORIG_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(ORIG_DIR)

os.makedirs(MERGED_MASK_DIR, exist_ok=True)

all_imgs, all_msks = [], []
if os.path.exists(ORIG_DIR):
    folder_names = sorted(os.listdir(ORIG_DIR))
    for folder in tqdm(folder_names, desc="Merging Masks"):
        folder_path = os.path.join(ORIG_DIR, folder)
        if not os.path.isdir(folder_path): continue
        img_file = os.path.join(folder_path, "images", folder + ".png")
        mask_files = glob(os.path.join(folder_path, "masks", "*.png"))
        if not os.path.exists(img_file) or len(mask_files) == 0: continue

        merged_path = os.path.join(MERGED_MASK_DIR, folder + ".png")
        if not os.path.exists(merged_path):
            merged = None
            for mf in mask_files:
                m = np.array(Image.open(mf).convert('L'))
                merged = m if merged is None else np.maximum(merged, m)
            Image.fromarray(merged).save(merged_path)
            
        all_imgs.append(img_file)
        all_msks.append(merged_path)

images = all_imgs
masks = all_msks
assert len(images) > 0, "No images found for DSB-2018!"

random.seed(42)
combined = list(zip(images, masks))
random.shuffle(combined)
images, masks = zip(*combined)

total = len(images)
split = max(1, int(total * 0.8))
train_x, train_y = list(images[:split]), list(masks[:split])
valid_x, valid_y = list(images[split:]), list(masks[split:])

train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True, repeat_factor=REPEAT_FACTOR)
valid_dataset = UniversalDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False, repeat_factor=1)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=__WORKERS__, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                          num_workers=__WORKERS__, pin_memory=True)

print(f"DSB-2018: Train={len(train_dataset)} (augmented), Val={len(valid_dataset)}")
"""

TEMPLATE_ISIC = r"""# ================================================================
# ISIC-2018 — Skin lesion segmentation
# Paper Table I: 2596 images, Train 1815, Test 259
# Resolution: 512x512 | LR: 1e-4 | Epochs: 60 (robust completion)
# ================================================================
DATASET_NAME = "ISIC-2018"
IMAGE_SIZE = (512, 512)
EPOCHS = 60
BATCH_SIZE = 8
LR = 1e-4
REPEAT_FACTOR = 1

DATA_DIR = "__DATA_DIR__"

N_TRAIN = 1815
N_TEST = 259

all_imgs = sorted(glob(os.path.join(DATA_DIR, "**", "*.jpg"), recursive=True))
all_msks = sorted(glob(os.path.join(DATA_DIR, "**", "*_segmentation.png"), recursive=True))

def isic_id(path):
    base = os.path.basename(path).replace("_segmentation", "")
    return os.path.splitext(base)[0]

img_dict = {isic_id(p): p for p in all_imgs}
msk_dict = {isic_id(p): p for p in all_msks}
common_ids = sorted(set(img_dict.keys()) & set(msk_dict.keys()))

images = [img_dict[k] for k in common_ids]
masks  = [msk_dict[k] for k in common_ids]

print(f"ISIC-2018: Found {len(images)} matched image-mask pairs")

random.seed(42)
combined = list(zip(images, masks))
random.shuffle(combined)
images, masks = zip(*combined)

total = len(images)
if total >= N_TRAIN + N_TEST:
    train_x, train_y = list(images[:N_TRAIN]), list(masks[:N_TRAIN])
    valid_x, valid_y = list(images[N_TRAIN:N_TRAIN+N_TEST]), list(masks[N_TRAIN:N_TRAIN+N_TEST])
else:
    split = max(1, int(total * 0.8))
    train_x, train_y = list(images[:split]), list(masks[:split])
    valid_x, valid_y = list(images[split:]), list(masks[split:])

train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True, repeat_factor=REPEAT_FACTOR)
valid_dataset = UniversalDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False, repeat_factor=1)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=__WORKERS__, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                          num_workers=__WORKERS__, pin_memory=True)

print(f"ISIC-2018: Train={len(train_dataset)}, Val={len(valid_dataset)}")
"""

COLAB_SETUP_CODE = r"""# ================================================================
# [Colab Setup] Drive Mount, Secrets, GPU, Deps & Data Download
# ================================================================
import os, sys, subprocess, zipfile, glob as _glob
from google.colab import drive, userdata
drive.mount('/content/drive')

WORK_DIR = "__WORK_DIR__"
OUT_DIR = WORK_DIR
DATA_ROOT = "__DATA_ROOT__"
os.makedirs(WORK_DIR, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "albumentations", "tifffile", "opencv-python-headless"], check=True)

import torch
print("CUDA:", torch.cuda.is_available(), "|",
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
torch.backends.cudnn.benchmark = True
NUM_WORKERS = min(8, os.cpu_count() or 2)

os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
import json as _json
try:
    with open(os.path.expanduser('~/.kaggle/kaggle.json'), 'w') as f:
        _json.dump({"username": userdata.get("KAGGLE_USERNAME"), "key": userdata.get("KAGGLE_KEY")}, f)
    os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
    print("✅ Kaggle credentials configured from Colab Secrets.")
except Exception as e:
    print("⚠️ Colab Secrets not found or Kaggle keys missing:", e)

def _unzip_all(root, skip=()):
    for z in _glob.glob(os.path.join(root, "*.zip")):
        if os.path.basename(z) in skip: continue
        print("Extracting", z)
        with zipfile.ZipFile(z) as f:
            f.extractall(root)
        os.remove(z)

DRIVE_DATA = "/content/drive/MyDrive/FANet_data/__SLUG__"
if not os.listdir(DATA_ROOT):
    if os.path.isdir(DRIVE_DATA) and os.listdir(DRIVE_DATA):
        print("Copying data from Google Drive:", DRIVE_DATA)
        subprocess.run(["cp", "-r", DRIVE_DATA + "/.", DATA_ROOT], check=True)
    else:
        print("Downloading via Kaggle API...")
        __DL_CMD__
    _unzip_all(DATA_ROOT, skip=("stage1_train.zip",))
print("Data ready at:", DATA_ROOT, "| entries:", sorted(os.listdir(DATA_ROOT))[:10])
"""

# ════════════════════════════════════════════════════════════════
# CONFIG RESOLVERS
# ════════════════════════════════════════════════════════════════

def resolve_config(ds_key, is_colab):
    workers = "NUM_WORKERS" if is_colab else "2"
    if ds_key == "drive":
        d = "/content/drive_data" if is_colab else "/kaggle/input/datasets/namnguynnnn/drive-vessel"
        return TEMPLATE_DRIVE.replace("__DATA_DIR__", d).replace("__WORKERS__", workers)
    elif ds_key == "chase_db1":
        d = "/content/chasedb1" if is_colab else "/kaggle/input/datasets/namnguynnnn/chase-db1"
        return TEMPLATE_CHASE.replace("__DATA_DIR__", d).replace("__WORKERS__", workers)
    elif ds_key == "em_dataset":
        d = "/content/em-dataset" if is_colab else "/kaggle/input/datasets/kmader/electron-microscopy-3d-segmentation"
        return TEMPLATE_EM.replace("__DATA_DIR__", d).replace("__WORKERS__", workers)
    elif ds_key == "cvc_clinicdb":
        d = "/content/cvc-clinicdb" if is_colab else "/kaggle/input/datasets/namnguynnnn/cvc-clinicdb"
        return TEMPLATE_CVC.replace("__DATA_DIR__", d).replace("__WORKERS__", workers)
    elif ds_key == "kvasir_seg":
        d = "/content/kvasir-seg" if is_colab else "/kaggle/input/datasets/namnguynnnn/kvasir-seg"
        return TEMPLATE_KVASIR.replace("__DATA_DIR__", d).replace("__WORKERS__", workers)
    elif ds_key == "dsb2018":
        zp = "/content/dsb2018/stage1_train.zip" if is_colab else "/kaggle/input/data-science-bowl-2018/stage1_train.zip"
        od = "/content/dsb2018_work/stage1_train" if is_colab else "/kaggle/working/stage1_train"
        md = "/content/dsb2018_work/dsb2018_masks" if is_colab else "/kaggle/working/dsb2018_masks"
        return TEMPLATE_DSB.replace("__ZIP_PATH__", zp).replace("__ORIG_DIR__", od).replace("__MERGED_DIR__", md).replace("__WORKERS__", workers)
    elif ds_key == "isic_2018":
        d = "/content/isic-2018" if is_colab else "/kaggle/input/datasets/tschandl/isic2018-challenge-task1-data-segmentation"
        return TEMPLATE_ISIC.replace("__DATA_DIR__", d).replace("__WORKERS__", workers)
    else:
        raise ValueError(f"Unknown dataset: {ds_key}")


DATASETS = {
    "chase_db1":    {"title": "CHASE-DB1",    "slug": "chasedb1",     "baseline": 0.8108, "kaggle_ds": "namnguynnnn/chase-db1", "is_comp": False},
    "cvc_clinicdb": {"title": "CVC-ClinicDB", "slug": "cvc-clinicdb", "baseline": 0.9355, "kaggle_ds": "namnguynnnn/cvc-clinicdb", "is_comp": False},
    "drive":        {"title": "DRIVE",        "slug": "drive",        "baseline": 0.8183, "kaggle_ds": "namnguynnnn/drive-vessel", "is_comp": False},
    "dsb2018":      {"title": "DSB-2018",     "slug": "dsb2018",      "baseline": 0.9176, "kaggle_ds": "data-science-bowl-2018", "is_comp": True},
    "em_dataset":   {"title": "EM-Dataset",   "slug": "em-dataset",   "baseline": 0.9547, "kaggle_ds": "kmader/electron-microscopy-3d-segmentation", "is_comp": False},
    "isic_2018":    {"title": "ISIC-2018",    "slug": "isic-2018",    "baseline": 0.8731, "kaggle_ds": "tschandl/isic2018-challenge-task1-data-segmentation", "is_comp": False},
    "kvasir_seg":   {"title": "Kvasir-SEG",   "slug": "kvasir-seg",   "baseline": 0.8803, "kaggle_ds": "namnguynnnn/kvasir-seg", "is_comp": False},
}


# ════════════════════════════════════════════════════════════════
# BUILDERS
# ════════════════════════════════════════════════════════════════

def build_kaggle_notebook(ds_key, ds_info):
    nb = nbf.v4.new_notebook()
    title = ds_info['title']
    baseline = ds_info['baseline']

    md_header = f"""# FANet-MFAD Benchmark: {title}
**Proposed Architecture:** ResNet-34 Encoder + Multi-scale Feedback Attention Decoder with Gradient Decoupling (.detach())
**Enhanced Data Augmentation:** Paper Table I specifications (repeat_factor, dynamic Albumentations, CLAHE)

| Property | Value |
|----------|-------|
| Target Model | FANet_MFAD (Proposed Only) |
| Loss Function | 0.5×BCE + 0.5×Dice |
| Paper Baseline Dice | {baseline} |
| Metrics Reported | Dice, mIoU, Sensitivity, Specificity, FPR, Precision (Final Feedback) |
"""
    nb.cells.append(nbf.v4.new_markdown_cell(md_header))
    nb.cells.append(nbf.v4.new_code_cell(CELL_IMPORTS.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_DATASET.strip()))
    nb.cells.append(nbf.v4.new_code_cell(resolve_config(ds_key, is_colab=False).strip()))

    nb.cells.append(nbf.v4.new_markdown_cell("## Model Definition & Loss Functions"))
    nb.cells.append(nbf.v4.new_code_cell(CELL_MFAD.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_LOSS_METRICS.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_TRAIN_FN.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_EVAL_VIS.strip()))

    nb.cells.append(nbf.v4.new_markdown_cell(f"## Execute Training & Benchmark Evaluation: {title}"))
    nb.cells.append(nbf.v4.new_code_cell(CELL_RUN_BENCHMARK.strip()))

    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "file_extension": ".py", "mimetype": "text/x-python"}
    }

    out_path = f"notebooks/benchmark_v2/fanet_vs_mfad_{ds_key}.ipynb"
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"  [Kaggle] Wrote {out_path}")
    return out_path


def build_colab_notebook(ds_key, ds_info):
    nb = nbf.v4.new_notebook()
    title = ds_info['title']
    slug = ds_info['slug']
    is_comp = ds_info['is_comp']
    kaggle_ds = ds_info['kaggle_ds']

    work_dir = f"/content/drive/MyDrive/FANet_runs_MFAD/{slug}"
    data_root = f"/content/{slug}"

    colab_setup_md = f"""# Google Colab (A100 / T4) Runner — FANet-MFAD: {title}
Trained with paper-grade data augmentation and evaluation of Feedback Refinement.

### Instructions:
1. **Runtime -> Change runtime type -> A100 GPU (or T4)**
2. In the left sidebar, click the **Key icon (Secrets)** and add:
   - `KAGGLE_USERNAME`: Your Kaggle username
   - `KAGGLE_KEY`: Your Kaggle API key
   - Enable "Notebook access" toggle.
3. Checkpoints, CSV results, and figures are automatically saved to Google Drive:
   `{work_dir}`
4. **Runtime -> Run all**
"""

    if is_comp:
        dl_cmd = f'subprocess.run(["kaggle", "competitions", "download", "-c", "{kaggle_ds}", "-f", "stage1_train.zip", "-p", DATA_ROOT], check=True)'
    else:
        dl_cmd = f'subprocess.run(["kaggle", "datasets", "download", "-d", "{kaggle_ds}", "-p", DATA_ROOT], check=True)'

    setup_code = COLAB_SETUP_CODE.replace("__WORK_DIR__", work_dir) \
                                 .replace("__DATA_ROOT__", data_root) \
                                 .replace("__SLUG__", slug) \
                                 .replace("__DL_CMD__", dl_cmd)

    nb.cells.append(nbf.v4.new_markdown_cell(colab_setup_md))
    nb.cells.append(nbf.v4.new_code_cell(setup_code.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_IMPORTS.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_DATASET.strip()))
    nb.cells.append(nbf.v4.new_code_cell(resolve_config(ds_key, is_colab=True).strip()))

    nb.cells.append(nbf.v4.new_markdown_cell("## Model Definition & Loss Functions"))
    nb.cells.append(nbf.v4.new_code_cell(CELL_MFAD.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_LOSS_METRICS.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_TRAIN_FN.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_EVAL_VIS.strip()))

    nb.cells.append(nbf.v4.new_markdown_cell(f"## Execute Training & Benchmark Evaluation: {title}"))
    nb.cells.append(nbf.v4.new_code_cell(CELL_RUN_BENCHMARK.strip()))

    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "provenance": []}
    }

    out_path = f"notebooks/colab/fanet_mfad_{ds_key}_colab.ipynb"
    with open(out_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"  [Colab]  Wrote {out_path}")
    return out_path


def main():
    print("=" * 70)
    print("  BUILDING MFAD-ONLY BENCHMARK NOTEBOOKS (Kaggle & Colab)")
    print("  Enhanced Data Augmentation (DRIVE/CHASE 32x, EM 16x)")
    print("=" * 70)

    os.makedirs("notebooks/benchmark_v2", exist_ok=True)
    os.makedirs("notebooks/colab", exist_ok=True)
    os.makedirs("kaggle/benchmark_kernels", exist_ok=True)

    for key, info in DATASETS.items():
        k_path = build_kaggle_notebook(key, info)
        c_path = build_colab_notebook(key, info)

        # Copy to kaggle/benchmark_kernels/<slug>/
        slug = info['slug']
        target_dir = os.path.join("kaggle", "benchmark_kernels", slug)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy(k_path, os.path.join(target_dir, os.path.basename(k_path)))
        print(f"  [Synced] {k_path} -> {target_dir}")

    print("\n[OK] Successfully generated all 7 Kaggle and 7 Colab notebooks!")


if __name__ == '__main__':
    main()
