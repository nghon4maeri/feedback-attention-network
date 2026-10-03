"""
generate_benchmark_v2.py
========================
Generates 7 Kaggle-ready benchmark notebooks for the T-MI submission.

Each notebook:
  1. Defines both FANet_Original and FANet_MFAD inline
  2. Trains both on the SAME data split with the SAME loss/optimizer
  3. Evaluates: Dice, mIoU, Sensitivity, Specificity, FPR
  4. Outputs benchmark_results.csv + qualitative comparison figure
  5. Uses dataset-specific configs (split ratios, image sizes, LR)
     matching the original FANet paper exactly.

Usage:
    python generate_benchmark_v2.py
"""

import nbformat as nbf
import os

# ════════════════════════════════════════════════════════════════
# SHARED CODE CELLS (used across all 7 notebooks)
# ════════════════════════════════════════════════════════════════

CELL_IMPORTS = r"""
import os, sys, random, time, csv, json
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

# ────────────────────────────────────────────────────────────────
# Universal Dataset (handles all formats)
# ────────────────────────────────────────────────────────────────
CELL_DATASET = r"""
# ================================================================
# UniversalDataset — handles jpg/png/tif/gif, auto-binarises masks
# ================================================================
class UniversalDataset(Dataset):
    def __init__(self, images_path, masks_path, image_size, is_train=False):
        self.images_path = list(images_path)
        self.masks_path  = list(masks_path)
        self.image_size  = image_size
        self.is_train    = is_train
        self.n_samples   = len(self.images_path)

        # Paper augmentation suite
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,
                               rotate_limit=45, p=0.6),
            A.ElasticTransform(p=0.3, alpha=120, sigma=120*0.05,
                               alpha_affine=120*0.03),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3, distort_limit=0.05, shift_limit=0.05),
            A.ToGray(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.2),
        ])

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
        image = self._read_image(self.images_path[index])
        mask  = self._read_mask(self.masks_path[index])

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
    def __init__(self, volume, labels, indices, image_size, is_train=False):
        self.volume  = volume
        self.labels  = labels
        self.indices = indices
        self.image_size = image_size
        self.is_train = is_train
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,
                               rotate_limit=45, p=0.5),
            A.ElasticTransform(p=0.3, alpha=120, sigma=120*0.05,
                               alpha_affine=120*0.03),
            A.GridDistortion(p=0.2),
        ])

    def __getitem__(self, idx):
        i = self.indices[idx]
        img = self.volume[i]
        msk = self.labels[i]

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
        return len(self.indices)
"""

# ────────────────────────────────────────────────────────────────
# FANet Original (exact reproduction from src/fanet/models/)
# ────────────────────────────────────────────────────────────────
CELL_FANET_ORIGINAL = r"""
# ================================================================
# FANet Original — Exact reproduction from the paper's codebase
# Custom Encoder/Decoder + SE + MixPool with hard binary gating
# Input: [image, mask] as a list
# ================================================================
class SELayer(nn.Module):
    #Squeeze-and-Excitation channel attention.#
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    #3x3->3x3 Residual block with SE attention.#
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.se = SELayer(out_c, out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.bn2(self.conv2(x1))
        x3 = self.se(self.bn3(self.conv3(x)))
        return self.relu(x2 + x3)


class MixPool_Original(nn.Module):
    #MixPool with HARD BINARY gating (original FANet behaviour).#
    def __init__(self, in_c, out_c):
        super().__init__()
        self.fmask = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c // 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, m):
        fmask = self.fmask(x)
        stride_h = m.shape[2] // x.shape[2]
        stride_w = m.shape[3] // x.shape[3]
        m = nn.MaxPool2d((stride_h, stride_w))(m)
        m_fg = m[:, 0:1]

        # HARD BINARY: gradient killed by (> 0.5)
        fmask_g = (fmask > 0.5).float()
        keep = torch.maximum(fmask_g, m_fg)

        x1 = self.conv1(x * keep)
        x2 = self.conv2(x)
        return torch.cat([x1, x2], dim=1)


class EncoderBlock_Orig(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.r1 = ResidualBlock(in_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)
        self.p1 = MixPool_Original(out_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs, masks):
        x = self.r1(inputs)
        x = self.r2(x)
        p = self.p1(x, masks)
        o = self.pool(p)
        return o, x


class DecoderBlock_Orig(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_c, in_c, kernel_size=4,
                                            stride=2, padding=1)
        self.r1 = ResidualBlock(in_c + in_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)
        self.p1 = MixPool_Original(out_c, out_c)

    def forward(self, inputs, skip, masks):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.r1(x)
        x = self.r2(x)
        return self.p1(x, masks)


class FANet_Original(nn.Module):
    #FANet with hard binary MixPool gating — exact paper reproduction.#
    def __init__(self):
        super().__init__()
        self.e1 = EncoderBlock_Orig(3, 32)
        self.e2 = EncoderBlock_Orig(32, 64)
        self.e3 = EncoderBlock_Orig(64, 128)
        self.e4 = EncoderBlock_Orig(128, 256)
        self.d1 = DecoderBlock_Orig(256, 128)
        self.d2 = DecoderBlock_Orig(128, 64)
        self.d3 = DecoderBlock_Orig(64, 32)
        self.d4 = DecoderBlock_Orig(32, 16)
        self.output = nn.Conv2d(16 + 1, 1, kernel_size=1, padding=0)

    def forward(self, x, m_fg=None):
        if m_fg is None:
            m_fg = torch.zeros(x.size(0), 1, x.size(2), x.size(3),
                               device=x.device)
        masks = m_fg
        p1, s1 = self.e1(x, masks)
        p2, s2 = self.e2(p1, masks)
        p3, s3 = self.e3(p2, masks)
        p4, s4 = self.e4(p3, masks)
        d1 = self.d1(p4, s4, masks)
        d2 = self.d2(d1, s3, masks)
        d3 = self.d3(d2, s2, masks)
        d4 = self.d4(d3, s1, masks)
        d5 = torch.cat([d4, masks[:, 0:1]], dim=1)
        return self.output(d5)
"""

# ────────────────────────────────────────────────────────────────
# MFAD Architecture (ResNet-34 Encoder + Multi-scale Feedback
# Attention Decoder with Gradient Decoupling)
# ────────────────────────────────────────────────────────────────
CELL_MFAD = r"""
# ================================================================
# FANet-MFAD (Proposed) — ResNet-34 + Multi-scale Feedback Attention
# Decoder with Gradient Decoupling (.detach())
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
    #Multi-scale Feedback Attention Decoder with Gradient Decoupling.
    
    Key innovations:
      1. ResNet-34 pre-trained encoder (transfer learning)
      2. 4-channel input (RGB + detached prev_mask) for spatial guidance
      3. Learned gate on prev_mask with .detach() — Feedback Firewall
      4. Multi-scale attention: feat * (1 + att(mask)) at every decoder level
    #
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
        #Multi-scale Feedback Attention: feat * (1 + att(mask))#
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

# ────────────────────────────────────────────────────────────────
# Loss & Metrics
# ────────────────────────────────────────────────────────────────
CELL_LOSS_METRICS = r"""
# ================================================================
# Loss: 0.5*BCE + 0.5*Dice (matches original FANet paper exactly)
# ================================================================
class DiceBCELoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
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
    #Compute Dice, mIoU, Sensitivity, Specificity, FPR.
    
    Args:
        y_true_np: binary GT mask [H, W] or [1, H, W]
        y_pred_np: binary pred mask [H, W] or [1, H, W]
    Returns:
        dict with keys: dice, miou, sensitivity, specificity, fpr, precision
    #
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

# ────────────────────────────────────────────────────────────────
# Training Loop (shared between both models)
# ────────────────────────────────────────────────────────────────
CELL_TRAIN_FN = r"""
# ================================================================
# Training function — works for both FANet_Original and FANet_MFAD
# ================================================================
def train_model(model, train_loader, valid_loader, model_name,
                epochs=EPOCHS, lr=LR, device=DEVICE):
    #Train a model and return best checkpoint path + training history.#
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10, min_lr=1e-6)

    # Mixed precision for speed on T4/P100
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    use_amp = (device.type == 'cuda')

    ckpt_path = f"{model_name}_{DATASET_NAME.replace(' ', '_')}_best.pth"
    best_dice = 0.0
    history = {'train_loss': [], 'val_dice': []}

    for epoch in range(epochs):
        # ---- TRAIN ----
        model.train()
        train_loss = 0.0
        for x, y in tqdm(train_loader,
                         desc=f'[{model_name}] E{epoch+1}/{epochs} Train',
                         leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(x)                       # iter 1 (no feedback)
                out = model(x, torch.sigmoid(out))   # iter 2 (with feedback)
                loss = criterion(out, y)
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
                    out = model(x)
                    out = model(x, torch.sigmoid(out))
                pred = (torch.sigmoid(out).cpu().numpy() > 0.5).astype(np.float32)
                gt   = y.cpu().numpy()
                # Per-image dice
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

    print(f"  [{model_name}] Best Val Dice = {best_dice:.4f}")
    return ckpt_path, history
"""

# ────────────────────────────────────────────────────────────────
# Evaluation & Visualisation
# ────────────────────────────────────────────────────────────────
CELL_EVAL_VIS = r"""
# ================================================================
# Full Evaluation — loads best checkpoint, computes all metrics
# ================================================================
def evaluate_model(model, ckpt_path, valid_loader, model_name, device=DEVICE):
    #Load best checkpoint and compute per-image metrics on val set.#
    model.load_state_dict(torch.load(ckpt_path, map_location=device,
                                     weights_only=True))
    model.eval()

    all_metrics = []
    all_preds   = []
    all_gts     = []
    all_images  = []

    with torch.no_grad():
        for x, y in tqdm(valid_loader,
                         desc=f'[{model_name}] Evaluating', leave=False):
            x, y = x.to(device), y.to(device)
            out = model(x)
            out = model(x, torch.sigmoid(out))  # 2-iter feedback
            pred = (torch.sigmoid(out).cpu().numpy() > 0.5).astype(np.float32)
            gt   = y.cpu().numpy()
            imgs = x.cpu().numpy()

            for b_i in range(pred.shape[0]):
                m = compute_metrics(gt[b_i], pred[b_i])
                all_metrics.append(m)
                all_preds.append(pred[b_i])
                all_gts.append(gt[b_i])
                all_images.append(imgs[b_i])

    # Aggregate
    agg = {}
    for key in all_metrics[0]:
        vals = [m[key] for m in all_metrics]
        agg[key] = {'mean': np.mean(vals), 'std': np.std(vals)}

    return agg, all_metrics, all_images, all_gts, all_preds


def print_results(name, agg):
    #Pretty-print aggregated results.#
    print(f"\n{'='*60}")
    print(f"  {name} — Final Results on {DATASET_NAME}")
    print(f"{'='*60}")
    for k, v in agg.items():
        print(f"  {k:>15s}: {v['mean']:.4f} ± {v['std']:.4f}")
    print(f"{'='*60}\n")


# ================================================================
# Qualitative Comparison (side-by-side visualisation)
# ================================================================
def plot_comparison(images_orig, gts_orig, preds_orig,
                    images_mfad, gts_mfad, preds_mfad,
                    n_samples=5):
    #Plot: Image | GT | FANet Mask | MFAD Mask.#
    n = min(n_samples, len(images_orig), len(images_mfad))
    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for i in range(n):
        img = np.transpose(images_orig[i], (1, 2, 0))
        img = np.clip(img, 0, 1)
        gt  = gts_orig[i][0]
        p_orig = preds_orig[i][0]
        p_mfad = preds_mfad[i][0]

        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Input Image' if i == 0 else '')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(gt, cmap='gray')
        axes[i, 1].set_title('Ground Truth' if i == 0 else '')
        axes[i, 1].axis('off')

        axes[i, 2].imshow(p_orig, cmap='gray')
        axes[i, 2].set_title('FANet Original' if i == 0 else '')
        axes[i, 2].axis('off')

        axes[i, 3].imshow(p_mfad, cmap='gray')
        axes[i, 3].set_title('MFAD (Ours)' if i == 0 else '')
        axes[i, 3].axis('off')

    plt.suptitle(f'Qualitative Comparison — {DATASET_NAME}', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(f'comparison_{DATASET_NAME.replace(" ", "_")}.png',
                dpi=150, bbox_inches='tight')
    plt.show()


# ================================================================
# Save results to CSV
# ================================================================
def save_benchmark_csv(agg_orig, agg_mfad, dataset_name):
    #Save benchmark results to CSV.#
    csv_path = f'benchmark_results_{dataset_name.replace(" ", "_")}.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Model', 'Metric', 'Mean', 'Std'])
        for model_name, agg in [('FANet_Original', agg_orig),
                                  ('FANet_MFAD', agg_mfad)]:
            for k, v in agg.items():
                writer.writerow([model_name, k,
                                f"{v['mean']:.6f}", f"{v['std']:.6f}"])
    print(f"Results saved to {csv_path}")
    return csv_path
"""

# ────────────────────────────────────────────────────────────────
# Main Benchmark Runner
# ────────────────────────────────────────────────────────────────
CELL_RUN_BENCHMARK = r"""
# ================================================================
# RUN BENCHMARK: Train + Evaluate + Compare + Save
# ================================================================
print(f"\n{'#'*70}")
print(f"#  BENCHMARK: {DATASET_NAME}")
print(f"#  Image Size: {IMAGE_SIZE} | Epochs: {EPOCHS} | LR: {LR}")
print(f"#  Train: {len(train_loader.dataset)} | Val: {len(valid_loader.dataset)}")
print(f"{'#'*70}\n")

# ---- 1. Instantiate both models ----
model_orig = FANet_Original().to(DEVICE)
model_mfad = FANet_MFAD().to(DEVICE)

orig_params = sum(p.numel() for p in model_orig.parameters())
mfad_params = sum(p.numel() for p in model_mfad.parameters())
print(f"FANet Original: {orig_params/1e6:.2f}M params")
print(f"FANet MFAD:     {mfad_params/1e6:.2f}M params\n")

# ---- 2. Train FANet Original ----
print("=" * 60)
print("  TRAINING: FANet Original")
print("=" * 60)
ckpt_orig, hist_orig = train_model(model_orig, train_loader, valid_loader,
                                    "FANet_Orig", epochs=EPOCHS, lr=LR)

# ---- 3. Train FANet MFAD ----
print("\n" + "=" * 60)
print("  TRAINING: FANet MFAD (Proposed)")
print("=" * 60)
ckpt_mfad, hist_mfad = train_model(model_mfad, train_loader, valid_loader,
                                    "FANet_MFAD", epochs=EPOCHS, lr=LR)

# ---- 4. Evaluate both models ----
print("\n" + "=" * 60)
print("  EVALUATION")
print("=" * 60)

# Re-create fresh models for loading
model_orig_eval = FANet_Original().to(DEVICE)
model_mfad_eval = FANet_MFAD().to(DEVICE)

agg_orig, metrics_orig, imgs_orig, gts_orig, preds_orig = \
    evaluate_model(model_orig_eval, ckpt_orig, valid_loader, "FANet_Orig")
agg_mfad, metrics_mfad, imgs_mfad, gts_mfad, preds_mfad = \
    evaluate_model(model_mfad_eval, ckpt_mfad, valid_loader, "FANet_MFAD")

print_results("FANet Original", agg_orig)
print_results("FANet MFAD (Proposed)", agg_mfad)

# ---- 5. Save CSV ----
csv_path = save_benchmark_csv(agg_orig, agg_mfad, DATASET_NAME)

# ---- 6. Qualitative comparison ----
plot_comparison(imgs_orig, gts_orig, preds_orig,
                imgs_mfad, gts_mfad, preds_mfad, n_samples=5)

# ---- 7. Training curves ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(hist_orig['train_loss'], label='FANet Original', alpha=0.8)
ax1.plot(hist_mfad['train_loss'], label='MFAD (Ours)', alpha=0.8)
ax1.set_xlabel('Epoch'); ax1.set_ylabel('Train Loss')
ax1.set_title('Training Loss'); ax1.legend(); ax1.grid(True, alpha=0.3)

ax2.plot(hist_orig['val_dice'], label='FANet Original', alpha=0.8)
ax2.plot(hist_mfad['val_dice'], label='MFAD (Ours)', alpha=0.8)
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Val Dice')
ax2.set_title('Validation Dice Score'); ax2.legend(); ax2.grid(True, alpha=0.3)

plt.suptitle(f'Training Curves — {DATASET_NAME}', fontsize=14)
plt.tight_layout()
plt.savefig(f'training_curves_{DATASET_NAME.replace(" ", "_")}.png',
            dpi=150, bbox_inches='tight')
plt.show()

# ---- 8. Summary ----
print(f"\n{'='*70}")
print(f"  BENCHMARK SUMMARY — {DATASET_NAME}")
print(f"{'='*70}")
print(f"  {'Metric':<15s} | {'FANet Orig':>12s} | {'MFAD (Ours)':>12s} | {'Delta':>10s}")
print(f"  {'-'*15}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
for k in ['dice', 'miou', 'sensitivity', 'specificity', 'fpr', 'precision']:
    vo = agg_orig[k]['mean']
    vm = agg_mfad[k]['mean']
    delta = vm - vo
    sign = '+' if delta >= 0 else ''
    print(f"  {k:<15s} | {vo:>12.4f} | {vm:>12.4f} | {sign}{delta:>9.4f}")
print(f"{'='*70}")
"""


# ════════════════════════════════════════════════════════════════
# PER-DATASET CONFIG CELLS
# ════════════════════════════════════════════════════════════════

def config_cvc_clinicdb():
    return r"""
# ================================================================
# CVC-ClinicDB — Polyp segmentation
# Paper: 612 images, Train 490, Test 61 (remaining 61 val)
# Image size: 384x288 (paper's original resolution)
# ================================================================
DATASET_NAME = "CVC-ClinicDB"
IMAGE_SIZE = (384, 288)
EPOCHS = 200
BATCH_SIZE = 8
LR = 1e-4

DATA_DIR = "/kaggle/input/cvc-clinicdb"
IMG_SUBDIR = "Original"
MASK_SUBDIR = "Ground Truth"

N_TRAIN = 490
N_TEST = 61

# Locate files
exts = ['*.jpg', '*.png', '*.tif', '*.tiff', '*.jpeg', '*.bmp']
images, masks = [], []
for ext in exts:
    images.extend(glob(os.path.join(DATA_DIR, IMG_SUBDIR, '**', ext), recursive=True))
    masks.extend(glob(os.path.join(DATA_DIR, MASK_SUBDIR, '**', ext), recursive=True))

# Fallback auto-search
if len(images) == 0:
    for root, dirs, files in os.walk(DATA_DIR):
        d_lower = [d.lower() for d in dirs]
        if any(x in d_lower for x in ['original', 'images']):
            for ext in exts:
                images.extend(glob(os.path.join(root,
                    dirs[d_lower.index(next(x for x in ['original', 'images']
                         if x in d_lower))], '**', ext), recursive=True))
        if any(x in d_lower for x in ['ground truth', 'masks']):
            for ext in exts:
                masks.extend(glob(os.path.join(root,
                    dirs[d_lower.index(next(x for x in ['ground truth', 'masks']
                         if x in d_lower))], '**', ext), recursive=True))

images = sorted(images)
masks = sorted(masks)
assert len(images) > 0, f"No images found in {DATA_DIR}!"

if len(images) != len(masks):
    n = min(len(images), len(masks))
    images, masks = images[:n], masks[:n]

# Shuffle + split
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

train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True)
valid_dataset = UniversalDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"CVC-ClinicDB: Train={len(train_dataset)}, Val={len(valid_dataset)}")
"""


def config_isic2018():
    return r"""
# ================================================================
# ISIC-2018 — Skin lesion segmentation
# Paper: 2594 images, Train 1815, Test 259, remaining val
# Image size: 512x512
# ================================================================
DATASET_NAME = "ISIC-2018"
IMAGE_SIZE = (512, 512)
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-4

DATA_DIR = "/kaggle/input/isic2018-challenge-task1-data-segmentation"

N_TRAIN = 1815
N_TEST = 259

# ISIC: images are .jpg, masks are .png with "_segmentation" suffix
all_imgs = sorted(glob(os.path.join(DATA_DIR, "**", "*.jpg"), recursive=True))
all_msks = sorted(glob(os.path.join(DATA_DIR, "**", "*_segmentation.png"), recursive=True))

# Match by base ID
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

train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True)
valid_dataset = UniversalDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"ISIC-2018: Train={len(train_dataset)}, Val={len(valid_dataset)}")
"""


def config_drive():
    return r"""
# ================================================================
# DRIVE — Retinal vessel segmentation
# Paper: 20 training, 20 test (predefined sets)
# Image size: 512x512 (higher res to preserve thin vessels)
# LR: 1e-3 (paper: "learning rate adjusted to 1e-3 due to small size")
# ================================================================
DATASET_NAME = "DRIVE"
IMAGE_SIZE = (512, 512)
EPOCHS = 200
BATCH_SIZE = 2
LR = 1e-3

DATA_DIR = "/kaggle/input/drive-digital-retinal-images-for-vessel-extraction"

# DRIVE standard structure:
#   training/images/ + training/1st_manual/ (GT)
#   test/images/     + test/1st_manual/ or test/2nd_manual/

# Find all image/mask files
all_files = []
for ext in ['*.jpg', '*.png', '*.tif', '*.tiff', '*.gif', '*.ppm']:
    all_files.extend(glob(os.path.join(DATA_DIR, '**', ext), recursive=True))

# Filter: images vs ground truth
images_all = [f for f in all_files if '/images/' in f.replace(os.sep, '/').lower()]
masks_all  = [f for f in all_files if '/1st_manual/' in f.replace(os.sep, '/').lower()
              or '/manual1/' in f.replace(os.sep, '/').lower()]

# Split by train/test directory
train_x = sorted([p for p in images_all if 'train' in p.lower()])
train_y = sorted([p for p in masks_all if 'train' in p.lower()])
valid_x = sorted([p for p in images_all if 'test' in p.lower()])
valid_y = sorted([p for p in masks_all if 'test' in p.lower()])

# Fallback: if test masks not found, split training set 16:4
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

# Align counts
n_tr = min(len(train_x), len(train_y))
train_x, train_y = train_x[:n_tr], train_y[:n_tr]
n_va = min(len(valid_x), len(valid_y))
valid_x, valid_y = valid_x[:n_va], valid_y[:n_va]

print(f"DRIVE: Train={n_tr}, Val={n_va}")
assert n_tr > 0, "No training images found!"

train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True)
valid_dataset = UniversalDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                          num_workers=2, pin_memory=True)
"""


def config_chase_db1():
    return r"""
# ================================================================
# CHASE-DB1 — Retinal vessel segmentation
# Paper: 28 images total, Train 20, Test 8
# Image size: 512x512
# LR: 1e-3 (paper: "learning rate adjusted to 1e-3 due to small size")
# ================================================================
DATASET_NAME = "CHASE-DB1"
IMAGE_SIZE = (512, 512)
EPOCHS = 200
BATCH_SIZE = 2
LR = 1e-3

DATA_DIR = "/kaggle/input/chase-db1"

N_TRAIN = 20
N_TEST = 8

# CHASE-DB1: images + 1st_manual annotations
exts = ['*.jpg', '*.png', '*.tif', '*.tiff', '*.bmp']
all_files = []
for ext in exts:
    all_files.extend(glob(os.path.join(DATA_DIR, '**', ext), recursive=True))

# Separate images from masks (masks typically have '1stHO' in name)
images = sorted([f for f in all_files
                 if '1stho' not in os.path.basename(f).lower()
                 and '2ndho' not in os.path.basename(f).lower()
                 and 'mask' not in os.path.basename(f).lower()])
masks = sorted([f for f in all_files
                if '1stho' in os.path.basename(f).lower()])

# Fallback: try images/ and masks/ subdirectories
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

print(f"CHASE-DB1: Train={len(train_x)}, Val={len(valid_x)}")

train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True)
valid_dataset = UniversalDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                          num_workers=2, pin_memory=True)
"""


def config_em_dataset():
    return r"""
# ================================================================
# EM Dataset — Electron microscopy 3D segmentation
# 30 slices from 3D TIFF volumes. Paper: Train 24, Test 6
# Image size: 512x512
# ================================================================
DATASET_NAME = "EM-Dataset"
IMAGE_SIZE = (512, 512)
EPOCHS = 100
BATCH_SIZE = 2
LR = 1e-4

DATA_DIR = "/kaggle/input/electron-microscopy-3d-segmentation"

train_vol = tiff.imread(os.path.join(DATA_DIR, "training.tif"))
train_gt  = tiff.imread(os.path.join(DATA_DIR, "training_groundtruth.tif"))

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
                                IMAGE_SIZE, is_train=True)
valid_dataset = EMVolumeDataset(train_vol, train_gt, valid_idx,
                                IMAGE_SIZE, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"EM-Dataset: Train={len(train_dataset)}, Val={len(valid_dataset)}")
"""


def config_kvasir_seg():
    return r"""
# ================================================================
# Kvasir-SEG — Polyp segmentation (full dataset)
# Paper: 1000 images, Train 880, Test 120
# Image size: 256x256
# ================================================================
DATASET_NAME = "Kvasir-SEG"
IMAGE_SIZE = (256, 256)
EPOCHS = 200
BATCH_SIZE = 8
LR = 1e-4

DATA_DIR = "/kaggle/input/kvasir-seg"

N_TRAIN = 880
N_TEST = 120

# Find images and masks
exts = ['*.jpg', '*.png', '*.jpeg']
images, masks = [], []
for ext in exts:
    images.extend(glob(os.path.join(DATA_DIR, 'images', '**', ext), recursive=True))
    masks.extend(glob(os.path.join(DATA_DIR, 'masks', '**', ext), recursive=True))

# Fallback: search deeper
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

train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True)
valid_dataset = UniversalDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Kvasir-SEG: Train={len(train_dataset)}, Val={len(valid_dataset)}")
"""


def config_kvasir_sessile():
    return r"""
# ================================================================
# Kvasir-Sessile — Sessile polyp subset (harder, flat polyps)
# ~196 images, 80:20 split (no paper-defined split)
# Image size: 256x256
# ================================================================
DATASET_NAME = "Kvasir-Sessile"
IMAGE_SIZE = (256, 256)
EPOCHS = 200
BATCH_SIZE = 8
LR = 1e-4

DATA_DIR = "/kaggle/input/sessile-main-kvasir-seg"

# Find images and masks
exts = ['*.jpg', '*.png', '*.jpeg']
images, masks = [], []

# Try common directory structures
for subdir_img in ['images', 'sessile-main-Kvasir-SEG/images', 'Images']:
    for ext in exts:
        images.extend(glob(os.path.join(DATA_DIR, subdir_img, '**', ext),
                          recursive=True))
for subdir_msk in ['masks', 'sessile-main-Kvasir-SEG/masks', 'Masks']:
    for ext in exts:
        masks.extend(glob(os.path.join(DATA_DIR, subdir_msk, '**', ext),
                         recursive=True))

# Deep fallback
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

images = sorted(set(images))
masks = sorted(set(masks))
assert len(images) > 0, f"No images found in {DATA_DIR}!"

if len(images) != len(masks):
    n = min(len(images), len(masks))
    images, masks = images[:n], masks[:n]

random.seed(42)
combined = list(zip(images, masks))
random.shuffle(combined)
images, masks = zip(*combined)

total = len(images)
split = max(1, int(total * 0.8))
train_x, train_y = list(images[:split]), list(masks[:split])
valid_x, valid_y = list(images[split:]), list(masks[split:])

train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True)
valid_dataset = UniversalDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                          num_workers=2, pin_memory=True)

print(f"Kvasir-Sessile: Train={len(train_dataset)}, Val={len(valid_dataset)}")
"""


# ════════════════════════════════════════════════════════════════
# NOTEBOOK BUILDER
# ════════════════════════════════════════════════════════════════

DATASETS = {
    "cvc_clinicdb":    {"title": "CVC-ClinicDB",    "config_fn": config_cvc_clinicdb},
    "isic_2018":       {"title": "ISIC-2018",        "config_fn": config_isic2018},
    "drive":           {"title": "DRIVE",             "config_fn": config_drive},
    "chase_db1":       {"title": "CHASE-DB1",         "config_fn": config_chase_db1},
    "em_dataset":      {"title": "EM-Dataset",        "config_fn": config_em_dataset},
    "kvasir_seg":      {"title": "Kvasir-SEG",        "config_fn": config_kvasir_seg},
    "kvasir_sessile":  {"title": "Kvasir-Sessile",    "config_fn": config_kvasir_sessile},
}

BASELINE_SCORES = {
    "CVC-ClinicDB":   0.9355,
    "ISIC-2018":      0.8731,
    "DRIVE":          0.8183,
    "CHASE-DB1":      0.8108,
    "EM-Dataset":     0.9547,
    "Kvasir-SEG":     0.8803,
    "Kvasir-Sessile": 0.3428,  # sessile subset (Dice on hard polyps)
}


def build_notebook(ds_key, ds_info):
    nb = nbf.v4.new_notebook()
    title = ds_info['title']
    baseline = BASELINE_SCORES.get(title, "N/A")

    # Header markdown
    nb.cells.append(nbf.v4.new_markdown_cell(
        f"# FANet vs MFAD Benchmark: {title}\n\n"
        f"**Purpose:** Head-to-head comparison for IEEE T-MI submission\n\n"
        f"| Property | Value |\n"
        f"|----------|-------|\n"
        f"| FANet Original | Custom Encoder + SE + MixPool (hard binary gating) |\n"
        f"| MFAD (Proposed) | ResNet-34 + Multi-scale Feedback Attention Decoder + Gradient Decoupling |\n"
        f"| Loss | 0.5×BCE + 0.5×Dice |\n"
        f"| Baseline Dice (Paper) | {baseline} |\n"
        f"| Metrics | Dice, mIoU, Sensitivity, Specificity, FPR, Precision |\n"
    ))

    # Code cells
    nb.cells.append(nbf.v4.new_code_cell(CELL_IMPORTS.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_DATASET.strip()))
    nb.cells.append(nbf.v4.new_code_cell(ds_info['config_fn']().strip()))

    nb.cells.append(nbf.v4.new_markdown_cell(
        "## Model Definitions\n"
        "Both architectures are defined inline for reproducibility."
    ))
    nb.cells.append(nbf.v4.new_code_cell(CELL_FANET_ORIGINAL.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_MFAD.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_LOSS_METRICS.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_TRAIN_FN.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_EVAL_VIS.strip()))

    nb.cells.append(nbf.v4.new_markdown_cell(
        f"## Run Benchmark: {title}\n"
        f"Training both models on the same data split, then evaluating."
    ))
    nb.cells.append(nbf.v4.new_code_cell(CELL_RUN_BENCHMARK.strip()))

    # Save
    os.makedirs('notebooks/benchmark_v2', exist_ok=True)
    path = f'notebooks/benchmark_v2/fanet_vs_mfad_{ds_key}.ipynb'
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"  ✅ {path}")
    return path


if __name__ == '__main__':
    print("=" * 60)
    print("  Generating 7 Benchmark Notebooks (FANet vs MFAD)")
    print("  Target: IEEE T-MI Submission")
    print("=" * 60)
    print()
    for key, info in DATASETS.items():
        build_notebook(key, info)
    print()
    print("Done! All notebooks saved to notebooks/benchmark_v2/")
    print("Upload each .ipynb to Kaggle with the corresponding dataset attached.")
