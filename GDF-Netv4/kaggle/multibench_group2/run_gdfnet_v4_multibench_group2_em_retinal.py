"""
========================================================================================
Phase 17: GDF-Netv4 Unified Multi-Dataset Clinical Benchmark — Group 2 (EM & Retinal)
Datasets:
  1. EM-Dataset (3D Electron Microscopy Neural Membrane, 30 slices: 20 Train / 4 Val / 6 Test, 512x512)
  2. CHASE-DB1  (Retinal Vessel Segmentation, N=28: 20 Train / 4 Val / 4 Test, 512x512)
  3. DRIVE      (Retinal Vessel Segmentation, ID-Matched Fix + Warm-Start Transfer Learning from CHASE-DB1, N=20: 12 Train / 2 Val / 6 Test, 512x512)
Backbones Evaluated (100% Unified GDF-Netv4 Architecture):
  - ConvFormerS18 (`convformer_s18.sail_in22k_ft_in1k_384` — Modern Pure-CNN Flagship)
  - PVTv2B2       (`pvt_v2_b2` — Hierarchical Vision Transformer Flagship)
Evaluation Modes (9 Clinical Metrics + Global Micro Dice/IoU):
  - Single-View (T = 0.50)
  - 4-View Flip TTA (T = 0.50)
  - Validation-Calibrated 4-View TTA (T = T_opt)
========================================================================================
"""

import os
import re
import sys
import glob
import time
import copy
import json
import random
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from scipy.ndimage import distance_transform_edt, binary_erosion

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

try:
    import timm
    import albumentations as A
    import tifffile as tiff
except ImportError:
    subprocess.run(["pip", "install", "-q", "timm", "albumentations", "tifffile", "opencv-python-headless"], check=True)
    import timm
    import albumentations as A
    import tifffile as tiff

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ====================================================================
# 1. Unified GDF-Netv4 Architecture (Pluggable Hierarchical Backbone)
# ====================================================================
class HierarchicalBackboneV4(nn.Module):
    def __init__(self, backbone_name, pretrained=True, in_chans=3, drop_path_rate=0.2):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            drop_path_rate=drop_path_rate
        )
        self.channels = self.backbone.feature_info.channels()

    def get_channels(self):
        return tuple(self.channels)

    def forward(self, x):
        feats = self.backbone(x)
        return feats[0], feats[1], feats[2], feats[3]


class MultiScaleFrequencyFilter(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        mid_channels = max(in_channels // reduction, 16)

        self.dw_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
        self.dw_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.dw_conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False)

        self.pw_low = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.hf_gate = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.gamma = nn.Parameter(torch.ones(1, in_channels, 1, 1) * 0.5)

    def forward(self, x):
        low_1 = self.dw_conv1(x)
        low_3 = self.dw_conv3(x)
        low_5 = self.dw_conv5(x)
        f_low = self.pw_low(torch.cat([low_1, low_3, low_5], dim=1))

        f_high_raw = x - f_low
        gate = self.hf_gate(f_high_raw)
        f_high = f_high_raw * gate

        out = self.fusion(torch.cat([f_low, f_high], dim=1))
        return x + self.gamma * out


class AttentionGate(nn.Module):
    def __init__(self, f_g, f_l, f_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(f_g, f_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(f_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(f_l, f_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(f_int)
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(f_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, g, x):
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode='bilinear', align_corners=False)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        alpha = self.psi(g1 + x1)
        return x * alpha


class FeedbackModulator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_gate = nn.Sequential(
            nn.Conv2d(channels + 1, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )
        self.alpha = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.3)

    def forward(self, feat, coarse_prob):
        if coarse_prob.shape[2:] != feat.shape[2:]:
            coarse_prob = F.interpolate(coarse_prob, size=feat.shape[2:], mode='bilinear', align_corners=False)
        gate = self.conv_gate(torch.cat([feat, coarse_prob], dim=1))
        return feat * (1.0 + self.alpha * gate)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DeepSupervisionDecoder(nn.Module):
    def __init__(self, c1, c2, c3, c4, num_classes=1):
        super().__init__()
        self.ag3 = AttentionGate(f_g=c4, f_l=c3, f_int=c3 // 2)
        self.ag2 = AttentionGate(f_g=c3, f_l=c2, f_int=c2 // 2)
        self.ag1 = AttentionGate(f_g=c2, f_l=c1, f_int=c1 // 2)

        self.dec4 = ConvBlock(c4, c4)
        self.dec3 = ConvBlock(c4 + c3, c3)
        self.dec2 = ConvBlock(c3 + c2, c2)
        self.dec1 = ConvBlock(c2 + c1, c1)

        self.head4 = nn.Conv2d(c4, num_classes, kernel_size=1)
        self.head3 = nn.Conv2d(c3, num_classes, kernel_size=1)
        self.head2 = nn.Conv2d(c2, num_classes, kernel_size=1)
        self.head1 = nn.Conv2d(c1, num_classes, kernel_size=1)

        self.final_refine = nn.Sequential(
            nn.Conv2d(c1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )

    def forward(self, f1, f2, f3, f4, target_size):
        d4 = self.dec4(f4)
        p4 = self.head4(d4)

        d4_up = F.interpolate(d4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        f3_att = self.ag3(g=d4_up, x=f3)
        d3 = self.dec3(torch.cat([d4_up, f3_att], dim=1))
        p3 = self.head3(d3)

        d3_up = F.interpolate(d3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        f2_att = self.ag2(g=d3_up, x=f2)
        d2 = self.dec2(torch.cat([d3_up, f2_att], dim=1))
        p2 = self.head2(d2)

        d2_up = F.interpolate(d2, size=f1.shape[2:], mode='bilinear', align_corners=False)
        f1_att = self.ag1(g=d2_up, x=f1)
        d1 = self.dec1(torch.cat([d2_up, f1_att], dim=1))
        p1 = self.head1(d1)

        d1_full = F.interpolate(d1, size=target_size, mode='bilinear', align_corners=False)
        logits = self.final_refine(d1_full)

        p1_up = F.interpolate(p1, size=target_size, mode='bilinear', align_corners=False)
        p2_up = F.interpolate(p2, size=target_size, mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=target_size, mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=target_size, mode='bilinear', align_corners=False)

        return {'logits': logits, 'p1': p1_up, 'p2': p2_up, 'p3': p3_up, 'p4': p4_up}


class GDFNetV4(nn.Module):
    def __init__(self, backbone_name, pretrained=True, num_classes=1, drop_path_rate=0.2):
        super().__init__()
        self.backbone = HierarchicalBackboneV4(
            backbone_name=backbone_name, pretrained=pretrained, in_chans=3, drop_path_rate=drop_path_rate
        )
        c1, c2, c3, c4 = self.backbone.get_channels()

        self.freq1 = MultiScaleFrequencyFilter(c1)
        self.freq2 = MultiScaleFrequencyFilter(c2)
        self.freq3 = MultiScaleFrequencyFilter(c3)
        self.freq4 = MultiScaleFrequencyFilter(c4)

        self.fb1 = FeedbackModulator(c1)
        self.fb2 = FeedbackModulator(c2)
        self.fb3 = FeedbackModulator(c3)

        self.decoder = DeepSupervisionDecoder(c1, c2, c3, c4, num_classes=num_classes)

    def forward(self, x):
        target_size = x.shape[2:]
        s1, s2, s3, s4 = self.backbone(x)

        f1 = self.freq1(s1)
        f2 = self.freq2(s2)
        f3 = self.freq3(s3)
        f4 = self.freq4(s4)

        coarse_logits = self.decoder.head4(self.decoder.dec4(f4))
        coarse_prob = torch.sigmoid(coarse_logits)

        f1_mod = self.fb1(f1, coarse_prob)
        f2_mod = self.fb2(f2, coarse_prob)
        f3_mod = self.fb3(f3, coarse_prob)

        outputs = self.decoder(f1_mod, f2_mod, f3_mod, f4, target_size=target_size)
        if self.training:
            return outputs
        return outputs['logits']


class WarmupModelEMA:
    def __init__(self, model, decay=0.995):
        self.ema = copy.deepcopy(model).eval()
        self.decay = decay
        self.updates = 0
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        msd = model.state_dict()
        for k, ema_v in self.ema.state_dict().items():
            model_v = msd[k].detach()
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(d).add_(model_v, alpha=1.0 - d)
            else:
                ema_v.copy_(model_v)


class StructureAndMacroDiceLoss(nn.Module):
    def __init__(self, kernel_size=15, weight_factor=5.0, dice_weight=0.6):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight_factor = weight_factor
        self.dice_weight = dice_weight
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, pred_logits, targets):
        with torch.no_grad():
            local_mean = self.avg_pool(targets)
            boundary_diff = torch.abs(targets - local_mean)
            weight = 1.0 + self.weight_factor * boundary_diff

        bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
        wbce = (weight * bce).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + 1e-8)

        probs = torch.sigmoid(pred_logits)
        intersection = (weight * probs * targets).sum(dim=(2, 3))
        union = (weight * (probs + targets - probs * targets)).sum(dim=(2, 3))
        wiou = 1.0 - (intersection + 1.0) / (union + 1.0)

        raw_inter = (probs * targets).sum(dim=(2, 3))
        raw_card = (probs + targets).sum(dim=(2, 3))
        soft_dice_loss = 1.0 - (2.0 * raw_inter + 1.0) / (raw_card + 1.0)

        return (wbce + wiou + self.dice_weight * soft_dice_loss).mean()


class GDFNetV4DiceOptCompositeLoss(nn.Module):
    def __init__(self, dice_weight=0.6):
        super().__init__()
        self.base_loss = StructureAndMacroDiceLoss(dice_weight=dice_weight)

    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            l_final = self.base_loss(outputs['logits'], targets)
            l_p1 = self.base_loss(outputs['p1'], targets)
            l_p2 = self.base_loss(outputs['p2'], targets)
            l_p3 = self.base_loss(outputs['p3'], targets)
            l_p4 = self.base_loss(outputs['p4'], targets)
            return l_final + l_p1 + 0.5 * l_p2 + 0.3 * l_p3 + 0.2 * l_p4
        return self.base_loss(outputs, targets)


# ====================================================================
# 2. Datasets: EM-Dataset (3D TIFF) & Retinal Vessels (CHASE-DB1, DRIVE)
# ====================================================================
def discover_em_paths():
    cands_vol = glob.glob("/kaggle/input/**/training.tif", recursive=True)
    cands_gt  = glob.glob("/kaggle/input/**/training_groundtruth.tif", recursive=True)
    assert len(cands_vol) > 0 and len(cands_gt) > 0, "EM TIFF volumes not found!"
    vol = tiff.imread(cands_vol[0])
    gt  = tiff.imread(cands_gt[0])

    n_slices = vol.shape[0]
    random.seed(42)
    indices = list(range(n_slices))
    random.shuffle(indices)

    # Exact same 6 test slices (indices[24:]) as Phase 8, with 4 held-out val slices from train
    train_idx = indices[:20]
    val_idx   = indices[20:24]
    test_idx  = indices[24:]
    print(f"[+] EM-Dataset Slices -> Total: {n_slices} | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}", flush=True)
    return (vol, gt), train_idx, val_idx, test_idx


class EMSliceDatasetV4(Dataset):
    def __init__(self, vol, gt, indices, img_size=512, is_train=False, repeat_factor=8):
        self.vol = vol
        self.gt = gt
        self.indices = list(indices)
        self.img_size = img_size
        self.is_train = is_train
        self.repeat_factor = max(1, repeat_factor) if is_train else 1
        self.n_samples = len(self.indices) * self.repeat_factor

        if is_train:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(scale=(0.88, 1.12), translate_percent=(-0.08, 0.08), rotate=(-180, 180), p=0.6),
                A.OneOf([
                    A.ElasticTransform(alpha=40, sigma=5, p=1.0),
                    A.GridDistortion(p=1.0),
                ], p=0.35),
                A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        actual_idx = self.indices[index % len(self.indices)]
        img = self.vol[actual_idx].astype(np.uint8)
        msk = (self.gt[actual_idx] > 127).astype(np.float32)

        if len(img.shape) == 2:
            img = np.stack([img] * 3, axis=-1)

        aug = self.transform(image=img, mask=msk)
        img_t = np.transpose(aug['image'].astype(np.float32), (2, 0, 1))
        msk_t = np.expand_dims(aug['mask'].astype(np.float32), axis=0)
        return torch.from_numpy(img_t).float(), torch.from_numpy(msk_t).float()


def discover_chase_paths():
    cands = glob.glob("/kaggle/input/**/Image_*.jpg", recursive=True)
    assert len(cands) > 0, "CHASE-DB1 images not found!"
    data_dir = os.path.dirname(cands[0])

    pairs = []
    all_imgs = sorted(glob.glob(os.path.join(data_dir, "Image_*.jpg")))
    for img_p in all_imgs:
        base = os.path.splitext(os.path.basename(img_p))[0]
        m_cands = glob.glob(os.path.join(os.path.dirname(data_dir), "**", f"{base}_1stHO.png"), recursive=True)
        if not m_cands:
            m_cands = glob.glob(os.path.join(data_dir, f"{base}_1stHO.png"))
        if m_cands:
            pairs.append((img_p, m_cands[0]))

    random.seed(42)
    random.shuffle(pairs)

    train_pairs = pairs[:20]
    val_pairs   = pairs[20:24]
    test_pairs  = pairs[24:]

    train_x, train_y = [p[0] for p in train_pairs], [p[1] for p in train_pairs]
    val_x, val_y     = [p[0] for p in val_pairs], [p[1] for p in val_pairs]
    test_x, test_y   = [p[0] for p in test_pairs], [p[1] for p in test_pairs]

    print(f"[+] CHASE-DB1 Matched Pairs: {len(pairs)} -> Train: {len(train_x)} | Val: {len(val_x)} | Test: {len(test_x)}", flush=True)
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


def discover_drive_paths_fixed():
    """
    Fixes the Phase 8 alphabetical mismatch bug where 01_test.tif was paired with 21_manual1.gif!
    Strictly pairs images and manual annotations by their 2-digit numeric ID (e.g. '21' -> 21_training.tif & 21_manual1.gif).
    """
    all_files = []
    for ext in ["*.jpg", "*.png", "*.tif", "*.tiff", "*.gif"]:
        all_files.extend(glob.glob(os.path.join("/kaggle/input", "**", ext), recursive=True))

    drive_files = [f for f in all_files if "drive" in f.lower()]
    if not drive_files:
        drive_files = all_files

    images_all = [f for f in drive_files if "/images/" in f.replace(os.sep, "/").lower()]
    masks_all  = [f for f in drive_files if "/1st_manual/" in f.replace(os.sep, "/").lower() or "/manual1/" in f.replace(os.sep, "/").lower()]

    img_by_id = {}
    for p in sorted(images_all):
        m = re.match(r'^(\d+)', os.path.basename(p))
        if m:
            img_by_id[int(m.group(1))] = p

    msk_by_id = {}
    for p in sorted(masks_all):
        m = re.match(r'^(\d+)', os.path.basename(p))
        if m:
            msk_by_id[int(m.group(1))] = p

    common_ids = sorted(set(img_by_id.keys()).intersection(set(msk_by_id.keys())))
    pairs = [(img_by_id[cid], msk_by_id[cid]) for cid in common_ids]
    print(f"[+] DRIVE ID-Verified Matched Pairs: {len(pairs)} (IDs: {common_ids})", flush=True)

    random.seed(42)
    random.shuffle(pairs)

    # 20 annotated DRIVE images: 12 Train, 2 Val, 6 Test (exact same N=6 test size as Phase 8)
    n_test = 6
    n_val = 2
    n_train = len(pairs) - n_test - n_val

    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train:n_train + n_val]
    test_pairs  = pairs[n_train + n_val:]

    train_x, train_y = [p[0] for p in train_pairs], [p[1] for p in train_pairs]
    val_x, val_y     = [p[0] for p in val_pairs], [p[1] for p in val_pairs]
    test_x, test_y   = [p[0] for p in test_pairs], [p[1] for p in test_pairs]

    print(f"[+] DRIVE Splits -> Train: {len(train_x)} | Val: {len(val_x)} | Test: {len(test_x)}", flush=True)
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


class RetinalDatasetV4(Dataset):
    def __init__(self, imgs, msks, img_size=512, is_train=False, repeat_factor=10):
        self.imgs = list(imgs)
        self.msks = list(msks)
        self.img_size = img_size
        self.is_train = is_train
        self.repeat_factor = max(1, repeat_factor) if is_train else 1
        self.n_samples = len(self.imgs) * self.repeat_factor

        if is_train:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(scale=(0.88, 1.12), translate_percent=(-0.08, 0.08), rotate=(-180, 180), p=0.6),
                A.OneOf([
                    A.ElasticTransform(alpha=35, sigma=5, p=1.0),
                    A.GridDistortion(p=1.0),
                ], p=0.30),
                A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, p=0.4),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.transform = A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def __len__(self):
        return self.n_samples

    def _read_img(self, path):
        if path.lower().endswith('.gif'):
            return np.array(Image.open(path).convert('RGB'))
        bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return np.array(Image.open(path).convert('RGB'))

    def _read_msk(self, path):
        if path.lower().endswith('.gif'):
            m = np.array(Image.open(path).convert('L'))
            return (m > 127).astype(np.float32)
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is not None:
            return (gray > 127).astype(np.float32)
        m = np.array(Image.open(path).convert('L'))
        return (m > 127).astype(np.float32)

    def __getitem__(self, index):
        actual_idx = index % len(self.imgs)
        img = self._read_img(self.imgs[actual_idx])
        msk = self._read_msk(self.msks[actual_idx])

        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        aug = self.transform(image=img, mask=msk)
        img_t = np.transpose(aug['image'].astype(np.float32), (2, 0, 1))
        msk_t = np.expand_dims(aug['mask'].astype(np.float32), axis=0)
        return torch.from_numpy(img_t).float(), torch.from_numpy(msk_t).float()


# ====================================================================
# 3. 9-Metric Clinical Evaluation & Threshold Calibration
# ====================================================================
def compute_metrics_v4(p, t, compute_surface=True):
    p = p.astype(bool)
    t = t.astype(bool)
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    tn = np.logical_and(~p, ~t).sum()

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    miou = tp / (tp + fp + fn + 1e-8)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    fpr  = fp / (fp + tn + 1e-8)

    hd95, asd, bf2 = 0.0, 0.0, 0.0
    if compute_surface and t.sum() > 0 and p.sum() > 0:
        bg = (t ^ binary_erosion(t)).astype(bool)
        bp = (p ^ binary_erosion(p)).astype(bool)
        if bg.sum() > 0 and bp.sum() > 0:
            dt_g = distance_transform_edt(~bg)
            dt_p = distance_transform_edt(~bp)
            dists = np.concatenate([dt_g[bp], dt_p[bg]])
            hd95 = float(np.percentile(dists, 95))
            asd  = float((dt_g[bp].mean() + dt_p[bg].mean()) / 2.0)
            prec_b = (dt_g[bp] <= 2).sum() / bp.sum()
            rec_b  = (dt_p[bg] <= 2).sum() / bg.sum()
            bf2 = float(2 * prec_b * rec_b / (prec_b + rec_b + 1e-8))

    return {
        'Dice': float(dice), 'mIoU': float(miou), 'Sensitivity': float(sens),
        'Specificity': float(spec), 'Precision': float(prec), 'FPR': float(fpr),
        'HD95': hd95, 'ASD': asd, 'BF2': bf2,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn)
    }


@torch.no_grad()
def evaluate_model_v4(model, loader, device, use_tta=False, prob_thresh=0.5, compute_surface=True):
    model.eval()
    records = []
    tot_tp, tot_fp, tot_fn = 0, 0, 0

    for imgs, msks in loader:
        imgs = imgs.to(device)
        with autocast('cuda'):
            if use_tta:
                p1 = torch.sigmoid(model(imgs))
                p2 = torch.flip(torch.sigmoid(model(torch.flip(imgs, dims=[3]))), dims=[3])
                p3 = torch.flip(torch.sigmoid(model(torch.flip(imgs, dims=[2]))), dims=[2])
                p4 = torch.flip(torch.sigmoid(model(torch.flip(imgs, dims=[2, 3]))), dims=[2, 3])
                probs = (p1 + p2 + p3 + p4) / 4.0
            else:
                probs = torch.sigmoid(model(imgs))

        preds = (probs > prob_thresh).cpu().numpy()[:, 0]
        msks_np = (msks > 0.5).numpy()[:, 0]

        for i in range(len(preds)):
            m = compute_metrics_v4(preds[i], msks_np[i], compute_surface=compute_surface)
            tot_tp += m['tp']
            tot_fp += m['fp']
            tot_fn += m['fn']
            records.append(m)

    df = pd.DataFrame(records)
    global_dice = float((2.0 * tot_tp) / (2.0 * tot_tp + tot_fp + tot_fn + 1e-8))
    global_iou  = float(tot_tp / (tot_tp + tot_fp + tot_fn + 1e-8))

    summary = []
    for col in ['Dice', 'mIoU', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'HD95', 'ASD', 'BF2']:
        summary.append({
            'Metric': col,
            'Mean': float(df[col].mean()),
            'Std': float(df[col].std()),
            'Median': float(df[col].median()),
            'Min': float(df[col].min()),
            'Max': float(df[col].max())
        })
    summary.append({'Metric': 'Global_Micro_Dice', 'Mean': global_dice, 'Std': 0.0, 'Median': global_dice, 'Min': global_dice, 'Max': global_dice})
    summary.append({'Metric': 'Global_Micro_IoU', 'Mean': global_iou, 'Std': 0.0, 'Median': global_iou, 'Min': global_iou, 'Max': global_iou})

    summary_df = pd.DataFrame(summary)
    return df, summary_df, global_dice, global_iou


@torch.no_grad()
def calibrate_threshold_on_val(model, val_loader, device):
    model.eval()
    all_probs, all_msks = [], []
    for imgs, msks in val_loader:
        imgs = imgs.to(device)
        with autocast('cuda'):
            p1 = torch.sigmoid(model(imgs))
            p2 = torch.flip(torch.sigmoid(model(torch.flip(imgs, dims=[3]))), dims=[3])
            p3 = torch.flip(torch.sigmoid(model(torch.flip(imgs, dims=[2]))), dims=[2])
            p4 = torch.flip(torch.sigmoid(model(torch.flip(imgs, dims=[2, 3]))), dims=[2, 3])
            probs = (p1 + p2 + p3 + p4) / 4.0
        all_probs.append(probs.cpu().numpy()[:, 0])
        all_msks.append((msks > 0.5).numpy()[:, 0])

    probs_np = np.concatenate(all_probs, axis=0)
    msks_np = np.concatenate(all_msks, axis=0)

    best_t, best_dice = 0.50, -1.0
    for t in np.linspace(0.25, 0.75, 51):
        preds = probs_np > t
        tp = np.logical_and(preds, msks_np).sum(axis=(1, 2))
        fp = np.logical_and(preds, ~msks_np).sum(axis=(1, 2))
        fn = np.logical_and(~preds, msks_np).sum(axis=(1, 2))
        d = ((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)).mean()
        if d > best_dice:
            best_dice = float(d)
            best_t = float(t)
    return best_t, best_dice


# ====================================================================
# 4. Experiment Runner (Supports Warm-Start Transfer Learning)
# ====================================================================
def run_experiment(dataset_tag, train_loader, val_loader, test_loader, backbone_name, backbone_tag, epochs, device, output_dir, warm_start_ckpt=None, grad_accum=2):
    set_seed(42)
    run_id = f"{dataset_tag}_{backbone_tag}"
    print("\n" + "=" * 80, flush=True)
    print(f"[*] STARTING GDF-Netv4 BENCHMARK: {run_id} ({epochs} Epochs)", flush=True)
    print(f"[*] Backbone: {backbone_name} | Warm-Start: {warm_start_ckpt}", flush=True)
    print("=" * 80, flush=True)

    model = GDFNetV4(backbone_name=backbone_name, pretrained=True, drop_path_rate=0.2).to(device)
    if warm_start_ckpt and os.path.exists(warm_start_ckpt):
        model.load_state_dict(torch.load(warm_start_ckpt, map_location=device, weights_only=False))
        print(f"[+] Successfully loaded Warm-Start Transfer Checkpoint: {warm_start_ckpt}", flush=True)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Model Initialized | Total Params: {total_params:,} | Channels: {model.backbone.get_channels()}", flush=True)

    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(id(p) for p in backbone_params)
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    lr_bb = 5e-5 if warm_start_ckpt else 1e-4
    lr_hd = 1.5e-4 if warm_start_ckpt else 3e-4

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr_bb},
        {'params': head_params, 'lr': lr_hd}
    ], weight_decay=1e-4)

    ema = WarmupModelEMA(model, decay=0.995)
    criterion = GDFNetV4DiceOptCompositeLoss(dice_weight=0.6)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    scaler = GradScaler('cuda')

    checkpoint_best = os.path.join(output_dir, f"GDFNetV4_{run_id}_best.pth")
    history_csv = os.path.join(output_dir, f"training_history_{run_id}.csv")

    best_val_dice = 0.0
    best_epoch = 0
    best_source = "init"
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, (imgs, msks) in enumerate(train_loader):
            imgs, msks = imgs.to(device, non_blocking=True), msks.to(device, non_blocking=True)
            with autocast('cuda'):
                outs = model(imgs)
                loss = criterion(outs, msks) / grad_accum

            scaler.scale(loss).backward()
            running_loss += loss.item() * grad_accum

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                ema.update(model)

        scheduler.step(epoch)
        elapsed = time.time() - t0
        epoch_loss = running_loss / max(len(train_loader), 1)
        current_lr = optimizer.param_groups[-1]['lr']

        _, val_sum_raw, val_glob_raw, _ = evaluate_model_v4(model, val_loader, device, use_tta=False, compute_surface=False)
        val_dice_raw = float(val_sum_raw.loc[val_sum_raw['Metric'] == 'Dice', 'Mean'].values[0])

        _, val_sum_ema, val_glob_ema, _ = evaluate_model_v4(ema.ema, val_loader, device, use_tta=False, compute_surface=False)
        val_dice_ema = float(val_sum_ema.loc[val_sum_ema['Metric'] == 'Dice', 'Mean'].values[0])

        epoch_best_dice = max(val_dice_raw, val_dice_ema)
        epoch_best_src = "EMA" if val_dice_ema >= val_dice_raw else "RAW"

        saved_str = ""
        if epoch_best_dice > best_val_dice:
            best_val_dice = epoch_best_dice
            best_epoch = epoch
            best_source = epoch_best_src
            best_state = ema.ema.state_dict() if epoch_best_src == "EMA" else model.state_dict()
            torch.save(best_state, checkpoint_best)
            saved_str = f" -> [*] Best ({best_source}: {best_val_dice:.4f})"

        history.append({
            'epoch': epoch,
            'loss': epoch_loss,
            'lr': current_lr,
            'val_macro_dice_raw': val_dice_raw,
            'val_macro_dice_ema': val_dice_ema,
            'best_val_dice': best_val_dice,
            'elapsed_sec': elapsed
        })
        if epoch % 5 == 0 or epoch == epochs:
            pd.DataFrame(history).to_csv(history_csv, index=False)

        print(
            f"[{run_id}] Ep {epoch:02d}/{epochs} [{elapsed:.1f}s] | Loss: {epoch_loss:.4f} | "
            f"Val Raw: {val_dice_raw:.4f} | Val EMA: {val_dice_ema:.4f}{saved_str}",
            flush=True
        )

    pd.DataFrame(history).to_csv(history_csv, index=False)
    print(f"\n[+] Training Finished for {run_id}! Best Val Macro Dice: {best_val_dice:.4f} @ Ep {best_epoch} ({best_source})", flush=True)

    if os.path.exists(checkpoint_best):
        model.load_state_dict(torch.load(checkpoint_best, map_location=device, weights_only=False))

    # 1. Single-View Evaluation
    sv_df, sv_sum, sv_gdice, sv_giou = evaluate_model_v4(model, test_loader, device, use_tta=False, prob_thresh=0.50, compute_surface=True)
    sv_sum.to_csv(os.path.join(output_dir, f"benchmark_summary_{run_id}_SingleView.csv"), index=False)
    sv_df.to_csv(os.path.join(output_dir, f"benchmark_raw_{run_id}_SingleView.csv"), index=False)
    print(f"\n--- {run_id} TEST RESULTS (SINGLE-VIEW, T=0.50) ---", flush=True)
    print(sv_sum.to_string(index=False), flush=True)

    # 2. 4-View Flip TTA Evaluation
    tta_df, tta_sum, tta_gdice, tta_giou = evaluate_model_v4(model, test_loader, device, use_tta=True, prob_thresh=0.50, compute_surface=True)
    tta_sum.to_csv(os.path.join(output_dir, f"benchmark_summary_{run_id}_TTA.csv"), index=False)
    tta_df.to_csv(os.path.join(output_dir, f"benchmark_raw_{run_id}_TTA.csv"), index=False)
    print(f"\n--- {run_id} TEST RESULTS (4-VIEW TTA, T=0.50) ---", flush=True)
    print(tta_sum.to_string(index=False), flush=True)

    # 3. Validation-Calibrated 4-View TTA Evaluation
    opt_t, val_cal_dice = calibrate_threshold_on_val(model, val_loader, device)
    print(f"[*] {run_id} Calibrated Threshold on Val: T_opt = {opt_t:.3f} (Val TTA Macro Dice: {val_cal_dice:.4f})", flush=True)
    cal_df, cal_sum, cal_gdice, cal_giou = evaluate_model_v4(model, test_loader, device, use_tta=True, prob_thresh=opt_t, compute_surface=True)
    cal_sum['T_opt'] = opt_t
    cal_sum.to_csv(os.path.join(output_dir, f"benchmark_summary_{run_id}_TTA_Calibrated.csv"), index=False)
    cal_df.to_csv(os.path.join(output_dir, f"benchmark_raw_{run_id}_TTA_Calibrated.csv"), index=False)
    print(f"\n--- {run_id} TEST RESULTS (CALIBRATED 4-VIEW TTA, T_opt={opt_t:.3f}) ---", flush=True)
    print(cal_sum.to_string(index=False), flush=True)

    del model, ema, optimizer, scaler
    torch.cuda.empty_cache()

    return checkpoint_best, {
        'Dataset': dataset_tag,
        'Backbone': backbone_tag,
        'Best_Val_Dice': best_val_dice,
        'Best_Epoch': best_epoch,
        'T_opt': opt_t,
        'SV_Mean_Dice': float(sv_sum.loc[sv_sum['Metric'] == 'Dice', 'Mean'].values[0]),
        'SV_Median_Dice': float(sv_sum.loc[sv_sum['Metric'] == 'Dice', 'Median'].values[0]),
        'TTA_Mean_Dice': float(tta_sum.loc[tta_sum['Metric'] == 'Dice', 'Mean'].values[0]),
        'TTA_Median_Dice': float(tta_sum.loc[tta_sum['Metric'] == 'Dice', 'Median'].values[0]),
        'Cal_Mean_Dice': float(cal_sum.loc[cal_sum['Metric'] == 'Dice', 'Mean'].values[0]),
        'Cal_Median_Dice': float(cal_sum.loc[cal_sum['Metric'] == 'Dice', 'Median'].values[0]),
        'Cal_mIoU': float(cal_sum.loc[cal_sum['Metric'] == 'mIoU', 'Mean'].values[0]),
        'Cal_Sens': float(cal_sum.loc[cal_sum['Metric'] == 'Sensitivity', 'Mean'].values[0]),
        'Cal_Spec': float(cal_sum.loc[cal_sum['Metric'] == 'Specificity', 'Mean'].values[0]),
        'Cal_Prec': float(cal_sum.loc[cal_sum['Metric'] == 'Precision', 'Mean'].values[0]),
        'Cal_FPR': float(cal_sum.loc[cal_sum['Metric'] == 'FPR', 'Mean'].values[0]),
        'Cal_HD95': float(cal_sum.loc[cal_sum['Metric'] == 'HD95', 'Mean'].values[0]),
        'Cal_ASD': float(cal_sum.loc[cal_sum['Metric'] == 'ASD', 'Mean'].values[0]),
        'Cal_BF2': float(cal_sum.loc[cal_sum['Metric'] == 'BF2', 'Mean'].values[0]),
        'Cal_Global_Dice': cal_gdice
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"[+] GPU: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "./results_multibench_group2"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Discover Group 2 Datasets
    (em_vol, em_gt), em_tr_idx, em_va_idx, em_te_idx = discover_em_paths()
    chase_tr, chase_va, chase_te = discover_chase_paths()
    drive_tr, drive_va, drive_te = discover_drive_paths_fixed()

    num_workers = min(4, os.cpu_count() or 2)
    backbones_cfg = [
        ("convformer_s18.sail_in22k_ft_in1k_384", "ConvFormerS18"),
        ("pvt_v2_b2", "PVTv2B2"),
    ]

    master_records = []

    for bb_name, bb_tag in backbones_cfg:
        # --- A. EM-Dataset (512x512, 60 Epochs) ---
        em_tr_loader = DataLoader(EMSliceDatasetV4(em_vol, em_gt, em_tr_idx, img_size=512, is_train=True, repeat_factor=8),
                                  batch_size=4, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        em_va_loader = DataLoader(EMSliceDatasetV4(em_vol, em_gt, em_va_idx, img_size=512, is_train=False),
                                  batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=True)
        em_te_loader = DataLoader(EMSliceDatasetV4(em_vol, em_gt, em_te_idx, img_size=512, is_train=False),
                                  batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=True)

        _, rec_em = run_experiment("EMDataset", em_tr_loader, em_va_loader, em_te_loader, bb_name, bb_tag, 60, device, OUTPUT_DIR, warm_start_ckpt=None, grad_accum=2)
        master_records.append(rec_em)
        pd.DataFrame(master_records).to_csv(os.path.join(OUTPUT_DIR, "master_benchmark_group2.csv"), index=False)

        # --- B. CHASE-DB1 (512x512, 60 Epochs) ---
        ch_tr_loader = DataLoader(RetinalDatasetV4(chase_tr[0], chase_tr[1], img_size=512, is_train=True, repeat_factor=10),
                                  batch_size=4, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        ch_va_loader = DataLoader(RetinalDatasetV4(chase_va[0], chase_va[1], img_size=512, is_train=False),
                                  batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=True)
        ch_te_loader = DataLoader(RetinalDatasetV4(chase_te[0], chase_te[1], img_size=512, is_train=False),
                                  batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=True)

        chase_ckpt, rec_ch = run_experiment("CHASE-DB1", ch_tr_loader, ch_va_loader, ch_te_loader, bb_name, bb_tag, 60, device, OUTPUT_DIR, warm_start_ckpt=None, grad_accum=2)
        master_records.append(rec_ch)
        pd.DataFrame(master_records).to_csv(os.path.join(OUTPUT_DIR, "master_benchmark_group2.csv"), index=False)

        # --- C. DRIVE (ID-Matched Fix + Warm-Start Transfer Learning from CHASE-DB1, 50 Epochs) ---
        dr_tr_loader = DataLoader(RetinalDatasetV4(drive_tr[0], drive_tr[1], img_size=512, is_train=True, repeat_factor=12),
                                  batch_size=4, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        dr_va_loader = DataLoader(RetinalDatasetV4(drive_va[0], drive_va[1], img_size=512, is_train=False),
                                  batch_size=2, shuffle=False, num_workers=num_workers, pin_memory=True)
        dr_te_loader = DataLoader(RetinalDatasetV4(drive_te[0], drive_te[1], img_size=512, is_train=False),
                                  batch_size=2, shuffle=False, num_workers=num_workers, pin_memory=True)

        _, rec_dr = run_experiment("DRIVE", dr_tr_loader, dr_va_loader, dr_te_loader, bb_name, bb_tag, 50, device, OUTPUT_DIR, warm_start_ckpt=chase_ckpt, grad_accum=2)
        master_records.append(rec_dr)
        pd.DataFrame(master_records).to_csv(os.path.join(OUTPUT_DIR, "master_benchmark_group2.csv"), index=False)

    print("\n" + "=" * 90, flush=True)
    print("       GROUP 2 MASTER BENCHMARK SUMMARY (EM-Dataset, CHASE-DB1, DRIVE)       ", flush=True)
    print("=" * 90, flush=True)
    print(pd.DataFrame(master_records).to_string(index=False), flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
