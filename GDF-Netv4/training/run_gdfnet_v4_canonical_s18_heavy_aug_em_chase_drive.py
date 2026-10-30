"""
========================================================================================
Phase 20: Canonical GDF-Netv4 ConvFormer-S18 with Heavy Data Augmentation on EM-Dataset, CHASE-DB1, and DRIVE
Architecture: 100% Canonical GDFNetV4 (Mask-Guided LGAG + Cascaded MSCAM Decoder)
Key Fixes & Improvements:
  1. EM-Dataset: Corrected split -> 153 Train / 6 Val / 6 Test (restores Phase 8 test protocol!)
  2. CHASE-DB1: High-Resolution (768x768) + Vessel-Specific Thin-Border Loss (kernel=7)
  3. DRIVE: Near-Native Resolution (576x576) + Strict ID-Matching Fix + ISIC-2018 Pretrained Weights
Backbones Evaluated:
  - ConvFormerS18 (`convformer_s18.sail_in22k_ft_in1k_384` — Modern Pure-CNN Flagship)
  - PVTv2B2       (`pvt_v2_b2` — Hierarchical Vision Transformer Flagship)
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
# 1. Canonical GDF-Netv4 Architecture (Mask-Guided LGAG + MSCAM Decoder)
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

    def forward(self, x):
        low1 = self.dw_conv1(x)
        low3 = self.dw_conv3(x)
        low5 = self.dw_conv5(x)
        x_low = self.pw_low(torch.cat([low1, low3, low5], dim=1))
        x_high = x - x_low
        gate = self.hf_gate(x_low)
        x_high_filtered = x_high * gate
        out = self.fusion(torch.cat([x_low, x_high_filtered], dim=1))
        return out + x


class MaskGuidedLGAG(nn.Module):
    def __init__(self, skip_channels, gating_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or skip_channels
        self.skip_channels = skip_channels
        self.gating_channels = gating_channels

        self.g_proj = nn.Sequential(
            nn.Conv2d(gating_channels, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels)
        )

        mid_ch = max(skip_channels // 2, 16)
        self.conv1x1 = nn.Conv2d(skip_channels + 1, mid_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu = nn.ReLU(inplace=True)

        self.dw_dil1 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, dilation=1, groups=mid_ch, bias=False)
        self.dw_dil2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=2, dilation=2, groups=mid_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(mid_ch)

        self.attn_proj = nn.Sequential(
            nn.Conv2d(mid_ch, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.Sigmoid()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, skip, gating, mask_prob=None):
        B, C, H, W = skip.shape
        g_up = F.interpolate(gating, size=(H, W), mode='bilinear', align_corners=False)
        g_proj = self.g_proj(g_up)

        if mask_prob is not None:
            m_up = F.interpolate(mask_prob, size=(H, W), mode='bilinear', align_corners=False)
        else:
            m_up = torch.zeros((B, 1, H, W), device=skip.device, dtype=skip.dtype)

        feat = skip + g_proj
        feat_with_mask = torch.cat([feat, m_up], dim=1)

        x = self.relu(self.bn1(self.conv1x1(feat_with_mask)))
        x = self.relu(self.bn_dw(self.dw_dil1(x) + self.dw_dil2(x)))
        attn = self.attn_proj(x)

        gated_skip = skip * attn
        return self.out_conv(gated_skip)


class MultiScaleConvolutionalAttentionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dw1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
        self.dw3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.dw5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_channels * 3)

        self.pw = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        mid_ch = max(out_channels // 4, 16)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, mid_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.sa = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        d1 = self.dw1(x)
        d3 = self.dw3(x)
        d5 = self.dw5(x)
        dw_cat = F.relu(self.bn_dw(torch.cat([d1, d3, d5], dim=1)), inplace=True)
        feat = self.pw(dw_cat)
        feat = feat * self.ca(feat)
        feat = feat * self.sa(feat)
        return self.refine(feat)


class CascadedMSCAMDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mscam = MultiScaleConvolutionalAttentionModule(in_channels + skip_channels, out_channels)
        self.head = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x, gated_skip):
        x_up = self.upsample(x)
        if x_up.shape[2:] != gated_skip.shape[2:]:
            x_up = F.interpolate(x_up, size=gated_skip.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([x_up, gated_skip], dim=1)
        out_feat = self.mscam(fused)
        pred_logits = self.head(out_feat)
        return out_feat, pred_logits


class CanonicalGDFNetV4(nn.Module):
    def __init__(self, backbone_name="convformer_s18.sail_in22k_ft_in1k_384", pretrained=True, in_chans=3, decoder_channels=64, drop_path_rate=0.2):
        super().__init__()
        self.backbone_name = backbone_name
        self.decoder_channels = decoder_channels

        self.backbone = HierarchicalBackboneV4(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            drop_path_rate=drop_path_rate
        )
        c1, c2, c3, c4 = self.backbone.get_channels()

        self.fdf1 = MultiScaleFrequencyFilter(c1)
        self.fdf2 = MultiScaleFrequencyFilter(c2)
        self.fdf3 = MultiScaleFrequencyFilter(c3)
        self.fdf4 = MultiScaleFrequencyFilter(c4)

        dec4_ch = decoder_channels * 4
        self.dec4_proj = nn.Sequential(
            nn.Conv2d(c4, dec4_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec4_ch),
            nn.ReLU(inplace=True)
        )
        self.head4 = nn.Conv2d(dec4_ch, 1, kernel_size=1)

        dec3_ch = decoder_channels * 3
        dec2_ch = decoder_channels * 2
        dec1_ch = decoder_channels

        self.gate3 = MaskGuidedLGAG(skip_channels=c3, gating_channels=dec4_ch, out_channels=c3)
        self.gate2 = MaskGuidedLGAG(skip_channels=c2, gating_channels=dec3_ch, out_channels=c2)
        self.gate1 = MaskGuidedLGAG(skip_channels=c1, gating_channels=dec2_ch, out_channels=c1)

        self.dec3 = CascadedMSCAMDecoderBlock(in_channels=dec4_ch, skip_channels=c3, out_channels=dec3_ch)
        self.dec2 = CascadedMSCAMDecoderBlock(in_channels=dec3_ch, skip_channels=c2, out_channels=dec2_ch)
        self.dec1 = CascadedMSCAMDecoderBlock(in_channels=dec2_ch, skip_channels=c1, out_channels=dec1_ch)

        self.final_refine = nn.Sequential(
            nn.Conv2d(dec1_ch, dec1_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec1_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(dec1_ch, 1, kernel_size=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        f1, f2, f3, f4 = self.backbone(x)

        s1 = self.fdf1(f1)
        s2 = self.fdf2(f2)
        s3 = self.fdf3(f3)
        s4 = self.fdf4(f4)

        d4 = self.dec4_proj(s4)
        p4 = self.head4(d4)
        m4_prob = torch.sigmoid(p4)

        gated_s3 = self.gate3(s3, d4, m4_prob)
        d3, p3 = self.dec3(d4, gated_s3)
        m3_prob = torch.sigmoid(p3)

        gated_s2 = self.gate2(s2, d3, m3_prob)
        d2, p2 = self.dec2(d3, gated_s2)
        m2_prob = torch.sigmoid(p2)

        gated_s1 = self.gate1(s1, d2, m2_prob)
        d1, p1 = self.dec1(d2, gated_s1)

        p1_up = F.interpolate(p1, size=(H, W), mode='bilinear', align_corners=False)
        p2_up = F.interpolate(p2, size=(H, W), mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=(H, W), mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=(H, W), mode='bilinear', align_corners=False)

        d1_up = F.interpolate(d1, size=(H, W), mode='bilinear', align_corners=False)
        refined_logits = self.final_refine(d1_up)

        out_final = refined_logits + p1_up + p2_up + p3_up + p4_up

        if self.training:
            return {
                'logits': out_final,
                'p1': p1_up,
                'p2': p2_up,
                'p3': p3_up,
                'p4': p4_up
            }
        return out_final


class ModelEMA:
    def __init__(self, model, decay=0.995):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        msd = model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])


# ====================================================================
# 2. Loss Functions: Structure & Soft Dice Loss (Adapted for Thin Structures)
# ====================================================================
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
    def __init__(self, kernel_size=15, weight_factor=5.0, dice_weight=0.6):
        super().__init__()
        self.base_loss = StructureAndMacroDiceLoss(
            kernel_size=kernel_size, weight_factor=weight_factor, dice_weight=dice_weight
        )

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
# 3. Dataset Loaders & Discovery
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

    # Restores Phase 8 test protocol: exact same 6 test slices (indices[n_slices - 6:])
    # Plus 6 clean held-out val slices for threshold calibration
    n_test = 6
    n_val = 6
    n_train = n_slices - n_test - n_val

    train_idx = indices[:n_train]
    val_idx   = indices[n_train:n_train + n_val]
    test_idx  = indices[n_slices - n_test:]

    print(f"[+] EM-Dataset Slices -> Total: {n_slices} | Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}", flush=True)
    return (vol, gt), train_idx, val_idx, test_idx


class EMSliceDatasetV4(Dataset):
    def __init__(self, vol, gt, indices, img_size=512, is_train=False, repeat_factor=2):
        self.vol = vol
        self.gt = gt
        self.indices = list(indices)
        self.img_size = img_size
        self.is_train = is_train
        self.repeat_factor = max(1, repeat_factor) if is_train else 1
        self.n_samples = len(self.indices) * self.repeat_factor

        if is_train:
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.70, 1.0), p=0.6),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(scale=(0.85, 1.15), translate_percent=(-0.08, 0.08), rotate=(-180, 180), p=0.6),
                A.OneOf([
                    A.ElasticTransform(alpha=40, sigma=5, p=1.0),
                    A.GridDistortion(p=1.0),
                ], p=0.35),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.25),
                A.OneOf([
                    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=1.0),
                    A.RandomGamma(gamma_limit=(85, 115), p=1.0),
                ], p=0.4),
                A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(12, 28), hole_width_range=(12, 28), fill=0, fill_mask=0, p=0.25),
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
    def __init__(self, imgs, msks, img_size=576, is_train=False, repeat_factor=10):
        self.imgs = list(imgs)
        self.msks = list(msks)
        self.img_size = img_size
        self.is_train = is_train
        self.repeat_factor = max(1, repeat_factor) if is_train else 1
        self.n_samples = len(self.imgs) * self.repeat_factor

        if is_train:
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.75, 1.0), p=0.5),
                A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.6),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(scale=(0.88, 1.12), translate_percent=(-0.06, 0.06), rotate=(-180, 180), p=0.6),
                A.OneOf([
                    A.ElasticTransform(alpha=30, sigma=5, p=1.0),
                    A.GridDistortion(p=1.0),
                ], p=0.30),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.25),
                A.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.20, hue=0.05, p=0.4),
                A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(10, 24), hole_width_range=(10, 24), fill=0, fill_mask=0, p=0.25),
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
# 4. Evaluation Function (9 Metrics)
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
# 5. Training Engine with ISIC Pretrained Warm-Start
# ====================================================================
def find_pretrained_isic_checkpoint(backbone_tag):
    search_dirs = [
        "/kaggle/input/gdfnetv4-isic-pretrained",
        "/kaggle/input/gdfnet-v4-isic2018-best",
        "/kaggle/input",
        "./archive/scratch/kaggle_upload_checkpoints",
        "./GDF-Netv4/results"
    ]
    patterns = {
        "ConvFormerS18": ["*ConvFormer*ISIC*.pth", "*ConvFormer*300ep*.pth", "*ConvFormerS18*.pth"],
        "PVTv2B2": ["*PVTv2*ISIC*.pth", "*ISIC2018*best*.pth", "*ISIC2018*300ep*.pth"]
    }

    for base in search_dirs:
        if not os.path.exists(base):
            continue
        for pat in patterns.get(backbone_tag, []):
            matches = glob.glob(os.path.join(base, "**", pat), recursive=True)
            if matches:
                return matches[0]
    return None


def run_experiment(dataset_tag, train_loader, val_loader, test_loader, backbone_name, backbone_tag, epochs, device, output_dir, warm_start_mode="isic2018", loss_kernel_size=15, grad_accum=2):
    set_seed(42)
    run_id = f"{dataset_tag}_{backbone_tag}_{warm_start_mode}"
    print("\n" + "=" * 80, flush=True)
    print(f"[*] STARTING GDF-Netv4 TRAINING: {run_id} ({epochs} Epochs)", flush=True)
    print(f"[*] Backbone: {backbone_name} | Warm-Start Mode: {warm_start_mode} | Loss Kernel: {loss_kernel_size}", flush=True)
    print("=" * 80, flush=True)

    model = CanonicalGDFNetV4(backbone_name=backbone_name, pretrained=True, drop_path_rate=0.2).to(device)

    # Load Warm-Start Checkpoint if requested
    loaded_warm_start = False
    if warm_start_mode == "isic2018":
        ckpt_path = find_pretrained_isic_checkpoint(backbone_tag)
        if ckpt_path and os.path.exists(ckpt_path):
            print(f"[*] Found ISIC-2018 Checkpoint: {ckpt_path}", flush=True)
            sd = torch.load(ckpt_path, map_location=device, weights_only=False)
            if "model_state" in sd:
                sd = sd["model_state"]
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"[+] Successfully initialized from ISIC-2018! (Missing: {len(missing)}, Unexpected: {len(unexpected)})", flush=True)
            loaded_warm_start = True
        else:
            print("[!] ISIC-2018 Checkpoint not found, falling back to ImageNet pretraining.", flush=True)
    elif warm_start_mode.endswith(".pth") and os.path.exists(warm_start_mode):
        print(f"[*] Loading Custom Warm-Start Checkpoint: {warm_start_mode}", flush=True)
        sd = torch.load(warm_start_mode, map_location=device, weights_only=False)
        if "model_state" in sd:
            sd = sd["model_state"]
        model.load_state_dict(sd, strict=False)
        loaded_warm_start = True

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Model Initialized | Total Params: {total_params:,} | Channels: {model.backbone.get_channels()}", flush=True)

    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(id(p) for p in backbone_params)
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]

    lr_bb = 5e-5 if loaded_warm_start else 1e-4
    lr_hd = 1.5e-4 if loaded_warm_start else 3e-4

    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': lr_bb},
        {'params': head_params, 'lr': lr_hd}
    ], weight_decay=1e-4)

    ema = ModelEMA(model, decay=0.995)
    criterion = GDFNetV4DiceOptCompositeLoss(kernel_size=loss_kernel_size, weight_factor=4.0, dice_weight=0.6)
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
        'Warm_Start': warm_start_mode if not warm_start_mode.endswith(".pth") else "custom_ckpt",
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

    OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "./results_retrain_em_chase_drive"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Discover Data
    (em_vol, em_gt), em_tr_idx, em_va_idx, em_te_idx = discover_em_paths()
    chase_tr, chase_va, chase_te = discover_chase_paths()
    drive_tr, drive_va, drive_te = discover_drive_paths_fixed()

    num_workers = min(4, os.cpu_count() or 2)
    backbones_cfg = [
        ("convformer_s18.sail_in22k_ft_in1k_384", "ConvFormerS18"),
    ]

    master_records = []

    for bb_name, bb_tag in backbones_cfg:
        print(f"\n>>> Running Experiments for Backbone: {bb_tag} ({bb_name}) <<<\n", flush=True)

        # -------------------------------------------------------------------------
        # A. EM-Dataset (512x512, 153 Train / 6 Val / 6 Test) — 60 Epochs
        # -------------------------------------------------------------------------
        em_tr_loader = DataLoader(EMSliceDatasetV4(em_vol, em_gt, em_tr_idx, img_size=512, is_train=True, repeat_factor=2),
                                  batch_size=4, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        em_va_loader = DataLoader(EMSliceDatasetV4(em_vol, em_gt, em_va_idx, img_size=512, is_train=False),
                                  batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=True)
        em_te_loader = DataLoader(EMSliceDatasetV4(em_vol, em_gt, em_te_idx, img_size=512, is_train=False),
                                  batch_size=4, shuffle=False, num_workers=num_workers, pin_memory=True)

        _, rec_em = run_experiment("EMDataset", em_tr_loader, em_va_loader, em_te_loader,
                                   bb_name, bb_tag, epochs=60, device=device, output_dir=OUTPUT_DIR,
                                   warm_start_mode="isic2018", loss_kernel_size=15, grad_accum=2)
        master_records.append(rec_em)
        pd.DataFrame(master_records).to_csv(os.path.join(OUTPUT_DIR, "master_retrain_summary.csv"), index=False)

        # -------------------------------------------------------------------------
        # B. CHASE-DB1 (High-Res 768x768, Thin Vessel Loss kernel=7) — 60 Epochs
        # -------------------------------------------------------------------------
        ch_tr_loader = DataLoader(RetinalDatasetV4(chase_tr[0], chase_tr[1], img_size=768, is_train=True, repeat_factor=10),
                                  batch_size=2, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        ch_va_loader = DataLoader(RetinalDatasetV4(chase_va[0], chase_va[1], img_size=768, is_train=False),
                                  batch_size=2, shuffle=False, num_workers=num_workers, pin_memory=True)
        ch_te_loader = DataLoader(RetinalDatasetV4(chase_te[0], chase_te[1], img_size=768, is_train=False),
                                  batch_size=2, shuffle=False, num_workers=num_workers, pin_memory=True)

        chase_ckpt, rec_ch = run_experiment("CHASE-DB1", ch_tr_loader, ch_va_loader, ch_te_loader,
                                            bb_name, bb_tag, epochs=60, device=device, output_dir=OUTPUT_DIR,
                                            warm_start_mode="isic2018", loss_kernel_size=7, grad_accum=4)
        master_records.append(rec_ch)
        pd.DataFrame(master_records).to_csv(os.path.join(OUTPUT_DIR, "master_retrain_summary.csv"), index=False)

        # -------------------------------------------------------------------------
        # C. DRIVE (Near-Native 576x576, Direct ISIC-2018 Pretrained Warm-Start) — 50 Epochs
        # -------------------------------------------------------------------------
        dr_tr_loader = DataLoader(RetinalDatasetV4(drive_tr[0], drive_tr[1], img_size=576, is_train=True, repeat_factor=12),
                                  batch_size=4, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        dr_va_loader = DataLoader(RetinalDatasetV4(drive_va[0], drive_va[1], img_size=576, is_train=False),
                                  batch_size=2, shuffle=False, num_workers=num_workers, pin_memory=True)
        dr_te_loader = DataLoader(RetinalDatasetV4(drive_te[0], drive_te[1], img_size=576, is_train=False),
                                  batch_size=2, shuffle=False, num_workers=num_workers, pin_memory=True)

        # Run DRIVE with intra-domain CHASE-DB1 warm-start (retinal-to-retinal)
        _, rec_dr_chase = run_experiment("DRIVE", dr_tr_loader, dr_va_loader, dr_te_loader,
                                         bb_name, bb_tag, epochs=50, device=device, output_dir=OUTPUT_DIR,
                                         warm_start_mode=chase_ckpt, loss_kernel_size=7, grad_accum=2)
        master_records.append(rec_dr_chase)
        pd.DataFrame(master_records).to_csv(os.path.join(OUTPUT_DIR, "master_retrain_summary.csv"), index=False)

        # Run DRIVE with direct ISIC-2018 warm-start
        _, rec_dr_isic = run_experiment("DRIVE", dr_tr_loader, dr_va_loader, dr_te_loader,
                                        bb_name, bb_tag, epochs=50, device=device, output_dir=OUTPUT_DIR,
                                        warm_start_mode="isic2018", loss_kernel_size=7, grad_accum=2)
        master_records.append(rec_dr_isic)
        pd.DataFrame(master_records).to_csv(os.path.join(OUTPUT_DIR, "master_retrain_summary.csv"), index=False)

    print("\n" + "=" * 90, flush=True)
    print("       REVISED RETRAINING MASTER SUMMARY (EM-Dataset, CHASE-DB1, DRIVE)       ", flush=True)
    print("=" * 90, flush=True)
    print(pd.DataFrame(master_records).to_string(index=False), flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    main()
