"""
GDF-Netv4 Training & Official ISIC-2018 Blind Test Benchmark Script
Author: Deep Learning Research Team
Framework: PyTorch + timm + albumentations

Features:
- Hierarchical Transformer Backbone: PVTv2-B2 (or ConvNeXt-V2-Tiny)
- Multi-Scale Frequency-Decoupled Filter (MS-FDF with kernels [1, 3, 5])
- Mask-Guided Large-Kernel Grouped Attention Gates (LGAG)
- Cascaded Multi-Scale Convolutional Attention Decoders (MSCAM)
- StructureLoss (Weighted BCE + Weighted IoU) with Deep Supervision
- ImageNet standard normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
- CosineAnnealingLR with Warmup
- Robust Resumption: Seamlessly resumes epoch, model, optimizer, scheduler, scaler
- Google Drive Mirroring: Automatic checkpoint synchronization
- 4-View Flip Test-Time Augmentation (TTA)
- Full 9-Metric Protocol (Both Per-Image Macro and Dataset-Level Global Micro)
"""

import os
import sys
import glob
import time
import json
import zipfile
import shutil
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt, binary_erosion
from tqdm import tqdm

import timm
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

# ====================================================================
# 1. Output & Device Configuration
# ====================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[+] Active Compute Device: {device}", flush=True)
if torch.cuda.is_available():
    print(f"[+] GPU Hardware: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)", flush=True)

OUTPUT_DIR = "/content/results_gdfnet_v4" if os.path.exists("/content") else "./results_gdfnet_v4"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DRIVE_DIR = "/content/drive/MyDrive/GDFNetV4_ISIC2018"
if os.path.exists("/content/drive/MyDrive"):
    os.makedirs(DRIVE_DIR, exist_ok=True)
    print(f"[+] Google Drive Connected: Syncing to {DRIVE_DIR}", flush=True)

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ====================================================================
# 2. Dataset Acquisition & Pairing
# ====================================================================
DATA_ROOT = "/content/isic-2018" if os.path.exists("/content") else "./isic-2018"
os.makedirs(DATA_ROOT, exist_ok=True)

kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
k_user = os.environ.get("KAGGLE_USERNAME", "namnguynnnn")
k_key  = os.environ.get("KAGGLE_KEY", "KGAT_dad33ce94fb8b25d8a9ecd4ca26ce03f")
with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
    json.dump({"username": k_user, "key": k_key}, f)
os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

def _unzip_all(root):
    for z in glob.glob(os.path.join(root, "*.zip")):
        print(f"[*] Extracting: {os.path.basename(z)}...", flush=True)
        try:
            with zipfile.ZipFile(z) as f:
                f.extractall(root)
            os.remove(z)
        except Exception as e:
            print(f"[!] Warning while unzipping {z}: {e}", flush=True)

has_train = any("ISIC2018_Task1-2_Training_Input" in root for root, dirs, _ in os.walk(DATA_ROOT))
has_test  = any("ISIC2018_Task1_Test_GroundTruth" in root for root, dirs, _ in os.walk(DATA_ROOT))

if not (has_train and has_test):
    print("[*] Downloading ISIC-2018 Datasets via Kaggle API...", flush=True)
    subprocess.run(["kaggle", "datasets", "download", "-d",
                    "tschandl/isic2018-challenge-task1-data-segmentation", "-p", DATA_ROOT], check=True)
    subprocess.run(["kaggle", "datasets", "download", "-d",
                    "mfahimhossen/isic2018-test-groundtruth", "-p", DATA_ROOT], check=True)
    _unzip_all(DATA_ROOT)

val_img_url = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip"
val_gt_url  = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip"

has_val_img = any("ISIC2018_Task1-2_Validation_Input" in root for root, dirs, _ in os.walk(DATA_ROOT))
has_val_gt  = any("ISIC2018_Task1_Validation_GroundTruth" in root for root, dirs, _ in os.walk(DATA_ROOT))

if not has_val_img:
    print("[*] Downloading Official Val Input from S3...", flush=True)
    subprocess.run(["wget", "-q", "-O", f"{DATA_ROOT}/val_input.zip", val_img_url], check=True)
if not has_val_gt:
    print("[*] Downloading Official Val GT from S3...", flush=True)
    subprocess.run(["wget", "-q", "-O", f"{DATA_ROOT}/val_gt.zip", val_gt_url], check=True)
_unzip_all(DATA_ROOT)

def find_dir_by_name(target_name, base_search=DATA_ROOT):
    for root, dirs, files in os.walk(base_search):
        if os.path.basename(root).lower() == target_name.lower():
            return root
    return None

train_img_dir = find_dir_by_name("ISIC2018_Task1-2_Training_Input")
train_gt_dir  = find_dir_by_name("ISIC2018_Task1_Training_GroundTruth")
val_img_dir   = find_dir_by_name("ISIC2018_Task1-2_Validation_Input")
val_gt_dir    = find_dir_by_name("ISIC2018_Task1_Validation_GroundTruth")
test_img_dir  = find_dir_by_name("ISIC2018_Task1-2_Test_Input")
test_gt_dir   = find_dir_by_name("ISIC2018_Task1_Test_GroundTruth")

def isic_id(path):
    base = os.path.basename(path).replace("_segmentation", "")
    return os.path.splitext(base)[0]

def pair_data(img_dir, gt_dir):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    msks = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    id_map = {isic_id(m): m for m in msks}
    paired_x = []
    paired_y = []
    for im in imgs:
        imid = isic_id(im)
        if imid in id_map:
            paired_x.append(im)
            paired_y.append(id_map[imid])
    return paired_x, paired_y

train_x, train_y = pair_data(train_img_dir, train_gt_dir)
val_x, val_y     = pair_data(val_img_dir, val_gt_dir)
test_x, test_y   = pair_data(test_img_dir, test_gt_dir)

print(f"[+] Paired Datasets -> Train: {len(train_x)} | Val: {len(val_x)} | Test: {len(test_x)}", flush=True)

# ====================================================================
# 3. Model Architecture: GDF-Netv4
# ====================================================================
class HierarchicalBackboneV4(nn.Module):
    def __init__(self, backbone_name="pvt_v2_b2", pretrained=True, in_chans=3):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(0, 1, 2, 3)
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
        ca_weight = self.ca(feat)
        feat = feat * ca_weight
        sa_weight = self.sa(feat)
        feat = feat * sa_weight
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

class GDFNetV4(nn.Module):
    def __init__(self, backbone_name="pvt_v2_b2", pretrained=True, in_chans=3, decoder_channels=64):
        super().__init__()
        self.backbone_name = backbone_name
        self.decoder_channels = decoder_channels

        self.backbone = HierarchicalBackboneV4(backbone_name=backbone_name, pretrained=pretrained, in_chans=in_chans)
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
        else:
            return out_final

# ====================================================================
# 4. Structure Loss with Deep Supervision
# ====================================================================
class StructureLoss(nn.Module):
    def __init__(self, kernel_size=31, weight_factor=5.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight_factor = weight_factor
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

        return (wbce + wiou).mean()

class GDFNetV4CompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.structure_loss = StructureLoss()

    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            p_final = outputs['logits']
            p1 = outputs['p1']
            p2 = outputs['p2']
            p3 = outputs['p3']
            p4 = outputs['p4']

            l_final = self.structure_loss(p_final, targets)
            l_p1 = self.structure_loss(p1, targets)
            l_p2 = self.structure_loss(p2, targets)
            l_p3 = self.structure_loss(p3, targets)
            l_p4 = self.structure_loss(p4, targets)

            return l_final + l_p1 + 0.5 * l_p2 + 0.3 * l_p3 + 0.2 * l_p4
        else:
            return self.structure_loss(outputs, targets)

# ====================================================================
# 5. Dataset with ImageNet Normalization & Augmentations
# ====================================================================
IMG_SIZE = 384

class ISICDatasetV4(Dataset):
    def __init__(self, imgs, msks, is_train=False):
        self.imgs = imgs
        self.msks = msks
        self.is_train = is_train

        if is_train:
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.5, 1.0), p=0.7),
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.6),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs[idx], cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(Image.open(self.imgs[idx]).convert('RGB'))

        msk = cv2.imread(self.msks[idx], cv2.IMREAD_GRAYSCALE)
        if msk is None:
            msk = np.array(Image.open(self.msks[idx]).convert('L'))
        msk = (msk > 127).astype(np.float32)

        augmented = self.transform(image=img, mask=msk)
        img = augmented['image'].astype(np.float32)
        msk = augmented['mask'].astype(np.float32)

        img = np.transpose(img, (2, 0, 1))
        msk = np.expand_dims(msk, axis=0)
        return torch.from_numpy(img).float(), torch.from_numpy(msk).float()

def compute_metrics_v4(p, t):
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

    hd95 = 0.0
    asd = 0.0
    bf2 = 0.0

    if t.sum() > 0 and p.sum() > 0:
        bg = t ^ binary_erosion(t)
        bp = p ^ binary_erosion(p)
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
        'dice': float(dice), 'miou': float(miou), 'sensitivity': float(sens),
        'specificity': float(spec), 'precision': float(prec), 'fpr': float(fpr),
        'hd95': hd95, 'asd': asd, 'bf2': bf2,
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn)
    }

@torch.no_grad()
def evaluate_model_v4(model, loader, use_tta=False):
    model.eval()
    records = []
    tot_tp, tot_fp, tot_fn = 0, 0, 0

    for imgs, msks in loader:
        imgs = imgs.to(device)
        with autocast('cuda'):
            if use_tta:
                p1 = torch.sigmoid(model(imgs))
                p2 = torch.sigmoid(model(torch.flip(imgs, dims=[3])))
                p2 = torch.flip(p2, dims=[3])
                p3 = torch.sigmoid(model(torch.flip(imgs, dims=[2])))
                p3 = torch.flip(p3, dims=[2])
                p4 = torch.sigmoid(model(torch.flip(imgs, dims=[2, 3])))
                p4 = torch.flip(p4, dims=[2, 3])
                probs = (p1 + p2 + p3 + p4) / 4.0
            else:
                probs = torch.sigmoid(model(imgs))

        preds = probs.cpu().numpy()[:, 0]
        msks_np = msks.numpy()[:, 0]

        for i in range(len(preds)):
            m = compute_metrics_v4(preds[i] > 0.5, msks_np[i] > 0.5)
            tot_tp += m['tp']
            tot_fp += m['fp']
            tot_fn += m['fn']
            records.append(m)

    df = pd.DataFrame(records)
    global_dice = float((2.0 * tot_tp) / (2.0 * tot_tp + tot_fp + tot_fn + 1e-8))
    global_iou  = float(tot_tp / (tot_tp + tot_fp + tot_fn + 1e-8))

    summary = []
    for col in ['dice', 'miou', 'sensitivity', 'specificity', 'precision', 'fpr', 'hd95', 'asd', 'bf2']:
        summary.append({
            'Metric': col,
            'Mean': float(df[col].mean()),
            'Std': float(df[col].std()),
            'Median': float(df[col].median()),
            'Min': float(df[col].min()),
            'Max': float(df[col].max())
        })
    summary_df = pd.DataFrame(summary)
    return df, summary_df, global_dice, global_iou

# ====================================================================
# 6. Training Engine on A100 GPU (100 Epochs, Robust Resumption)
# ====================================================================
train_loader = DataLoader(ISICDatasetV4(train_x, train_y, is_train=True), batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(ISICDatasetV4(val_x, val_y, is_train=False), batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(ISICDatasetV4(test_x, test_y, is_train=False), batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

model = GDFNetV4(backbone_name="pvt_v2_b2", pretrained=True).to(device)
criterion = GDFNetV4CompositeLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
EPOCHS = 100
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
scaler = GradScaler('cuda')

best_val_dice = 0.0
best_epoch = 0
start_epoch = 1

checkpoint_best = f"{OUTPUT_DIR}/GDFNetV4_ISIC2018_best.pth"
checkpoint_latest = f"{OUTPUT_DIR}/GDFNetV4_ISIC2018_latest.pth"
history_csv = f"{OUTPUT_DIR}/training_history_ISIC2018.csv"

# Check for resume candidates (Drive > Output Dir > Root)
resume_candidates = [
    f"{DRIVE_DIR}/GDFNetV4_ISIC2018_latest.pth",
    checkpoint_latest,
    "/content/GDFNetV4_latest.pth"
]

for rpath in resume_candidates:
    if os.path.exists(rpath):
        print(f"[*] Found valid resume checkpoint at {rpath}. Restoring training state...", flush=True)
        ckpt = torch.load(rpath, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch = ckpt['epoch'] + 1
        best_val_dice = ckpt.get('best_val_dice', 0.0)
        best_epoch = ckpt.get('best_epoch', ckpt['epoch'])
        print(f"[+] Successfully Resumed from Epoch {start_epoch} (Baseline Best Val Dice: {best_val_dice:.4f} at Epoch {best_epoch})", flush=True)
        break

history = []
if os.path.exists(history_csv):
    try:
        history = pd.read_csv(history_csv).to_dict('records')
    except Exception:
        pass
elif os.path.exists(f"{DRIVE_DIR}/training_history_ISIC2018.csv"):
    try:
        history = pd.read_csv(f"{DRIVE_DIR}/training_history_ISIC2018.csv").to_dict('records')
        shutil.copy2(f"{DRIVE_DIR}/training_history_ISIC2018.csv", history_csv)
    except Exception:
        pass

print(f"\n[*] Starting GDF-Netv4 (PVTv2-B2 + MS-FDF + LGAG) Training on A100 (Epoch {start_epoch} to {EPOCHS})", flush=True)

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    t0 = time.time()

    for imgs, msks in train_loader:
        imgs, msks = imgs.to(device), msks.to(device)
        optimizer.zero_grad()
        with autocast('cuda'):
            outs = model(imgs)
            loss = criterion(outs, msks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    scheduler.step()
    elapsed = time.time() - t0
    epoch_loss = running_loss / len(train_loader)
    
    # Validation evaluation
    _, val_summary, val_global_dice, val_global_iou = evaluate_model_v4(model, val_loader)
    val_macro_dice = val_summary.loc[val_summary['Metric'] == 'dice', 'Mean'].values[0]
    val_macro_miou = val_summary.loc[val_summary['Metric'] == 'miou', 'Mean'].values[0]

    current_lr = scheduler.get_last_lr()[0]
    history.append({
        'epoch': epoch, 'loss': epoch_loss, 'lr': current_lr,
        'val_macro_dice': val_macro_dice, 'val_macro_miou': val_macro_miou,
        'val_global_dice': val_global_dice, 'val_global_iou': val_global_iou,
        'elapsed_sec': elapsed
    })
    pd.DataFrame(history).to_csv(history_csv, index=False)

    # Save latest checkpoint (complete state for seamless resumption)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'scaler_state': scaler.state_dict(),
        'best_val_dice': best_val_dice,
        'best_epoch': best_epoch
    }, checkpoint_latest)

    print(f"Epoch {epoch:03d}/{EPOCHS} [{elapsed:.1f}s] | Loss: {epoch_loss:.4f} | LR: {current_lr:.6f} | Val Macro Dice: {val_macro_dice:.4f} | Val Global Dice: {val_global_dice:.4f}", flush=True)

    if val_macro_dice > best_val_dice:
        best_val_dice = val_macro_dice
        best_epoch = epoch
        torch.save(model.state_dict(), checkpoint_best)
        print(f"[*] New Best Model Saved at Epoch {epoch} (Val Macro Dice: {val_macro_dice:.4f})", flush=True)

    # Mirror to Google Drive
    if os.path.exists(DRIVE_DIR):
        try:
            shutil.copy2(history_csv, DRIVE_DIR)
            shutil.copy2(checkpoint_latest, DRIVE_DIR)
            if epoch == best_epoch:
                shutil.copy2(checkpoint_best, DRIVE_DIR)
        except Exception as e:
            print(f"[!] Warning syncing to Drive: {e}", flush=True)

# Final Blind Test Evaluation on 1,000 Official Held-Out Test Images
print("\n[*] Evaluating Best GDF-Netv4 Checkpoint on 1,000 Official Blind Test Images (Single View)...", flush=True)
if os.path.exists(checkpoint_best):
    model.load_state_dict(torch.load(checkpoint_best, map_location=device, weights_only=False))
elif os.path.exists(f"{DRIVE_DIR}/GDFNetV4_ISIC2018_best.pth"):
    model.load_state_dict(torch.load(f"{DRIVE_DIR}/GDFNetV4_ISIC2018_best.pth", map_location=device, weights_only=False))

per_img_df, summary_df, test_global_dice, test_global_iou = evaluate_model_v4(model, test_loader, use_tta=False)
summary_df.to_csv(f"{OUTPUT_DIR}/benchmark_summary_ISIC2018.csv", index=False)
per_img_df.to_csv(f"{OUTPUT_DIR}/benchmark_raw_ISIC2018.csv", index=False)

print("\n================ GDF-NETV4 OFFICIAL TEST RESULTS (SINGLE VIEW) ================", flush=True)
print(summary_df.to_string(index=False), flush=True)
print(f"[+] Dataset-Level Global Micro Dice: {test_global_dice:.4f} ({test_global_dice*100:.2f}%)", flush=True)
print(f"[+] Dataset-Level Global Micro IoU:  {test_global_iou:.4f} ({test_global_iou*100:.2f}%)", flush=True)

# Also run with 4-View Flip TTA
print("\n[*] Evaluating Best GDF-Netv4 Checkpoint on 1,000 Test Images with 4-View Flip TTA...", flush=True)
per_img_tta_df, summary_tta_df, test_global_tta_dice, test_global_tta_iou = evaluate_model_v4(model, test_loader, use_tta=True)
summary_tta_df.to_csv(f"{OUTPUT_DIR}/benchmark_summary_ISIC2018_TTA.csv", index=False)
per_img_tta_df.to_csv(f"{OUTPUT_DIR}/benchmark_raw_ISIC2018_TTA.csv", index=False)

print("\n================ GDF-NETV4 OFFICIAL TEST RESULTS (4-VIEW TTA) ================", flush=True)
print(summary_tta_df.to_string(index=False), flush=True)
print(f"[+] TTA Dataset-Level Global Micro Dice: {test_global_tta_dice:.4f} ({test_global_tta_dice*100:.2f}%)", flush=True)
print(f"[+] TTA Dataset-Level Global Micro IoU:  {test_global_tta_iou:.4f} ({test_global_tta_iou*100:.2f}%)", flush=True)

# Final Drive Mirroring
if os.path.exists(DRIVE_DIR):
    try:
        shutil.copy2(f"{OUTPUT_DIR}/benchmark_summary_ISIC2018.csv", DRIVE_DIR)
        shutil.copy2(f"{OUTPUT_DIR}/benchmark_raw_ISIC2018.csv", DRIVE_DIR)
        shutil.copy2(f"{OUTPUT_DIR}/benchmark_summary_ISIC2018_TTA.csv", DRIVE_DIR)
        shutil.copy2(f"{OUTPUT_DIR}/benchmark_raw_ISIC2018_TTA.csv", DRIVE_DIR)
    except Exception:
        pass

print("\n[+] GDF-Netv4 ISIC-2018 Training & Official Blind Evaluation Complete!", flush=True)
