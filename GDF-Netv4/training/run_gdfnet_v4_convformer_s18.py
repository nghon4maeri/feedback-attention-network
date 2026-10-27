"""
========================================================================================
GDF-Netv4 Pure-CNN Backbone Pilot Screening (ConvFormer-S18 (MetaFormer Pure-CNN IN-22K + 384x384, 29.50M Total Params))
Backbone: convformer_s18.sail_in22k_ft_in1k_384
Pilot Horizon: 25 Epochs (Fast SSD Pre-Cached ~20s/epoch)
Official ISIC-2018 Training (2,594 Train / 100 Official Val) & 1,000 Blind Test Benchmark
========================================================================================
"""

import os
import sys
import glob
import time
import copy
import json
import random
import zipfile
import shutil
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import cv2

from scipy.ndimage import distance_transform_edt, binary_erosion

import timm
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

SCRIPT_START_TIME = time.time()
MAX_RUNTIME_SECONDS = 39600
BACKBONE_NAME = "convformer_s18.sail_in22k_ft_in1k_384"
MODEL_TAG = "convformer_s18"
PILOT_EPOCHS = 25

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ====================================================================
# 1. Expanded 15-Case Clinical Dermoscopy Augmentation Suite
# ====================================================================
class RandomVignettingTransform:
    def __init__(self, p=0.25, min_radius=0.50, max_radius=0.80):
        self.p = p
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __call__(self, image):
        if random.random() > self.p:
            return image
        H, W = image.shape[:2]
        center_x = random.uniform(W * 0.42, W * 0.58)
        center_y = random.uniform(H * 0.42, H * 0.58)
        radius = random.uniform(self.min_radius, self.max_radius) * min(H, W)
        Y, X = np.ogrid[:H, :W]
        dist = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        sigma = radius * 0.18
        mask = 1.0 - 1.0 / (1.0 + np.exp(-(dist - radius) / max(sigma, 1.0)))
        mask = np.clip(mask, 0.08, 1.0)
        if image.ndim == 3:
            mask = mask[..., None]
        return (image * mask).astype(np.uint8)


class SyntheticHairTransform:
    def __init__(self, p=0.30, max_hairs=5):
        self.p = p
        self.max_hairs = max_hairs

    def __call__(self, image):
        if random.random() > self.p:
            return image
        H, W = image.shape[:2]
        img_aug = image.copy()
        num_hairs = random.randint(1, self.max_hairs)
        for _ in range(num_hairs):
            x0, y0 = random.randint(0, W), random.randint(0, H)
            x1, y1 = random.randint(0, W), random.randint(0, H)
            ctrl_x, ctrl_y = random.randint(0, W), random.randint(0, H)
            points = []
            for t in np.linspace(0, 1, 30):
                px = int((1 - t) ** 2 * x0 + 2 * (1 - t) * t * ctrl_x + t ** 2 * x1)
                py = int((1 - t) ** 2 * y0 + 2 * (1 - t) * t * ctrl_y + t ** 2 * y1)
                points.append((px, py))
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            thickness = random.randint(1, 2)
            if random.random() < 0.8:
                c = random.randint(10, 45)
                color = (c, max(0, c - 5), max(0, c - 10))
            else:
                c = random.randint(180, 230)
                color = (c, c, c)
            cv2.polylines(img_aug, [pts], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)
        return img_aug


class SyntheticRulerMarkTransform:
    def __init__(self, p=0.15):
        self.p = p

    def __call__(self, image):
        if random.random() > self.p:
            return image
        H, W = image.shape[:2]
        img_aug = image.copy()
        horizontal = random.random() < 0.5
        step = random.randint(12, 24)
        color = (random.randint(15, 50),) * 3
        if horizontal:
            y_base = random.choice([random.randint(5, int(H * 0.15)), random.randint(int(H * 0.85), H - 5)])
            for i, x in enumerate(range(10, W - 10, step)):
                tick_h = 14 if i % 5 == 0 else 7
                cv2.line(img_aug, (x, y_base), (x, min(H - 1, y_base + tick_h)), color, thickness=random.randint(1, 2))
        else:
            x_base = random.choice([random.randint(5, int(W * 0.15)), random.randint(int(W * 0.85), W - 5)])
            for i, y in enumerate(range(10, H - 10, step)):
                tick_w = 14 if i % 5 == 0 else 7
                cv2.line(img_aug, (x_base, y), (min(W - 1, x_base + tick_w), y), color, thickness=random.randint(1, 2))
        return img_aug


class SpecularGlareTransform:
    def __init__(self, p=0.15, max_spots=4):
        self.p = p
        self.max_spots = max_spots

    def __call__(self, image):
        if random.random() > self.p:
            return image
        H, W = image.shape[:2]
        img_aug = image.copy()
        for _ in range(random.randint(1, self.max_spots)):
            cx = random.randint(int(W * 0.15), int(W * 0.85))
            cy = random.randint(int(H * 0.15), int(H * 0.85))
            rx = random.randint(3, 10)
            ry = random.randint(2, 7)
            angle = random.randint(0, 180)
            overlay = img_aug.copy()
            cv2.ellipse(overlay, (cx, cy), (rx, ry), angle, 0, 360, (245, 245, 245), -1)
            alpha = random.uniform(0.4, 0.75)
            img_aug = cv2.addWeighted(overlay, alpha, img_aug, 1.0 - alpha, 0)
        return img_aug


class ImageOnlyCutoutTransform:
    def __init__(self, p=0.25, max_holes=4, max_size=28):
        self.p = p
        self.max_holes = max_holes
        self.max_size = max_size

    def __call__(self, image):
        if random.random() > self.p:
            return image
        H, W = image.shape[:2]
        img_aug = image.copy()
        mean_color = img_aug.mean(axis=(0, 1)).astype(np.uint8).tolist()
        for _ in range(random.randint(1, self.max_holes)):
            h = random.randint(8, self.max_size)
            w = random.randint(8, self.max_size)
            y0 = random.randint(0, max(1, H - h))
            x0 = random.randint(0, max(1, W - w))
            img_aug[y0:y0 + h, x0:x0 + w] = mean_color
        return img_aug


# ====================================================================
# 2. Model Architecture: GDF-Netv4 with Pluggable Pure-CNN Backbone
# ====================================================================
class HierarchicalBackboneV4(nn.Module):
    def __init__(self, backbone_name=BACKBONE_NAME, pretrained=True, in_chans=3, drop_path_rate=0.2):
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
    def __init__(self, backbone_name=BACKBONE_NAME, pretrained=True, num_classes=1, drop_path_rate=0.2):
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
    """EMA with dynamic warmup decay so early random decoder weights do not pollute the average."""
    def __init__(self, model, decay=0.998):
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
    def __init__(self, kernel_size=31, weight_factor=5.0, dice_weight=0.5):
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
    def __init__(self, dice_weight=0.5):
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
# 3. Fast Local SSD Cache & Dataset Engine
# ====================================================================
IMG_SIZE = 384
TRAIN_CACHE_SIZE = 512

def prepare_fast_ssd_cache(imgs, msks, cache_dir, target_size, desc="Dataset"):
    os.makedirs(cache_dir, exist_ok=True)
    cached_imgs, cached_msks = [], []
    need_build = False
    for im_p, mk_p in zip(imgs, msks):
        base_id = os.path.splitext(os.path.basename(im_p))[0]
        c_im = os.path.join(cache_dir, f"{base_id}.jpg")
        c_mk = os.path.join(cache_dir, f"{base_id}_mask.png")
        cached_imgs.append(c_im)
        cached_msks.append(c_mk)
        if not (os.path.exists(c_im) and os.path.exists(c_mk)):
            need_build = True

    if need_build:
        print(f"[*] Building fast local SSD cache for ConvFormer-S18 (MetaFormer Pure-CNN IN-22K + 384x384, 29.50M Total Params) ({len(imgs)} pairs -> {target_size}x{target_size})...", flush=True)
        t_cache = time.time()
        for im_p, mk_p, c_im, c_mk in zip(imgs, msks, cached_imgs, cached_msks):
            if not (os.path.exists(c_im) and os.path.exists(c_mk)):
                img = cv2.imread(im_p, cv2.IMREAD_COLOR)
                if img is None:
                    img = cv2.cvtColor(np.array(Image.open(im_p).convert('RGB')), cv2.COLOR_RGB2BGR)
                msk = cv2.imread(mk_p, cv2.IMREAD_GRAYSCALE)
                if msk is None:
                    msk = np.array(Image.open(mk_p).convert('L'))
                img_r = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
                msk_r = cv2.resize(msk, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(c_im, img_r, [int(cv2.IMWRITE_JPEG_QUALITY), 96])
                cv2.imwrite(c_mk, msk_r)
        print(f"[+] Fast SSD cache ready for ConvFormer-S18 (MetaFormer Pure-CNN IN-22K + 384x384, 29.50M Total Params) in {time.time() - t_cache:.1f}s!", flush=True)
    else:
        print(f"[+] Using existing fast SSD cache for ConvFormer-S18 (MetaFormer Pure-CNN IN-22K + 384x384, 29.50M Total Params) at {cache_dir}", flush=True)
    return cached_imgs, cached_msks


class ISICDatasetV4HeavyAug(Dataset):
    def __init__(self, imgs, msks, is_train=False):
        self.imgs = imgs
        self.msks = msks
        self.is_train = is_train

        self.vignetting = RandomVignettingTransform(p=0.25)
        self.hair = SyntheticHairTransform(p=0.30, max_hairs=5)
        self.ruler = SyntheticRulerMarkTransform(p=0.15)
        self.glare = SpecularGlareTransform(p=0.15)
        self.cutout = ImageOnlyCutoutTransform(p=0.25, max_holes=4, max_size=28)

        if is_train:
            self.spatial_transform = A.Compose([
                A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.5, 1.0), p=0.75),
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.Affine(
                    scale=(0.82, 1.18),
                    translate_percent=(-0.10, 0.10),
                    rotate=(-180, 180),
                    shear=(-10, 10),
                    p=0.65
                ),
                A.OneOf([
                    A.GridDistortion(num_steps=5, distort_limit=0.25, p=1.0),
                    A.OpticalDistortion(distort_limit=0.25, p=1.0),
                    A.ElasticTransform(alpha=30, sigma=5, p=1.0),
                ], p=0.35),
            ])
            self.photo_transform = A.Compose([
                A.OneOf([
                    A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.08, p=1.0),
                    A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=20, p=1.0),
                    A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                ], p=0.60),
                A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.30),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.GaussNoise(p=1.0),
                ], p=0.20),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ])
        else:
            self.eval_transform = A.Compose([
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

        if self.is_train:
            sp = self.spatial_transform(image=img, mask=msk)
            img, msk = sp['image'], sp['mask']
            img = self.vignetting(img)
            img = self.hair(img)
            img = self.ruler(img)
            img = self.glare(img)
            img = self.cutout(img)
            augmented = self.photo_transform(image=img, mask=msk)
        else:
            augmented = self.eval_transform(image=img, mask=msk)

        img = np.transpose(augmented['image'].astype(np.float32), (2, 0, 1))
        msk = np.expand_dims(augmented['mask'].astype(np.float32), axis=0)
        return torch.from_numpy(img).float(), torch.from_numpy(msk).float()


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
        'dice': float(dice), 'miou': float(miou), 'sensitivity': float(sens),
        'specificity': float(spec), 'precision': float(prec), 'fpr': float(fpr),
        'hd95': hd95, 'asd': asd, 'bf2': bf2,
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
    for col in ['dice', 'miou', 'sensitivity', 'specificity', 'precision', 'fpr', 'hd95', 'asd', 'bf2']:
        summary.append({
            'Metric': col,
            'Mean': float(df[col].mean()),
            'Std': float(df[col].std()),
            'Median': float(df[col].median()),
            'Min': float(df[col].min()),
            'Max': float(df[col].max())
        })
    j_vals = df['miou'].values
    j_thresh = float(np.where(j_vals >= 0.65, j_vals, 0.0).mean())
    failed = float((j_vals < 0.65).sum())
    summary.append({'Metric': 'j_thresh', 'Mean': j_thresh, 'Std': 0.0, 'Median': j_thresh, 'Min': 0.0, 'Max': 1.0})
    summary.append({'Metric': 'failed_cases_j_lt_0.65', 'Mean': failed, 'Std': 0.0, 'Median': failed, 'Min': 0.0, 'Max': float(len(df))})

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
    for t in np.linspace(0.35, 0.75, 41):
        preds = probs_np > t
        tp = np.logical_and(preds, msks_np).sum(axis=(1, 2))
        fp = np.logical_and(preds, ~msks_np).sum(axis=(1, 2))
        fn = np.logical_and(~preds, msks_np).sum(axis=(1, 2))
        d = ((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)).mean()
        if d > best_dice:
            best_dice = float(d)
            best_t = float(t)
    return best_t, best_dice


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Active Compute Device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"[+] GPU Hardware: {torch.cuda.get_device_name(0)} (VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB)", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    OUTPUT_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else f"./results_{MODEL_TAG}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    DATA_ROOT = "/kaggle/temp/isic-2018" if os.path.exists("/kaggle") else "./isic-2018"
    os.makedirs(DATA_ROOT, exist_ok=True)

    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)
    k_user = os.environ.get("KAGGLE_USERNAME", "namnguynnnn")
    k_key  = os.environ.get("KAGGLE_KEY", "KGAT_dad33ce94fb8b25d8a9ecd4ca26ce03f")
    with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
        json.dump({"username": k_user, "key": k_key}, f)
    os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

    def find_dir_by_name(target_name, search_roots):
        for base in search_roots:
            if not os.path.exists(base):
                continue
            for root, dirs, files in os.walk(base):
                if os.path.basename(root).lower() == target_name.lower():
                    return root
        return None

    def _unzip_all(root):
        for z in glob.glob(os.path.join(root, "*.zip")):
            print(f"[*] Extracting: {os.path.basename(z)}...", flush=True)
            try:
                with zipfile.ZipFile(z) as f:
                    f.extractall(root)
                os.remove(z)
            except Exception as e:
                print(f"[!] Warning while unzipping {z}: {e}", flush=True)

    search_roots = ["/kaggle/input", DATA_ROOT]
    train_img_dir = find_dir_by_name("ISIC2018_Task1-2_Training_Input", search_roots)
    test_gt_dir   = find_dir_by_name("ISIC2018_Task1_Test_GroundTruth", search_roots)

    if not train_img_dir:
        subprocess.run(["kaggle", "datasets", "download", "-d",
                        "tschandl/isic2018-challenge-task1-data-segmentation", "-p", DATA_ROOT], check=True)
        _unzip_all(DATA_ROOT)
    if not test_gt_dir:
        subprocess.run(["kaggle", "datasets", "download", "-d",
                        "mfahimhossen/isic2018-test-groundtruth", "-p", DATA_ROOT], check=True)
        _unzip_all(DATA_ROOT)

    val_img_dir = find_dir_by_name("ISIC2018_Task1-2_Validation_Input", search_roots)
    val_gt_dir  = find_dir_by_name("ISIC2018_Task1_Validation_GroundTruth", search_roots)
    if not val_img_dir:
        subprocess.run(["wget", "-q", "-O", f"{DATA_ROOT}/val_input.zip",
                        "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip"], check=True)
    if not val_gt_dir:
        subprocess.run(["wget", "-q", "-O", f"{DATA_ROOT}/val_gt.zip",
                        "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip"], check=True)
    _unzip_all(DATA_ROOT)

    train_img_dir = find_dir_by_name("ISIC2018_Task1-2_Training_Input", search_roots)
    train_gt_dir  = find_dir_by_name("ISIC2018_Task1_Training_GroundTruth", search_roots)
    val_img_dir   = find_dir_by_name("ISIC2018_Task1-2_Validation_Input", search_roots)
    val_gt_dir    = find_dir_by_name("ISIC2018_Task1_Validation_GroundTruth", search_roots)
    test_img_dir  = find_dir_by_name("ISIC2018_Task1-2_Test_Input", search_roots)
    test_gt_dir   = find_dir_by_name("ISIC2018_Task1_Test_GroundTruth", search_roots)

    def isic_id(path):
        base = os.path.basename(path).replace("_segmentation", "")
        return os.path.splitext(base)[0]

    def pair_data(img_dir, gt_dir):
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        msks = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
        id_map = {isic_id(m): m for m in msks}
        paired_x, paired_y = [], []
        for im in imgs:
            imid = isic_id(im)
            if imid in id_map:
                paired_x.append(im)
                paired_y.append(id_map[imid])
        return paired_x, paired_y

    train_x_raw, train_y_raw = pair_data(train_img_dir, train_gt_dir)
    val_x_raw, val_y_raw     = pair_data(val_img_dir, val_gt_dir)
    test_x_raw, test_y_raw   = pair_data(test_img_dir, test_gt_dir)
    print(f"[+] Paired Datasets -> Train: {len(train_x_raw)} | Official Val: {len(val_x_raw)} | Blind Test: {len(test_x_raw)}", flush=True)

    CACHE_ROOT = os.path.join(DATA_ROOT, "fast_ssd_cache")
    train_x, train_y = prepare_fast_ssd_cache(train_x_raw, train_y_raw, os.path.join(CACHE_ROOT, "train_512"), TRAIN_CACHE_SIZE, desc="Train Set")
    val_x, val_y     = prepare_fast_ssd_cache(val_x_raw, val_y_raw, os.path.join(CACHE_ROOT, "val_384"), IMG_SIZE, desc="Official Val Set")
    test_x, test_y   = prepare_fast_ssd_cache(test_x_raw, test_y_raw, os.path.join(CACHE_ROOT, "test_384"), IMG_SIZE, desc="Official Test Set")

    BATCH_SIZE = 16
    EPOCHS = PILOT_EPOCHS
    NUM_WORKERS = min(4, os.cpu_count() or 2)
    train_loader = DataLoader(ISICDatasetV4HeavyAug(train_x, train_y, is_train=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=(NUM_WORKERS > 0), drop_last=True)
    val_loader   = DataLoader(ISICDatasetV4HeavyAug(val_x, val_y, is_train=False),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=(NUM_WORKERS > 0))
    test_loader  = DataLoader(ISICDatasetV4HeavyAug(test_x, test_y, is_train=False),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=(NUM_WORKERS > 0))

    model = GDFNetV4(backbone_name=BACKBONE_NAME, pretrained=True, drop_path_rate=0.2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    bb_params = sum(p.numel() for p in model.backbone.parameters())
    print(f"[+] GDFNetV4 ({MODEL_TAG}) Initialized | Backbone: {BACKBONE_NAME}", flush=True)
    print(f"[+] Channels: {model.backbone.get_channels()} | Backbone Params: {bb_params:,} | Total Params: {total_params:,}", flush=True)

    checkpoint_best = os.path.join(OUTPUT_DIR, f"GDFNetV4_{MODEL_TAG}_best.pth")
    checkpoint_latest = os.path.join(OUTPUT_DIR, f"GDFNetV4_{MODEL_TAG}_latest.pth")
    history_csv = os.path.join(OUTPUT_DIR, f"training_history_{MODEL_TAG}.csv")

    backbone_params = list(model.backbone.parameters())
    backbone_ids = set(id(p) for p in backbone_params)
    head_params = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = optim.AdamW([
        {'params': backbone_params, 'lr': 1e-4},
        {'params': head_params, 'lr': 3e-4}
    ], weight_decay=2e-4)
    ema = WarmupModelEMA(model, decay=0.998)
    criterion = GDFNetV4DiceOptCompositeLoss(dice_weight=0.5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    scaler = GradScaler('cuda')

    best_val_dice = 0.0
    best_epoch = 0
    best_source = "init"
    history = []

    print("\n=======================================================", flush=True)
    print(f"[*] STARTING {EPOCHS}-EPOCH PILOT SCREENING FOR {MODEL_TAG.upper()}", flush=True)
    print("=======================================================", flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()

        for imgs, msks in train_loader:
            imgs, msks = imgs.to(device, non_blocking=True), msks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast('cuda'):
                outs = model(imgs)
                loss = criterion(outs, msks)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            running_loss += loss.item()

        scheduler.step()
        elapsed = time.time() - t0
        epoch_loss = running_loss / len(train_loader)
        current_lr = scheduler.get_last_lr()[-1]

        _, val_sum_raw, val_glob_raw, _ = evaluate_model_v4(model, val_loader, device, use_tta=False, compute_surface=False)
        val_dice_raw = float(val_sum_raw.loc[val_sum_raw['Metric'] == 'dice', 'Mean'].values[0])

        _, val_sum_ema, val_glob_ema, _ = evaluate_model_v4(ema.ema, val_loader, device, use_tta=False, compute_surface=False)
        val_dice_ema = float(val_sum_ema.loc[val_sum_ema['Metric'] == 'dice', 'Mean'].values[0])

        epoch_best_dice = max(val_dice_raw, val_dice_ema)
        epoch_best_src = "EMA" if val_dice_ema >= val_dice_raw else "RAW"

        history.append({
            'epoch': epoch,
            'loss': epoch_loss,
            'lr': current_lr,
            'val_macro_dice_raw': val_dice_raw,
            'val_global_dice_raw': val_glob_raw,
            'val_macro_dice_ema': val_dice_ema,
            'val_global_dice_ema': val_glob_ema,
            'elapsed_sec': elapsed
        })
        pd.DataFrame(history).to_csv(history_csv, index=False)

        print(
            f"Epoch {epoch:03d}/{EPOCHS} [{elapsed:.1f}s] | Loss: {epoch_loss:.4f} | LR: {current_lr:.6f} "
            f"| Val Raw Dice: {val_dice_raw:.4f} | Val EMA Dice: {val_dice_ema:.4f} | Best: {max(best_val_dice, epoch_best_dice):.4f}",
            flush=True
        )

        if epoch_best_dice > best_val_dice:
            best_val_dice = epoch_best_dice
            best_epoch = epoch
            best_source = epoch_best_src
            best_state = ema.ema.state_dict() if epoch_best_src == "EMA" else model.state_dict()
            torch.save(best_state, checkpoint_best)
            print(f"[*] New Best Model ({best_source}) Saved at Epoch {epoch} (Official Val Macro Dice: {best_val_dice:.4f})", flush=True)

        if epoch == EPOCHS or (epoch % 5 == 0):
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'ema_state': ema.ema.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'best_val_dice': best_val_dice,
                'best_epoch': best_epoch,
                'best_source': best_source
            }, checkpoint_latest)

    print("\n=======================================================", flush=True)
    print(f"[*] STARTING OFFICIAL 1,000 BLIND TEST EVALUATION FOR {MODEL_TAG.upper()} (Best Val: {best_val_dice:.4f} @ Ep {best_epoch} [{best_source}])", flush=True)
    print("=======================================================", flush=True)

    if os.path.exists(checkpoint_best):
        model.load_state_dict(torch.load(checkpoint_best, map_location=device, weights_only=False))

    sv_df, sv_summary, sv_glob_dice, sv_glob_iou = evaluate_model_v4(
        model, test_loader, device, use_tta=False, prob_thresh=0.50, compute_surface=True
    )
    sv_summary.to_csv(os.path.join(OUTPUT_DIR, f"benchmark_summary_{MODEL_TAG}_SingleView.csv"), index=False)
    sv_df.to_csv(os.path.join(OUTPUT_DIR, f"benchmark_raw_{MODEL_TAG}_SingleView.csv"), index=False)
    print(f"\n================ {MODEL_TAG.upper()} BLIND TEST (SINGLE VIEW, T=0.50) ================", flush=True)
    print(sv_summary.to_string(index=False), flush=True)
    print(f"[+] Dataset Global Micro Dice: {sv_glob_dice:.4f} | IoU: {sv_glob_iou:.4f}", flush=True)

    tta_df, tta_summary, tta_glob_dice, tta_glob_iou = evaluate_model_v4(
        model, test_loader, device, use_tta=True, prob_thresh=0.50, compute_surface=True
    )
    tta_summary.to_csv(os.path.join(OUTPUT_DIR, f"benchmark_summary_{MODEL_TAG}_TTA.csv"), index=False)
    tta_df.to_csv(os.path.join(OUTPUT_DIR, f"benchmark_raw_{MODEL_TAG}_TTA.csv"), index=False)
    print(f"\n================ {MODEL_TAG.upper()} BLIND TEST (4-VIEW TTA, T=0.50) ================", flush=True)
    print(tta_summary.to_string(index=False), flush=True)
    print(f"[+] TTA Dataset Global Micro Dice: {tta_glob_dice:.4f} | IoU: {tta_glob_iou:.4f}", flush=True)

    opt_t, val_cal_dice = calibrate_threshold_on_val(model, val_loader, device)
    print(f"\n[*] Calibrated Optimal Threshold on 100 Official Val Images: T_opt = {opt_t:.3f} (Val TTA Macro Dice: {val_cal_dice:.4f})", flush=True)
    cal_df, cal_summary, cal_glob_dice, cal_glob_iou = evaluate_model_v4(
        model, test_loader, device, use_tta=True, prob_thresh=opt_t, compute_surface=True
    )
    cal_summary.to_csv(os.path.join(OUTPUT_DIR, f"benchmark_summary_{MODEL_TAG}_TTA_Calibrated.csv"), index=False)
    cal_df.to_csv(os.path.join(OUTPUT_DIR, f"benchmark_raw_{MODEL_TAG}_TTA_Calibrated.csv"), index=False)
    print(f"\n================ {MODEL_TAG.upper()} BLIND TEST (4-VIEW TTA, CALIBRATED T={opt_t:.2f}) ================", flush=True)
    print(cal_summary.to_string(index=False), flush=True)
    print(f"[+] Calibrated TTA Global Micro Dice: {cal_glob_dice:.4f} | IoU: {cal_glob_iou:.4f}", flush=True)


if __name__ == "__main__":
    main()
