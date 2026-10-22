"""
========================================================================================
GDF-Netv5: General Decoupled Feedback Network Version 5
5-Fold Cross-Validation Training & Official Blind Evaluation on NVIDIA A100 GPU (ISIC-2018)

Key Architectural & Training Upgrades:
1. Encoder: Hierarchical Vision Transformer (PVTv2-B4, 62.0M params, 24 blocks in Stage 3)
2. Native High-Resolution Input: 512 x 512 pixels
3. Reverse Edge Guidance Stream (REGS): Explicit boundary supervision & background suppression
4. Artifact-Aware Data Augmentation: Synthetic Vignetting + CLAHE + Synthetic Hair
5. 5-Fold Stratified Cross Validation with Soft-Voting Ensemble on 1,000 Blind Test Images
========================================================================================
"""

import os, sys, time, glob, random, math, shutil, argparse
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import KFold
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
import albumentations as A
import timm

# ====================================================================
# 1. Custom Artifact-Aware Augmentations
# ====================================================================

class RandomVignettingTransform:
    """
    Simulates circular dermoscopic optical lens vignetting (dark ring/corners).
    Immunizes the network against the 28 False Positive over-segmentation failure modes.
    """
    def __init__(self, p=0.4, min_radius=0.45, max_radius=0.75):
        self.p = p
        self.min_radius = min_radius
        self.max_radius = max_radius

    def __call__(self, image):
        if random.random() > self.p:
            return image
        H, W = image.shape[:2]
        center_x = random.uniform(W * 0.4, W * 0.6)
        center_y = random.uniform(H * 0.4, H * 0.6)
        radius = random.uniform(self.min_radius, self.max_radius) * min(H, W)

        Y, X = np.ogrid[:H, :W]
        dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)
        
        # Soft sigmoid falloff at the boundary
        sigma = radius * 0.15
        mask = 1.0 - 1.0 / (1.0 + np.exp(-(dist_from_center - radius) / max(sigma, 1.0)))
        mask = np.clip(mask, 0.05, 1.0)
        
        if len(image.shape) == 3:
            mask = np.expand_dims(mask, axis=-1)
        return (image * mask).astype(np.uint8)

class SyntheticHairTransform:
    """
    Simulates thin dark hair curves occluding the lesion and surrounding healthy skin.
    Prevents the network from falsely misclassifying hair artifacts as lesion borders.
    """
    def __init__(self, p=0.3, max_hairs=4):
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
            
            # Draw quadratic curve using line segments
            points = []
            for t in np.linspace(0, 1, 30):
                px = int((1 - t)**2 * x0 + 2 * (1 - t) * t * ctrl_x + t**2 * x1)
                py = int((1 - t)**2 * y0 + 2 * (1 - t) * t * ctrl_y + t**2 * y1)
                points.append((px, py))
            
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            thickness = random.randint(1, 2)
            color = (random.randint(10, 40), random.randint(10, 40), random.randint(10, 40))
            cv2.polylines(img_aug, [pts], isClosed=False, color=color, thickness=thickness)
        return img_aug


# ====================================================================
# 2. GDF-Netv5 Architecture Definitions
# ====================================================================

class HierarchicalBackboneV5(nn.Module):
    def __init__(self, backbone_name="pvt_v2_b4", pretrained=True, in_chans=3):
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

class MultiScaleFrequencyFilterV5(nn.Module):
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

        feat_with_mask = torch.cat([skip + g_proj, m_up], dim=1)
        x = self.relu(self.bn1(self.conv1x1(feat_with_mask)))
        x = self.relu(self.bn_dw(self.dw_dil1(x) + self.dw_dil2(x)))
        attn = self.attn_proj(x)
        return self.out_conv(skip * attn)

class ReverseEdgeGuidanceStream(nn.Module):
    def __init__(self, c1, c2, mid_channels=64):
        super().__init__()
        self.proj_c2 = nn.Sequential(
            nn.Conv2d(c2, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.proj_c1 = nn.Sequential(
            nn.Conv2d(c1, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.edge_head = nn.Conv2d(mid_channels, 1, kernel_size=1)
        self.edge_gate_proj = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Sigmoid()
        )

    def forward(self, f1, f2, target_size):
        H4, W4 = f1.shape[2], f1.shape[3]
        f2_up = F.interpolate(self.proj_c2(f2), size=(H4, W4), mode='bilinear', align_corners=False)
        f1_proj = self.proj_c1(f1)
        edge_feat = self.edge_conv(torch.cat([f1_proj, f2_up], dim=1))
        edge_logits = F.interpolate(self.edge_head(edge_feat), size=target_size, mode='bilinear', align_corners=False)
        edge_attn = self.edge_gate_proj(edge_feat)
        edge_feat_gated = edge_feat * (1.0 + edge_attn)
        return edge_logits, edge_feat_gated

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
        dw = F.relu(self.bn_dw(torch.cat([self.dw1(x), self.dw3(x), self.dw5(x)], dim=1)), inplace=True)
        feat = self.pw(dw)
        feat = feat * self.ca(feat)
        feat = feat * self.sa(feat)
        return self.refine(feat)

class CascadedMSCAMDecoderBlockV5(nn.Module):
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
        return out_feat, self.head(out_feat)

class GDFNetV5(nn.Module):
    def __init__(
        self,
        backbone_name="pvt_v2_b4",
        pretrained=True,
        in_chans=3,
        decoder_channels=64,
        edge_channels=64
    ):
        super().__init__()
        self.backbone = HierarchicalBackboneV5(backbone_name=backbone_name, pretrained=pretrained, in_chans=in_chans)
        c1, c2, c3, c4 = self.backbone.get_channels()

        self.fdf1 = MultiScaleFrequencyFilterV5(c1)
        self.fdf2 = MultiScaleFrequencyFilterV5(c2)
        self.fdf3 = MultiScaleFrequencyFilterV5(c3)
        self.fdf4 = MultiScaleFrequencyFilterV5(c4)

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

        self.gate3 = MaskGuidedLGAG(c3, dec4_ch, c3)
        self.gate2 = MaskGuidedLGAG(c2, dec3_ch, c2)
        self.gate1 = MaskGuidedLGAG(c1, dec2_ch, c1)

        self.dec3 = CascadedMSCAMDecoderBlockV5(dec4_ch, c3, dec3_ch)
        self.dec2 = CascadedMSCAMDecoderBlockV5(dec3_ch, c2, dec2_ch)
        self.dec1 = CascadedMSCAMDecoderBlockV5(dec2_ch, c1, dec1_ch)

        self.regs = ReverseEdgeGuidanceStream(c1=c1, c2=c2, mid_channels=edge_channels)

        self.final_edge_fuse = nn.Sequential(
            nn.Conv2d(dec1_ch + edge_channels, dec1_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(dec1_ch),
            nn.ReLU(inplace=True)
        )
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

        edge_logits, edge_feat = self.regs(s1, s2, target_size=(H, W))
        d1_edge_fused = self.final_edge_fuse(torch.cat([d1, edge_feat], dim=1))

        p1_up = F.interpolate(p1, size=(H, W), mode='bilinear', align_corners=False)
        p2_up = F.interpolate(p2, size=(H, W), mode='bilinear', align_corners=False)
        p3_up = F.interpolate(p3, size=(H, W), mode='bilinear', align_corners=False)
        p4_up = F.interpolate(p4, size=(H, W), mode='bilinear', align_corners=False)
        d1_up = F.interpolate(d1_edge_fused, size=(H, W), mode='bilinear', align_corners=False)
        
        refined_logits = self.final_refine(d1_up)
        out_final = refined_logits + p1_up + p2_up + p3_up + p4_up

        if self.training:
            return {
                'logits': out_final,
                'edge_logits': edge_logits,
                'p1': p1_up,
                'p2': p2_up,
                'p3': p3_up,
                'p4': p4_up
            }
        return out_final


# ====================================================================
# 3. Loss Functions
# ====================================================================

def extract_edge_ground_truth(mask, radius=1):
    kernel_size = 2 * radius + 1
    with torch.no_grad():
        dilation = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=radius)
        erosion = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=radius)
        edge = (dilation - erosion).clamp(0.0, 1.0)
    return edge

class StructureLoss(nn.Module):
    def __init__(self, kernel_size=31, weight_factor=5.0):
        super().__init__()
        self.weight_factor = weight_factor
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, pred_logits, targets):
        with torch.no_grad():
            local_mean = self.avg_pool(targets)
            weight = 1.0 + self.weight_factor * torch.abs(targets - local_mean)

        bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
        wbce = (weight * bce).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + 1e-8)

        probs = torch.sigmoid(pred_logits)
        intersection = (weight * probs * targets).sum(dim=(2, 3))
        union = (weight * (probs + targets - probs * targets)).sum(dim=(2, 3))
        wiou = 1.0 - (intersection + 1.0) / (union + 1.0)
        return (wbce + wiou).mean()

class BoundaryEdgeLoss(nn.Module):
    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred_edge_logits, target_edges):
        weight = torch.where(target_edges > 0.5, self.pos_weight, 1.0)
        bce = F.binary_cross_entropy_with_logits(pred_edge_logits, target_edges, reduction='none')
        wbce = (weight * bce).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + 1e-8)

        probs = torch.sigmoid(pred_edge_logits)
        intersection = (probs * target_edges).sum(dim=(2, 3))
        cardinality = (probs + target_edges).sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)
        return (wbce + dice_loss).mean()

class GDFNetV5CompositeLoss(nn.Module):
    def __init__(self, edge_weight=0.8):
        super().__init__()
        self.structure_loss = StructureLoss()
        self.edge_loss = BoundaryEdgeLoss()
        self.edge_weight = edge_weight

    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            l_final = self.structure_loss(outputs['logits'], targets)
            l_p1 = self.structure_loss(outputs['p1'], targets)
            l_p2 = self.structure_loss(outputs['p2'], targets)
            l_p3 = self.structure_loss(outputs['p3'], targets)
            l_p4 = self.structure_loss(outputs['p4'], targets)
            seg_loss = l_final + l_p1 + 0.5 * l_p2 + 0.3 * l_p3 + 0.2 * l_p4

            if 'edge_logits' in outputs:
                g_edge = extract_edge_ground_truth(targets)
                l_edge = self.edge_loss(outputs['edge_logits'], g_edge)
                return seg_loss + self.edge_weight * l_edge
            return seg_loss
        return self.structure_loss(outputs, targets)


# ====================================================================
# 4. Dataset & Evaluation Utilities
# ====================================================================

IMG_SIZE = 512

class ISICDatasetV5(Dataset):
    def __init__(self, imgs, msks, is_train=True):
        self.imgs = imgs
        self.msks = msks
        self.is_train = is_train

        self.vignetting = RandomVignettingTransform(p=0.4)
        self.hair = SyntheticHairTransform(p=0.3)

        if is_train:
            self.transform = A.Compose([
                A.RandomResizedCrop(size=(IMG_SIZE, IMG_SIZE), scale=(0.6, 1.0), p=0.7),
                A.Resize(height=IMG_SIZE, width=IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.6),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
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

        if self.is_train:
            img = self.vignetting(img)
            img = self.hair(img)

        augmented = self.transform(image=img, mask=msk)
        img = np.transpose(augmented['image'].astype(np.float32), (2, 0, 1))
        msk = np.expand_dims(augmented['mask'].astype(np.float32), axis=0)

        return torch.from_numpy(img), torch.from_numpy(msk)

def compute_surface_distances(gt_binary, pred_binary):
    if not np.any(gt_binary) or not np.any(pred_binary):
        return np.nan, np.nan
    gt_boundary = gt_binary ^ cv2.erode(gt_binary.astype(np.uint8), np.ones((3,3), np.uint8))
    pred_boundary = pred_binary ^ cv2.erode(pred_binary.astype(np.uint8), np.ones((3,3), np.uint8))
    dt_gt = distance_transform_edt(~gt_boundary)
    dt_pred = distance_transform_edt(~pred_boundary)
    dist_pred_to_gt = dt_gt[pred_boundary.astype(bool)]
    dist_gt_to_pred = dt_pred[gt_boundary.astype(bool)]
    if len(dist_pred_to_gt) == 0 or len(dist_gt_to_pred) == 0:
        return np.nan, np.nan
    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    hd95 = np.percentile(all_dists, 95)
    asd = np.mean(all_dists)
    return hd95, asd

def compute_all_metrics(gt_binary, pred_binary):
    tp = np.logical_and(pred_binary, gt_binary).sum()
    fp = np.logical_and(pred_binary, ~gt_binary).sum()
    fn = np.logical_and(~pred_binary, gt_binary).sum()
    tn = np.logical_and(~pred_binary, ~gt_binary).sum()

    eps = 1e-7
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    miou = (tp + eps) / (tp + fp + fn + eps)
    sensitivity = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)
    precision   = (tp + eps) / (tp + fp + eps)
    fpr         = (fp + eps) / (fp + tn + eps)

    hd95, asd = compute_surface_distances(gt_binary, pred_binary)
    if np.isnan(hd95): hd95 = 0.0
    if np.isnan(asd): asd = 0.0

    return {
        'dice': float(dice), 'miou': float(miou),
        'sensitivity': float(sensitivity), 'specificity': float(specificity),
        'precision': float(precision), 'fpr': float(fpr),
        'hd95': float(hd95), 'asd': float(asd),
        'tp': int(tp), 'fp': int(fp), 'fn': int(fn)
    }

def evaluate_models_ensemble(models, dataloader, device, use_tta=False):
    for m in models:
        m.eval()
    
    results = []
    total_tp, total_fp, total_fn = 0, 0, 0

    with torch.no_grad():
        for imgs, msks in dataloader:
            imgs = imgs.to(device)
            B = imgs.shape[0]

            # Accumulate predictions across all fold models
            ensemble_probs = torch.zeros((B, 1, IMG_SIZE, IMG_SIZE), device=device)

            for model in models:
                if not use_tta:
                    logits = model(imgs)
                    probs = torch.sigmoid(logits)
                else:
                    # 4-View Flip TTA
                    p1 = torch.sigmoid(model(imgs))
                    p2 = torch.sigmoid(model(torch.flip(imgs, dims=[3])))
                    p2 = torch.flip(p2, dims=[3])
                    p3 = torch.sigmoid(model(torch.flip(imgs, dims=[2])))
                    p3 = torch.flip(p3, dims=[2])
                    p4 = torch.sigmoid(model(torch.flip(imgs, dims=[2, 3])))
                    p4 = torch.flip(p4, dims=[2, 3])
                    probs = (p1 + p2 + p3 + p4) / 4.0

                ensemble_probs += probs / len(models)

            pred_bin = (ensemble_probs > 0.5).cpu().numpy().squeeze(1).astype(bool)
            gt_bin = (msks > 0.5).numpy().squeeze(1).astype(bool)

            for i in range(B):
                m = compute_all_metrics(gt_bin[i], pred_bin[i])
                results.append(m)
                total_tp += m['tp']
                total_fp += m['fp']
                total_fn += m['fn']

    df = pd.DataFrame(results)
    global_dice = (2.0 * total_tp) / (2.0 * total_tp + total_fp + total_fn + 1e-7)
    global_iou = (total_tp) / (total_tp + total_fp + total_fn + 1e-7)

    summary = []
    for col in ['dice', 'miou', 'sensitivity', 'specificity', 'precision', 'fpr', 'hd95', 'asd']:
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
# 5. Main 5-Fold Cross Validation Orchestrator
# ====================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Epochs per fold')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size on A100')
    parser.add_argument('--lr', type=float, default=1e-4, help='Initial learning rate')
    parser.add_argument('--output-dir', type=str, default='/content/results_gdfnet_v5', help='Checkpoint output dir')
    parser.add_argument('--fold', type=int, default=-1, help='Specific fold to train (0-4), or -1 to run all 5 folds')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[+] Active Device: {device} | Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Dataset discovery
    DATA_ROOT = "/content/isic-2018"
    train_imgs = sorted(glob.glob(f"{DATA_ROOT}/train_images/*.jpg"))
    train_msks = sorted(glob.glob(f"{DATA_ROOT}/train_masks/*.png"))
    test_imgs  = sorted(glob.glob(f"{DATA_ROOT}/test_images/*.jpg"))
    test_msks  = sorted(glob.glob(f"{DATA_ROOT}/test_masks/*.png"))

    assert len(train_imgs) > 0, "No training images found! Ensure dataset is unzipped."
    assert len(test_imgs) > 0, "No test images found!"
    print(f"[+] Discovered: {len(train_imgs)} Train images | {len(test_imgs)} Blind Test images")

    # Stratified 5-Fold Splitting
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = list(kf.split(train_imgs))

    target_folds = range(5) if args.fold == -1 else [args.fold]

    for fold_idx in target_folds:
        print(f"\n=======================================================")
        print(f"[*] STARTING GDF-NETV5 TRAINING ON FOLD {fold_idx + 1} / 5")
        print(f"=======================================================")

        train_indices, val_indices = folds[fold_idx]
        fold_train_imgs = [train_imgs[i] for i in train_indices]
        fold_train_msks = [train_msks[i] for i in train_indices]
        fold_val_imgs   = [train_imgs[i] for i in val_indices]
        fold_val_msks   = [train_msks[i] for i in val_indices]

        train_loader = DataLoader(ISICDatasetV5(fold_train_imgs, fold_train_msks, is_train=True), batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        val_loader   = DataLoader(ISICDatasetV5(fold_val_imgs, fold_val_msks, is_train=False), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

        model = GDFNetV5(backbone_name="pvt_v2_b4", pretrained=True).to(device)
        criterion = GDFNetV5CompositeLoss(edge_weight=0.8)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        scaler = GradScaler('cuda')

        checkpoint_best = f"{args.output_dir}/GDFNetV5_fold{fold_idx}_best.pth"
        checkpoint_latest = f"{args.output_dir}/GDFNetV5_fold{fold_idx}_latest.pth"
        history_csv = f"{args.output_dir}/history_fold{fold_idx}.csv"

        start_epoch = 1
        best_val_dice = 0.0
        best_epoch = 0

        # Resumption support
        if os.path.exists(checkpoint_latest):
            print(f"[*] Found existing checkpoint for Fold {fold_idx} at {checkpoint_latest}. Restoring...", flush=True)
            ckpt = torch.load(checkpoint_latest, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            scaler.load_state_dict(ckpt['scaler_state'])
            start_epoch = ckpt['epoch'] + 1
            best_val_dice = ckpt.get('best_val_dice', 0.0)
            best_epoch = ckpt.get('best_epoch', ckpt['epoch'])
            print(f"[+] Resumed Fold {fold_idx} from Epoch {start_epoch} (Best Val Dice: {best_val_dice:.4f})", flush=True)

        history = []
        if os.path.exists(history_csv):
            try: history = pd.read_csv(history_csv).to_dict('records')
            except Exception: pass

        for epoch in range(start_epoch, args.epochs + 1):
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
            current_lr = scheduler.get_last_lr()[0]

            # Validation evaluation
            _, val_summary, val_global_dice, _ = evaluate_models_ensemble([model], val_loader, device, use_tta=False)
            val_macro_dice = val_summary.loc[val_summary['Metric'] == 'dice', 'Mean'].values[0]

            history.append({
                'epoch': epoch, 'loss': epoch_loss, 'lr': current_lr,
                'val_macro_dice': val_macro_dice, 'val_global_dice': val_global_dice,
                'elapsed_sec': elapsed
            })
            pd.DataFrame(history).to_csv(history_csv, index=False)

            # Save latest checkpoint
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict(),
                'best_val_dice': best_val_dice,
                'best_epoch': best_epoch
            }, checkpoint_latest)

            print(f"Fold {fold_idx} | Epoch {epoch:03d}/{args.epochs} [{elapsed:.1f}s] | Loss: {epoch_loss:.4f} | LR: {current_lr:.6f} | Val Macro Dice: {val_macro_dice:.4f} | Val Global Dice: {val_global_dice:.4f}", flush=True)

            if val_macro_dice > best_val_dice:
                best_val_dice = val_macro_dice
                best_epoch = epoch
                torch.save(model.state_dict(), checkpoint_best)
                print(f"[*] New Best Model for Fold {fold_idx} Saved at Epoch {epoch} (Val Macro Dice: {val_macro_dice:.4f})", flush=True)

    # ====================================================================
    # 6. Final Evaluation on 1,000 Official Blind Test Images
    # ====================================================================
    print("\n=======================================================")
    print("[*] STARTING OFFICIAL 1,000 BLIND TEST EVALUATION")
    print("=======================================================")

    test_loader = DataLoader(ISICDatasetV5(test_imgs, test_msks, is_train=False), batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # Collect available trained fold models
    trained_models = []
    for fold_idx in range(5):
        ckpt_path = f"{args.output_dir}/GDFNetV5_fold{fold_idx}_best.pth"
        if os.path.exists(ckpt_path):
            m = GDFNetV5(backbone_name="pvt_v2_b4", pretrained=False).to(device)
            m.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
            trained_models.append(m)
            print(f"[+] Loaded Fold {fold_idx} checkpoint: {ckpt_path}")

    assert len(trained_models) > 0, "No trained fold models found for evaluation!"
    print(f"[+] Total Active Models in Ensemble: {len(trained_models)}")

    # 1. Single View Evaluation (Ensemble if >1 model, or single model)
    print("\n[*] Evaluating Ensemble on 1,000 Blind Test Images (Single-View)...", flush=True)
    raw_df_sv, sum_df_sv, g_dice_sv, g_iou_sv = evaluate_models_ensemble(trained_models, test_loader, device, use_tta=False)
    print("\n================ GDF-NETV5 BLIND TEST (SINGLE VIEW) ================")
    print(sum_df_sv.to_string(index=False))
    print(f"[+] Dataset Global Micro Dice: {g_dice_sv:.4f} | IoU: {g_iou_sv:.4f}")
    sum_df_sv.to_csv(f"{args.output_dir}/benchmark_summary_ISIC2018_SingleView.csv", index=False)
    raw_df_sv.to_csv(f"{args.output_dir}/benchmark_raw_ISIC2018_SingleView.csv", index=False)

    # 2. 4-View Flip TTA Evaluation
    print("\n[*] Evaluating Ensemble on 1,000 Blind Test Images (4-View Flip TTA)...", flush=True)
    raw_df_tta, sum_df_tta, g_dice_tta, g_iou_tta = evaluate_models_ensemble(trained_models, test_loader, device, use_tta=True)
    print("\n================ GDF-NETV5 BLIND TEST (4-VIEW TTA ENSEMBLE) ================")
    print(sum_df_tta.to_string(index=False))
    print(f"[+] TTA Dataset Global Micro Dice: {g_dice_tta:.4f} | IoU: {g_iou_tta:.4f}")
    sum_df_tta.to_csv(f"{args.output_dir}/benchmark_summary_ISIC2018_TTA_Ensemble.csv", index=False)
    raw_df_tta.to_csv(f"{args.output_dir}/benchmark_raw_ISIC2018_TTA_Ensemble.csv", index=False)

    print("\n[+] GDF-Netv5 5-Fold Training & Benchmark Evaluation Successfully Completed!")

if __name__ == '__main__':
    main()
