"""
GDF-Netv2 (Ultra-GDFNet): CVC-ClinicDB Polyp Segmentation Runner
Architecture: ConvNeXt-V2-Tiny + Spatial FDF + SBA + MSF
Loss: Composite Focal-Tversky + Boundary Difference over Union (BD-IoU)
Metrics: 9-metric clinical protocol (Dice, mIoU, Sens, Spec, Prec, FPR, HD95, ASD, BF2)
"""

import os
import sys
import time
import glob
import math
import random
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial.distance import directed_hausdorff

# Dependencies setup for Kaggle
try:
    import timm
    import albumentations as A
except ImportError:
    subprocess.run(["pip", "install", "-q", "timm", "albumentations", "opencv-python-headless"], check=True)
    import timm
    import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

# Hardware setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[+] Device: {device} | CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[+] GPU Model: {torch.cuda.get_device_name(0)}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ====================================================================
# Architecture: GDF-Netv2
# ====================================================================
class HierarchicalBackbone(nn.Module):
    def __init__(self, name="convnextv2_tiny", pretrained=True, in_chans=3):
        super().__init__()
        self.encoder = timm.create_model(
            name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        self.feature_info = self.encoder.feature_info.channels()

    def forward(self, x):
        feats = self.encoder(x)
        return feats[0], feats[1], feats[2], feats[3]

class FrequencyDecoupledFilter(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        mid_channels = max(in_channels // reduction, 16)
        self.low_pass = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )
        self.hf_gate = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, groups=mid_channels),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x_low = self.low_pass(x)
        x_high = x - x_low
        gate = self.hf_gate(x_low)
        out = self.fusion(torch.cat([x_low, x_high * gate], dim=1))
        return out + x

class DifferentiableEdgeStream(nn.Module):
    def __init__(self, in_channels=3, edge_channels=32):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x.repeat(in_channels, 1, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(in_channels, 1, 1, 1))
        self.in_channels = in_channels
        self.edge_proj = nn.Sequential(
            nn.Conv2d(in_channels, edge_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(edge_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=self.in_channels)
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=self.in_channels)
        grad_mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
        return self.edge_proj(grad_mag)

class SpatialBoundaryAttention(nn.Module):
    def __init__(self, in_channels, edge_channels=32):
        super().__init__()
        self.edge_conv = nn.Sequential(
            nn.Conv2d(edge_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feat, edge):
        if edge.shape[-2:] != feat.shape[-2:]:
            edge = F.interpolate(edge, size=feat.shape[-2:], mode='bilinear', align_corners=False)
        e_weight = self.edge_conv(edge)
        feat_edge = feat * (1.0 + e_weight)
        avg_out = torch.mean(feat_edge, dim=1, keepdim=True)
        max_out, _ = torch.max(feat_edge, dim=1, keepdim=True)
        sp_weight = self.spatial_gate(torch.cat([avg_out, max_out], dim=1))
        out = feat_edge * sp_weight
        return self.proj(out) + feat

class MultiScaleFusionGate(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels * 2, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, deep, shallow):
        if deep.shape[-2:] != shallow.shape[-2:]:
            deep = F.interpolate(deep, size=shallow.shape[-2:], mode='bilinear', align_corners=False)
        cat = torch.cat([deep, shallow], dim=1)
        w = self.gate(cat)
        w_deep, w_shallow = torch.split(w, deep.shape[1], dim=1)
        refined = torch.cat([deep * w_deep, shallow * w_shallow], dim=1)
        return self.out_conv(refined)

class GDFNetV2(nn.Module):
    def __init__(self, backbone_name="convnextv2_tiny", pretrained=True, in_chans=3, decoder_channels=64, edge_channels=32):
        super().__init__()
        self.backbone = HierarchicalBackbone(backbone_name, pretrained, in_chans)
        c1, c2, c3, c4 = self.backbone.feature_info

        self.fdf1 = FrequencyDecoupledFilter(c1)
        self.fdf2 = FrequencyDecoupledFilter(c2)
        self.fdf3 = FrequencyDecoupledFilter(c3)
        self.fdf4 = FrequencyDecoupledFilter(c4)

        self.dec4_conv = nn.Sequential(
            nn.Conv2d(c4, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        self.msf3 = MultiScaleFusionGate(decoder_channels)
        self.msf2 = MultiScaleFusionGate(decoder_channels)
        self.msf1 = MultiScaleFusionGate(decoder_channels)

        self.proj3 = nn.Conv2d(c3, decoder_channels, kernel_size=1)
        self.proj2 = nn.Conv2d(c2, decoder_channels, kernel_size=1)
        self.proj1 = nn.Conv2d(c1, decoder_channels, kernel_size=1)

        self.edge_stream = DifferentiableEdgeStream(in_channels=in_chans, edge_channels=edge_channels)
        self.sba1 = SpatialBoundaryAttention(decoder_channels, edge_channels)
        self.sba2 = SpatialBoundaryAttention(decoder_channels, edge_channels)
        self.sba3 = SpatialBoundaryAttention(decoder_channels, edge_channels)

        self.seg_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1)
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        e_feat = self.edge_stream(x)
        e1, e2, e3, e4 = self.backbone(x)

        f1 = self.fdf1(e1)
        f2 = self.fdf2(e2)
        f3 = self.fdf3(e3)
        f4 = self.fdf4(e4)

        d4 = self.dec4_conv(f4)
        p3 = self.sba3(self.proj3(f3), e_feat)
        d3 = self.msf3(d4, p3)

        p2 = self.sba2(self.proj2(f2), e_feat)
        d2 = self.msf2(d3, p2)

        p1 = self.sba1(self.proj1(f1), e_feat)
        d1 = self.msf1(d2, p1)

        out_seg = self.seg_head(d1)
        out_seg = F.interpolate(out_seg, size=(h, w), mode='bilinear', align_corners=False)
        return out_seg

# ====================================================================
# Loss Functions
# ====================================================================
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=4.0/3.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs_flat) * targets_flat).sum()

        tversky = (tp + self.eps) / (tp + self.alpha * fn + self.beta * fp + self.eps)
        return torch.pow((1.0 - tversky), self.gamma)

class BoundaryDifferenceIoULoss(nn.Module):
    def __init__(self, kernel_size=3, eps=1e-6):
        super().__init__()
        self.kernel_size = kernel_size
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pad = self.kernel_size // 2
        p_eroded = -F.max_pool2d(-probs, kernel_size=self.kernel_size, stride=1, padding=pad)
        t_eroded = -F.max_pool2d(-targets, kernel_size=self.kernel_size, stride=1, padding=pad)

        p_boundary = probs - p_eroded
        t_boundary = targets - t_eroded

        inter = (p_boundary * t_boundary).sum(dim=(-2, -1))
        union = (p_boundary + t_boundary - p_boundary * t_boundary).sum(dim=(-2, -1))
        biou = (inter + self.eps) / (union + self.eps)
        return (1.0 - biou).mean()

class CompositeLoss(nn.Module):
    def __init__(self, bce_w=0.4, ft_w=0.3, bdiou_w=0.3):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ft = FocalTverskyLoss()
        self.bdiou = BoundaryDifferenceIoULoss()
        self.bce_w = bce_w
        self.ft_w = ft_w
        self.bdiou_w = bdiou_w

    def forward(self, logits, targets):
        l_bce = self.bce(logits, targets)
        l_ft = self.ft(logits, targets)
        l_bdiou = self.bdiou(logits, targets)
        return self.bce_w * l_bce + self.ft_w * l_ft + self.bdiou_w * l_bdiou

# ====================================================================
# Data Pipeline: CVC-ClinicDB
# ====================================================================
IMG_SIZE = 384

def discover_cvc_paths():
    base_search = "/kaggle/input"
    img_dir, msk_dir = None, None
    for root, dirs, files in os.walk(base_search):
        dirs_lower = [d.lower() for d in dirs]
        if any(x in dirs_lower for x in ['original', 'images']) and any(x in dirs_lower for x in ['ground truth', 'masks']):
            for d in dirs:
                if d.lower() in ['original', 'images']:
                    img_dir = os.path.join(root, d)
                elif d.lower() in ['ground truth', 'masks']:
                    msk_dir = os.path.join(root, d)
            break

    if img_dir is None or msk_dir is None:
        raise FileNotFoundError(f"Could not automatically locate CVC-ClinicDB in {base_search}")

    exts = ('*.png', '*.jpg', '*.tif', '*.tiff', '*.bmp')
    imgs, msks = [], []
    for ext in exts:
        imgs.extend(glob.glob(os.path.join(img_dir, ext)))
        msks.extend(glob.glob(os.path.join(msk_dir, ext)))

    imgs = sorted(imgs)
    msks = sorted(msks)
    min_len = min(len(imgs), len(msks))
    imgs, msks = imgs[:min_len], msks[:min_len]

    # Deterministic split: 490 Train, 61 Val, 61 Test
    combined = list(zip(imgs, msks))
    random.seed(42)
    random.shuffle(combined)
    shuffled_imgs, shuffled_msks = zip(*combined)

    train_imgs = list(shuffled_imgs[:490])
    train_msks = list(shuffled_msks[:490])

    val_imgs   = list(shuffled_imgs[490:490+61])
    val_msks   = list(shuffled_msks[490:490+61])

    test_imgs  = list(shuffled_imgs[490+61:])
    test_msks  = list(shuffled_msks[490+61:])

    print(f"[+] CVC-ClinicDB Matched Pairs: {len(imgs)}")
    print(f"[+] Splits -> Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")
    return (train_imgs, train_msks), (val_imgs, val_msks), (test_imgs, test_msks)

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=30, p=0.5),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, p=0.5),
    A.ElasticTransform(alpha=1.0, sigma=50, p=0.3)
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE)
])

class MedicalDataset(Dataset):
    def __init__(self, imgs, msks, transform=None):
        self.imgs = imgs
        self.msks = msks
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_bgr = cv2.imread(self.imgs[idx], cv2.IMREAD_COLOR)
        if img_bgr is not None:
            img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            img = np.array(Image.open(self.imgs[idx]).convert('RGB'), dtype=np.float32) / 255.0

        msk_gray = cv2.imread(self.msks[idx], cv2.IMREAD_GRAYSCALE)
        if msk_gray is not None:
            msk = (msk_gray.astype(np.float32) > 127.0).astype(np.float32)
        else:
            msk = np.array(Image.open(self.msks[idx]).convert('L'), dtype=np.float32)
            msk = (msk > 127.0).astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=img, mask=msk)
            img, msk = aug['image'], aug['mask']

        img = np.transpose(img, (2, 0, 1))
        msk = np.expand_dims(msk, axis=0)
        return torch.from_numpy(img).float(), torch.from_numpy(msk).float()

# ====================================================================
# Evaluation Function (9 Metrics)
# ====================================================================
def compute_metrics(p, t):
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

    p_b = p ^ binary_erosion(p)
    t_b = t ^ binary_erosion(t)
    p_pts = np.argwhere(p_b)
    t_pts = np.argwhere(t_b)

    if len(p_pts) > 0 and len(t_pts) > 0:
        d1 = directed_hausdorff(p_pts, t_pts)[0]
        d2 = directed_hausdorff(t_pts, p_pts)[0]
        hd95 = max(d1, d2)

        dt_t = distance_transform_edt(~t_b)
        dt_p = distance_transform_edt(~p_b)
        asd = (dt_t[p_b].sum() + dt_p[t_b].sum()) / (p_b.sum() + t_b.sum() + 1e-8)

        dt_target = distance_transform_edt(~t_b)
        prec_b = (dt_target[p_b] <= 2.0).sum() / (p_b.sum() + 1e-8)
        dt_pred = distance_transform_edt(~p_b)
        rec_b = (dt_pred[t_b] <= 2.0).sum() / (t_b.sum() + 1e-8)
        bf2 = (2.0 * prec_b * rec_b) / (prec_b + rec_b + 1e-8)
    else:
        hd95 = 50.0 if (len(p_pts) != 0 or len(t_pts) != 0) else 0.0
        asd = 25.0 if (len(p_pts) != 0 or len(t_pts) != 0) else 0.0
        bf2 = 0.0 if (len(p_pts) != 0 or len(t_pts) != 0) else 1.0

    return {
        'Dice': float(dice), 'mIoU': float(miou), 'Sensitivity': float(sens),
        'Specificity': float(spec), 'Precision': float(prec), 'FPR': float(fpr),
        'HD95': float(hd95), 'ASD': float(asd), 'BF2': float(bf2)
    }

def evaluate(model, loader):
    model.eval()
    records = []
    with torch.no_grad():
        for imgs, msks in loader:
            imgs = imgs.to(device)
            with autocast('cuda'):
                preds = torch.sigmoid(model(imgs)).cpu().numpy()[:, 0]
            msks = msks.numpy()[:, 0]

            for i in range(len(preds)):
                records.append(compute_metrics(preds[i] > 0.5, msks[i] > 0.5))

    df = pd.DataFrame(records)
    summary = []
    for m in df.columns:
        summary.append({'Metric': m, 'Mean': df[m].mean(), 'Std': df[m].std(), 'Median': df[m].median(), 'Min': df[m].min(), 'Max': df[m].max()})
    return df, pd.DataFrame(summary)

# ====================================================================
# Main Execution Pipeline
# ====================================================================
def main():
    OUTPUT_DIR = "/kaggle/working"
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = discover_cvc_paths()

    train_loader = DataLoader(MedicalDataset(train_x, train_y, train_transform), batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(MedicalDataset(val_x, val_y, val_transform), batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(MedicalDataset(test_x, test_y, val_transform), batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

    model = GDFNetV2(backbone_name="convnextv2_tiny", pretrained=True).to(device)
    criterion = CompositeLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
    scaler = GradScaler('cuda')

    EPOCHS = 150
    best_val_dice = 0.0
    history = []

    print("\n" + "=" * 65)
    print(f"[*] Starting GDF-Netv2 CVC-ClinicDB Training on {device} ({EPOCHS} Epochs)")
    print(f"[*] Resolution: {IMG_SIZE}x{IMG_SIZE} | Batch Size: 8 | Backbone: ConvNeXt-V2-Tiny")
    print("=" * 65 + "\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        n_batches = 0

        for imgs, msks in train_loader:
            imgs, msks = imgs.to(device), msks.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                preds = model(imgs)
                loss = criterion(preds, msks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)

        # Validation
        model.eval()
        val_dice = 0.0
        val_batches = 0
        with torch.no_grad():
            for imgs, msks in val_loader:
                imgs, msks = imgs.to(device), msks.to(device)
                with autocast('cuda'):
                    probs = torch.sigmoid(model(imgs))
                p = (probs > 0.5).float()
                inter = (p * msks).sum()
                d = (2.0 * inter + 1e-6) / (p.sum() + msks.sum() + 1e-6)
                val_dice += d.item()
                val_batches += 1

        val_dice /= max(val_batches, 1)
        scheduler.step(val_dice)
        dur = time.time() - t0

        saved_msg = ""
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "GDFNetV2_CVC_best.pth"))
            saved_msg = f" -> [*] Saved Best (Val Dice: {val_dice:.4f})"

        print(f"Epoch {epoch:03d}/{EPOCHS} [{dur:.1f}s] | Loss: {train_loss:.4f} | Val Dice: {val_dice:.4f}{saved_msg}")
        history.append({'epoch': epoch, 'loss': train_loss, 'val_dice': val_dice, 'duration_sec': dur})

    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "training_history_CVC.csv"), index=False)
    print("\n[+] Training complete! Best Val Dice:", best_val_dice)

    # Test Evaluation
    best_ckpt = os.path.join(OUTPUT_DIR, "GDFNetV2_CVC_best.pth")
    if os.path.exists(best_ckpt):
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        print("[+] Loaded best checkpoint for 9-metric test evaluation.")

    raw_df, sum_df = evaluate(model, test_loader)
    raw_df.to_csv(os.path.join(OUTPUT_DIR, "benchmark_raw_CVC.csv"), index=False)
    sum_df.to_csv(os.path.join(OUTPUT_DIR, "benchmark_summary_CVC.csv"), index=False)

    print("\n" + "=" * 65)
    print("       GDF-Netv2 CVC-ClinicDB Clinical Benchmark Summary        ")
    print("=" * 65)
    print(sum_df.to_string(index=False))
    print("=" * 65)

if __name__ == '__main__':
    main()
