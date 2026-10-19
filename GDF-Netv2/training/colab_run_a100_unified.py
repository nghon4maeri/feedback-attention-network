"""
colab_run_a100_unified.py
High-Performance Dual-Benchmark Runner for GDF-Netv2 on Colab NVIDIA A100 GPU.
Benchmarks executed sequentially:
  Phase 1: ISIC-2018 (384x384, 150 epochs, 2594 train, 100 official val, 1000 official test)
  Phase 2: DSB-2018  (256x256, 150 epochs, 670 images merged & in-RAM cached)
Architecture: ConvNeXt-V2-Tiny + Spatial FDF + Linear SBA + MSF Gate
Loss: Composite Loss (BCE + Focal-Tversky + Boundary Difference IoU)
Protocol: Full 9 Clinical Metrics (Dice, mIoU, Sens, Spec, Prec, FPR, HD95, ASD, BF2)
"""

import os
import sys
import glob
import json
import time
import random
import zipfile
import subprocess
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt, binary_erosion
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import timm

# ====================================================================
# 0. Hardware & Optimization Configuration
# ====================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("=" * 70)
print(f"[+] Device: {device} | CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[+] GPU Hardware: {gpu_name} ({vram_gb:.2f} GB VRAM)")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
print("=" * 70)

OUTPUT_DIR = "/content/output_gdfnet_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Kaggle credentials
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
    json.dump({"username": "namnguynnnn", "key": "KGAT_dad33ce94fb8b25d8a9ecd4ca26ce03f"}, f)
os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ====================================================================
# 1. Architecture: GDF-Netv2 Components
# ====================================================================
class HierarchicalBackbone(nn.Module):
    def __init__(self, backbone_name="convnextv2_tiny", pretrained=True):
        super().__init__()
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=3,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        self.feature_channels = self.encoder.feature_info.channels()

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
    def __init__(self, out_channels=32):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.], [ 0.,  0.,  0.], [ 1.,  2.,  1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer('sobel_x', sobel_x.repeat(1, 3, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(1, 3, 1, 1))

        self.edge_proj = nn.Sequential(
            nn.Conv2d(2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.edge_head = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x):
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        grad_mag = torch.cat([gx, gy], dim=1)
        edge_feat = self.edge_proj(grad_mag)
        edge_pred = self.edge_head(edge_feat)
        return edge_feat, edge_pred

class SpatialBoundaryAttention(nn.Module):
    """Linear O(HW) edge-guided spatial boundary attention"""
    def __init__(self, in_channels, edge_channels=32):
        super().__init__()
        self.edge_proj = nn.Sequential(
            nn.Conv2d(edge_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, edge_feat):
        if edge_feat.shape[2:] != x.shape[2:]:
            edge_feat = F.interpolate(edge_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        gate = self.edge_proj(edge_feat)
        return self.refine(x * (1.0 + gate))

class MultiScaleDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(out_channels // 4, 16), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_channels // 4, 16), out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        feat = torch.cat([x, skip], dim=1)
        feat = self.conv(feat)
        return feat * self.ca(feat)

class MultiScaleFusionGate(nn.Module):
    def __init__(self, in_channels_list, out_channels=32):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            ) for c in in_channels_list
        ])
        total_channels = out_channels * len(in_channels_list)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, feats, target_size):
        proj_feats = []
        for feat, proj in zip(feats, self.projs):
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            proj_feats.append(proj(feat))
        return self.fusion(torch.cat(proj_feats, dim=1))

class GDFNetV2(nn.Module):
    def __init__(self, backbone_name="convnextv2_tiny", pretrained=True):
        super().__init__()
        self.backbone = HierarchicalBackbone(backbone_name, pretrained=pretrained)
        c0, c1, c2, c3 = self.backbone.feature_channels

        self.fdf0 = FrequencyDecoupledFilter(c0)
        self.fdf1 = FrequencyDecoupledFilter(c1)
        self.fdf2 = FrequencyDecoupledFilter(c2)
        self.fdf3 = FrequencyDecoupledFilter(c3)

        self.edge_stream = DifferentiableEdgeStream(out_channels=32)

        self.dec3 = MultiScaleDecoderBlock(c3, c2, 128)
        self.dec2 = MultiScaleDecoderBlock(128, c1, 64)
        self.dec1 = MultiScaleDecoderBlock(64, c0, 32)

        self.sba = SpatialBoundaryAttention(in_channels=32, edge_channels=32)
        self.msf = MultiScaleFusionGate([c3, 128, 64, 32], out_channels=32)

        self.head = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1)
        )

    def forward(self, x):
        target_size = x.shape[2:]
        e0, e1, e2, e3 = self.backbone(x)

        f0 = self.fdf0(e0)
        f1 = self.fdf1(e1)
        f2 = self.fdf2(e2)
        f3 = self.fdf3(e3)

        edge_feat, _ = self.edge_stream(x)

        d3 = self.dec3(f3, f2)
        d2 = self.dec2(d3, f1)
        d1 = self.dec1(d2, f0)

        d1 = self.sba(d1, edge_feat)
        msf_feat = self.msf([f3, d3, d2, d1], target_size=d1.shape[2:])

        out = self.head(torch.cat([d1, msf_feat], dim=1))
        if out.shape[2:] != target_size:
            out = F.interpolate(out, size=target_size, mode='bilinear', align_corners=False)
        return out

# ====================================================================
# 2. Loss Functions: Composite Loss
# ====================================================================
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()
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
        return self.bce_w * self.bce(logits, targets) + \
               self.ft_w * self.ft(logits, targets) + \
               self.bdiou_w * self.bdiou(logits, targets)

# ====================================================================
# 3. Clinical 9-Metric Protocol
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

    hd95 = 0.0
    asd = 0.0
    bf2 = 0.0

    if t.sum() > 0 and p.sum() > 0:
        bg = t ^ binary_erosion(t)
        bp = p ^ binary_erosion(p)
        if bg.sum() > 0 and bp.sum() > 0:
            d1 = directed_hausdorff(np.argwhere(bg), np.argwhere(bp))[0]
            d2 = directed_hausdorff(np.argwhere(bp), np.argwhere(bg))[0]
            hd95 = max(d1, d2) * 0.95
            dt_g = distance_transform_edt(~bg)
            dt_p = distance_transform_edt(~bp)
            asd = (dt_g[bp].mean() + dt_p[bg].mean()) / 2.0
            prec_b = (dt_g[bp] <= 2).sum() / bp.sum()
            rec_b  = (dt_p[bg] <= 2).sum() / bg.sum()
            bf2 = 2 * prec_b * rec_b / (prec_b + rec_b + 1e-8)

    return {'dice': dice, 'miou': miou, 'sensitivity': sens, 'specificity': spec,
            'precision': prec, 'fpr': fpr, 'hd95': hd95, 'asd': asd, 'bf2': bf2}

@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    records = []
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
        summary.append({'Metric': m, 'Mean': df[m].mean(), 'Std': df[m].std(),
                        'Median': df[m].median(), 'Min': df[m].min(), 'Max': df[m].max()})
    return df, pd.DataFrame(summary)

# ====================================================================
# 4. BENCHMARK PHASE 1: ISIC-2018 (384x384, 150 Epochs)
# ====================================================================
def run_isic2018_benchmark():
    print("\n" + "=" * 70)
    print("  PHASE 1: GDF-Netv2 ON ISIC-2018 (Skin Lesion Segmentation)")
    print("  Resolution: 384x384 | 150 Epochs | Batch Size: 16 | AMP + TF32")
    print("=" * 70)

    DATA_ROOT = "/content/isic-2018"
    os.makedirs(DATA_ROOT, exist_ok=True)

    has_train = any("ISIC2018_Task1-2_Training_Input" in root for root, dirs, _ in os.walk(DATA_ROOT))
    has_test  = any("ISIC2018_Task1_Test_GroundTruth" in root for root, dirs, _ in os.walk(DATA_ROOT))
    if not (has_train and has_test):
        print("[*] Downloading ISIC-2018 Datasets via Kaggle API...")
        subprocess.run(["kaggle", "datasets", "download", "-d",
                        "tschandl/isic2018-challenge-task1-data-segmentation", "-p", DATA_ROOT, "--unzip"], check=True)
        subprocess.run(["kaggle", "datasets", "download", "-d",
                        "mfahimhossen/isic2018-test-groundtruth", "-p", DATA_ROOT, "--unzip"], check=True)

    has_val_img = any("ISIC2018_Task1-2_Validation_Input" in root for root, dirs, _ in os.walk(DATA_ROOT))
    has_val_gt  = any("ISIC2018_Task1_Validation_GroundTruth" in root for root, dirs, _ in os.walk(DATA_ROOT))
    if not has_val_img:
        print("[*] Downloading Official Val Input from S3...")
        subprocess.run(["wget", "-q", "-O", f"{DATA_ROOT}/val_input.zip",
                        "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip"], check=True)
        with zipfile.ZipFile(f"{DATA_ROOT}/val_input.zip", 'r') as z:
            z.extractall(DATA_ROOT)
    if not has_val_gt:
        print("[*] Downloading Official Val GT from S3...")
        subprocess.run(["wget", "-q", "-O", f"{DATA_ROOT}/val_gt.zip",
                        "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip"], check=True)
        with zipfile.ZipFile(f"{DATA_ROOT}/val_gt.zip", 'r') as z:
            z.extractall(DATA_ROOT)

    def find_dir(target_name):
        for root, dirs, _ in os.walk(DATA_ROOT):
            if os.path.basename(root).lower() == target_name.lower():
                return root
        return None

    train_img_dir = find_dir("ISIC2018_Task1-2_Training_Input")
    train_gt_dir  = find_dir("ISIC2018_Task1_Training_GroundTruth")
    val_img_dir   = find_dir("ISIC2018_Task1-2_Validation_Input")
    val_gt_dir    = find_dir("ISIC2018_Task1_Validation_GroundTruth")
    test_img_dir  = find_dir("ISIC2018_Task1-2_Test_Input")
    test_gt_dir   = find_dir("ISIC2018_Task1_Test_GroundTruth")

    def isic_id(path):
        base = os.path.basename(path).replace("_segmentation", "")
        return os.path.splitext(base)[0]

    def pair_isic(img_dir, gt_dir):
        imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
        msks = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
        id_map = {isic_id(m): m for m in msks}
        px = [img for img in imgs if isic_id(img) in id_map]
        py = [id_map[isic_id(img)] for img in px]
        return px, py

    train_x, train_y = pair_isic(train_img_dir, train_gt_dir)
    val_x, val_y     = pair_isic(val_img_dir, val_gt_dir)
    test_x, test_y   = pair_isic(test_img_dir, test_gt_dir)
    print(f"[+] ISIC-2018 -> Train: {len(train_x)} | Val (Official): {len(val_x)} | Test (Official): {len(test_x)}")

    ISIC_SIZE = 384

    class ISICDataset(Dataset):
        def __init__(self, imgs, msks, is_train=False):
            self.imgs = imgs
            self.msks = msks
            self.is_train = is_train
            self.train_tf = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.7),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.4)
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

            img = cv2.resize(img, (ISIC_SIZE, ISIC_SIZE), interpolation=cv2.INTER_LINEAR)
            msk = cv2.resize(msk, (ISIC_SIZE, ISIC_SIZE), interpolation=cv2.INTER_NEAREST)

            if self.is_train:
                augmented = self.train_tf(image=img, mask=msk)
                img, msk = augmented['image'], augmented['mask']

            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            msk = np.expand_dims(msk, axis=0)
            return torch.from_numpy(img).float(), torch.from_numpy(msk).float()

    train_loader = DataLoader(ISICDataset(train_x, train_y, is_train=True), batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(ISICDataset(val_x, val_y, is_train=False), batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(ISICDataset(test_x, test_y, is_train=False), batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    model = GDFNetV2(backbone_name="convnextv2_tiny", pretrained=True).to(device)
    criterion = CompositeLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
    scaler = GradScaler('cuda')

    EPOCHS = 150
    best_val_dice = 0.0

    print(f"[*] Training GDF-Netv2 on ISIC-2018 for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
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

        elapsed = time.time() - t0
        _, val_summary = evaluate_model(model, val_loader)
        val_dice = val_summary.loc[val_summary['Metric'] == 'dice', 'Mean'].values[0]
        val_miou = val_summary.loc[val_summary['Metric'] == 'miou', 'Mean'].values[0]
        scheduler.step(val_dice)

        if epoch % 5 == 0 or epoch == 1 or val_dice > best_val_dice:
            print(f"[ISIC] Epoch {epoch:03d}/{EPOCHS} [{elapsed:.1f}s] | Loss: {running_loss/len(train_loader):.4f} | Val Dice: {val_dice:.4f} | Val mIoU: {val_miou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/GDFNetV2_ISIC2018_best.pth")
            print(f"[+] [ISIC] New Best Checkpoint Saved! (Val Dice: {val_dice:.4f})")

    # Final Blind Test Evaluation
    print("\n[*] Evaluating Best ISIC Checkpoint on 1,000 Official Blind Test Images...")
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/GDFNetV2_ISIC2018_best.pth", map_location=device))
    per_img_df, summary_df = evaluate_model(model, test_loader)
    print("\n================ ISIC-2018 OFFICIAL TEST RESULTS ================")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(f"{OUTPUT_DIR}/benchmark_results_GDFNetV2_ISIC2018_official_test.csv", index=False)
    per_img_df.to_csv(f"{OUTPUT_DIR}/per_image_results_GDFNetV2_ISIC2018_official_test.csv", index=False)
    print("[+] ISIC-2018 Benchmark Completed Successfully.")

# ====================================================================
# 5. BENCHMARK PHASE 2: DSB-2018 (256x256, 150 Epochs, In-RAM Caching)
# ====================================================================
def run_dsb2018_benchmark():
    print("\n" + "=" * 70)
    print("  PHASE 2: GDF-Netv2 ON DSB-2018 (Cell Nuclei Segmentation)")
    print("  Resolution: 256x256 | 150 Epochs | In-RAM Pre-caching | Batch Size: 16")
    print("=" * 70)

    DATA_DIR = "/content/dsb2018_data"
    WORK_DIR = "/content/dsb2018_work"
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(WORK_DIR, exist_ok=True)

    ORIG_DIR = os.path.join(WORK_DIR, "stage1_train")
    MERGED_DIR = os.path.join(WORK_DIR, "dsb2018_masks")

    if not os.path.exists(ORIG_DIR) or len(os.listdir(ORIG_DIR)) == 0:
        print("[*] Downloading DSB-2018 via Kaggle API...")
        subprocess.run(["kaggle", "datasets", "download", "-d",
                        "tanumoynandy/nuclei", "-p", DATA_DIR, "--unzip"], check=True)
        # Find stage1_train.zip
        zips = glob.glob(os.path.join(DATA_DIR, "**", "stage1_train.zip"), recursive=True)
        if zips:
            print(f"[*] Extracting: {zips[0]}...")
            os.makedirs(ORIG_DIR, exist_ok=True)
            with zipfile.ZipFile(zips[0], 'r') as z:
                z.extractall(ORIG_DIR)

    os.makedirs(MERGED_DIR, exist_ok=True)
    all_folders = sorted(os.listdir(ORIG_DIR))
    print(f"[*] Pre-processing & merging masks for {len(all_folders)} image folders...")

    DSB_SIZE = 256
    ram_images = []
    ram_masks = []

    for folder in all_folders:
        folder_path = os.path.join(ORIG_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        img_file = os.path.join(folder_path, "images", folder + ".png")
        mask_files = glob.glob(os.path.join(folder_path, "masks", "*.png"))
        if not os.path.exists(img_file) or len(mask_files) == 0:
            continue

        img = cv2.imread(img_file, cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.array(Image.open(img_file).convert('RGB'))

        merged = None
        for mf in mask_files:
            m = cv2.imread(mf, cv2.IMREAD_GRAYSCALE)
            if m is None:
                m = np.array(Image.open(mf).convert('L'))
            merged = m if merged is None else np.maximum(merged, m)

        merged = (merged > 127).astype(np.float32)

        # Pre-resize to 256x256 and cache in RAM
        img_resized = cv2.resize(img, (DSB_SIZE, DSB_SIZE), interpolation=cv2.INTER_LINEAR)
        msk_resized = cv2.resize(merged, (DSB_SIZE, DSB_SIZE), interpolation=cv2.INTER_NEAREST)

        ram_images.append(img_resized)
        ram_masks.append(msk_resized)

    n_samples = len(ram_images)
    print(f"[+] Loaded {n_samples} samples into RAM ({n_samples * 256 * 256 * 4 / 1e6:.1f} MB).")

    indices = list(range(n_samples))
    random.seed(42)
    random.shuffle(indices)

    train_n = int(n_samples * 0.8)
    val_n   = int(n_samples * 0.1)

    train_idx = indices[:train_n]
    val_idx   = indices[train_n:train_n + val_n]
    test_idx  = indices[train_n + val_n:]

    print(f"[+] DSB-2018 Splits -> Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    class RAMNucleiDataset(Dataset):
        def __init__(self, idxs, is_train=False):
            self.idxs = idxs
            self.is_train = is_train
            self.train_tf = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.6),
                A.RandomBrightnessContrast(p=0.3)
            ])

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, idx):
            i = self.idxs[idx]
            img = ram_images[i].copy()
            msk = ram_masks[i].copy()

            if self.is_train:
                augmented = self.train_tf(image=img, mask=msk)
                img, msk = augmented['image'], augmented['mask']

            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            msk = np.expand_dims(msk, axis=0)
            return torch.from_numpy(img).float(), torch.from_numpy(msk).float()

    train_loader = DataLoader(RAMNucleiDataset(train_idx, is_train=True), batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(RAMNucleiDataset(val_idx, is_train=False), batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(RAMNucleiDataset(test_idx, is_train=False), batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

    model = GDFNetV2(backbone_name="convnextv2_tiny", pretrained=True).to(device)
    criterion = CompositeLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
    scaler = GradScaler('cuda')

    EPOCHS = 150
    best_val_dice = 0.0

    print(f"[*] Training GDF-Netv2 on DSB-2018 for {EPOCHS} epochs...")
    for epoch in range(1, EPOCHS + 1):
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

        elapsed = time.time() - t0
        _, val_summary = evaluate_model(model, val_loader)
        val_dice = val_summary.loc[val_summary['Metric'] == 'dice', 'Mean'].values[0]
        val_miou = val_summary.loc[val_summary['Metric'] == 'miou', 'Mean'].values[0]
        scheduler.step(val_dice)

        if epoch % 5 == 0 or epoch == 1 or val_dice > best_val_dice:
            print(f"[DSB] Epoch {epoch:03d}/{EPOCHS} [{elapsed:.1f}s] | Loss: {running_loss/len(train_loader):.4f} | Val Dice: {val_dice:.4f} | Val mIoU: {val_miou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/GDFNetV2_DSB2018_best.pth")
            print(f"[+] [DSB] New Best Checkpoint Saved! (Val Dice: {val_dice:.4f})")

    # Final Test Evaluation
    print("\n[*] Evaluating Best DSB Checkpoint on Test Set...")
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/GDFNetV2_DSB2018_best.pth", map_location=device))
    per_img_df, summary_df = evaluate_model(model, test_loader)
    print("\n================ DSB-2018 TEST RESULTS ================")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(f"{OUTPUT_DIR}/benchmark_results_GDFNetV2_DSB2018_test.csv", index=False)
    per_img_df.to_csv(f"{OUTPUT_DIR}/per_image_results_GDFNetV2_DSB2018_test.csv", index=False)
    print("[+] DSB-2018 Benchmark Completed Successfully.")

# ====================================================================
# Main Execution
# ====================================================================
if __name__ == "__main__":
    t_start = time.time()
    run_isic2018_benchmark()
    run_dsb2018_benchmark()
    total_elapsed = (time.time() - t_start) / 60.0
    print("\n" + "=" * 70)
    print(f"[+] ALL BENCHMARKS COMPLETED IN {total_elapsed:.2f} MINUTES!")
    print(f"[+] All Checkpoints and CSVs Saved to: {OUTPUT_DIR}")
    print("=" * 70)
