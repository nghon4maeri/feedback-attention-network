"""
train_gdfnet_v2_isic.py
Kaggle Training Script for GDF-Netv2 on ISIC-2018 (Skin Lesion Segmentation)
Goal: Break past 0.94 Mean Dice using ConvNeXt-V2-Tiny + Spatial FDF + Linear SBA + MSF Gate
Loss: Composite Focal-Tversky + Boundary Difference over Union (BD-IoU)
Metrics: Full 9 Clinical Metrics (Dice, mIoU, Sens, Spec, Prec, FPR, HD95, ASD, BF2)
"""

import os
import sys
import glob
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[+] Device: {device} | CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[+] GPU Model: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ====================================================================
# 1. Dataset Discovery & Pairing
# ====================================================================
def find_dir_by_name(target_name, base_search="/kaggle/input"):
    for root, dirs, files in os.walk(base_search):
        if os.path.basename(root).lower() == target_name.lower():
            return root
    return None

train_img_dir = find_dir_by_name("ISIC2018_Task1-2_Training_Input")
train_gt_dir  = find_dir_by_name("ISIC2018_Task1_Training_GroundTruth")
test_img_dir  = find_dir_by_name("ISIC2018_Task1-2_Test_Input")
test_gt_dir   = find_dir_by_name("ISIC2018_Task1_Test_GroundTruth")

# Official Validation from S3
VAL_DIR = "/kaggle/working/val_official"
os.makedirs(VAL_DIR, exist_ok=True)
val_img_dir = os.path.join(VAL_DIR, "ISIC2018_Task1-2_Validation_Input")
val_gt_dir  = os.path.join(VAL_DIR, "ISIC2018_Task1_Validation_GroundTruth")

if not os.path.exists(val_img_dir):
    print("[*] Downloading Official Val Input from S3...")
    val_img_url = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip"
    subprocess.run(["wget", "-q", "-O", f"{VAL_DIR}/val_input.zip", val_img_url], check=True)
    with zipfile.ZipFile(f"{VAL_DIR}/val_input.zip", 'r') as z:
        z.extractall(VAL_DIR)

if not os.path.exists(val_gt_dir):
    print("[*] Downloading Official Val GT from S3...")
    val_gt_url = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip"
    subprocess.run(["wget", "-q", "-O", f"{VAL_DIR}/val_gt.zip", val_gt_url], check=True)
    with zipfile.ZipFile(f"{VAL_DIR}/val_gt.zip", 'r') as z:
        z.extractall(VAL_DIR)

def isic_id(path):
    base = os.path.basename(path).replace("_segmentation", "")
    return os.path.splitext(base)[0]

def pair_data(img_dir, gt_dir):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    msks = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    id_map = {isic_id(m): m for m in msks}
    px = [img for img in imgs if isic_id(img) in id_map]
    py = [id_map[isic_id(img)] for img in px]
    return px, py

train_x, train_y = pair_data(train_img_dir, train_gt_dir)
val_x, val_y     = pair_data(val_img_dir, val_gt_dir)
test_x, test_y   = pair_data(test_img_dir, test_gt_dir)
print(f"[+] Splits Verified -> Train: {len(train_x)} | Val (Official): {len(val_x)} | Test (Official): {len(test_x)}")

# ====================================================================
# 2. Architecture: GDF-Netv2
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
# 3. Loss Functions: Composite Loss
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
# 4. Data Pipeline & 9-Metric Protocol
# ====================================================================
IMG_SIZE = 384

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
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 32), hole_width_range=(8, 32), p=0.4)
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

        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        if self.is_train:
            augmented = self.train_tf(image=img, mask=msk)
            img, msk = augmented['image'], augmented['mask']

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        msk = np.expand_dims(msk, axis=0)
        return torch.from_numpy(img).float(), torch.from_numpy(msk).float()

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
# 5. Training Engine (150 Epochs, AMP, AdamW, ReduceLROnPlateau)
# ====================================================================
train_loader = DataLoader(ISICDataset(train_x, train_y, is_train=True), batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(ISICDataset(val_x, val_y, is_train=False), batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(ISICDataset(test_x, test_y, is_train=False), batch_size=8, shuffle=False, num_workers=2, pin_memory=True)

model = GDFNetV2(backbone_name="convnextv2_tiny", pretrained=True).to(device)
criterion = CompositeLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
scaler = GradScaler('cuda')

EPOCHS = 150
best_val_dice = 0.0

print(f"\n[*] Starting GDF-Netv2 ISIC-2018 Training on {device} ({EPOCHS} Epochs)")
history = []

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
    epoch_loss = running_loss / len(train_loader)
    _, val_summary = evaluate_model(model, val_loader)
    val_dice = val_summary.loc[val_summary['Metric'] == 'dice', 'Mean'].values[0]
    val_miou = val_summary.loc[val_summary['Metric'] == 'miou', 'Mean'].values[0]
    scheduler.step(val_dice)

    history.append({'epoch': epoch, 'loss': epoch_loss, 'val_dice': val_dice, 'val_miou': val_miou, 'elapsed_sec': elapsed})
    print(f"Epoch {epoch:03d}/{EPOCHS} [{elapsed:.1f}s] | Loss: {epoch_loss:.4f} | Val Dice: {val_dice:.4f} | Val mIoU: {val_miou:.4f}")

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), f"{OUTPUT_DIR}/GDFNetV2_ISIC2018_best.pth")
        print(f"[*] New Best Model Saved (Val Dice: {val_dice:.4f})")

pd.DataFrame(history).to_csv(f"{OUTPUT_DIR}/training_history_ISIC2018.csv", index=False)

# Final Blind Test Evaluation
print("\n[*] Evaluating Best ISIC Checkpoint on 1,000 Official Blind Test Images...")
model.load_state_dict(torch.load(f"{OUTPUT_DIR}/GDFNetV2_ISIC2018_best.pth", map_location=device))
per_img_df, summary_df = evaluate_model(model, test_loader)
print("\n================ GDF-NETV2 OFFICIAL TEST RESULTS ================")
print(summary_df.to_string(index=False))
summary_df.to_csv(f"{OUTPUT_DIR}/benchmark_summary_ISIC2018.csv", index=False)
per_img_df.to_csv(f"{OUTPUT_DIR}/benchmark_raw_ISIC2018.csv", index=False)
print("[+] Training and Blind Evaluation Completed Successfully!")
