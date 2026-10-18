"""
GDF-Netv2 (Ultra-GDFNet): CVC-ClinicDB Benchmark Runner
Architecture: ConvNeXt-V2-Tiny + FDF + SBA + MSF
Loss: Composite Focal-Tversky + Boundary Difference over Union (BD-IoU)
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
from scipy.ndimage import binary_erosion, distance_transform_edt
from scipy.spatial.distance import directed_hausdorff

# 0. Dependencies setup for Kaggle
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

# ====================================================================
# 1. Hardware & Reproducibility
# ====================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[+] Device: {device} | CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[+] GPU Model: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

seed_everything(42)

# ====================================================================
# 2. GDF-Netv2 Architecture Components
# ====================================================================
class HierarchicalBackbone(nn.Module):
    def __init__(self, backbone_name="convnextv2_tiny", pretrained=True, in_chans=3):
        super().__init__()
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True
        )
        self.feature_info = self.encoder.feature_info.channels()

    def forward(self, x):
        feats = self.encoder(x)
        return feats[0], feats[1], feats[2], feats[3]

class FrequencyDecoupledFilter(nn.Module):
    def __init__(self, channels, cutoff_ratio=0.35):
        super().__init__()
        self.channels = channels
        self.cutoff_ratio = cutoff_ratio
        self.gamma_low = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.gamma_high = nn.Parameter(torch.ones(1, channels, 1, 1) * 0.1)

    def forward(self, x):
        B, C, H, W = x.shape
        X = torch.fft.rfft2(x, norm="ortho")
        u = torch.fft.fftfreq(H, device=x.device)
        v = torch.fft.rfftfreq(W, device=x.device)
        U, V = torch.meshgrid(u, v, indexing='ij')
        dist = torch.sqrt(U**2 + V**2).unsqueeze(0).unsqueeze(0)
        mask = (dist <= self.cutoff_ratio).float()

        X_low = X * mask
        X_high = X * (1.0 - mask)
        x_low = torch.fft.irfft2(X_low, s=(H, W), norm="ortho")
        x_high = torch.fft.irfft2(X_high, s=(H, W), norm="ortho")
        return self.gamma_low * x_low + self.gamma_high * x_high

class DifferentiableEdgeStream(nn.Module):
    def __init__(self, in_channels=3, edge_channels=32):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, edge_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(edge_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(edge_channels, edge_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(edge_channels),
            nn.ReLU(inplace=True)
        )
        self.edge_classifier = nn.Conv2d(edge_channels, 1, kernel_size=1)

    def forward(self, x):
        feat = self.stem(x)
        pred = self.edge_classifier(feat)
        return feat, pred

class SelectiveBoundaryAggregation(nn.Module):
    def __init__(self, decoder_channels, edge_channels):
        super().__init__()
        self.query_conv = nn.Conv2d(decoder_channels, decoder_channels // 4, kernel_size=1)
        self.key_conv   = nn.Conv2d(edge_channels, decoder_channels // 4, kernel_size=1)
        self.val_conv   = nn.Conv2d(edge_channels, decoder_channels, kernel_size=1)
        self.gamma      = nn.Parameter(torch.zeros(1))
        self.fuse_head  = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1)
        )

    def forward(self, dec_feat, edge_feat):
        B, C, H, W = dec_feat.shape
        if edge_feat.shape[2:] != (H, W):
            edge_feat = F.interpolate(edge_feat, size=(H, W), mode='bilinear', align_corners=False)
        proj_query = self.query_conv(dec_feat).view(B, -1, H * W).permute(0, 2, 1)
        proj_key   = self.key_conv(edge_feat).view(B, -1, H * W)
        energy     = torch.bmm(proj_query, proj_key)
        attention  = F.softmax(energy * ((C // 4) ** -0.5), dim=-1)
        proj_val   = self.val_conv(edge_feat).view(B, -1, H * W)
        out = torch.bmm(proj_val, attention.permute(0, 2, 1)).view(B, C, H, W)
        enhanced = dec_feat + self.gamma * out
        return self.fuse_head(enhanced)

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
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([x, skip], dim=1)
        out = self.conv(out)
        return out * self.ca(out)

class GDFNetV2(nn.Module):
    def __init__(self, backbone_name="convnextv2_tiny", pretrained=True, in_chans=3,
                 feedback_channels=32, decoder_channels=128):
        super().__init__()
        self.backbone = HierarchicalBackbone(backbone_name, pretrained, in_chans)
        c1, c2, c3, c4 = self.backbone.feature_info

        self.fdf1 = FrequencyDecoupledFilter(c1, cutoff_ratio=0.35)
        self.fdf2 = FrequencyDecoupledFilter(c2, cutoff_ratio=0.35)
        self.fdf3 = FrequencyDecoupledFilter(c3, cutoff_ratio=0.35)
        self.fdf4 = FrequencyDecoupledFilter(c4, cutoff_ratio=0.35)

        self.dec4_conv = nn.Sequential(
            nn.Conv2d(c4, decoder_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU(inplace=True)
        )
        self.dec3 = MultiScaleDecoderBlock(decoder_channels, c3, decoder_channels)
        self.dec2 = MultiScaleDecoderBlock(decoder_channels, c2, decoder_channels)
        self.dec1 = MultiScaleDecoderBlock(decoder_channels, c1, decoder_channels)

        self.edge_stream = DifferentiableEdgeStream(in_channels=in_chans, edge_channels=32)
        self.sba = SelectiveBoundaryAggregation(decoder_channels, edge_channels=32)

        self.feedback_adapter = nn.Sequential(
            nn.Conv2d(decoder_channels, feedback_channels, kernel_size=1),
            nn.BatchNorm2d(feedback_channels),
            nn.ReLU(inplace=True)
        )
        self.feedback_gate = nn.Sequential(
            nn.Conv2d(c1 + feedback_channels, c1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.aux_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1)
        )

    def _decode(self, s1, s2, s3, s4):
        d4 = self.dec4_conv(self.fdf4(s4))
        d3 = self.dec3(d4, self.fdf3(s3))
        d2 = self.dec2(d3, self.fdf2(s2))
        return self.dec1(d2, self.fdf1(s1))

    def forward(self, x):
        B, _, H, W = x.shape
        edge_feat, edge_pred = self.edge_stream(x)
        s1, s2, s3, s4 = self.backbone(x)

        d1_t1 = self._decode(s1, s2, s3, s4)
        pred_t1 = F.interpolate(self.aux_head(d1_t1), size=(H, W), mode='bilinear', align_corners=False)

        fb_feat = self.feedback_adapter(d1_t1).detach()
        if fb_feat.shape[2:] != s1.shape[2:]:
            fb_feat = F.interpolate(fb_feat, size=s1.shape[2:], mode='bilinear', align_corners=False)
        gate = self.feedback_gate(torch.cat([s1, fb_feat], dim=1))
        s1_refined = s1 + (s1 * gate)

        d1_t2 = self._decode(s1_refined, s2, s3, s4)
        final_pred = self.sba(d1_t2, edge_feat)

        if self.training:
            return {'final_pred': final_pred, 'pred_t1': pred_t1, 'edge_pred': edge_pred}
        return final_pred

# ====================================================================
# 3. Boundary & Region Losses
# ====================================================================
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1e-5):
        super().__init__()
        self.alpha, self.beta, self.gamma, self.smooth = alpha, beta, gamma, smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        tp = (probs * targets).sum()
        fp = (probs * (1.0 - targets)).sum()
        fn = ((1.0 - probs) * targets).sum()
        ti = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1.0 - ti, 1.0 / self.gamma)

class BoundaryDifferenceIoULoss(nn.Module):
    def __init__(self, kernel_size=5, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        self.dilation = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        dilated = self.dilation(targets)
        eroded = 1.0 - self.dilation(1.0 - targets)
        band = (dilated - eroded).detach()

        p_band = probs * band
        g_band = targets * band
        intersection = (p_band * g_band).sum(dim=(2, 3))
        union = (p_band + g_band - (p_band * g_band)).sum(dim=(2, 3))
        return (1.0 - (intersection + self.smooth) / (union + self.smooth)).mean()

class CompositeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ftl = FocalTverskyLoss(alpha=0.7, beta=0.3, gamma=1.33)
        self.bdiou = BoundaryDifferenceIoULoss(kernel_size=5)
        self.bce = nn.BCEWithLogitsLoss()
        self.dilation = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, outputs, targets):
        p_final = outputs['final_pred']
        p_t1 = outputs['pred_t1']
        p_edge = outputs['edge_pred']

        loss_final = self.ftl(p_final, targets) + 0.5 * self.bdiou(p_final, targets)
        loss_t1 = self.ftl(p_t1, targets)
        edge_gt = self.dilation(targets) - (1.0 - self.dilation(1.0 - targets))
        loss_edge = self.bce(p_edge, edge_gt)
        return loss_final + 0.4 * loss_t1 + 0.3 * loss_edge

# ====================================================================
# 4. Dataset Discovery & Transforms
# ====================================================================
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

    exts = ('*.png', '*.jpg', '*.tif', '*.bmp')
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

IMG_SIZE = 384

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=30, p=0.5),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, p=0.5),
    A.ElasticTransform(alpha=1.0, sigma=50, p=0.3),
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 32), hole_width_range=(8, 32), fill=0, p=0.25)
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
        img = np.array(Image.open(self.imgs[idx]).convert('RGB'), dtype=np.float32) / 255.0
        msk = np.array(Image.open(self.msks[idx]).convert('L'), dtype=np.float32)
        msk = (msk > 127.0).astype(np.float32)

        if self.transform is not None:
            aug = self.transform(image=img, mask=msk)
            img, msk = aug['image'], aug['mask']

        img = np.transpose(img, (2, 0, 1))
        msk = np.expand_dims(msk, axis=0)
        return torch.from_numpy(img).float(), torch.from_numpy(msk).float()

# ====================================================================
# 5. Full Evaluation Function (9 Metrics)
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

    if t.sum() == 0 and p.sum() == 0:
        hd95, asd, bf2 = 0.0, 0.0, 1.0
    elif t.sum() == 0 or p.sum() == 0:
        diag = np.sqrt(t.shape[0]**2 + t.shape[1]**2)
        hd95, asd, bf2 = float(diag), float(diag / 2.0), 0.0
    else:
        bg = t ^ binary_erosion(t, iterations=1)
        bp = p ^ binary_erosion(p, iterations=1)
        if len(np.argwhere(bg)) == 0 or len(np.argwhere(bp)) == 0:
            hd95, asd, bf2 = 0.0, 0.0, 0.0
        else:
            d1 = directed_hausdorff(np.argwhere(bg), np.argwhere(bp))[0]
            d2 = directed_hausdorff(np.argwhere(bp), np.argwhere(bg))[0]
            hd95 = max(d1, d2) * 0.95
            dt_g = distance_transform_edt(~bg)
            dt_p = distance_transform_edt(~bp)
            asd = (dt_g[bp].mean() + dt_p[bg].mean()) / 2.0
            prec_b = (dt_g[bp] <= 2).sum() / bp.sum()
            rec_b  = (dt_p[bg] <= 2).sum() / bg.sum()
            bf2 = 2 * prec_b * rec_b / (prec_b + rec_b + 1e-8)

    return {'dice': dice, 'miou': miou, 'sensitivity': sens, 'specificity': spec, 'precision': prec, 'fpr': fpr, 'hd95': hd95, 'asd': asd, 'bf2': bf2}

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
        summary.append({'Metric': m, 'Mean': df[m].mean(), 'Std': df[m].std(), 'Median': df[m].median(), 'Min': df[m].min(), 'Max': df[m].max()})
    return df, pd.DataFrame(summary)

# ====================================================================
# 6. Main Execution Pipeline
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

    print("=" * 65)
    print(f"[*] Starting GDF-Netv2 CVC-ClinicDB Training on {device} ({EPOCHS} Epochs)")
    print(f"[*] Resolution: {IMG_SIZE}x{IMG_SIZE} | Batch Size: 8 | Backbone: ConvNeXt-V2-Tiny")
    print("=" * 65)

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

        print(f"Epoch {epoch:03d}/{EPOCHS} [{elapsed:.1f}s] | Loss: {running_loss/len(train_loader):.4f} | Val Dice: {val_dice:.4f} | Val mIoU: {val_miou:.4f}")

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            torch.save(model.state_dict(), f"{OUTPUT_DIR}/GDFNetV2_CVC_best.pth")
            print(f"[*] New Best Model Saved (Val Dice: {val_dice:.4f})")

    torch.save(model.state_dict(), f"{OUTPUT_DIR}/GDFNetV2_CVC_latest.pth")

    # Final Benchmark on Test Set
    print("\n" + "=" * 65)
    print("[*] Loading Best Checkpoint for Final CVC-ClinicDB Test Benchmark...")
    print("=" * 65)
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/GDFNetV2_CVC_best.pth", map_location=device))
    per_img_df, summary_df = evaluate_model(model, test_loader)

    print("\n================ GDF-NETV2 CVC-CLINICDB TEST RESULTS ================")
    print(summary_df.to_string(index=False))

    summary_df.to_csv(f"{OUTPUT_DIR}/benchmark_results_GDFNetV2_CVC-ClinicDB.csv", index=False)
    per_img_df.to_csv(f"{OUTPUT_DIR}/per_image_results_GDFNetV2_CVC-ClinicDB.csv", index=False)
    print(f"[*] Results saved successfully in: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
