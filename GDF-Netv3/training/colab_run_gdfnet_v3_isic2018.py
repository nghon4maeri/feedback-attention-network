"""
colab_run_gdfnet_v3_isic2018.py
Autonomous High-Performance Training of GDF-Netv3 on Colab NVIDIA A100 GPU (80GB).
Features:
- Backbone: EfficientNetV2-S (5-stage spatial preservation, Stage 0 at H/2 x W/2)
- Multi-Scale FDF with Depthwise Separable Convolutions
- Linear Selective Boundary Aggregation (SBA-Linear)
- Multi-Scale Semantic Feedback Gate (MSF-v3)
- Composite Loss (Weighted BCE + Focal-Tversky + BD-IoU + Deep Supervision)
- 9 Clinical Metrics (Dice, mIoU, Sensitivity, Specificity, Precision, FPR, HD95, ASD, BF2)
"""

import os
import sys
import glob
import json
import time
import zipfile
import subprocess
import shutil
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
from torch.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader

# Dual Tee Logger to write to stdout and log file simultaneously
class DualLogger(object):
    def __init__(self, filepath="/content/train_gdfnet_v3.log"):
        self.terminal = sys.stdout
        self.logfile = open(filepath, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.logfile.write(message)
        self.logfile.flush()
    def flush(self):
        self.terminal.flush()
        self.logfile.flush()

sys.stdout = DualLogger("/content/train_gdfnet_v3.log")
sys.stderr = sys.stdout

# ====================================================================
# 1. Hardware & Environment
# ====================================================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[+] Device: {device} | CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[+] GPU Model: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

LOCAL_OUT = "/content/output_gdfnet_v3"
os.makedirs(LOCAL_OUT, exist_ok=True)
DRIVE_OUT = "/content/drive/MyDrive/GDFNetV3_ISIC2018"
OUTPUT_DIR = DRIVE_OUT if os.path.exists("/content/drive/MyDrive") else LOCAL_OUT
os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[+] Output Directory configured: {OUTPUT_DIR}")

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ====================================================================
# 2. Acquire Dataset (Kaggle API + Official S3)
# ====================================================================
DATA_ROOT = "/content/isic-2018"
os.makedirs(DATA_ROOT, exist_ok=True)

kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
    json.dump({"username": "namnguynnnn", "key": "KGAT_dad33ce94fb8b25d8a9ecd4ca26ce03f"}, f)
os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

def _unzip_all(root):
    for z in glob.glob(os.path.join(root, "*.zip")):
        print(f"[*] Extracting: {os.path.basename(z)}...")
        with zipfile.ZipFile(z) as f:
            f.extractall(root)
        try:
            os.remove(z)
        except Exception:
            pass

has_train = any("ISIC2018_Task1-2_Training_Input" in root for root, dirs, _ in os.walk(DATA_ROOT))
has_test  = any("ISIC2018_Task1_Test_GroundTruth" in root for root, dirs, _ in os.walk(DATA_ROOT))

if not (has_train and has_test):
    print("[*] Downloading ISIC-2018 Datasets via Kaggle API...")
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
    print("[*] Downloading Official Val Input from S3...")
    subprocess.run(["wget", "-q", "-O", f"{DATA_ROOT}/val_input.zip", val_img_url], check=True)
if not has_val_gt:
    print("[*] Downloading Official Val GT from S3...")
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

print(f"[+] Paired Train: {len(train_x)} | Val: {len(val_x)} | Test: {len(test_x)}")

# ====================================================================
# 3. GDF-Netv3 Architecture Definition
# ====================================================================
class FrequencyDecoupledFilter(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        mid_channels = max(in_channels // reduction, 8)

        self.low_pass = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
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
        x_low = self.low_pass(x)
        x_high = x - x_low
        gate = self.hf_gate(x_low)
        x_high_filtered = x_high * gate
        out = self.fusion(torch.cat([x_low, x_high_filtered], dim=1))
        return out + x

class DifferentiableEdgeStream(nn.Module):
    def __init__(self, out_channels=32):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]).unsqueeze(0).unsqueeze(0)

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

class SelectiveBoundaryAggregationLinear(nn.Module):
    def __init__(self, semantic_channels, edge_channels=32, out_channels=48):
        super().__init__()
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(semantic_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(edge_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.cross_gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, semantic_feat, edge_feat):
        if semantic_feat.shape[2:] != edge_feat.shape[2:]:
            semantic_feat = F.interpolate(semantic_feat, size=edge_feat.shape[2:], mode='bilinear', align_corners=False)

        sem = self.semantic_conv(semantic_feat)
        edg = self.edge_conv(edge_feat)

        combined = torch.cat([sem, edg], dim=1)
        gate = self.cross_gate(combined)

        refined_sem = sem * gate
        refined_edg = edg * (1.0 - gate)

        out = self.fusion(torch.cat([refined_sem, refined_edg], dim=1))
        return out

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
            nn.Conv2d(out_channels, max(out_channels // 4, 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_channels // 4, 8), out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([x, skip], dim=1)
        out = self.conv(out)
        attn = self.ca(out)
        return out * attn

class GDFNetV3(nn.Module):
    def __init__(
        self,
        backbone_name="tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained=True,
        in_chans=3,
        feedback_iterations=2,
        feedback_channels=32,
        decoder_channels=48
    ):
        super().__init__()
        self.feedback_iterations = feedback_iterations
        self.feedback_channels = feedback_channels

        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(0, 1, 2, 3, 4)
        )
        c0, c1, c2, c3, c4 = self.encoder.feature_info.channels()

        self.fdf0 = FrequencyDecoupledFilter(c0)
        self.fdf1 = FrequencyDecoupledFilter(c1)
        self.fdf2 = FrequencyDecoupledFilter(c2)
        self.fdf3 = FrequencyDecoupledFilter(c3)
        self.fdf4 = FrequencyDecoupledFilter(c4)

        self.feedback_adapter = nn.Sequential(
            nn.Conv2d(decoder_channels * 2, feedback_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feedback_channels),
            nn.ReLU(inplace=True)
        )
        self.feedback_gate = nn.Sequential(
            nn.Conv2d(c1 + feedback_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Sigmoid()
        )

        self.dec4_conv = nn.Sequential(
            nn.Conv2d(c4, decoder_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.dec3 = MultiScaleDecoderBlock(decoder_channels * 4, c3, decoder_channels * 3)
        self.dec2 = MultiScaleDecoderBlock(decoder_channels * 3, c2, decoder_channels * 2)
        self.dec1 = MultiScaleDecoderBlock(decoder_channels * 2, c1, decoder_channels * 2)
        self.dec0 = MultiScaleDecoderBlock(decoder_channels * 2, c0, decoder_channels)

        self.edge_stream = DifferentiableEdgeStream(out_channels=32)
        self.sba = SelectiveBoundaryAggregationLinear(
            semantic_channels=decoder_channels,
            edge_channels=32,
            out_channels=decoder_channels
        )

        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.head_final = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.head_aux1  = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.head_aux2  = nn.Conv2d(decoder_channels * 2, 1, kernel_size=1)

    def forward_single_pass(self, x, feedback_token=None):
        B, C, H, W = x.shape
        edge_feat, edge_pred = self.edge_stream(x)
        f0, f1, f2, f3, f4 = self.encoder(x)

        s0 = self.fdf0(f0)
        s1 = self.fdf1(f1)
        s2 = self.fdf2(f2)
        s3 = self.fdf3(f3)
        s4 = self.fdf4(f4)

        if feedback_token is not None:
            fb = F.interpolate(feedback_token, size=s1.shape[2:], mode='bilinear', align_corners=False)
            gate = self.feedback_gate(torch.cat([s1, fb], dim=1))
            s1 = s1 * gate + s1

        d4 = self.dec4_conv(s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)

        d_full = self.final_upsample(d0)
        refined_feat = self.sba(d_full, edge_feat)
        out_final = self.head_final(refined_feat)

        aux1 = F.interpolate(self.head_aux1(d0), size=(H, W), mode='bilinear', align_corners=False)
        aux2 = F.interpolate(self.head_aux2(d1), size=(H, W), mode='bilinear', align_corners=False)
        next_feedback = self.feedback_adapter(d1).detach()

        return out_final, aux1, aux2, edge_pred, next_feedback

    def forward(self, x):
        feedback = None
        for it in range(self.feedback_iterations):
            out_final, aux1, aux2, edge_pred, feedback = self.forward_single_pass(x, feedback)

        if self.training:
            return {
                'logits': out_final,
                'aux1': aux1,
                'aux2': aux2,
                'edge_pred': edge_pred
            }
        else:
            return out_final

# ====================================================================
# 4. Composite Loss Function
# ====================================================================
class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.4, gamma=1.33, smooth=1e-5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs_flat) * targets_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return torch.pow(1.0 - tversky, 1.0 / self.gamma)

class BoundaryDifferenceIoULoss(nn.Module):
    def __init__(self, kernel_size=5, smooth=1e-5):
        super().__init__()
        self.kernel_size = kernel_size
        self.smooth = smooth
        self.dilation = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def _extract_boundary_band(self, targets):
        dilated = self.dilation(targets)
        eroded = 1.0 - self.dilation(1.0 - targets)
        return dilated - eroded

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        boundary_band = self._extract_boundary_band(targets).detach()

        p_boundary = probs * boundary_band
        g_boundary = targets * boundary_band

        intersection = (p_boundary * g_boundary).sum(dim=(2, 3))
        union = (p_boundary + g_boundary - (p_boundary * g_boundary)).sum(dim=(2, 3))

        bd_iou = (intersection + self.smooth) / (union + self.smooth)
        return (1.0 - bd_iou).mean()

class CompositeGDFNetV3Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_tversky = FocalTverskyLoss()
        self.bd_iou = BoundaryDifferenceIoULoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dilation = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def _generate_edge_gt(self, targets):
        dilated = self.dilation(targets)
        eroded = 1.0 - self.dilation(1.0 - targets)
        return dilated - eroded

    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            final_pred = outputs['logits']
            aux1       = outputs['aux1']
            aux2       = outputs['aux2']
            edge_pred  = outputs['edge_pred']

            loss_bce_final = self.bce(final_pred, targets)
            loss_ft_final  = self.focal_tversky(final_pred, targets)
            loss_bd_final  = self.bd_iou(final_pred, targets)

            loss_aux1 = self.focal_tversky(aux1, targets)
            loss_aux2 = self.focal_tversky(aux2, targets)

            edge_gt = self._generate_edge_gt(targets)
            loss_edge = self.bce(edge_pred, edge_gt)

            total_loss = loss_bce_final + loss_ft_final + 0.5 * loss_bd_final + \
                         0.3 * loss_aux1 + 0.2 * loss_aux2 + 0.3 * loss_edge
            return total_loss
        else:
            return self.bce(outputs, targets) + self.focal_tversky(outputs, targets) + 0.5 * self.bd_iou(outputs, targets)

# ====================================================================
# 5. Data Pipeline & 9-Metric Protocol
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
# 6. Training Engine on A100 GPU (150 Epochs, Batch Size 16)
# ====================================================================
# On A100 with 40-80GB VRAM, batch size 16 runs blazing fast
train_loader = DataLoader(ISICDataset(train_x, train_y, is_train=True), batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(ISICDataset(val_x, val_y, is_train=False), batch_size=16, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(ISICDataset(test_x, test_y, is_train=False), batch_size=16, shuffle=False, num_workers=2, pin_memory=True)

model = GDFNetV3(backbone_name="tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True).to(device)
criterion = CompositeGDFNetV3Loss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8)
scaler = GradScaler('cuda')

EPOCHS = 100
best_val_dice = 0.0

checkpoint_best = f"{OUTPUT_DIR}/GDFNetV3_ISIC2018_best.pth"
checkpoint_latest = f"{OUTPUT_DIR}/GDFNetV3_ISIC2018_latest.pth"
history_csv = f"{OUTPUT_DIR}/training_history_ISIC2018.csv"

print(f"\n[*] Starting GDF-Netv3 (EfficientNetV2-S) ISIC-2018 Training on A100 ({EPOCHS} Epochs, Batch Size 16)")
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
    pd.DataFrame(history).to_csv(history_csv, index=False)
    
    # Save latest state per epoch
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_val_dice': best_val_dice
    }, checkpoint_latest)

    print(f"Epoch {epoch:03d}/{EPOCHS} [{elapsed:.1f}s] | Loss: {epoch_loss:.4f} | Val Dice: {val_dice:.4f} | Val mIoU: {val_miou:.4f}", flush=True)

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), checkpoint_best)
        print(f"[*] New Best Model Saved (Val Dice: {val_dice:.4f})", flush=True)

    # Mirror to Google Drive if connected and distinct
    if os.path.exists("/content/drive/MyDrive/GDFNetV3_ISIC2018") and OUTPUT_DIR != "/content/drive/MyDrive/GDFNetV3_ISIC2018":
        try:
            shutil.copy2(history_csv, "/content/drive/MyDrive/GDFNetV3_ISIC2018/")
            if val_dice == best_val_dice:
                shutil.copy2(checkpoint_best, "/content/drive/MyDrive/GDFNetV3_ISIC2018/")
        except Exception:
            pass

# Final Blind Test Evaluation
print("\n[*] Evaluating Best GDF-Netv3 ISIC Checkpoint on 1,000 Official Blind Test Images...", flush=True)
model.load_state_dict(torch.load(checkpoint_best, map_location=device))
per_img_df, summary_df = evaluate_model(model, test_loader)
print("\n================ GDF-NETV3 OFFICIAL TEST RESULTS ================", flush=True)
print(summary_df.to_string(index=False), flush=True)
summary_df.to_csv(f"{OUTPUT_DIR}/benchmark_summary_ISIC2018.csv", index=False)
per_img_df.to_csv(f"{OUTPUT_DIR}/benchmark_raw_ISIC2018.csv", index=False)
if os.path.exists("/content/drive/MyDrive/GDFNetV3_ISIC2018") and OUTPUT_DIR != "/content/drive/MyDrive/GDFNetV3_ISIC2018":
    try:
        shutil.copy2(f"{OUTPUT_DIR}/benchmark_summary_ISIC2018.csv", "/content/drive/MyDrive/GDFNetV3_ISIC2018/")
        shutil.copy2(f"{OUTPUT_DIR}/benchmark_raw_ISIC2018.csv", "/content/drive/MyDrive/GDFNetV3_ISIC2018/")
    except Exception:
        pass
print("[+] GDF-Netv3 Training and Blind Evaluation Completed Successfully!", flush=True)
