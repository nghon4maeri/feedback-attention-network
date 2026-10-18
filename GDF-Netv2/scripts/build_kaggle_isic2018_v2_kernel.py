import os
import nbformat as nbf

def generate_gdfnet_v2_kaggle_notebook(output_path="GDF-Netv2/notebooks/gdfnet_v2_isic2018.ipynb"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nb = nbf.v4.new_notebook()

    # Markdown Intro
    nb.cells.append(nbf.v4.new_markdown_cell(
"""# GDF-Netv2 (Ultra-GDFNet): Skin Lesion Segmentation Benchmark on ISIC-2018
### Goal: Break past 0.94 Mean Dice using ConvNeXt-V2 Backbone, FDF, SBA, and Boundary-Difference Loss.
- Train: All 2,594 Training Images with 384x384 High-Resolution Data Augmentation
- Val: 100 Official Validation Images (Guided Checkpoint Selection)
- Test: 1,000 Official Blind Test Images (Zero-Leakage Benchmark)
"""
    ))

    # Cell 1: Environment & Dependencies
    nb.cells.append(nbf.v4.new_code_cell(
"""!pip install -q timm albumentations opencv-python-headless scipy pandas tqdm
import os, sys, glob, cv2, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt, binary_erosion
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device} | CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
"""
    ))

    # Cell 2: GDF-Netv2 Modules (Backbone, FDF, Edge, SBA, Model)
    nb.cells.append(nbf.v4.new_code_cell(
r"""import timm

class HierarchicalBackbone(nn.Module):
    def __init__(self, backbone_name="convnextv2_tiny", pretrained=True, in_chans=3):
        super().__init__()
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        self.feature_channels = self.encoder.feature_info.channels()

    def forward(self, x):
        return self.encoder(x)

    def get_channels(self):
        return self.feature_channels

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

class SelectiveBoundaryAggregation(nn.Module):
    def __init__(self, semantic_channels, edge_channels=32, out_channels=64):
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
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1)
        )

    def forward(self, semantic_feat, edge_feat):
        if semantic_feat.shape[2:] != edge_feat.shape[2:]:
            semantic_feat = F.interpolate(semantic_feat, size=edge_feat.shape[2:], mode='bilinear', align_corners=False)
        s_proj = self.semantic_conv(semantic_feat)
        e_proj = self.edge_conv(edge_feat)
        gate = self.cross_gate(torch.cat([s_proj, e_proj], dim=1))
        fused = torch.cat([s_proj, e_proj * gate], dim=1)
        return self.out_conv(fused)

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
        out = self.conv(torch.cat([x, skip], dim=1))
        return out * self.ca(out)

class GDFNetV2(nn.Module):
    def __init__(self, backbone_name="convnextv2_tiny", pretrained=True, feedback_channels=32, decoder_channels=64):
        super().__init__()
        self.backbone = HierarchicalBackbone(backbone_name=backbone_name, pretrained=pretrained)
        c1, c2, c3, c4 = self.backbone.get_channels()

        self.fdf1 = FrequencyDecoupledFilter(c1)
        self.fdf2 = FrequencyDecoupledFilter(c2)
        self.fdf3 = FrequencyDecoupledFilter(c3)
        self.fdf4 = FrequencyDecoupledFilter(c4)

        self.feedback_adapter = nn.Sequential(
            nn.Conv2d(decoder_channels, feedback_channels, kernel_size=1, bias=False),
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
        self.dec3 = MultiScaleDecoderBlock(decoder_channels * 4, c3, decoder_channels * 2)
        self.dec2 = MultiScaleDecoderBlock(decoder_channels * 2, c2, decoder_channels)
        self.dec1 = MultiScaleDecoderBlock(decoder_channels, c1, decoder_channels)

        self.edge_stream = DifferentiableEdgeStream(out_channels=32)
        self.sba = SelectiveBoundaryAggregation(decoder_channels, edge_channels=32, out_channels=64)
        self.aux_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1)
        )

    def _decode_features(self, s1, s2, s3, s4):
        d4 = self.dec4_conv(self.fdf4(s4))
        d3 = self.dec3(d4, self.fdf3(s3))
        d2 = self.dec2(d3, self.fdf2(s2))
        return self.dec1(d2, self.fdf1(s1))

    def forward(self, x):
        B, _, H, W = x.shape
        edge_feat, edge_pred = self.edge_stream(x)
        s1, s2, s3, s4 = self.backbone(x)

        d1_t1 = self._decode_features(s1, s2, s3, s4)
        pred_t1 = F.interpolate(self.aux_head(d1_t1), size=(H, W), mode='bilinear', align_corners=False)

        # Multi-Channel Semantic Feedback with Detached Gradient Firewall
        fb_feat = self.feedback_adapter(d1_t1).detach()
        if fb_feat.shape[2:] != s1.shape[2:]:
            fb_feat = F.interpolate(fb_feat, size=s1.shape[2:], mode='bilinear', align_corners=False)
        gate = self.feedback_gate(torch.cat([s1, fb_feat], dim=1))
        s1_refined = s1 + (s1 * gate)

        d1_t2 = self._decode_features(s1_refined, s2, s3, s4)
        final_pred = self.sba(d1_t2, edge_feat)

        if self.training:
            return {'final_pred': final_pred, 'pred_t1': pred_t1, 'edge_pred': edge_pred}
        return final_pred
"""
    ))

    # Cell 3: Losses (Focal-Tversky, BD-IoU, Composite)
    nb.cells.append(nbf.v4.new_code_cell(
r"""class FocalTverskyLoss(nn.Module):
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

class CompositeGDFNetV2Loss(nn.Module):
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
"""
    ))

    # Cell 4: Data Discovery & Loading
    nb.cells.append(nbf.v4.new_code_cell(
r"""def isic_id(path):
    base = os.path.basename(path).replace("_segmentation", "")
    return os.path.splitext(base)[0]

def find_dir_by_name(target_name, base_search="/kaggle/input"):
    for root, dirs, files in os.walk(base_search):
        if os.path.basename(root).lower() == target_name.lower():
            return root
    return None

# Locate Datasets
train_img_dir = find_dir_by_name("ISIC2018_Task1-2_Training_Input")
train_gt_dir  = find_dir_by_name("ISIC2018_Task1_Training_GroundTruth")
test_img_dir  = find_dir_by_name("ISIC2018_Task1-2_Test_Input")
test_gt_dir   = find_dir_by_name("ISIC2018_Task1_Test_GroundTruth")

# Ensure Official Val is present (download from S3 if missing)
val_img_dir = find_dir_by_name("ISIC2018_Task1-2_Validation_Input")
val_gt_dir  = find_dir_by_name("ISIC2018_Task1_Validation_GroundTruth")

if not val_img_dir:
    print("[*] Downloading Official Val Input from ISIC S3...")
    os.makedirs("isic2018_val", exist_ok=True)
    !wget -q -O val_input.zip "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip"
    !unzip -q val_input.zip -d isic2018_val/
    val_img_dir = "isic2018_val/ISIC2018_Task1-2_Validation_Input"

if not val_gt_dir:
    print("[*] Downloading Official Val GT from ISIC S3...")
    os.makedirs("isic2018_val", exist_ok=True)
    !wget -q -O val_gt.zip "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip"
    !unzip -q val_gt.zip -d isic2018_val/
    val_gt_dir = "isic2018_val/ISIC2018_Task1_Validation_GroundTruth"

def pair_images_and_masks(img_dir, gt_dir):
    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    msks = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    id_map = {isic_id(m): m for m in msks}
    paired_x = [img for img in imgs if isic_id(img) in id_map]
    paired_y = [id_map[isic_id(img)] for img in paired_x]
    return paired_x, paired_y

train_x, train_y = pair_images_and_masks(train_img_dir, train_gt_dir)
val_x, val_y     = pair_images_and_masks(val_img_dir, val_gt_dir)
test_x, test_y   = pair_images_and_masks(test_img_dir, test_gt_dir)

print(f"Verified Splits -> Train: {len(train_x)} | Val (Official): {len(val_x)} | Test (Official): {len(test_x)}")
"""
    ))

    # Cell 5: Dataset & Augmentations (384x384)
    nb.cells.append(nbf.v4.new_code_cell(
r"""IMG_SIZE = 384

train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=45, border_mode=0, p=0.6),
    A.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05, p=0.5),
    A.OneOf([
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=1.0),
        A.ElasticTransform(alpha=1.0, sigma=50, alpha_affine=30, p=1.0),
    ], p=0.3),
    A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(8, 32), hole_width_range=(8, 32), fill=0, p=0.25)
])

val_transform = A.Compose([A.Resize(IMG_SIZE, IMG_SIZE)])

class MedicalDataset(Dataset):
    def __init__(self, x, y, transform):
        self.x, self.y, self.transform = x, y, transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img = cv2.imread(self.x[idx], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        msk = cv2.imread(self.y[idx], cv2.IMREAD_GRAYSCALE)

        aug = self.transform(image=img, mask=msk)
        img, msk = aug['image'], aug['mask']

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        msk = (msk.astype(np.float32) > 127.5).astype(np.float32)
        msk = np.expand_dims(msk, axis=0)
        return torch.from_numpy(img).float(), torch.from_numpy(msk).float()

train_loader = DataLoader(MedicalDataset(train_x, train_y, train_transform), batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(MedicalDataset(val_x, val_y, val_transform), batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(MedicalDataset(test_x, test_y, val_transform), batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
"""
    ))

    # Cell 6: Evaluator Functions
    nb.cells.append(nbf.v4.new_code_cell(
r"""def compute_metrics(p, t):
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    tn = np.logical_and(~p, ~t).sum()

    dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
    miou = tp / (tp + fp + fn + 1e-8)
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    prec = tp / (tp + fp + 1e-8)

    # Surface distances
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

    return {'dice': dice, 'miou': miou, 'sensitivity': sens, 'specificity': spec, 'precision': prec, 'hd95': hd95, 'asd': asd, 'bf2': bf2}

@torch.no_grad()
def evaluate_model(model, loader):
    model.eval()
    records = []
    for imgs, msks in loader:
        imgs = imgs.to(device)
        with autocast():
            preds = torch.sigmoid(model(imgs)).cpu().numpy()[:, 0]
        msks = msks.numpy()[:, 0]

        for i in range(len(preds)):
            records.append(compute_metrics(preds[i] > 0.5, msks[i] > 0.5))

    df = pd.DataFrame(records)
    summary = []
    for m in df.columns:
        summary.append({'Metric': m, 'Mean': df[m].mean(), 'Std': df[m].std(), 'Median': df[m].median(), 'Min': df[m].min(), 'Max': df[m].max()})
    return df, pd.DataFrame(summary)
"""
    ))

    # Cell 7: Training Engine (150 Epochs)
    nb.cells.append(nbf.v4.new_code_cell(
r"""model = GDFNetV2(backbone_name="convnextv2_tiny", pretrained=True).to(device)
criterion = CompositeGDFNetV2Loss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, verbose=True)
scaler = GradScaler()

EPOCHS = 150
best_val_dice = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0

    for imgs, msks in train_loader:
        imgs, msks = imgs.to(device), msks.to(device)
        optimizer.zero_grad()
        with autocast():
            outs = model(imgs)
            loss = criterion(outs, msks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()

    # Fast Validation on 100 Official Images
    _, val_summary = evaluate_model(model, val_loader)
    val_dice = val_summary.loc[val_summary['Metric'] == 'dice', 'Mean'].values[0]
    val_miou = val_summary.loc[val_summary['Metric'] == 'miou', 'Mean'].values[0]
    scheduler.step(val_dice)

    print(f"Epoch {epoch:03d}/{EPOCHS} | Train Loss: {running_loss/len(train_loader):.4f} | Val Dice: {val_dice:.4f} | Val mIoU: {val_miou:.4f}")

    if val_dice > best_val_dice:
        best_val_dice = val_dice
        torch.save(model.state_dict(), "GDFNetV2_ISIC2018_best.pth")
        print(f"[*] Best Checkpoint Saved: Val Dice = {val_dice:.4f}")

torch.save(model.state_dict(), "GDFNetV2_ISIC2018_latest.pth")
"""
    ))

    # Cell 8: Final Official Test Benchmark
    nb.cells.append(nbf.v4.new_code_cell(
r"""print("[*] Loading Best Model for Official 1,000 Test Benchmark...")
model.load_state_dict(torch.load("GDFNetV2_ISIC2018_best.pth", map_location=device))
per_img_df, summary_df = evaluate_model(model, test_loader)

print("\n================ GDF-NETV2 OFFICIAL TEST BENCHMARK RESULTS ================")
print(summary_df.to_string(index=False))

summary_df.to_csv("benchmark_results_GDFNetV2_ISIC-2018_official_test.csv", index=False)
per_img_df.to_csv("per_image_results_GDFNetV2_ISIC-2018_official_test.csv", index=False)
print("[*] Benchmark artifacts saved to disk.")
"""
    ))

    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[*] GDF-Netv2 Kaggle Notebook successfully generated: {output_path}")

if __name__ == "__main__":
    generate_gdfnet_v2_kaggle_notebook()
