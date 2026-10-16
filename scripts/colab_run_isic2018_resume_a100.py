"""
Script: colab_run_isic2018_resume_a100.py
Self-contained script executed on Colab NVIDIA A100 GPU to resume training ISIC-2018
from Epoch 114 to Epoch 200 (BGDNet Protocol: 2,334 Train, 260 Val, 1,000 Official Test).
"""

import os, sys, time, json, random, csv, glob, zipfile, subprocess
from collections import OrderedDict
from scipy.ndimage import distance_transform_edt

print("=" * 65)
print("  GDF-Net ISIC-2018: A100 Resume Training (Epochs 114 -> 200)")
print("  Protocol: BGDNet Partitioning (2,334 Train | 260 Val | 1,000 Test)")
print("=" * 65)

# 1. Credentials & Dependencies
os.environ["KAGGLE_USERNAME"] = "namnguynnnn"
os.environ["KAGGLE_KEY"] = "KGAT_dad33ce94fb8b25d8a9ecd4ca26ce03f"
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
    json.dump({"username": "namnguynnnn", "key": "KGAT_dad33ce94fb8b25d8a9ecd4ca26ce03f"}, f)
os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
                "albumentations", "tifffile", "opencv-python-headless"], check=True)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[+] Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("[+] Enabled NVIDIA A100 TensorFloat-32 (TF32) Acceleration.")

# 2. Acquire Dataset
DATA_ROOT = "/content/isic-2018"
os.makedirs(DATA_ROOT, exist_ok=True)

def _unzip_all(root):
    for z in glob.glob(os.path.join(root, "*.zip")):
        print(f"[*] Extracting: {os.path.basename(z)}...")
        with zipfile.ZipFile(z) as f:
            f.extractall(root)
        os.remove(z)

has_train_data = any("ISIC2018_Task1-2_Training_Input" in root for root, dirs, _ in os.walk(DATA_ROOT))
has_test_gt    = any("ISIC2018_Task1_Test_GroundTruth" in root for root, dirs, _ in os.walk(DATA_ROOT))

if not (has_train_data and has_test_gt):
    print("[*] Downloading ISIC-2018 Challenge Data via Kaggle API...")
    subprocess.run(["kaggle", "datasets", "download", "-d",
                    "tschandl/isic2018-challenge-task1-data-segmentation", "-p", DATA_ROOT], check=True)
    subprocess.run(["kaggle", "datasets", "download", "-d",
                    "mfahimhossen/isic2018-test-groundtruth", "-p", DATA_ROOT], check=True)
    _unzip_all(DATA_ROOT)

print(f"[+] Datasets ready at: {DATA_ROOT}")

# 3. UniversalDataset Definition
class UniversalDataset(Dataset):
    def __init__(self, images_path, masks_path, image_size, is_train=False, repeat_factor=1):
        self.images_path = list(images_path)
        self.masks_path  = list(masks_path)
        self.image_size  = image_size
        self.is_train    = is_train
        self.repeat_factor = max(1, int(repeat_factor)) if is_train else 1
        self.n_samples   = len(self.images_path) * self.repeat_factor

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=45, p=0.6),
            A.ElasticTransform(p=0.3, alpha=120, sigma=120*0.05),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3, distort_limit=0.05),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(p=0.3),
        ])

    def _read_image(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return np.array(Image.open(path).convert('RGB'))
        if len(img.shape) == 2:
            return np.stack([img]*3, axis=-1)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _read_mask(self, path):
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is None:
            return np.array(Image.open(path).convert('L'))
        if len(m.shape) == 3:
            return cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        return m

    def __getitem__(self, index):
        actual_idx = index % len(self.images_path)
        image = self._read_image(self.images_path[actual_idx])
        mask  = self._read_mask(self.masks_path[actual_idx])

        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  self.image_size, interpolation=cv2.INTER_NEAREST)

        if self.is_train:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']

        image = image.astype(np.float32)
        mx = image.max()
        if mx > 0:
            image = image / (65535.0 if mx > 255 else 255.0)
        image = np.transpose(image, (2, 0, 1))

        mask = mask.astype(np.float32)
        if mask.max() > 0:
            mask = (mask > (mask.max() + mask.min()) / 2.0).astype(np.float32)
        if mask.ndim == 2:
            mask = np.expand_dims(mask, axis=0)

        return image, mask

    def __len__(self):
        return self.n_samples

# 4. Data Partitioning (BGDNet Split)
IMAGE_SIZE = (512, 512)
EPOCHS = 200
BATCH_SIZE = 16
LR = 1e-4

def find_dir_by_name(target_name, base_search=DATA_ROOT):
    for root, dirs, files in os.walk(base_search):
        if os.path.basename(root).lower() == target_name.lower():
            return root
    return None

def isic_id(path):
    base = os.path.basename(path).replace("_segmentation", "")
    return os.path.splitext(base)[0]

train_img_dir = find_dir_by_name("ISIC2018_Task1-2_Training_Input")
train_gt_dir  = find_dir_by_name("ISIC2018_Task1_Training_GroundTruth")
test_img_dir  = find_dir_by_name("ISIC2018_Task1-2_Test_Input")
test_gt_dir   = find_dir_by_name("ISIC2018_Task1_Test_GroundTruth")

all_train_jpgs = sorted(glob.glob(os.path.join(train_img_dir, "*.jpg"))) if train_img_dir else []
all_train_pngs = sorted(glob.glob(os.path.join(train_gt_dir, "*.png")))  if train_gt_dir else []

img_dict = {isic_id(p): p for p in all_train_jpgs}
msk_dict = {isic_id(p): p for p in all_train_pngs}
train_pool_ids = sorted(set(img_dict.keys()) & set(msk_dict.keys()))

random.seed(2026)
random.shuffle(train_pool_ids)

N_TRAIN = 2334
N_VAL   = 260

train_ids = train_pool_ids[:N_TRAIN]
val_ids   = train_pool_ids[N_TRAIN:N_TRAIN + N_VAL]

train_x = [img_dict[k] for k in train_ids]
train_y = [msk_dict[k] for k in train_ids]
val_x   = [img_dict[k] for k in val_ids]
val_y   = [msk_dict[k] for k in val_ids]

test_jpgs = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg"))) if test_img_dir else []
test_pngs = sorted(glob.glob(os.path.join(test_gt_dir, "*.png")))  if test_gt_dir else []
test_img_dict = {isic_id(p): p for p in test_jpgs}
test_msk_dict = {isic_id(p): p for p in test_pngs}
test_ids = sorted(set(test_img_dict.keys()) & set(test_msk_dict.keys()))
test_x = [test_img_dict[k] for k in test_ids]
test_y = [test_msk_dict[k] for k in test_ids]

print(f"[+] Data loaded -> Train: {len(train_x)} | Val: {len(val_x)} | Official Test: {len(test_x)}")

train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True)
valid_dataset = UniversalDataset(val_x, val_y, IMAGE_SIZE, is_train=False)
test_dataset  = UniversalDataset(test_x, test_y, IMAGE_SIZE, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# 5. GDF-Net Architecture
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))


class GDFNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        self.e1_conv = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.e1_conv.weight[:, :3] = resnet.conv1.weight
            self.e1_conv.weight[:, 3:4] = 0.0
        self.e1_bn   = resnet.bn1
        self.e1_relu = resnet.relu
        self.pool    = resnet.maxpool

        self.e2 = resnet.layer1
        self.e3 = resnet.layer2
        self.e4 = resnet.layer3
        self.b  = resnet.layer4

        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d1  = ConvBlock(512, 256)
        self.att1 = nn.Sequential(nn.Conv2d(1, 256, 1), nn.BatchNorm2d(256), nn.Sigmoid())

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d2  = ConvBlock(256, 128)
        self.att2 = nn.Sequential(nn.Conv2d(1, 128, 1), nn.BatchNorm2d(128), nn.Sigmoid())

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d3  = ConvBlock(128, 64)
        self.att3 = nn.Sequential(nn.Conv2d(1, 64, 1), nn.BatchNorm2d(64), nn.Sigmoid())

        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.d4  = ConvBlock(128, 64)
        self.att4 = nn.Sequential(nn.Conv2d(1, 64, 1), nn.BatchNorm2d(64), nn.Sigmoid())

        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d5  = ConvBlock(32, 32)
        self.att5 = nn.Sequential(nn.Conv2d(1, 32, 1), nn.BatchNorm2d(32), nn.Sigmoid())

        self.out = nn.Conv2d(32, 1, 1)
        self.learned_gate = nn.Conv2d(1, 1, 1)

    def _mfad(self, feat, prev_mask, att_layer):
        m = F.interpolate(prev_mask, size=feat.shape[2:], mode='nearest')
        return feat * (1.0 + att_layer(m))

    def forward(self, x, prev_mask=None):
        if prev_mask is None:
            prev_mask = torch.zeros(x.size(0), 1, x.size(2), x.size(3), device=x.device)
        else:
            prev_mask = torch.sigmoid(self.learned_gate(prev_mask.detach()))

        s1 = self.e1_relu(self.e1_bn(self.e1_conv(torch.cat([x, prev_mask], dim=1))))
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        b  = self.b(s4)

        d1 = self.d1(torch.cat([self.up1(b), s4], 1));  d1 = self._mfad(d1, prev_mask, self.att1)
        d2 = self.d2(torch.cat([self.up2(d1), s3], 1)); d2 = self._mfad(d2, prev_mask, self.att2)
        d3 = self.d3(torch.cat([self.up3(d2), s2], 1)); d3 = self._mfad(d3, prev_mask, self.att3)
        d4 = self.d4(torch.cat([self.up4(d3), s1], 1)); d4 = self._mfad(d4, prev_mask, self.att4)
        d5 = self.d5(self.up5(d4));                      d5 = self._mfad(d5, prev_mask, self.att5)

        return self.out(d5)

# 6. Loss & Metrics
class DiceBCELoss(nn.Module):
    @torch.autocast(device_type='cuda', enabled=False)
    def forward(self, inputs, targets, smooth=1e-5):
        inputs, targets = inputs.float(), targets.float()
        inputs_s = torch.sigmoid(inputs).view(-1)
        targets_f = targets.view(-1)
        intersection = (inputs_s * targets_f).sum()
        dice_loss = 1.0 - (2.0 * intersection + smooth) / (inputs_s.sum() + targets_f.sum() + smooth)
        bce = F.binary_cross_entropy(inputs_s, targets_f, reduction='mean')
        return 0.5 * bce + 0.5 * dice_loss

def compute_metrics(gt, pred, eval_boundary=True):
    gt   = (gt > 0.5).astype(np.float32)
    pred = (pred > 0.5).astype(np.float32)

    tp = np.sum((gt == 1) & (pred == 1))
    fp = np.sum((gt == 0) & (pred == 1))
    fn = np.sum((gt == 1) & (pred == 0))
    tn = np.sum((gt == 0) & (pred == 0))

    dice = (2.0 * tp + 1.0) / (2.0 * tp + fp + fn + 1.0)
    iou  = (tp + 1.0) / (tp + fp + fn + 1.0)
    sens = tp / (tp + fn + 1e-7)
    spec = tn / (tn + fp + 1e-7)
    prec = tp / (tp + fp + 1e-7)
    fpr  = fp / (fp + tn + 1e-7)

    res = {
        'dice': float(dice), 'miou': float(iou),
        'sensitivity': float(sens), 'specificity': float(spec),
        'precision': float(prec), 'fpr': float(fpr)
    }

    if eval_boundary and (np.sum(gt) > 0 and np.sum(pred) > 0):
        try:
            g_2d = gt.squeeze()
            p_2d = pred.squeeze()
            g352 = cv2.resize(g_2d, (352, 352), interpolation=cv2.INTER_NEAREST)
            p352 = cv2.resize(p_2d, (352, 352), interpolation=cv2.INTER_NEAREST)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            g_bnd = cv2.morphologyEx(g352, cv2.MORPH_GRADIENT, kernel) > 0
            p_bnd = cv2.morphologyEx(p352, cv2.MORPH_GRADIENT, kernel) > 0

            d_gt = distance_transform_edt(~g_bnd)
            d_pred = distance_transform_edt(~p_bnd)

            dist_p_to_g = d_gt[p_bnd]
            dist_g_to_p = d_pred[g_bnd]

            if len(dist_p_to_g) > 0 and len(dist_g_to_p) > 0:
                all_dists = np.concatenate([dist_p_to_g, dist_g_to_p])
                res['hd95'] = float(np.percentile(all_dists, 95))
                res['asd']  = 0.5 * (float(np.mean(dist_p_to_g)) + float(np.mean(dist_g_to_p)))
                tau = 2.0
                p_acc = float(np.mean(dist_p_to_g <= tau))
                r_acc = float(np.mean(dist_g_to_p <= tau))
                res['bf2'] = (2.0 * p_acc * r_acc) / (p_acc + r_acc + 1e-7)
        except Exception:
            pass

    return res

# 7. Model Instantiation & Resume Checkpoint Loading
model = GDFNet().to(DEVICE)
criterion = DiceBCELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=8, min_lr=1e-6
)
scaler = torch.amp.GradScaler('cuda')

best_dice = 0.0
start_epoch = 0
history = {'train_loss': [], 'val_dice': []}

# Find resume checkpoint from /content/kaggle_out or /content/
resume_cands = [
    "/content/kaggle_out/GDFNet_ISIC2018_latest.pth",
    "/content/GDFNet_ISIC2018_latest.pth"
]
resume_path = next((p for p in resume_cands if os.path.exists(p)), None)

if resume_path:
    print(f"[*] Found checkpoint to resume: {resume_path}")
    ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    best_dice = ckpt.get('best_dice', 0.0)
    history = ckpt.get('history', {'train_loss': [], 'val_dice': []})
    start_epoch = ckpt.get('epoch', 0)
    print(f"[+] Successfully resumed from Epoch {start_epoch} with Best Val Dice: {best_dice:.4f}")

ckpt_best = "/content/GDFNet_ISIC2018_best.pth"
ckpt_latest = "/content/GDFNet_ISIC2018_latest.pth"

# 8. Training Loop (Epochs start_epoch -> 200)
print(f"[*] Training will run from Epoch {start_epoch+1} to {EPOCHS}...")
for epoch in range(start_epoch, EPOCHS):
    model.train()
    train_loss = 0.0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda'):
            out1 = model(x)
            out2 = model(x, torch.sigmoid(out1))
            loss = criterion(out2, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    train_loss /= len(train_loader)

    # Validation (260 Hold-out images)
    model.eval()
    val_dice_sum = 0.0
    n_val = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast('cuda'):
                out1 = model(x)
                out2 = model(x, torch.sigmoid(out1))
            pred = (torch.sigmoid(out2).cpu().numpy() > 0.5).astype(np.float32)
            gt   = y.cpu().numpy()
            for b_i in range(pred.shape[0]):
                m = compute_metrics(gt[b_i], pred[b_i], eval_boundary=False)
                val_dice_sum += m['dice']
                n_val += 1

    val_dice = val_dice_sum / max(n_val, 1)
    lr_now = optimizer.param_groups[0]['lr']
    scheduler.step(val_dice)

    history['train_loss'].append(train_loss)
    history['val_dice'].append(val_dice)

    print(f"Epoch {epoch+1:03d}/{EPOCHS} | LR {lr_now:.1e} | Loss {train_loss:.4f} | Val Dice {val_dice:.4f}")

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), ckpt_best)
        print(f"  [+] New best model saved with Val Dice: {best_dice:.4f}")

    torch.save({
        'epoch': epoch + 1,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_dice': best_dice,
        'history': history
    }, ckpt_latest)

print(f"\n[+] Training Complete up to Epoch {EPOCHS}. Peak Val Dice: {best_dice:.4f}")

# 9. Evaluation on Official 1,000 Test Set
print(f"\n{'='*70}")
print(f"  EVALUATING ON OFFICIAL ISIC-2018 TEST SET (N={len(test_loader.dataset)})")
print(f"{'='*70}")

model.load_state_dict(torch.load(ckpt_best, map_location=DEVICE, weights_only=True))
model.eval()

all_metrics = []
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast('cuda'):
            out1 = model(x)
            out2 = model(x, torch.sigmoid(out1))
        pred = (torch.sigmoid(out2).cpu().numpy() > 0.5).astype(np.float32)
        gt   = y.cpu().numpy()
        for b_i in range(pred.shape[0]):
            m = compute_metrics(gt[b_i], pred[b_i], eval_boundary=True)
            all_metrics.append(m)

agg = {}
for k in all_metrics[0].keys():
    vals = [m[k] for m in all_metrics if k in m]
    agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

print(f"\n{'='*70}")
print(f"  FINAL BENCHMARK RESULTS ON OFFICIAL ISIC-2018 TEST SET (N={len(test_loader.dataset)})")
print(f"{'='*70}")
for k, v in agg.items():
    print(f"  {k:<20s} | {v['mean']:>10.4f} ± {v['std']:<10.4f}")
print(f"{'='*70}")

# Save CSV
csv_path = "/content/benchmark_results_ISIC-2018_official_test_a100.csv"
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model', 'Dataset', 'Cohort', 'Metric', 'Mean', 'Std'])
    for k, v in agg.items():
        writer.writerow(['GDF-Net', 'ISIC-2018', 'Official_Test_1000', k, f"{v['mean']:.6f}", f"{v['std']:.6f}"])
print(f"[+] Final results saved to: {csv_path}")
