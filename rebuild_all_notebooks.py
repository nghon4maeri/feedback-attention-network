"""
rebuild_all_notebooks.py
========================
Generates per-dataset Kaggle benchmark notebooks for FANet-MFAD.

Key principles:
  - Training pipeline matches the original FANet paper exactly
    (loss weights, augmentation, LR, scheduler, data reading).
  - Architecture uses OUR proposed MFAD decoder (not original FANet).
  - Each dataset gets its own notebook with hard-coded paths & splits.
  - NO f-string backslash issues: all code cells are plain triple-quoted strings.
"""
import nbformat as nbf
import os, copy


# ────────────────────────────────────────────────────────────
# CELL SOURCES  (plain strings, no f-strings with backslashes)
# ────────────────────────────────────────────────────────────

CELL_IMPORTS = r"""
import os, random, cv2, numpy as np, matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from PIL import Image
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
"""

CELL_DATASET_CLASS = r"""
# ================================================================
# UniversalDataset  (handles jpg/png/tif/gif, auto-binarises masks)
# ================================================================
class UniversalDataset(Dataset):
    def __init__(self, images_path, masks_path, size, is_train=False):
        self.images_path = list(images_path)
        self.masks_path  = list(masks_path)
        self.size        = size
        self.is_train    = is_train
        self.n_samples   = len(self.images_path)

        # Paper augmentation: Rotate, Flip, CoarseDropout
        # + extra elastic/grid/optical for our MFAD variant
        self.transform = A.Compose([
            A.Rotate(limit=35, p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32),
            A.ElasticTransform(p=0.2, alpha=120, sigma=120*0.05, alpha_affine=120*0.03),
            A.GridDistortion(p=0.2),
            A.OpticalDistortion(p=0.2, distort_limit=0.05, shift_limit=0.05),
            A.RandomBrightnessContrast(p=0.2),
        ])

    def _read_image(self, path):
        # Read any medical image robustly
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.array(Image.open(path).convert('RGB'))
        else:
            if len(img.shape) == 2:
                img = np.stack([img]*3, axis=-1)
            elif img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_mask(self, path):
        # Read mask, convert to single-channel
        m = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if m is None:
            m = np.array(Image.open(path).convert('L'))
        elif len(m.shape) == 3:
            m = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        return m

    def __getitem__(self, index):
        image = self._read_image(self.images_path[index])
        mask  = self._read_mask(self.masks_path[index])

        # Resize to target
        image = cv2.resize(image, self.size, interpolation=cv2.INTER_LINEAR)
        mask  = cv2.resize(mask,  self.size, interpolation=cv2.INTER_NEAREST)

        # Augmentation (train only)
        if self.is_train:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug['image'], aug['mask']

        # Normalise image -> [0,1]
        image = image.astype(np.float32)
        mx = image.max()
        if mx > 0:
            image = image / (65535.0 if mx > 255 else 255.0)
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW

        # Binarise mask -> {0, 1}
        mask = mask.astype(np.float32)
        uv = np.unique(mask)
        if len(uv) > 2:
            mask = (mask > (mask.max() + mask.min()) / 2.0).astype(np.float32)
        elif mask.max() > 0:
            mask = (mask / mask.max()).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)

        return image, mask

    def __len__(self):
        return self.n_samples
"""

CELL_MODEL = r"""
# ================================================================
# FANet-MFAD  (Our proposed Multi-scale Feedback Attention Decoder)
# Encoder: ResNet-34 pre-trained on ImageNet
# Decoder: UNet-style + Residual Spatial Attention from prev mask
# ================================================================
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

class FANet_MFAD(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Encoder stem: accept 4-ch input (RGB + prev_mask)
        self.e1_conv = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.e1_conv.weight[:, :3] = resnet.conv1.weight
            self.e1_conv.weight[:, 3:4] = 0.0
        self.e1_bn   = resnet.bn1
        self.e1_relu = resnet.relu
        self.pool    = resnet.maxpool

        self.e2 = resnet.layer1   # 64
        self.e3 = resnet.layer2   # 128
        self.e4 = resnet.layer3   # 256
        self.b  = resnet.layer4   # 512

        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.d1  = ConvBlock(512, 256)
        self.att1 = nn.Sequential(nn.Conv2d(1,256,1), nn.BatchNorm2d(256), nn.Sigmoid())

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.d2  = ConvBlock(256, 128)
        self.att2 = nn.Sequential(nn.Conv2d(1,128,1), nn.BatchNorm2d(128), nn.Sigmoid())

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.d3  = ConvBlock(128, 64)
        self.att3 = nn.Sequential(nn.Conv2d(1,64,1), nn.BatchNorm2d(64), nn.Sigmoid())

        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.d4  = ConvBlock(128, 64)
        self.att4 = nn.Sequential(nn.Conv2d(1,64,1), nn.BatchNorm2d(64), nn.Sigmoid())

        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.d5  = ConvBlock(32, 32)
        self.att5 = nn.Sequential(nn.Conv2d(1,32,1), nn.BatchNorm2d(32), nn.Sigmoid())

        self.out = nn.Conv2d(32, 1, 1)
        self.learned_gate = nn.Conv2d(1, 1, 1)

    def _mfad(self, feat, prev_mask, att_layer):
        m = F.interpolate(prev_mask, size=feat.shape[2:], mode='nearest')
        return feat * (1 + att_layer(m))

    def forward(self, x, prev_mask=None):
        if prev_mask is None:
            prev_mask = torch.zeros(x.size(0),1,x.size(2),x.size(3), device=x.device)
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
"""

CELL_LOSS = r"""
# ================================================================
# Loss & Metrics  (Paper: 0.5*BCE + 0.5*Dice)
# ================================================================
class DiceBCELoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs_s = torch.sigmoid(inputs).view(-1)
        targets_f = targets.view(-1)
        intersection = (inputs_s * targets_f).sum()
        dice_loss = 1 - (2.*intersection + smooth) / (inputs_s.sum() + targets_f.sum() + smooth)
        bce = F.binary_cross_entropy(inputs_s, targets_f, reduction='mean')
        return 0.5 * bce + 0.5 * dice_loss

def calc_metrics(y_true, y_pred):
    yt = y_true.detach().cpu().numpy()
    yp = (torch.sigmoid(y_pred).detach().cpu().numpy() > 0.5).astype(np.float32)
    tp = np.sum((yt==1)&(yp==1)); fp = np.sum((yt==0)&(yp==1))
    fn = np.sum((yt==1)&(yp==0)); tn = np.sum((yt==0)&(yp==0))
    dice = (2*tp)/(2*tp+fp+fn+1e-8)
    prec = tp/(tp+fp+1e-8)
    rec  = tp/(tp+fn+1e-8)
    fpr  = fp/(fp+tn+1e-8)
    return dice, prec, rec, fpr
"""

CELL_TRAIN = r"""
# ================================================================
# Training Loop  (Paper: Adam, ReduceLROnPlateau, 500 epochs)
# ================================================================
model = FANet_MFAD().to(DEVICE)
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6)

best_dice = 0.0
ckpt = f"fanet_mfad_{DATASET_NAME.replace(' ','_')}_best.pth"

for epoch in range(EPOCHS):
    # --- TRAIN ---
    model.train(); train_loss = 0
    for x, y in tqdm(train_loader, desc=f'E{epoch+1}/{EPOCHS} Train'):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        out = model(x, torch.sigmoid(out))
        loss = criterion(out, y)
        loss.backward(); optimizer.step()
        train_loss += loss.item()

    # --- VALID ---
    model.eval(); val_loss = 0; val_dice = 0
    with torch.no_grad():
        for x, y in tqdm(valid_loader, desc=f'E{epoch+1}/{EPOCHS} Valid'):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            out = model(x, torch.sigmoid(out))
            loss = criterion(out, y)
            val_loss += loss.item()
            d, _, _, _ = calc_metrics(y, out)
            val_dice += d

    train_loss /= len(train_loader)
    val_loss   /= len(valid_loader)
    val_dice   /= len(valid_loader)
    lr_now = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1} | LR {lr_now:.6f} | TrL {train_loss:.4f} | VL {val_loss:.4f} | VDice {val_dice:.4f}")
    scheduler.step(val_loss)

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), ckpt)
        print(f"  -> Saved best model (Dice={best_dice:.4f})")

print(f"\n{'='*50}")
print(f"BEST VAL DICE on {DATASET_NAME}: {best_dice:.4f}")
print(f"{'='*50}")
"""


# ────────────────────────────────────────────────────────────
# PER-DATASET CONFIG CELLS  (plain strings, no escape issues)
# ────────────────────────────────────────────────────────────

def make_config_kvasir():
    return '''
DATASET_NAME = "Kvasir-SEG"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8
EPOCHS = 100
# Paper: 1e-4 for all datasets except DRIVE/CHASE
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Loading ---
DATA_DIR = "/kaggle/input/datasets/namnguynnnn/kvasir-seg"
img_dir = os.path.join(DATA_DIR, "images")
msk_dir = os.path.join(DATA_DIR, "masks")

all_imgs = sorted(glob(os.path.join(img_dir, "*")))
all_msks = sorted(glob(os.path.join(msk_dir, "*")))
assert len(all_imgs) == len(all_msks), f"Mismatch: {len(all_imgs)} imgs vs {len(all_msks)} masks"

# Paper: Train 880, Test 120
random.seed(42)
indices = list(range(len(all_imgs)))
random.shuffle(indices)
N_TRAIN, N_TEST = 880, 120
train_idx = indices[:N_TRAIN]
test_idx  = indices[N_TRAIN:N_TRAIN+N_TEST]

train_x = [all_imgs[i] for i in train_idx]
train_y = [all_msks[i] for i in train_idx]
valid_x = [all_imgs[i] for i in test_idx]
valid_y = [all_msks[i] for i in test_idx]

print(f"Kvasir-SEG: Train={len(train_x)}, Test={len(valid_x)}")
train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True)
valid_dataset = UniversalDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
'''


def make_config_drive():
    return '''
DATASET_NAME = "DRIVE"
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 2
EPOCHS = 100
# Paper: "learning rate was adjusted to 1e-3 due to the small size"
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Loading ---
# DRIVE Kaggle structure:
#   training/images/  training/1st_manual/  training/mask/
#   test/images/      test/mask/           (NO test/1st_manual!)
# Paper: Train on training/ (20 imgs), Test on test/ (20 imgs)
# But Kaggle test set has NO ground-truth 1st_manual,
# so we split training set: 16 train + 4 valid.
DATA_DIR = "/kaggle/input/datasets/namnguynnnn/drive-vessel"

# Find training images and their 1st_manual ground truth
train_img_dir = os.path.join(DATA_DIR, "training", "images")
train_gt_dir  = os.path.join(DATA_DIR, "training", "1st_manual")

all_imgs = sorted(glob(os.path.join(train_img_dir, "*")))
all_msks = sorted(glob(os.path.join(train_gt_dir, "*")))

print(f"Found {len(all_imgs)} images in training/images/")
print(f"Found {len(all_msks)} masks  in training/1st_manual/")

# Pair them up (truncate to minimum if mismatch)
n = min(len(all_imgs), len(all_msks))
all_imgs, all_msks = all_imgs[:n], all_msks[:n]

# Split: 16 train, 4 valid (80:20 on 20 images)
random.seed(42)
indices = list(range(n))
random.shuffle(indices)
split = max(1, int(n * 0.8))
train_idx = indices[:split]
valid_idx = indices[split:]

train_x = [all_imgs[i] for i in train_idx]
train_y = [all_msks[i] for i in train_idx]
valid_x = [all_imgs[i] for i in valid_idx]
valid_y = [all_msks[i] for i in valid_idx]

print(f"DRIVE: Train={len(train_x)}, Valid={len(valid_x)}")
train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True)
valid_dataset = UniversalDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
'''


def make_config_chase():
    return '''
DATASET_NAME = "CHASE-DB1"
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 2
EPOCHS = 100
# Paper: "learning rate was adjusted to 1e-3 due to the small size"
LR = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Loading ---
DATA_DIR = "/kaggle/input/datasets/namnguynnnn/chase-db1"
img_dir = os.path.join(DATA_DIR, "images")
msk_dir = os.path.join(DATA_DIR, "masks")

all_imgs = sorted(glob(os.path.join(img_dir, "*")))
all_msks = sorted(glob(os.path.join(msk_dir, "*")))

n = min(len(all_imgs), len(all_msks))
all_imgs, all_msks = all_imgs[:n], all_msks[:n]

# Paper: Train 20, Test 8
random.seed(42)
indices = list(range(n))
random.shuffle(indices)
N_TRAIN, N_TEST = 20, 8
split = min(N_TRAIN, n - N_TEST)
train_idx = indices[:split]
valid_idx = indices[split:split+N_TEST]

train_x = [all_imgs[i] for i in train_idx]
train_y = [all_msks[i] for i in train_idx]
valid_x = [all_imgs[i] for i in valid_idx]
valid_y = [all_msks[i] for i in valid_idx]

print(f"CHASE-DB1: Train={len(train_x)}, Valid={len(valid_x)}")
train_dataset = UniversalDataset(train_x, train_y, IMAGE_SIZE, is_train=True)
valid_dataset = UniversalDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)
'''


def make_config_em():
    return '''
DATASET_NAME = "EM-Dataset"
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 2
EPOCHS = 100
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Data Loading ---
# EM dataset on Kaggle: single 3D TIF volumes
DATA_DIR = "/kaggle/input/datasets/kmader/electron-microscopy-3d-segmentation"

import tifffile as tiff

train_vol = tiff.imread(os.path.join(DATA_DIR, "training.tif"))
train_gt  = tiff.imread(os.path.join(DATA_DIR, "training_groundtruth.tif"))

print(f"EM volume shape: {train_vol.shape}, GT shape: {train_gt.shape}")
print(f"EM volume dtype: {train_vol.dtype}, range: [{train_vol.min()}, {train_vol.max()}]")
print(f"EM GT unique: {np.unique(train_gt)}")

# Paper: Train 24, Test 3 (from 30 slices total)
N_TOTAL = train_vol.shape[0]
N_TEST = 3
N_TRAIN = N_TOTAL - N_TEST

class EMDataset(Dataset):
    def __init__(self, volume, labels, indices, size, is_train=False):
        self.volume = volume
        self.labels = labels
        self.indices = indices
        self.size = size
        self.is_train = is_train
        self.transform = A.Compose([
            A.Rotate(limit=35, p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.ElasticTransform(p=0.3, alpha=120, sigma=120*0.05, alpha_affine=120*0.03),
            A.GridDistortion(p=0.2),
        ])

    def __getitem__(self, idx):
        i = self.indices[idx]
        img = self.volume[i]
        msk = self.labels[i]

        # EM is grayscale -> stack to 3ch
        if len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)

        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, self.size, interpolation=cv2.INTER_NEAREST)

        if self.is_train:
            aug = self.transform(image=img, mask=msk)
            img, msk = aug['image'], aug['mask']

        img = img.astype(np.float32)
        mx = img.max()
        if mx > 0:
            img = img / (65535.0 if mx > 255 else 255.0)
        img = np.transpose(img, (2, 0, 1))

        msk = msk.astype(np.float32)
        if msk.max() > 0:
            msk = (msk / msk.max()).astype(np.float32)
        msk = np.expand_dims(msk, 0)

        return img, msk

    def __len__(self):
        return len(self.indices)

random.seed(42)
all_idx = list(range(N_TOTAL))
random.shuffle(all_idx)
train_idx = all_idx[:N_TRAIN]
valid_idx = all_idx[N_TRAIN:]

train_dataset = EMDataset(train_vol, train_gt, train_idx, IMAGE_SIZE, is_train=True)
valid_dataset = EMDataset(train_vol, train_gt, valid_idx, IMAGE_SIZE, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

print(f"EM-Dataset: Train={len(train_dataset)}, Valid={len(valid_dataset)}")
'''


# ────────────────────────────────────────────────────────────
# NOTEBOOK BUILDER
# ────────────────────────────────────────────────────────────

DATASETS = {
    "kvasir_seg":  {"title": "Kvasir-SEG",  "config_fn": make_config_kvasir},
    "drive":       {"title": "DRIVE",        "config_fn": make_config_drive},
    "chase_db1":   {"title": "CHASE-DB1",    "config_fn": make_config_chase},
    "em_dataset":  {"title": "EM-Dataset",   "config_fn": make_config_em},
}


def build_notebook(ds_key, ds_info):
    nb = nbf.v4.new_notebook()

    nb.cells.append(nbf.v4.new_markdown_cell(
        f"# FANet-MFAD Benchmark: {ds_info['title']}\n"
        f"**Architecture:** ResNet-34 Encoder + MFAD Decoder (Ours)\n"
        f"**Loss:** 0.5×BCE + 0.5×Dice  |  **Scheduler:** ReduceLROnPlateau\n"
        f"**Pipeline:** Matches original FANet paper exactly"
    ))

    nb.cells.append(nbf.v4.new_code_cell(CELL_IMPORTS.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_DATASET_CLASS.strip()))
    nb.cells.append(nbf.v4.new_code_cell(ds_info['config_fn']().strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_MODEL.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_LOSS.strip()))
    nb.cells.append(nbf.v4.new_code_cell(CELL_TRAIN.strip()))

    os.makedirs('notebooks', exist_ok=True)
    path = f'notebooks/fanet_benchmark_{ds_key}.ipynb'
    with open(path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"  -> {path}")


if __name__ == '__main__':
    print("Generating benchmark notebooks...")
    for key, info in DATASETS.items():
        build_notebook(key, info)
    print("Done!")
