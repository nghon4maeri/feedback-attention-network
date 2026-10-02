"""
Generate ONLY the ISIC-2018 benchmark notebook with:
  - Checkpoint Resume (save/load epoch, optimizer, scheduler, best_dice)
  - Mixed Precision (AMP) for ~2x speed
  - Exact paper pipeline: 512x512, 1815 train / 259 test, LR 1e-4, 100 epochs
"""
import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# ── Cell 0: Title ──
nb.cells.append(nbf.v4.new_markdown_cell(
"""# FANet-MFAD Benchmark: ISIC-2018 (with Checkpoint Resume)

**Architecture:** ResNet-34 Encoder + MFAD Decoder (Ours)
**Loss:** 0.5×BCE + 0.5×Dice | **Scheduler:** ReduceLROnPlateau
**Resume:** If Kaggle times out, save this notebook's output as a new dataset,
attach it as input, and re-run. Training will auto-resume from the last saved epoch.
"""))

# ── Cell 1: Imports ──
nb.cells.append(nbf.v4.new_code_cell(
r"""import os, random, cv2, numpy as np
from glob import glob
from tqdm import tqdm
from PIL import Image
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
"""))

# ── Cell 2: Config ──
nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# CONFIG  (Paper: ISIC-2018 at 512x512, LR 1e-4, 100 epochs)
# ================================================================
DATASET_NAME = "ISIC-2018"
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 8
EPOCHS = 100
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Checkpoint resume paths
CKPT_RESUME = "fanet_mfad_ISIC-2018_resume.pth"
CKPT_BEST   = "fanet_mfad_ISIC-2018_best.pth"

# Search for previous checkpoint in Kaggle input datasets
PREV_CKPT = None
for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        if f == CKPT_RESUME:
            PREV_CKPT = os.path.join(root, f)
            break

if PREV_CKPT:
    print(f"Found previous checkpoint: {PREV_CKPT}")
else:
    print("No previous checkpoint found. Training from scratch.")
"""))

# ── Cell 3: Dataset class ──
nb.cells.append(nbf.v4.new_code_cell(
r"""class ISICDataset(Dataset):
    def __init__(self, images, masks, size, is_train=False):
        self.images = list(images)
        self.masks  = list(masks)
        self.size   = size
        self.is_train = is_train
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

    def __getitem__(self, idx):
        img = np.array(Image.open(self.images[idx]).convert('RGB'))
        msk = np.array(Image.open(self.masks[idx]).convert('L'))

        img = cv2.resize(img, self.size, interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, self.size, interpolation=cv2.INTER_NEAREST)

        if self.is_train:
            aug = self.transform(image=img, mask=msk)
            img, msk = aug['image'], aug['mask']

        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        msk = msk.astype(np.float32)
        if msk.max() > 0:
            msk = msk / msk.max()
        msk = np.expand_dims(msk, 0)

        return img, msk

    def __len__(self):
        return len(self.images)
"""))

# ── Cell 4: Data loading ──
nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# DATA LOADING  (Paper: Train 1815, Test 259)
# ================================================================
DATA_DIR = "/kaggle/input/datasets/tschandl/isic2018-challenge-task1-data-segmentation"
img_dir = os.path.join(DATA_DIR, "ISIC2018_Task1-2_Training_Input")
msk_dir = os.path.join(DATA_DIR, "ISIC2018_Task1_Training_GroundTruth")

all_imgs = sorted(glob(os.path.join(img_dir, "*.jpg")))
all_msks = sorted(glob(os.path.join(msk_dir, "*.png")))

print(f"Found {len(all_imgs)} images, {len(all_msks)} masks")

# Match by ISIC ID
img_ids = {os.path.basename(f).split('.')[0]: f for f in all_imgs}
msk_ids = {os.path.basename(f).split('_segmentation')[0]: f for f in all_msks}
common = sorted(set(img_ids.keys()) & set(msk_ids.keys()))
print(f"Matched {len(common)} image-mask pairs")

all_imgs = [img_ids[k] for k in common]
all_msks = [msk_ids[k] for k in common]

# Split: 1815 train, 259 test
random.seed(42)
indices = list(range(len(all_imgs)))
random.shuffle(indices)
N_TRAIN, N_TEST = 1815, 259
train_idx = indices[:N_TRAIN]
test_idx  = indices[N_TRAIN:N_TRAIN+N_TEST]

train_x = [all_imgs[i] for i in train_idx]
train_y = [all_msks[i] for i in train_idx]
valid_x = [all_imgs[i] for i in test_idx]
valid_y = [all_msks[i] for i in test_idx]

print(f"ISIC-2018: Train={len(train_x)}, Test={len(valid_x)}")

train_dataset = ISICDataset(train_x, train_y, IMAGE_SIZE, is_train=True)
valid_dataset = ISICDataset(valid_x, valid_y, IMAGE_SIZE, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False,
                          num_workers=2, pin_memory=True)
"""))

# ── Cell 5: Model ──
nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# FANet-MFAD  (Our proposed architecture)
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
"""))

# ── Cell 6: Loss + Metrics ──
nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# Loss (Paper: 0.5*BCE + 0.5*Dice) & Metrics
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
"""))

# ── Cell 7: Training with AMP + Checkpoint Resume ──
nb.cells.append(nbf.v4.new_code_cell(
r"""# ================================================================
# Training Loop with AMP + Checkpoint Resume
# ================================================================
model = FANet_MFAD().to(DEVICE)
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
scaler = torch.amp.GradScaler('cuda')
use_amp = DEVICE.type == 'cuda'

start_epoch = 0
best_dice = 0.0

# ── RESUME from previous checkpoint if available ──
if PREV_CKPT and os.path.exists(PREV_CKPT):
    print(f"Loading checkpoint from {PREV_CKPT}...")
    ckpt_data = torch.load(PREV_CKPT, map_location=DEVICE)
    model.load_state_dict(ckpt_data['model'])
    optimizer.load_state_dict(ckpt_data['optimizer'])
    scheduler.load_state_dict(ckpt_data['scheduler'])
    scaler.load_state_dict(ckpt_data['scaler'])
    start_epoch = ckpt_data['epoch'] + 1
    best_dice   = ckpt_data['best_dice']
    print(f"Resumed from epoch {start_epoch}, best_dice={best_dice:.4f}")
else:
    print("Starting training from scratch.")

print(f"Will train epochs {start_epoch+1} -> {EPOCHS}")

for epoch in range(start_epoch, EPOCHS):
    # --- TRAIN ---
    model.train(); train_loss = 0
    for x, y in tqdm(train_loader, desc=f'E{epoch+1}/{EPOCHS} Train'):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = model(x)
            out = model(x, torch.sigmoid(out))
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    # --- VALID ---
    model.eval(); val_loss = 0; val_dice = 0
    with torch.no_grad():
        for x, y in tqdm(valid_loader, desc=f'E{epoch+1}/{EPOCHS} Valid'):
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.amp.autocast('cuda', enabled=use_amp):
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
        torch.save(model.state_dict(), CKPT_BEST)
        print(f"  -> Saved best model (Dice={best_dice:.4f})")

    # Save resume checkpoint EVERY epoch (overwrites previous)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'best_dice': best_dice,
    }, CKPT_RESUME)

print(f"\n{'='*50}")
print(f"BEST VAL DICE on {DATASET_NAME}: {best_dice:.4f}")
print(f"{'='*50}")
"""))

# ── Write notebook ──
os.makedirs('notebooks', exist_ok=True)
out = 'notebooks/fanet_benchmark_isic_2018.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Generated: {out}")
