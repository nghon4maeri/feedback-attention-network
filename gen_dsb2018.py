import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Markdown Header
nb.cells.append(nbf.v4.new_markdown_cell(
    "# FANet-MFAD Benchmark: DSB-2018\n"
    "**Architecture:** ResNet-34 Encoder + MFAD Decoder (Ours)\n"
    "**Loss:** 0.5xBCE + 0.5xDice  |  **Scheduler:** ReduceLROnPlateau\n"
    "**Pipeline:** Matches original FANet paper exactly"
))

# ── Cell 1: Install & Imports ──
nb.cells.append(nbf.v4.new_code_cell(
r"""!pip install -q albumentations

import os, random, cv2, glob, zipfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import albumentations as A

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

import warnings
warnings.filterwarnings('ignore')
"""))

# ── Cell 2: Dataset Config & Preprocessing ──
nb.cells.append(nbf.v4.new_code_cell(
r"""DATASET_NAME = "DSB-2018"
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 16 
EPOCHS = 100
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CKPT_RESUME = "/kaggle/working/fanet_mfad_DSB2018_resume.pth"
CKPT_BEST   = "/kaggle/working/fanet_mfad_DSB2018_best.pth"

ZIP_PATH = "/kaggle/input/data-science-bowl-2018/stage1_train.zip"
ORIG_DIR = "/kaggle/working/stage1_train"
MERGED_MASK_DIR = "/kaggle/working/dsb2018_masks"

if os.path.exists(ZIP_PATH) and not os.path.exists(ORIG_DIR):
    print(f"Đang giải nén {ZIP_PATH}...")
    os.makedirs(ORIG_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        zip_ref.extractall(ORIG_DIR)
elif not os.path.exists(ZIP_PATH) and not os.path.exists(ORIG_DIR):
    print("LỖI: Bạn chưa Add data-science-bowl-2018 vào Input Kaggle!")

os.makedirs(MERGED_MASK_DIR, exist_ok=True)

all_imgs = []
all_msks = []

print("Đang gộp các masks rời rạc thành 1 mask chung (nếu chưa gộp)...")
if os.path.exists(ORIG_DIR):
    folder_names = sorted(os.listdir(ORIG_DIR))
    for folder in tqdm(folder_names, desc="Merging Masks"):
        folder_path = os.path.join(ORIG_DIR, folder)
        if not os.path.isdir(folder_path): continue

        img_file = os.path.join(folder_path, "images", folder + ".png")
        mask_files = glob.glob(os.path.join(folder_path, "masks", "*.png"))

        if not os.path.exists(img_file) or len(mask_files) == 0: continue

        merged_path = os.path.join(MERGED_MASK_DIR, folder + ".png")
        
        # Tránh gộp lại nếu chạy lại cell
        if not os.path.exists(merged_path):
            merged = None
            for mf in mask_files:
                m = np.array(Image.open(mf).convert('L'))
                if merged is None:
                    merged = m
                else:
                    merged = np.maximum(merged, m)
            Image.fromarray(merged).save(merged_path)
            
        all_imgs.append(img_file)
        all_msks.append(merged_path)

print(f"Tổng số ảnh hợp lệ: {len(all_imgs)}")
"""))

# ── Cell 3: Dataset Class & Dataloader (80:20 Split) ──
nb.cells.append(nbf.v4.new_code_cell(
r"""random.seed(42)
indices = list(range(len(all_imgs)))
random.shuffle(indices)

# Tỷ lệ 80-20 giống đúng paper gốc
N_TRAIN = int(len(all_imgs) * 0.8)
train_idx = indices[:N_TRAIN]
test_idx  = indices[N_TRAIN:]

class DSBDataset(Dataset):
    def __init__(self, images, masks, size, is_train=False):
        self.images = list(images)
        self.masks  = list(masks)
        self.size   = size
        self.is_train = is_train
        self.transform = A.Compose([
            A.Rotate(limit=35, p=0.3),
            A.HorizontalFlip(p=0.3),
            A.VerticalFlip(p=0.3),
            A.CoarseDropout(p=0.3, num_holes_range=(1, 10), hole_height_range=(1, 32), hole_width_range=(1, 32)),
            A.ElasticTransform(p=0.2, alpha=120, sigma=120*0.05),
            A.GridDistortion(p=0.2),
            A.OpticalDistortion(p=0.2, distort_limit=0.05),
            A.RandomBrightnessContrast(p=0.2),
        ])

    def __getitem__(self, idx):
        # Ảnh gốc DSB có kênh Alpha (RGBA), bắt buộc convert RGB
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

train_loader = DataLoader(
    DSBDataset([all_imgs[i] for i in train_idx], [all_msks[i] for i in train_idx], IMAGE_SIZE, True),
    batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
)
valid_loader = DataLoader(
    DSBDataset([all_imgs[i] for i in test_idx], [all_msks[i] for i in test_idx], IMAGE_SIZE, False),
    batch_size=1, shuffle=False, num_workers=2, pin_memory=True
)
print(f"DSB 2018 Loaded: Train={len(train_idx)}, Test={len(test_idx)}")
"""))

# ── Cell 4: Model (FANet-MFAD) ──
nb.cells.append(nbf.v4.new_code_cell(
r"""class ConvBlock(nn.Module):
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
        self.e1_bn, self.e1_relu, self.pool = resnet.bn1, resnet.relu, resnet.maxpool
        self.e2, self.e3, self.e4, self.b = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

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
        s2, s3, s4 = self.e2(self.pool(s1)), self.e3(s2), self.e4(s3)
        b  = self.b(s4)

        d1 = self.d1(torch.cat([self.up1(b), s4], 1));  d1 = self._mfad(d1, prev_mask, self.att1)
        d2 = self.d2(torch.cat([self.up2(d1), s3], 1)); d2 = self._mfad(d2, prev_mask, self.att2)
        d3 = self.d3(torch.cat([self.up3(d2), s2], 1)); d3 = self._mfad(d3, prev_mask, self.att3)
        d4 = self.d4(torch.cat([self.up4(d3), s1], 1)); d4 = self._mfad(d4, prev_mask, self.att4)
        d5 = self.d5(self.up5(d4));                     d5 = self._mfad(d5, prev_mask, self.att5)
        return self.out(d5)
"""))

# ── Cell 5: Loss & Metrics ──
nb.cells.append(nbf.v4.new_code_cell(
r"""class DiceBCELoss(nn.Module):
    def forward(self, inputs, targets, smooth=1):
        inputs_s, targets_f = torch.sigmoid(inputs).view(-1), targets.view(-1)
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

# ── Cell 6: Training Loop (With AMP & Resume) ──
nb.cells.append(nbf.v4.new_code_cell(
r"""model = FANet_MFAD().to(DEVICE)
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
scaler = torch.amp.GradScaler('cuda')
use_amp = DEVICE.type == 'cuda'

start_epoch, best_dice = 0, 0.0

# Resume từ Kaggle Input nếu phiên trước bị ngắt
RESUME_INPUT_DIR = "/kaggle/input/fanet-mfad-dsb2018-output"
if os.path.exists(os.path.join(RESUME_INPUT_DIR, "fanet_mfad_DSB2018_resume.pth")):
    import shutil
    print("Đã tìm thấy checkpoint cũ từ phiên trước, đang chép sang ổ working...")
    shutil.copy(os.path.join(RESUME_INPUT_DIR, "fanet_mfad_DSB2018_resume.pth"), CKPT_RESUME)
    if os.path.exists(os.path.join(RESUME_INPUT_DIR, "fanet_mfad_DSB2018_best.pth")):
        shutil.copy(os.path.join(RESUME_INPUT_DIR, "fanet_mfad_DSB2018_best.pth"), CKPT_BEST)

if os.path.exists(CKPT_RESUME):
    print(f"Loading checkpoint from {CKPT_RESUME}...")
    ckpt_data = torch.load(CKPT_RESUME, map_location=DEVICE)
    model.load_state_dict(ckpt_data['model'])
    optimizer.load_state_dict(ckpt_data['optimizer'])
    scheduler.load_state_dict(ckpt_data['scheduler'])
    scaler.load_state_dict(ckpt_data['scaler'])
    start_epoch, best_dice = ckpt_data['epoch'] + 1, ckpt_data['best_dice']
    print(f"Resumed from epoch {start_epoch}, best_dice={best_dice:.4f}")

for epoch in range(start_epoch, EPOCHS):
    model.train(); train_loss = 0
    for x, y in tqdm(train_loader, desc=f'E{epoch+1}/{EPOCHS} Train'):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = model(x)
            loss = criterion(model(x, torch.sigmoid(out)), y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    model.eval(); val_loss, val_dice = 0, 0
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

    train_loss /= len(train_loader); val_loss /= len(valid_loader); val_dice /= len(valid_loader)
    print(f"Epoch {epoch+1} | LR {optimizer.param_groups[0]['lr']:.6f} | TrL {train_loss:.4f} | VL {val_loss:.4f} | VDice {val_dice:.4f}")
    scheduler.step(val_loss)

    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), CKPT_BEST)
        print(f"  -> Saved best model (Dice={best_dice:.4f})")

    torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'best_dice': best_dice}, CKPT_RESUME)

print(f"\n{'='*50}")
print(f"BEST VAL DICE on {DATASET_NAME}: {best_dice:.4f}")
print(f"{'='*50}")
"""))

os.makedirs('notebooks', exist_ok=True)
out = 'notebooks/fanet_benchmark_dsb2018.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Generated: {out}")
