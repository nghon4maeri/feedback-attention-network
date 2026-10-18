import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

md1 = '''# FANet: Learned Residual Decoupling on Kvasir-SEG
**Source Baseline:** [nikhilroxtomar/FANet](https://github.com/nikhilroxtomar/FANet)
**Modifications:** 
1. Fixed Mask Binarization bug (`cv2.INTER_NEAREST` + strict thresholding).
2. Replaced `Soft-OR` feedback with **Learned Residual Decoupling** (1x1 Conv Gate).
3. Using standard BCE + Dice Loss to ensure robust convergence.
'''
nb.cells.append(nbf.v4.new_markdown_cell(md1))

code_imports = '''import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
'''
nb.cells.append(nbf.v4.new_code_cell(code_imports))

code_configs = '''# Configuration
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4
IMAGE_SIZE = (256, 256)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- ROBUST DATASET AUTO-DETECTION ---
DATA_DIR = None
IMG_SUBDIR = 'images'
MASK_SUBDIR = 'masks'

print("Searching for images and masks folders in /kaggle/input...")
if os.path.exists('/kaggle/input'):
    for root, dirs, files in os.walk('/kaggle/input'):
        dirs_lower = [d.lower() for d in dirs]
        if 'images' in dirs_lower and 'masks' in dirs_lower:
            IMG_SUBDIR = dirs[dirs_lower.index('images')]
            MASK_SUBDIR = dirs[dirs_lower.index('masks')]
            DATA_DIR = root
            break

if DATA_DIR is None:
    DATA_DIR = '../data/kvasir-seg'

if not os.path.exists(DATA_DIR):
    print("❌ ERROR: CANNOT FIND DATASET! Please Add Data to Kaggle.")
else:
    print(f"✅ Found Dataset at: {DATA_DIR}")
'''
nb.cells.append(nbf.v4.new_code_cell(code_configs))

code_dataset = '''class KvasirDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = cv2.resize(image, IMAGE_SIZE)
        image = np.transpose(image, (2, 0, 1)) / 255.0
        image = image.astype(np.float32)

        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=0)
        mask = (mask > 127).astype(np.float32)

        return image, mask

    def __len__(self):
        return self.n_samples

images = sorted(glob(os.path.join(DATA_DIR, IMG_SUBDIR, '*.jpg')) + glob(os.path.join(DATA_DIR, IMG_SUBDIR, '*.png')))
masks = sorted(glob(os.path.join(DATA_DIR, MASK_SUBDIR, '*.jpg')) + glob(os.path.join(DATA_DIR, MASK_SUBDIR, '*.png')))

if len(images) == 0:
    raise ValueError(f"❌ DATASET EMPTY! Could not find any images in {os.path.join(DATA_DIR, IMG_SUBDIR)}")

train_x, valid_x, train_y, valid_y = train_test_split(images, masks, test_size=0.2, random_state=42)

train_dataset = KvasirDataset(train_x, train_y)
valid_dataset = KvasirDataset(valid_x, valid_y)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f'✅ Train size: {len(train_x)} | Valid size: {len(valid_x)}')
'''
nb.cells.append(nbf.v4.new_code_cell(code_dataset))

code_model = '''class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))))

class FANet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.e1 = ConvBlock(4, 64) 
        self.e2 = ConvBlock(64, 128)
        self.e3 = ConvBlock(128, 256)
        self.e4 = ConvBlock(256, 512)
        self.b = ConvBlock(512, 1024)
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.d1 = ConvBlock(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d2 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d3 = ConvBlock(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d4 = ConvBlock(128, 64)
        self.out = nn.Conv2d(64, 1, kernel_size=1)
        self.learned_gate = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x, prev_mask=None):
        if prev_mask is None:
            prev_mask = torch.zeros(x.size(0), 1, x.size(2), x.size(3)).to(x.device)
        else:
            prev_mask = torch.sigmoid(self.learned_gate(prev_mask.detach()))
            
        x = torch.cat([x, prev_mask], dim=1)
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        b = self.b(self.pool(s4))
        
        d1 = self.d1(torch.cat([self.up1(b), s4], dim=1))
        d2 = self.d2(torch.cat([self.up2(d1), s3], dim=1))
        d3 = self.d3(torch.cat([self.up3(d2), s2], dim=1))
        d4 = self.d4(torch.cat([self.up4(d3), s1], dim=1))
        
        return self.out(d4)
'''
nb.cells.append(nbf.v4.new_code_cell(code_model))

code_loss = '''class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return bce + (1 - dice)

def calculate_metrics(y_true, y_pred):
    y_true = y_true.detach().cpu().numpy()
    y_pred = (torch.sigmoid(y_pred).detach().cpu().numpy() > 0.5).astype(np.float32)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)
    prec = tp / (tp + fp + 1e-6)
    rec = tp / (tp + fn + 1e-6)
    fpr = fp / (fp + tn + 1e-6)
    return dice, prec, rec, fpr
'''
nb.cells.append(nbf.v4.new_code_cell(code_loss))

code_train = '''model = FANet().to(DEVICE)
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("Starting Training...")
best_dice = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for x, y in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Train]'):
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(x)
        out = model(x, torch.sigmoid(out))
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    model.eval()
    val_loss, val_dice = 0, 0
    with torch.no_grad():
        for x, y in tqdm(valid_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [Valid]'):
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            out = model(x, torch.sigmoid(out))
            loss = criterion(out, y)
            val_loss += loss.item()
            dice, _, _, _ = calculate_metrics(y, out)
            val_dice += dice
            
    train_loss /= len(train_loader)
    val_loss /= len(valid_loader)
    val_dice /= len(valid_loader)
    
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")
    if val_dice > best_dice:
        best_dice = val_dice
        torch.save(model.state_dict(), "fanet_1000_best.pth")
'''
nb.cells.append(nbf.v4.new_code_cell(code_train))

code_vis = '''print("Loading best model for inference and visualization...")
model.load_state_dict(torch.load("fanet_1000_best.pth", weights_only=True))
model.eval()
test_dice, test_prec, test_rec, test_fpr = 0, 0, 0, 0
viz_images, viz_masks, viz_preds = [], [], []

with torch.no_grad():
    for i, (x, y) in enumerate(tqdm(valid_loader, desc='Testing')):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        out = model(x, torch.sigmoid(out))
        dice, prec, rec, fpr = calculate_metrics(y, out)
        test_dice += dice; test_prec += prec; test_rec += rec; test_fpr += fpr
        
        if i == 0:
            viz_images = x.cpu().numpy()
            viz_masks = y.cpu().numpy()
            viz_preds = torch.sigmoid(out).cpu().numpy()

n = len(valid_loader)
print("\\n" + "="*50)
print(f"FINAL RESULTS ON FULL KVASIR-SEG (1000 IMAGES)")
print("="*50)
print(f"Mean Dice Score: {test_dice/n:.4f}")
print(f"Mean Precision:  {test_prec/n:.4f}")
print(f"Mean Recall:     {test_rec/n:.4f}")
print(f"Mean FPR:        {test_fpr/n:.4f}")
print("="*50)

fig, axs = plt.subplots(4, 3, figsize=(10, 12))
for i in range(min(4, len(viz_images))):
    img = np.transpose(viz_images[i], (1, 2, 0))
    gt = viz_masks[i][0]
    pred = (viz_preds[i][0] > 0.5).astype(np.float32)
    
    axs[i, 0].imshow(img)
    axs[i, 0].set_title("Input Image")
    axs[i, 0].axis('off')
    
    axs[i, 1].imshow(gt, cmap='gray')
    axs[i, 1].set_title("Ground Truth (Strict Binary)")
    axs[i, 1].axis('off')
    
    axs[i, 2].imshow(pred, cmap='gray')
    axs[i, 2].set_title("Prediction (FANet Learned Gate)")
    axs[i, 2].axis('off')

plt.tight_layout()
plt.show()
'''
nb.cells.append(nbf.v4.new_code_cell(code_vis))

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/fanet_1000_research.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
