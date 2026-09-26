import nbformat as nbf
import os

nb = nbf.v4.new_notebook()

# Cell 1: Markdown
md1 = '''# FANet: Comprehensive Benchmark (Learned Residual Decoupling)
**Architecture:** ResNet-34 Encoder + U-Net Decoder + 1x1 Conv Learned Gating
**Loss Function:** BCE + Dice Loss
**Ground Truth:** Strict Binary Masks (cv2.INTER_NEAREST)

### 🎯 Mục Tiêu Đánh Giá (So với Baseline FANet gốc)
* **Kvasir-SEG:** > 0.8803
* **CVC-ClinicDB:** > 0.9355
* **ISIC 2018:** > 0.8731
* **2018 Data Science Bowl:** > 0.9176
* **DRIVE Database:** > 0.8183
* **CHASE-DB1:** > 0.8108
* **EM dataset:** > 0.9547
'''
nb.cells.append(nbf.v4.new_markdown_cell(md1))

# Cell 2: Imports
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
import torchvision.models as models
'''
nb.cells.append(nbf.v4.new_code_cell(code_imports))

# Cell 3: Configs
code_configs = '''# ==============================================================================
# BƯỚC 1: CẤU HÌNH DATASET
# (Hãy uncomment đúng khối Dataset mà bạn đã Add vào Kaggle)
# ==============================================================================

DATASET_NAME = "Kvasir-SEG"
DATA_DIR = "/kaggle/input/kvasir-seg"
IMG_SUBDIR = "images"
MASK_SUBDIR = "masks"

# DATASET_NAME = "CVC-ClinicDB"
# DATA_DIR = "/kaggle/input/cvc-clinicdb" # Thay bằng tên thư mục chuẩn nếu bạn add bộ khác
# IMG_SUBDIR = "Original" # Có thể là 'images', 'Original'
# MASK_SUBDIR = "Ground Truth" # Có thể là 'masks', 'Ground Truth'

# DATASET_NAME = "ISIC-2018"
# DATA_DIR = "/kaggle/input/isic2018-challenge-task1-data-segmentation"
# IMG_SUBDIR = "ISIC2018_Task1-2_Training_Input"
# MASK_SUBDIR = "ISIC2018_Task1_Training_GroundTruth"

# DATASET_NAME = "DSB-2018"
# DATA_DIR = "/kaggle/input/data-science-bowl-2018" # Hãy dùng bản pre-processed 2D images/masks
# IMG_SUBDIR = "images"
# MASK_SUBDIR = "masks"

# DATASET_NAME = "DRIVE"
# DATA_DIR = "/kaggle/input/drive-digital-retinal-images-for-vessel-extraction"
# IMG_SUBDIR = "images"
# MASK_SUBDIR = "masks"

# DATASET_NAME = "CHASE-DB1"
# DATA_DIR = "/kaggle/input/chasedb1"
# IMG_SUBDIR = "images"
# MASK_SUBDIR = "masks"

# ==============================================================================
# BƯỚC 2: CẤU HÌNH TRAINING
# ==============================================================================
EPOCHS = 30 # Các tập khó như ISIC có thể cần 30-40 epochs
BATCH_SIZE = 8
LR = 1e-4
IMAGE_SIZE = (256, 256)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Auto-fallback search if the exact subdirs aren't found
print(f"Checking dataset: {DATASET_NAME} at {DATA_DIR}...")
if not os.path.exists(os.path.join(DATA_DIR, IMG_SUBDIR)):
    print(f"Warning: {IMG_SUBDIR} not found. Initiating auto-search...")
    for root, dirs, files in os.walk('/kaggle/input'):
        dirs_lower = [d.lower() for d in dirs]
        if any(x in dirs_lower for x in ['images', 'original', 'input']) and \\
           any(x in dirs_lower for x in ['masks', 'ground truth', 'labels']):
            IMG_SUBDIR = dirs[dirs_lower.index(next(x for x in ['images', 'original', 'input'] if x in dirs_lower))]
            MASK_SUBDIR = dirs[dirs_lower.index(next(x for x in ['masks', 'ground truth', 'labels'] if x in dirs_lower))]
            DATA_DIR = root
            break

print(f"✅ FINAL PATHS: Images -> {os.path.join(DATA_DIR, IMG_SUBDIR)} | Masks -> {os.path.join(DATA_DIR, MASK_SUBDIR)}")
'''
nb.cells.append(nbf.v4.new_code_cell(code_configs))

# Cell 4: Dataset
code_dataset = '''class UniversalDataset(Dataset):
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
        # Strict binary conversion
        mask = (mask > 127).astype(np.float32)

        return image, mask

    def __len__(self):
        return self.n_samples

# Dynamically find extensions
exts = ['*.jpg', '*.png', '*.tif', '*.tiff', '*.jpeg']
images, masks = [], []
for ext in exts:
    images.extend(glob(os.path.join(DATA_DIR, IMG_SUBDIR, '**', ext), recursive=True))
    masks.extend(glob(os.path.join(DATA_DIR, MASK_SUBDIR, '**', ext), recursive=True))

images = sorted(images)
masks = sorted(masks)

if len(images) == 0:
    raise ValueError("❌ DATASET EMPTY! Please check Kaggle Data tab and paths.")

# Clean up mismatched files if datasets are unaligned (some sets have hidden files)
if len(images) != len(masks):
    print(f"⚠️ Mismatch: {len(images)} images vs {len(masks)} masks. Truncating to minimum.")
    min_len = min(len(images), len(masks))
    images, masks = images[:min_len], masks[:min_len]

train_x, valid_x, train_y, valid_y = train_test_split(images, masks, test_size=0.2, random_state=42)

train_dataset = UniversalDataset(train_x, train_y)
valid_dataset = UniversalDataset(valid_x, valid_y)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f'✅ {DATASET_NAME} loaded! Train size: {len(train_x)} | Valid size: {len(valid_x)}')
'''
nb.cells.append(nbf.v4.new_code_cell(code_dataset))

# Cell 5: Model
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

class FANet_ResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        
        # Modify Conv1 for 4 channels (RGB + Mask)
        self.e1_conv = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.e1_conv.weight[:, :3, :, :] = resnet.conv1.weight
            self.e1_conv.weight[:, 3:4, :, :] = 0.0
            
        self.e1_bn = resnet.bn1
        self.e1_relu = resnet.relu
        self.pool = resnet.maxpool
        
        self.e2 = resnet.layer1
        self.e3 = resnet.layer2
        self.e4 = resnet.layer3
        self.b  = resnet.layer4
        
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.d1 = ConvBlock(256 + 256, 256)
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.d2 = ConvBlock(128 + 128, 128)
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.d3 = ConvBlock(64 + 64, 64)
        
        self.up4 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.d4 = ConvBlock(64 + 64, 64)
        
        self.up5 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.d5 = ConvBlock(32, 32)
        
        self.out = nn.Conv2d(32, 1, kernel_size=1)
        # LEARNED GATE
        self.learned_gate = nn.Conv2d(1, 1, kernel_size=1)

    def forward(self, x, prev_mask=None):
        if prev_mask is None:
            prev_mask = torch.zeros(x.size(0), 1, x.size(2), x.size(3)).to(x.device)
        else:
            prev_mask = torch.sigmoid(self.learned_gate(prev_mask.detach()))
            
        x = torch.cat([x, prev_mask], dim=1)
        s1 = self.e1_relu(self.e1_bn(self.e1_conv(x)))
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(s2)
        s4 = self.e4(s3)
        b = self.b(s4)
        
        d1 = self.d1(torch.cat([self.up1(b), s4], dim=1))
        d2 = self.d2(torch.cat([self.up2(d1), s3], dim=1))
        d3 = self.d3(torch.cat([self.up3(d2), s2], dim=1))
        d4 = self.d4(torch.cat([self.up4(d3), s1], dim=1))
        d5 = self.d5(self.up5(d4))
        
        return self.out(d5)
'''
nb.cells.append(nbf.v4.new_code_cell(code_model))

# Cell 6: Loss and Metrics
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

# Cell 7: Train loop
code_train = '''model = FANet_ResNet34().to(DEVICE)
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
ckpt_name = f"fanet_{DATASET_NAME.replace(' ', '_')}_best.pth"

print(f"Starting Training on {DATASET_NAME}...")
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
        torch.save(model.state_dict(), ckpt_name)
'''
nb.cells.append(nbf.v4.new_code_cell(code_train))

# Cell 8: Visualization
code_vis = '''print("Loading best model for inference...")
model.load_state_dict(torch.load(ckpt_name, weights_only=True))
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
print(f"FINAL RESULTS ON {DATASET_NAME}")
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
    axs[i, 1].set_title("Ground Truth")
    axs[i, 1].axis('off')
    
    axs[i, 2].imshow(pred, cmap='gray')
    axs[i, 2].set_title("FANet (Learned Gate)")
    axs[i, 2].axis('off')

plt.tight_layout()
plt.show()
'''
nb.cells.append(nbf.v4.new_code_cell(code_vis))

os.makedirs('notebooks', exist_ok=True)
with open('notebooks/fanet_universal_benchmark.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
