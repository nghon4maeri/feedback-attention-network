# %% [markdown]
# # FANet Phase 6 - Advisor feedback: no-FB baseline + asymmetric Tversky loss
#
# Two cells, both seed=43, 200 epochs, batch=2, same split/aug/lr as Phase 4/5:
#   TC | FB × asymmetric loss     | gate=binary, single-path, 0.5*DiceBCE + 0.5*Tversky(alpha=0.7, beta=0.3)
#   TD | no-FB × asymmetric loss  | same loss, feedback mask = zeros (mask-at-input removed)
#
# Pre-registration (docs/reports/2026-09-09_phase6-preregistration.md, Plan B):
#   - Reuse T00 (FB × orig, seed42) and T0N (no-FB × orig, seed43) as the two
#     orig-loss arms (caveat: seed 42 vs 43 remains in Plan B).
#   - The MISSING gap from Phase 5 is "no-FB × new-loss": TD fills it, so a
#     possible FP improvement can be attributed to loss vs feedback.
#   - Tversky alpha=0.7/beta=0.3 penalizes FP ~2.3x harder than FN (verified
#     finite-difference: FP/FN penalty ratio 2.308). Blend lambda=0.5 with
#     DiceBCE keeps Dice scale stable (avoids TB far-weighted collapse).
#
# Loss provenance (verified full-text):
#   - Salehi et al., "Tversky loss function for image segmentation using 3D
#     fully convolutional deep networks", MICCAI-MLMI 2017,
#     TI = |PG|/(|PG| + alpha|P\G| + beta|G\P|).
#   - Unified-Focal (Yeung 2022) pattern: lambda*CE-family + (1-lambda)*Tversky.
#
# Diverge rule (Phase 5 lesson): soft val-Dice hides collapse (TB: soft 0.19 but
# binary 0.0086). Phase 6 logs BINARY val-Dice/Precision/Recall/FPR every epoch;
# diverge gate is applied at epoch 40 on binary val-Dice, NOT on loss scale.
#
# Run on Kaggle with GPU T4. Attach namnguynnnn/kvasir-sessile dataset.

# %% [markdown]
# ## 1. Setup

# %%
!pip install -q albumentations opencv-python-headless tqdm scikit-learn

# %%
import os, time, random
import numpy as np
import cv2
import albumentations as A
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import shuffle as sk_shuffle

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
# ## 2. Dataset path & split (same as Phase 4 / 5 / local)

# %%
CANDIDATES = [
    "/kaggle/input/kvasir-sessile",
    "/kaggle/input/kvasir-seg",
    "/kaggle/input/datasets/namnguynnnn/kvasir/kvasir-sessile/sessile-main-Kvasir-SEG",
]
SRC = None
for p in CANDIDATES:
    if os.path.isdir(p) and os.path.isdir(f"{p}/images"):
        SRC = p
        break
if SRC is None:
    for root, dirs, _ in os.walk("/kaggle/input"):
        if "images" in dirs and "masks" in dirs:
            SRC = root
            break
assert SRC is not None, "dataset not found"
print(f"Source: {SRC}")

DATASET_PATH = "/kaggle/working/Kvasir-SEG"
if not os.path.exists(DATASET_PATH):
    !cp -r "{SRC}" "{DATASET_PATH}"

n_img = len([f for f in os.listdir(f"{DATASET_PATH}/images")
             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))])
print(f"Images: {n_img}")

# Deterministic split identical to local & Phase 4/5 (sorted names, seed 42, 80/20).
names = sorted([os.path.splitext(f)[0] for f in os.listdir(f"{DATASET_PATH}/images")
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))])
np.random.seed(42); np.random.shuffle(names)
split = int(len(names) * 0.8)
if not os.path.exists(f"{DATASET_PATH}/train.txt"):
    with open(f"{DATASET_PATH}/train.txt", 'w') as f:
        f.write("\n".join(names[:split]) + "\n")
    with open(f"{DATASET_PATH}/val.txt", 'w') as f:
        f.write("\n".join(names[split:]) + "\n")
print(f"Split: train={split}, val={len(names)-split}")

# %% [markdown]
# ## 3. Blocks (MixPool, binary gate) - identical to Phase 4/5

# %%
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid())
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.conv3 = nn.Conv2d(in_c, out_c, 1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)
        self.se = SELayer(out_c, out_c)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.bn2(self.conv2(x1))
        x3 = self.se(self.bn3(self.conv3(x)))
        return self.relu(x2 + x3)

class MixPool(nn.Module):
    """gate: binary | ste | soft ; dual_path: bool (Phase 6 uses binary single-path)."""
    def __init__(self, in_c, out_c, gate="binary", dual_path=False):
        super().__init__()
        self.gate = gate
        self.dual_path = dual_path
        self.fmask = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True), nn.Conv2d(out_c, 1, 1, padding=0), nn.Sigmoid())
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, 3, padding=1), nn.BatchNorm2d(out_c // 2),
            nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c // 2, 3, padding=1), nn.BatchNorm2d(out_c // 2),
            nn.ReLU(inplace=True))
    def forward(self, x, m):
        fmask = self.fmask(x)
        m = nn.MaxPool2d((m.shape[2] // x.shape[2], m.shape[3] // x.shape[3]))(m)
        m_fg = m[:, 0:1]
        m_bg = m[:, 1:2] if (self.dual_path and m.shape[1] > 1) else torch.zeros_like(m_fg)
        if self.gate == "binary":
            fmask_g = (fmask > 0.5).float()
        elif self.gate == "ste":
            fmask_g = (fmask > 0.5).float() + fmask - fmask.detach()
        else:
            fmask_g = fmask
        keep = torch.maximum(fmask_g, m_fg)
        if self.dual_path:
            keep = keep * (1.0 - m_bg)
        x1 = self.conv1(x * keep)
        x2 = self.conv2(x)
        return torch.cat([x1, x2], dim=1)

# %% [markdown]
# ## 4. FANet model (identical to Phase 4/5)

# %%
class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, gate="binary", dual_path=False):
        super().__init__()
        self.r1 = ResidualBlock(in_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)
        self.p1 = MixPool(out_c, out_c, gate=gate, dual_path=dual_path)
        self.pool = nn.MaxPool2d(2)
    def forward(self, inputs, masks):
        x = self.r2(self.r1(inputs))
        p = self.p1(x, masks)
        return self.pool(p), x

class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, gate="binary", dual_path=False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_c, in_c, 4, stride=2, padding=1)
        self.r1 = ResidualBlock(in_c * 2, out_c)
        self.r2 = ResidualBlock(out_c, out_c)
        self.p1 = MixPool(out_c, out_c, gate=gate, dual_path=dual_path)
    def forward(self, inputs, skip, masks):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], dim=1)
        x = self.r2(self.r1(x))
        return self.p1(x, masks)

class FANet(nn.Module):
    def __init__(self, gate="binary", dual_path=False):
        super().__init__()
        self.gate = gate
        self.dual_path = dual_path
        self.e1 = EncoderBlock(3, 32, gate, dual_path)
        self.e2 = EncoderBlock(32, 64, gate, dual_path)
        self.e3 = EncoderBlock(64, 128, gate, dual_path)
        self.e4 = EncoderBlock(128, 256, gate, dual_path)
        self.d1 = DecoderBlock(256, 128, gate, dual_path)
        self.d2 = DecoderBlock(128, 64, gate, dual_path)
        self.d3 = DecoderBlock(64, 32, gate, dual_path)
        self.d4 = DecoderBlock(32, 16, gate, dual_path)
        self.output = nn.Conv2d(16 + 1, 1, kernel_size=1)
    def forward(self, x):
        inputs, masks = x[0], x[1]
        p1, s1 = self.e1(inputs, masks)
        p2, s2 = self.e2(p1, masks)
        p3, s3 = self.e3(p2, masks)
        p4, s4 = self.e4(p3, masks)
        d1 = self.d1(p4, s4, masks)
        d2 = self.d2(d1, s3, masks)
        d3 = self.d3(d2, s2, masks)
        d4 = self.d4(d3, s1, masks)
        m_fg = masks[:, 0:1]
        return self.output(torch.cat([d4, m_fg], dim=1))

# %% [markdown]
# ## 5. Losses (Phase 6) + utils

# %%
class DiceBCELoss(nn.Module):
    def forward(self, inputs, targets, smooth=1.0):
        p = torch.sigmoid(inputs).view(-1); g = targets.view(-1)
        inter = (p * g).sum()
        dice = 1 - (2. * inter + smooth) / (p.sum() + g.sum() + smooth)
        bce = F.binary_cross_entropy(p, g, reduction='mean')
        return 0.5 * bce + 0.5 * dice

class TverskyLoss(nn.Module):
    """Asymmetric Tversky (Salehi 2017): alpha=0.7 penalizes FP ~2.3x > beta=0.3 FN."""
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__(); self.alpha = alpha; self.beta = beta; self.smooth = smooth
    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs).view(-1); g = targets.view(-1)
        tp = (p * g).sum(); fp = (p * (1 - g)).sum(); fn = ((1 - p) * g).sum()
        tv = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tv

class Phase6AsymmetricBCELoss(nn.Module):
    """0.5*DiceBCE + 0.5*Tversky(alpha=0.7, beta=0.3). DiceBCE keeps Dice scale stable."""
    def __init__(self, alpha=0.7, beta=0.3, lam=0.5, smooth=1.0):
        super().__init__(); self.lam = lam
        self.dice_bce = DiceBCELoss(); self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)
    def forward(self, inputs, targets):
        return self.lam * self.dice_bce(inputs, targets) + (1.0 - self.lam) * self.tversky(inputs, targets)

def rle_encode(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run = []; prev = -2
    for b in dots:
        if b > prev + 1:
            run.extend((b + 1, 0))
        run[-1] += 1; prev = b
    return run

def rle_decode(mask_rle, shape):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1; ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')

def init_mask(paths, size=(256, 256)):
    masks = []
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, size)
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        masks.append(rle_encode((th / 255.0 > 0.5).astype(np.int32)))
    return masks

def load_data(path):
    def _load(fname):
        with open(f"{path}/{fname}") as f:
            data = f.read().split("\n")[:-1]
        imgs, msks = [], []
        for n in data:
            for ext in (".jpg", ".jpeg", ".png", ".tif"):
                if os.path.exists(os.path.join(path, "images", n + ext)):
                    imgs.append(os.path.join(path, "images", n + ext))
                    msks.append(os.path.join(path, "masks", n + ext))
                    break
        return imgs, msks
    return _load("train.txt"), _load("val.txt")

def seeding(seed=42):
    random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.backends.cudnn.deterministic = True

class KvasirDataset(Dataset):
    def __init__(self, img_paths, mask_paths, size=(256, 256), transform=None):
        self.img_paths = img_paths; self.mask_paths = mask_paths
        self.size = size; self.transform = transform
    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx], cv2.IMREAD_COLOR)
        msk = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img, msk = aug["image"], aug["mask"]
        img = cv2.resize(img, self.size)
        img = np.transpose(img, (2, 0, 1)) / 255.0
        msk = cv2.resize(msk, self.size, interpolation=cv2.INTER_NEAREST)
        msk = np.expand_dims(msk, axis=0)
        msk = (msk > 127).astype(np.float32)
        return img.astype(np.float32), msk
    def __len__(self):
        return len(self.img_paths)

# %% [markdown]
# ## 6. Data loaders (batch=2, seed for TRAINING = 43)

# %%
SEED = 43
seeding(SEED)
(train_x, train_y), (valid_x, valid_y) = load_data(DATASET_PATH)
train_x, train_y = sk_shuffle(train_x, train_y, random_state=SEED)
print(f"Train: {len(train_x)}, Val: {len(valid_x)}")

transform = A.Compose([
    A.Rotate(limit=35, p=0.3),
    A.HorizontalFlip(p=0.3),
    A.VerticalFlip(p=0.3),
    A.CoarseDropout(p=0.3, num_holes=10, hole_height=32, hole_width=32,
                    fill_value=0, mask_fill_value=0),
])
BATCH = 2
train_ds = KvasirDataset(train_x, train_y, transform=transform)
valid_ds = KvasirDataset(valid_x, valid_y)
train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=False, num_workers=2)
valid_loader = DataLoader(valid_ds, batch_size=BATCH, shuffle=False, num_workers=2)

# %% [markdown]
# ## 7. Train / eval loop (no-feedback mode = zero mask; BINARY val metrics each epoch)

# %%
def feedback_tensor(mask_list, start, b, size=(256, 256), dual_path=False, no_feedback=False):
    if no_feedback:
        return torch.zeros(b, 1, size[0], size[1])
    ms = []
    for edata in mask_list[start:start + b]:
        dec = rle_decode(" ".join(str(d) for d in edata), size)
        ms.append(np.expand_dims(dec, 0))
    arr = np.array(ms, dtype=np.int32)
    arr = np.transpose(arr, (0, 1, 3, 2))
    if dual_path:
        bg = (1.0 - arr)
        arr = np.concatenate([arr, bg], axis=1)
    return torch.from_numpy(arr).float()

def train_epoch(model, loader, mask_list, opt, loss_fn, no_feedback):
    model.train(); total = 0; new_masks = []
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        b = y.shape[0]
        m = feedback_tensor(mask_list, i * BATCH, b, no_feedback=no_feedback).to(device)
        opt.zero_grad()
        pred = model([x, m])
        loss = loss_fn(pred, y)
        loss.backward(); opt.step()
        with torch.no_grad():
            for py in (torch.sigmoid(pred) > 0.5).cpu().numpy().astype(np.uint8):
                new_masks.append(rle_encode(np.squeeze(py, 0)))
        total += loss.item()
    return total / len(loader), new_masks

def binary_metrics_batch(pred, y):
    """Per-image binary (threshold 0.5) metrics: Dice, Precision, Recall, FPR, FN."""
    p = (torch.sigmoid(pred) > 0.5).float()
    g = (y > 0.5).float()
    tp = (p * g).sum(dim=(1, 2, 3))
    pred_pos = p.sum(dim=(1, 2, 3))
    gt_pos = g.sum(dim=(1, 2, 3))
    fp = (p * (1 - g)).sum(dim=(1, 2, 3))
    fn = ((1 - p) * g).sum(dim=(1, 2, 3))
    dice = 2 * tp / (pred_pos + gt_pos + 1e-15)
    prec = tp / (pred_pos + 1e-15)
    rec = tp / (gt_pos + 1e-15)
    fpr = fp / (g.numel() / g.shape[0] + 1e-15)
    fnr = fn / (g.numel() / g.shape[0] + 1e-15)
    return dice.mean().item(), prec.mean().item(), rec.mean().item(), \
        fpr.mean().item(), fnr.mean().item()

def eval_epoch(model, loader, mask_list, loss_fn, no_feedback):
    model.eval(); total = 0; new_masks = []
    b_dice = 0.0; b_prec = 0.0; b_rec = 0.0; b_fpr = 0.0; b_fnr = 0.0
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            b = y.shape[0]
            m = feedback_tensor(mask_list, i * BATCH, b, no_feedback=no_feedback).to(device)
            pred = model([x, m])
            loss = loss_fn(pred, y); total += loss.item()
            d, pr, rc, fpr, fnr = binary_metrics_batch(pred, y)
            b_dice += d; b_prec += pr; b_rec += rc; b_fpr += fpr; b_fnr += fnr
            for py in (torch.sigmoid(pred) > 0.5).cpu().numpy().astype(np.uint8):
                new_masks.append(rle_encode(np.squeeze(py, 0)))
    n = len(loader)
    return total / n, new_masks, {
        "bin_dice": b_dice / n, "bin_prec": b_prec / n, "bin_rec": b_rec / n,
        "bin_fpr": b_fpr / n, "bin_fnr": b_fnr / n,
    }

# %% [markdown]
# ## 8. Train 2 cells (seed 43 each)

# %%
import csv

CELLS = [
    {"name": "TC", "gate": "binary", "dual": False, "no_feedback": False,
     "loss": "tversky", "loss_kwargs": {"alpha": 0.7, "beta": 0.3, "lam": 0.5}},
    {"name": "TD", "gate": "binary", "dual": False, "no_feedback": True,
     "loss": "tversky", "loss_kwargs": {"alpha": 0.7, "beta": 0.3, "lam": 0.5}},
]
NUM_EPOCHS = 200
LR = 1e-4
CKPT_DIR = "/kaggle/working/checkpoints_phase6"
os.makedirs(CKPT_DIR, exist_ok=True)

def build_loss(name, kwargs):
    if name == "tversky":
        return Phase6AsymmetricBCELoss(**kwargs)
    raise ValueError(name)

for cell in CELLS:
    name = cell["name"]
    no_feedback = cell["no_feedback"]
    print(f"\n===== Training {name} (no_feedback={no_feedback}, loss={cell['loss']}) =====")

    seeding(SEED)
    model = FANet(gate=cell["gate"], dual_path=cell["dual"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', patience=5)
    loss_fn = build_loss(cell["loss"], cell["loss_kwargs"])

    train_mask = init_mask(train_x)
    valid_mask = init_mask(valid_x)
    best_val = float('inf')
    best_dice = 0.0
    log_rows = []

    for ep in range(NUM_EPOCHS):
        t0 = time.time()
        tr_loss, tr_new = train_epoch(model, train_loader, train_mask, opt, loss_fn, no_feedback)
        va_loss, va_new, bm = eval_epoch(model, valid_loader, valid_mask, loss_fn, no_feedback)
        sched.step(va_loss)
        improved = False
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), f"{CKPT_DIR}/ckpt_{name}.pth")
            if not no_feedback:
                train_mask, valid_mask = tr_new, va_new
            improved = True
        best_dice = max(best_dice, bm["bin_dice"])
        log_rows.append({
            "cell": name, "epoch": ep + 1, "train_loss": round(tr_loss, 5),
            "val_loss": round(va_loss, 5), "best_val": round(best_val, 5),
            "bin_val_dice": round(bm["bin_dice"], 5),
            "bin_val_prec": round(bm["bin_prec"], 5),
            "bin_val_rec": round(bm["bin_rec"], 5),
            "bin_val_fpr": round(bm["bin_fpr"], 5),
            "bin_val_fnr": round(bm["bin_fnr"], 5),
            "best_bin_dice": round(best_dice, 5),
            "lr": opt.param_groups[0]['lr'], "improved": int(improved),
            "time_s": round(time.time() - t0, 1),
        })
        if (ep + 1) % 20 == 0 or ep == 0:
            print(f"  ep {ep+1:>3}: train {tr_loss:.4f} val {va_loss:.4f} "
                  f"binDice {bm['bin_dice']:.4f} prec {bm['bin_prec']:.4f} "
                  f"rec {bm['bin_rec']:.4f} fpr {bm['bin_fpr']:.4f}")

        # Diverge rule at epoch 40 (pre-registered): binary val-Dice, not loss scale.
        if ep + 1 == 40 and bm["bin_dice"] < 0.02 and bm["bin_rec"] < 0.05:
            print(f"  !!! Diverge: ep40 bin_val_dice {bm['bin_dice']:.4f} ~ collapse. "
                  f"Stopping cell {name} per pre-registered diverge rule.")
            break

    with open(f"{CKPT_DIR}/train_log_{name}.csv", 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        wr.writeheader(); wr.writerows(log_rows)
    print(f"  Saved ckpt_{name}.pth + train_log_{name}.csv (best val {best_val:.4f}, "
          f"best binDice {best_dice:.4f})")

print("\nDone training 2 cells.")