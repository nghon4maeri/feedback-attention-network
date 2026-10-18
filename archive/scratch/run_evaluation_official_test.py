import os, sys, time, json, csv, glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[+] Device: {DEVICE} | GPU: {torch.cuda.get_device_name(0)}")

DATA_ROOT = "/content/isic-2018"
IMAGE_SIZE = (512, 512)

# Model Definition
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
        resnet = models.resnet34(weights=None)

        self.e1_conv = nn.Conv2d(4, 64, 7, stride=2, padding=3, bias=False)
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

# Dataset
class TestDataset(Dataset):
    def __init__(self, images_path, masks_path, image_size):
        self.images_path = list(images_path)
        self.masks_path  = list(masks_path)
        self.image_size  = image_size

    def __getitem__(self, index):
        p_img = self.images_path[index]
        p_msk = self.masks_path[index]

        img = cv2.imread(p_img, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.array(Image.open(p_img).convert('RGB'))
        elif len(img.shape) == 2:
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        msk = cv2.imread(p_msk, cv2.IMREAD_UNCHANGED)
        if msk is None:
            msk = np.array(Image.open(p_msk).convert('L'))
        elif len(msk.shape) == 3:
            msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, self.image_size, interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(msk, self.image_size, interpolation=cv2.INTER_NEAREST)

        img = img.astype(np.float32)
        mx = img.max()
        if mx > 0:
            img = img / (65535.0 if mx > 255 else 255.0)
        img = np.transpose(img, (2, 0, 1))

        msk = msk.astype(np.float32)
        if msk.max() > 0:
            msk = (msk > (msk.max() + msk.min()) / 2.0).astype(np.float32)
        if msk.ndim == 2:
            msk = np.expand_dims(msk, axis=0)

        return img, msk

    def __len__(self):
        return len(self.images_path)

def compute_metrics(gt, pred):
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

    if (np.sum(gt) > 0 and np.sum(pred) > 0):
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

# Locate official test set
def find_dir_by_name(target_name, base_search=DATA_ROOT):
    for root, dirs, files in os.walk(base_search):
        if os.path.basename(root).lower() == target_name.lower():
            return root
    return None

def isic_id(path):
    base = os.path.basename(path).replace("_segmentation", "")
    return os.path.splitext(base)[0]

test_img_dir = find_dir_by_name("ISIC2018_Task1-2_Test_Input")
test_gt_dir  = find_dir_by_name("ISIC2018_Task1_Test_GroundTruth")

test_jpgs = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg"))) if test_img_dir else []
test_pngs = sorted(glob.glob(os.path.join(test_gt_dir, "*.png")))  if test_gt_dir else []
test_img_dict = {isic_id(p): p for p in test_jpgs}
test_msk_dict = {isic_id(p): p for p in test_pngs}
test_ids = sorted(set(test_img_dict.keys()) & set(test_msk_dict.keys()))

test_x = [test_img_dict[k] for k in test_ids]
test_y = [test_msk_dict[k] for k in test_ids]
print(f"[+] Found {len(test_x)} official test pairs.")

test_dataset = TestDataset(test_x, test_y, IMAGE_SIZE)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

# Determine best checkpoint
best_cands = [
    "/content/GDFNet_ISIC2018_best.pth",
    "/content/kaggle_out/GDFNet_ISIC2018_best.pth",
    "/content/GDFNet_ISIC2018_latest.pth"
]
best_ckpt = next((p for p in best_cands if os.path.exists(p)), None)
print(f"[*] Selected checkpoint for test evaluation: {best_ckpt}")

model = GDFNet().to(DEVICE)
state_dict = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
if 'model_state' in state_dict:
    state_dict = state_dict['model_state']
model.load_state_dict(state_dict)
model.eval()

all_metrics = []
all_imgs, all_gts, all_preds = [], [], []

print(f"\n{'='*70}")
print(f"  RUNNING OFFICIAL TEST EVALUATION (N={len(test_loader)}) ON A100 GPU")
print(f"{'='*70}")

with torch.no_grad():
    for x, y in tqdm(test_loader, desc="[GDF-Net A100 Evaluation]"):
        x, y = x.to(DEVICE), y.to(DEVICE)
        with torch.amp.autocast('cuda'):
            out1 = model(x)
            out2 = model(x, torch.sigmoid(out1))
        pred = (torch.sigmoid(out2).cpu().numpy() > 0.5).astype(np.float32)
        gt   = y.cpu().numpy()
        imgs = x.cpu().numpy()

        for b_i in range(pred.shape[0]):
            m = compute_metrics(gt[b_i], pred[b_i])
            all_metrics.append(m)
            if len(all_imgs) < 10:
                all_imgs.append(imgs[b_i])
                all_gts.append(gt[b_i])
                all_preds.append(pred[b_i])

agg = {}
for k in all_metrics[0].keys():
    vals = [m[k] for m in all_metrics if k in m]
    agg[k] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals))}

print(f"\n{'='*70}")
print(f"  FINAL BENCHMARK RESULTS ON OFFICIAL ISIC-2018 TEST SET (N={len(test_loader)})")
print(f"{'='*70}")
print(f"  {'Metric':<20s} | {'Mean ± Std':>25s}")
print(f"  {'-'*20}-+-{'-'*25}")
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
print(f"[+] Saved results to: {csv_path}")

# Plot Qualitative Samples
n = min(5, len(all_imgs))
fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n))
for i in range(n):
    img = np.transpose(all_imgs[i], (1, 2, 0))
    img = np.clip(img, 0, 1)
    axes[i, 0].imshow(img)
    axes[i, 0].set_title('Test Input Image' if i == 0 else '')
    axes[i, 0].axis('off')

    axes[i, 1].imshow(all_gts[i][0], cmap='gray')
    axes[i, 1].set_title('Official Ground Truth' if i == 0 else '')
    axes[i, 1].axis('off')

    axes[i, 2].imshow(all_preds[i][0], cmap='gray')
    axes[i, 2].set_title('GDF-Net Prediction' if i == 0 else '')
    axes[i, 2].axis('off')

plt.tight_layout()
fig_path = "/content/comparison_ISIC-2018.png"
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"[+] Saved comparison plot to: {fig_path}")

# Copy to GDFNet_ISIC2018_best.pth for neatness
subprocess.run(["cp", best_ckpt, "/content/GDFNet_ISIC2018_best.pth"])
print("[+] Evaluation completed successfully!")
