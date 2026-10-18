"""Generate a FANet report (params, dataset, short training, refinement demo).

Usage:
    python scripts/report.py [--config configs/kvasir_sessile.yaml] [--epochs 15]
"""
import argparse
import time

import albumentations as A
import cv2
import numpy as np
import torch
from sklearn.utils import shuffle as sk_shuffle
from torch.utils.data import DataLoader

from fanet.config import load_config
from fanet.data import DATASET, load_data, rle_batch_to_tensor
from fanet.losses import DiceBCELoss
from fanet.models import FANet
from fanet.utils import seeding, create_dir, init_mask, rle_encode, print_and_save

BATCH = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, loader, mask, optimizer, loss_fn, size):
    epoch_loss = 0
    return_mask = []
    model.train()
    for i, (x, y) in enumerate(loader):
        x = x.to(DEVICE, dtype=torch.float32)
        y = y.to(DEVICE, dtype=torch.float32)
        m = rle_batch_to_tensor(mask, i * BATCH, y.shape[0], size).to(DEVICE)

        optimizer.zero_grad()
        y_pred = model([x, m])
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred_bin = (torch.sigmoid(y_pred) > 0.5).cpu().numpy().astype(np.uint8)
            for py in y_pred_bin:
                return_mask.append(rle_encode(np.squeeze(py, axis=0)))
        epoch_loss += loss.item()
    return epoch_loss / len(loader), return_mask


def eval_epoch(model, loader, mask, loss_fn, size):
    epoch_loss = 0
    return_mask = []
    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(DEVICE, dtype=torch.float32)
            y = y.to(DEVICE, dtype=torch.float32)
            m = rle_batch_to_tensor(mask, i * BATCH, y.shape[0], size).to(DEVICE)

            y_pred = model([x, m])
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            y_pred_bin = (torch.sigmoid(y_pred) > 0.5).cpu().numpy().astype(np.uint8)
            for py in y_pred_bin:
                return_mask.append(rle_encode(np.squeeze(py, axis=0)))
    return epoch_loss / len(loader), return_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kvasir_sessile.yaml")
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    cfg = load_config(args.config)
    size = tuple(cfg["dataset"]["image_size"])
    epochs = args.epochs
    lr = cfg["train"]["lr"]
    ckpt_path = cfg["paths"]["checkpoint"]
    log_path = cfg["paths"]["train_log"].replace("train_log.txt", "report_log.txt")
    aug = cfg["train"]["augmentation"]

    seeding(cfg["train"]["seed"])
    create_dir("logs")
    print_and_save(log_path, f"Report generated: {time.ctime()}")

    print_and_save(log_path, "\n===== CONFIGURATION =====")
    print_and_save(log_path, f"Device:       {DEVICE}")
    print_and_save(log_path, f"Image size:   {size}")
    print_and_save(log_path, f"Batch size:   {BATCH}")
    print_and_save(log_path, f"Epochs:       {epochs}")
    print_and_save(log_path, f"Learning rate:{lr}")
    print_and_save(log_path, f"Loss:         DiceBCELoss (0.5*Dice + 0.5*BCE)")

    model = FANet().to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print_and_save(log_path, f"\n===== MODEL ARCHITECTURE =====")
    print_and_save(log_path, f"Architecture: U-Net Encoder-Decoder (4 stages)")
    print_and_save(log_path, f"Total parameters: {total_params:,}")

    blocks = [
        ("Encoder 1 (3->32)", model.e1),
        ("Encoder 2 (32->64)", model.e2),
        ("Encoder 3 (64->128)", model.e3),
        ("Encoder 4 (128->256)", model.e4),
        ("Decoder 1 (256->128)", model.d1),
        ("Decoder 2 (128->64)", model.d2),
        ("Decoder 3 (64->32)", model.d3),
        ("Decoder 4 (32->16)", model.d4),
        ("Output (Conv1x1)", model.output),
    ]
    print_and_save(log_path, f"\nParameter breakdown:")
    for name, blk in blocks:
        n = sum(p.numel() for p in blk.parameters())
        print_and_save(log_path, f"  {name:<25s}: {n:>10,}")

    (train_x, train_y), (valid_x, valid_y) = load_data(cfg["dataset"]["root"])
    train_x, train_y = sk_shuffle(train_x, train_y, random_state=42)

    print_and_save(log_path, f"\n===== DATASET =====")
    print_and_save(log_path, f"Dataset:     {cfg['dataset']['root']}")
    print_and_save(log_path, f"Train:       {len(train_x)} images")
    print_and_save(log_path, f"Validation:  {len(valid_x)} images")

    transform = A.Compose([
        A.Rotate(limit=aug["rotate_limit"], p=aug["rotate_p"]),
        A.HorizontalFlip(p=aug["hflip_p"]),
        A.VerticalFlip(p=aug["vflip_p"]),
        A.CoarseDropout(p=aug["dropout_p"], max_holes=10, max_height=32, max_width=32),
    ])
    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)
    train_loader = DataLoader(train_dataset, batch_size=BATCH, shuffle=False, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    loss_fn = DiceBCELoss()

    print_and_save(log_path, f"\n===== INITIAL MASKS (OTSU) =====")
    print_and_save(log_path, f"Generating initial masks via Otsu thresholding...")
    train_mask = init_mask(train_x, size)
    valid_mask = init_mask(valid_x, size)
    print_and_save(log_path, f"Done. RLE-encoded masks ready.")

    print_and_save(log_path, f"\n===== TRAINING RESULTS ({epochs} epochs) =====")
    print_and_save(log_path,
                   f"{'Ep':<5}{'Train Loss':<14}{'Val Loss':<14}{'Best Val':<14}{'Time(s)':<10}{'Note'}")

    best_valid_loss = float('inf')

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, return_train_mask = train_epoch(
            model, train_loader, train_mask, optimizer, loss_fn, size)
        valid_loss, return_valid_mask = eval_epoch(
            model, valid_loader, valid_mask, loss_fn, size)
        scheduler.step(valid_loss)

        improved = ""
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), ckpt_path)
            train_mask = return_train_mask
            valid_mask = return_valid_mask
            improved = " * (mask updated)"

        elapsed = time.time() - t0
        print_and_save(log_path,
                       f"{epoch+1:<5}{train_loss:<14.4f}{valid_loss:<14.4f}"
                       f"{best_valid_loss:<14.4f}{elapsed:<10.1f}{improved}")

    print_and_save(log_path, f"\nBest validation loss: {best_valid_loss:.4f}")
    print_and_save(log_path, f"Checkpoint saved:     {ckpt_path}")

    print_and_save(log_path, f"\n===== TEST-TIME REFINEMENT (1 sample) =====")
    model.eval()

    test_img_path = valid_x[0]
    img_name = test_img_path.split("/")[-1]
    gt_mask_path = valid_y[0]

    img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, size)
    img_tensor = torch.from_numpy(np.transpose(img, (2, 0, 1)) / 255.0).float().unsqueeze(0).to(DEVICE)

    gt = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
    gt = cv2.resize(gt, size) / 255.0
    gt_fg = (gt > 0.5).sum()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    pm = (th / 255.0 > 0.5).astype(np.float32)
    pt = torch.from_numpy(pm).unsqueeze(0).unsqueeze(0).to(DEVICE)

    print_and_save(log_path, f"Test image:            {img_name}")
    print_and_save(log_path, f"Ground truth FG pixels: {gt_fg:.0f}")
    print_and_save(log_path, f"")
    print_and_save(log_path, f"{'Iter':<8}{'FG Pixels':<15}{'vs GT'}")
    print_and_save(log_path, f"{'Otsu':<8}{pm.sum():<15.0f}{pm.sum()-gt_fg:+.0f}")

    for it in range(5):
        with torch.no_grad():
            pred = torch.sigmoid(model([img_tensor, pt]))
            pred_bin = (pred > 0.5).float()
            n_fg = pred_bin.sum().item()
            print_and_save(log_path, f"{it+1:<8}{n_fg:<15.0f}{n_fg-gt_fg:+.0f}")
            pt = pred_bin

    print_and_save(log_path, f"\nReport complete.")


if __name__ == "__main__":
    main()
