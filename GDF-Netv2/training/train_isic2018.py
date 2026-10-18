import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure local GDF-Netv2 imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.gdfnet_v2 import GDFNetV2
from losses.boundary_losses import CompositeGDFNetV2Loss
from data.dataset import MedicalSegmentationDataset
from data.augmentations import get_gdfnet_v2_transforms
from training.evaluate import evaluate_binary_metrics

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=True):
            outputs = model(images)
            loss, loss_dict = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    preds_all = []
    targets_all = []

    for images, masks in tqdm(loader, desc="Validating", leave=False):
        images = images.to(device, non_blocking=True)
        with autocast(enabled=True):
            preds = model(images)
            probs = torch.sigmoid(preds)

        preds_all.append(probs.cpu().numpy()[:, 0])
        targets_all.append(masks.numpy()[:, 0])

    preds_all = np.concatenate(preds_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)

    _, summary_df = evaluate_binary_metrics(preds_all, targets_all)
    metrics = {row['Metric']: row['Mean'] for _, row in summary_df.iterrows()}
    return metrics, summary_df


def run_training(
    train_images, train_masks,
    val_images, val_masks,
    test_images, test_masks,
    backbone="convnextv2_tiny",
    img_size=384,
    batch_size=8,
    epochs=150,
    lr=1e-4,
    save_dir="checkpoints_gdfnet_v2"
):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training GDF-Netv2 [{backbone}] on Device: {device} | Resolution: {img_size}x{img_size}")

    train_ds = MedicalSegmentationDataset(train_images, train_masks, transform=get_gdfnet_v2_transforms(img_size, is_train=True))
    val_ds   = MedicalSegmentationDataset(val_images, val_masks, transform=get_gdfnet_v2_transforms(img_size, is_train=False))
    test_ds  = MedicalSegmentationDataset(test_images, test_masks, transform=get_gdfnet_v2_transforms(img_size, is_train=False))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = GDFNetV2(backbone_name=backbone, pretrained=True).to(device)
    criterion = CompositeGDFNetV2Loss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=8, verbose=True)
    scaler = GradScaler(enabled=True)

    best_dice = 0.0
    best_ckpt_path = os.path.join(save_dir, f"GDFNetV2_{backbone}_best.pth")

    for epoch in range(1, epochs + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_metrics, _ = validate(model, val_loader, device)

        val_dice = val_metrics['dice']
        val_miou = val_metrics['miou']
        scheduler.step(val_dice)

        print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {loss:.4f} | Val Dice: {val_dice:.4f} | Val mIoU: {val_miou:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), best_ckpt_path)
            print(f"[*] New Best Model Saved (Val Dice: {val_dice:.4f})")

    # Final Evaluation on Official Test Set using best checkpoint
    print(f"\n[*] Loading Best Model for Official Test Set Evaluation: {best_ckpt_path}")
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    test_metrics, test_summary_df = validate(model, test_loader, device)

    print("\n================ OFFICIAL TEST RESULTS ================")
    print(test_summary_df.to_string(index=False))

    test_csv_path = os.path.join(save_dir, "benchmark_results_test_gdfnet_v2.csv")
    test_summary_df.to_csv(test_csv_path, index=False)
    print(f"[*] Results saved to: {test_csv_path}")
    return test_metrics
