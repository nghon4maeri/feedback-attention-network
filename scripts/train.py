"""Train FANet with feedback attention.

Usage:
    python scripts/train.py [--config configs/kvasir_sessile.yaml]

Phase-7B new flags (can also be set via YAML config):
    --gating-mode   soft_or | soft_max | hard | hard_ste
    --detach-feedback
"""
import argparse
import datetime
import os
import time

import numpy as np
import albumentations as A
import torch
from torch.utils.data import DataLoader

from fanet.config import load_config
from fanet.data import DATASET, load_data, rle_batch_to_tensor
from fanet.losses import DiceBCELoss
from fanet.models import FANet
from fanet.utils import (
    seeding, shuffling, create_dir, init_mask,
    epoch_time, rle_encode, print_and_save,
)


def train(model, loader, mask, optimizer, loss_fn, device, size, dual_path=False,
          no_feedback=False):
    epoch_loss = 0
    return_mask = []

    model.train()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        b = y.shape[0]
        if no_feedback:
            m = torch.zeros(b, 1, *size)
        else:
            m = rle_batch_to_tensor(mask, i * b, b, size)
            if dual_path:
                bg = (1.0 - m)
                m = torch.cat([m, bg], dim=1)
        m = m.to(device)

        optimizer.zero_grad()
        y_pred = model([x, m])
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred = torch.sigmoid(y_pred).cpu().numpy()
            for py in y_pred:
                py = np.squeeze(py, axis=0)
                py = py > 0.5
                py = np.array(py, dtype=np.uint8)
                return_mask.append(rle_encode(py))

        epoch_loss += loss.item()

    return epoch_loss / len(loader), return_mask


def evaluate(model, loader, mask, loss_fn, device, size, dual_path=False,
             no_feedback=False):
    epoch_loss = 0
    return_mask = []
    bin_dice = 0.0
    bin_prec = 0.0
    bin_rec = 0.0
    bin_fpr = 0.0

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            b = y.shape[0]
            if no_feedback:
                m = torch.zeros(b, 1, *size)
            else:
                m = rle_batch_to_tensor(mask, i * b, b, size)
                if dual_path:
                    bg = (1.0 - m)
                    m = torch.cat([m, bg], dim=1)
            m = m.to(device)

            y_pred = model([x, m])
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            y_pred = torch.sigmoid(y_pred).cpu().numpy()
            for py, gy in zip(y_pred, y.cpu().numpy()):
                py = np.squeeze(py, axis=0)
                py = py > 0.5
                py = np.array(py, dtype=np.uint8)
                return_mask.append(rle_encode(py))

                gb = (np.squeeze(gy, axis=0) > 0.5).astype(np.uint8)
                pb = (py > 0).astype(np.uint8)
                tp = (pb & gb).sum()
                pred_pos = pb.sum(); gt_pos = gb.sum()
                fp = (pb & (1 - gb)).sum()
                bin_dice += 2.0 * tp / (pred_pos + gt_pos + 1e-15)
                bin_prec += tp / (pred_pos + 1e-15)
                bin_rec  += tp / (gt_pos + 1e-15)
                bin_fpr  += fp / (gb.size + 1e-15)

    n = len(loader.dataset)
    bin_metrics = {
        "bin_dice": bin_dice / n, "bin_prec": bin_prec / n,
        "bin_rec":  bin_rec  / n, "bin_fpr":  bin_fpr  / n,
    }
    return epoch_loss / len(loader), return_mask, bin_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kvasir_sessile.yaml")
    parser.add_argument("--resume", type=str, default=None,
                        help="checkpoint path to resume from")
    parser.add_argument("--gate", type=str, default=None,
                        choices=["binary", "ste", "soft"],
                        help="(legacy) MixPool gate mode")
    parser.add_argument("--gating-mode", type=str, default=None,
                        choices=["hard", "hard_ste", "soft_max", "soft_or"],
                        help="Phase-7B MixPool gating mode. Overrides --gate when set.")
    parser.add_argument("--detach-feedback", action="store_true", default=None,
                        help="Phase-7B: detach m_fg from gradient graph (severs Feedback Trap)")
    parser.add_argument("--dual-path", action="store_true",
                        help="dual-path feedback [m_fg, m_bg] (Phase 3/4)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="override num epochs (smoke test)")
    parser.add_argument("--loss", type=str, default=None,
                        choices=["dicebce", "negdice", "cella", "farwiou", "tversky"],
                        help="loss function override")
    parser.add_argument("--no-feedback", action="store_true",
                        help="T0N: feed zero mask (no cross-epoch feedback loop)")
    parser.add_argument("--seed", type=int, default=None,
                        help="override random seed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    tcfg = cfg["train"]                          # shorthand
    size = tuple(cfg["dataset"]["image_size"])
    batch_size = tcfg["batch_size"]
    num_epochs = args.epochs or tcfg["epochs"]
    lr = tcfg["lr"]
    checkpoint_path = cfg["paths"]["checkpoint"]
    train_log_path  = cfg["paths"]["train_log"]
    aug = tcfg["augmentation"]

    # ── Resolve gate / gating_mode (CLI > YAML > default) ───────────────
    # Priority: --gating-mode CLI > train.gating_mode YAML >
    #           --gate CLI > train.gate YAML > "binary"
    gating_mode = (args.gating_mode
                   or tcfg.get("gating_mode", None))
    gate = (args.gate
            or tcfg.get("gate", "binary"))

    # ── Resolve detach_feedback (CLI > YAML > False) ─────────────────────
    if args.detach_feedback:
        detach_feedback = True
    else:
        detach_feedback = bool(tcfg.get("detach_feedback", False))

    dual_path   = args.dual_path or bool(tcfg.get("dual_path", False))
    no_feedback = args.no_feedback or bool(tcfg.get("no_feedback", False))
    seed        = args.seed or tcfg.get("seed", 42)

    # ── Loss: CLI > YAML > "dicebce" ──────────────────────────────────────
    loss_key = args.loss or tcfg.get("loss", "dicebce")

    # ── Auto-tag checkpoint/log path to avoid overwriting old artifacts ───
    tag_parts = []
    if gating_mode and gating_mode != "hard":
        tag_parts.append(gating_mode)
    elif gate != "binary":
        tag_parts.append(gate)
    if dual_path:
        tag_parts.append("dual")
    if detach_feedback:
        tag_parts.append("detach")
    if loss_key != "dicebce":
        tag_parts.append(loss_key)
    if no_feedback:
        tag_parts.append("nofb")

    if tag_parts:
        tag = "_".join(tag_parts)
        base, ext = os.path.splitext(checkpoint_path)
        checkpoint_path = f"{base}_{tag}{ext}"
        train_log_path  = f"{os.path.splitext(train_log_path)[0]}_{tag}.txt"

    seeding(seed)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    os.makedirs(os.path.dirname(train_log_path),  exist_ok=True)

    # ── Config summary (logged to file + stdout) ─────────────────────────
    cfg_summary = (
        f"\n{'='*60}\n"
        f"FANet Training Config\n"
        f"{'='*60}\n"
        f"  Dataset root   : {cfg['dataset']['root']}\n"
        f"  Image size     : {size}\n"
        f"  Batch size     : {batch_size}\n"
        f"  Epochs         : {num_epochs}\n"
        f"  LR             : {lr}\n"
        f"  Seed           : {seed}\n"
        f"  --- MixPool ---\n"
        f"  gate (legacy)  : {gate}\n"
        f"  gating_mode    : {gating_mode or '(from gate alias)'}\n"
        f"  detach_feedback: {detach_feedback}   <- Feedback Trap severed: {detach_feedback}\n"
        f"  dual_path      : {dual_path}\n"
        f"  --- Training ---\n"
        f"  loss           : {loss_key}\n"
        f"  no_feedback    : {no_feedback}\n"
        f"  --- Paths ---\n"
        f"  checkpoint     : {checkpoint_path}\n"
        f"  train_log      : {train_log_path}\n"
        f"{'='*60}"
    )
    print_and_save(train_log_path, str(datetime.datetime.now()) + cfg_summary)

    # ── Data ──────────────────────────────────────────────────────────────
    (train_x, train_y), (valid_x, valid_y) = load_data(cfg["dataset"]["root"])
    train_x, train_y = shuffling(train_x, train_y)
    print_and_save(train_log_path,
                   f"Dataset: Train={len(train_x)}, Valid={len(valid_x)}")

    transform = A.Compose([
        A.Rotate(limit=aug["rotate_limit"], p=aug["rotate_p"]),
        A.HorizontalFlip(p=aug["hflip_p"]),
        A.VerticalFlip(p=aug["vflip_p"]),
        A.CoarseDropout(p=aug["dropout_p"], num_holes=10,
                        hole_height=32, hole_width=32),
    ])

    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)
    train_loader  = DataLoader(train_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=0)
    valid_loader  = DataLoader(valid_dataset, batch_size=batch_size,
                               shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FANet(
        gate=gate,
        dual_path=dual_path,
        gating_mode=gating_mode,
        detach_feedback=detach_feedback,
    ).to(device)

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print_and_save(train_log_path, f"Resumed from {args.resume}")

    # ── Loss ──────────────────────────────────────────────────────────────
    if loss_key == "negdice":
        from fanet.losses import NegativeAreaDiceBCELoss
        loss_fn = NegativeAreaDiceBCELoss(fp_weight=0.5)
        loss_name = "NegativeAreaDiceBCE"
    elif loss_key == "cella":
        from fanet.losses import Phase5CellALoss
        loss_fn = Phase5CellALoss(v1=0.3, lambda_w=1.0)
        loss_name = "CellA: IoU+BCE+WSDice"
    elif loss_key == "farwiou":
        from fanet.losses import FarWeightedIoUBCELoss
        loss_fn = FarWeightedIoUBCELoss(gamma=5.0)
        loss_name = "FarWeighted wIoU+wBCE"
    elif loss_key == "tversky":
        from fanet.losses import Phase6AsymmetricBCELoss
        alpha = float(tcfg.get("tversky_alpha", 0.7))
        beta  = float(tcfg.get("tversky_beta",  0.3))
        loss_fn = Phase6AsymmetricBCELoss(alpha=alpha, beta=beta, lam=0.5)
        loss_name = f"Tversky(alpha={alpha}, beta={beta}) + DiceBCE"
    elif loss_key == "adaptive_tversky":
        from fanet.losses import AdaptiveTverskyLoss
        loss_fn = AdaptiveTverskyLoss(total_epochs=num_epochs)
        loss_name = "Adaptive Tversky (Curriculum)"
    else:
        loss_fn = DiceBCELoss()
        loss_name = "DiceBCE"

    print_and_save(train_log_path, f"Loss: {loss_name}")

    # ── Optimiser ─────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

    best_valid_loss = float('inf')
    train_mask = init_mask(train_x, size)
    valid_mask = init_mask(valid_x, size)

    # ── Training Loop ─────────────────────────────────────────────────────
    for epoch in range(num_epochs):
        start_time = time.time()
        
        if hasattr(loss_fn, 'set_epoch'):
            loss_fn.set_epoch(epoch)

        train_loss, return_train_mask = train(
            model, train_loader, train_mask, optimizer, loss_fn, device, size,
            dual_path, no_feedback)
        valid_loss, return_valid_mask, bin_metrics = evaluate(
            model, valid_loader, valid_mask, loss_fn, device, size,
            dual_path, no_feedback)
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print_and_save(train_log_path, f"  [ckpt] Saving: {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
            if not no_feedback:
                train_mask = return_train_mask
                valid_mask = return_valid_mask

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str  = f"Epoch: {epoch+1:03}/{num_epochs} | {epoch_mins}m {epoch_secs}s\n"
        data_str += f"  Train Loss: {train_loss:.4f}\n"
        data_str += f"  Valid Loss: {valid_loss:.4f}\n"
        data_str += (f"  [BIN] Dice={bin_metrics['bin_dice']:.4f} | "
                     f"Prec={bin_metrics['bin_prec']:.4f} | "
                     f"Rec={bin_metrics['bin_rec']:.4f} | "
                     f"FPR={bin_metrics['bin_fpr']:.4f}\n")
        print_and_save(train_log_path, data_str)


if __name__ == "__main__":
    main()
