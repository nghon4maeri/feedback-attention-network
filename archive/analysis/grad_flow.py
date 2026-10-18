"""Analysis 1: gradient flow through MixPool's fmask branch.

Confirms whether the hard binary gate `(fmask > 0.5)` cuts gradient so the
fmask conv layers never learn. Two checks:
  A. live grad norms per training step (fmask vs conv1/conv2)
  B. weight change after optimizer steps + checkpoint-vs-init diff

Usage:
    python analysis/grad_flow.py [--config configs/kvasir_sessile.yaml]
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from fanet.config import load_config
from fanet.data import DATASET, load_data, rle_batch_to_tensor
from fanet.losses import DiceBCELoss
from fanet.models import FANet
from fanet.utils import seeding, init_mask

BATCH = 2
STEPS = 10


def mixpool_names(model):
    return [
        ("e1.p1", model.e1.p1), ("e2.p1", model.e2.p1),
        ("e3.p1", model.e3.p1), ("e4.p1", model.e4.p1),
        ("d1.p1", model.d1.p1), ("d2.p1", model.d2.p1),
        ("d3.p1", model.d3.p1), ("d4.p1", model.d4.p1),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kvasir_sessile.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    size = tuple(cfg["dataset"]["image_size"])
    ckpt = cfg["paths"]["checkpoint"]

    seeding(cfg["train"]["seed"])
    device = torch.device("cpu")

    print("=" * 80)
    print("CHECK A: live gradient norms per training step")
    print("=" * 80)

    (train_x, train_y), _ = load_data(cfg["dataset"]["root"])
    train_mask = init_mask(train_x, size)
    loader = DataLoader(DATASET(train_x, train_y, size),
                        batch_size=BATCH, shuffle=False, num_workers=0)

    model = FANet().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = DiceBCELoss()

    before = {n: p.detach().clone()
              for n, p in model.named_parameters() if "fmask" in n}

    mp_names = mixpool_names(model)
    rows = []
    model.train()
    for step, (x, y) in enumerate(loader):
        if step >= STEPS:
            break
        x = x.to(device)
        y = y.to(device)
        m = rle_batch_to_tensor(train_mask, step * BATCH, x.shape[0], size).to(device)

        optimizer.zero_grad()
        y_pred = model([x, m])
        loss = loss_fn(y_pred, y)
        loss.backward()

        row = {"step": step, "loss": float(loss.item())}
        for name, mp in mp_names:
            for tag, param_name in [("fmask_conv1", "fmask.0.weight"),
                                    ("fmask_conv2", "fmask.3.weight"),
                                    ("conv1", "conv1.0.weight"),
                                    ("conv2", "conv2.0.weight")]:
                p = dict(mp.named_parameters())[param_name]
                g = p.grad
                row[f"{name}.{tag}"] = float(g.norm().item()) if g is not None else None
        optimizer.step()
        rows.append(row)

    hdr_parts = []
    for name, _ in mp_names:
        for m in ["f1", "f2", "c1", "c2"]:
            hdr_parts.append(f"{name}.{m}")
    print(f"{'step':>5} {'loss':>7} | " + " | ".join(hdr_parts))
    for row in rows:
        vals = []
        for name, _ in mp_names:
            for tag in ["fmask_conv1", "fmask_conv2", "conv1", "conv2"]:
                v = row[f"{name}.{tag}"]
                vals.append(f"{v:.2e}" if v is not None else "   NONE ")
        print(f"{row['step']:>5} {row['loss']:>7.4f} | " + " | ".join(vals))

    fmask_grads = [r[k] for r in rows for k in r if "fmask" in k]
    live = [g for g in fmask_grads if g is not None and g > 1e-12]
    print(f"\nfmask params with any gradient over {STEPS} steps: {len(live)}")
    conv_grads = [r[k] for r in rows for k in r if ("conv1" in k or "conv2" in k)]
    live_conv = [g for g in conv_grads if g is not None and g > 1e-12]
    print(f"conv1/conv2 params with gradient: {len(live_conv)}")

    print("\n" + "=" * 80)
    print("CHECK B: weight change after training steps + checkpoint-vs-init diff")
    print("=" * 80)

    after = {n: p.detach().clone()
             for n, p in model.named_parameters() if "fmask" in n}
    print("\nMax |W_after - W_before| for fmask params after 10 optimizer steps:")
    for n in before:
        d = (after[n] - before[n]).abs().max().item()
        print(f"  {n:>35s}: {d:.3e}")

    fresh = FANet()
    trained = FANet()
    trained.load_state_dict(torch.load(ckpt, map_location=device))
    print("\nMax |W_checkpoint - W_init| (evidence of training):")
    for n, p in fresh.named_parameters():
        d = (trained.state_dict()[n] - p).abs().max().item()
        if "fmask" in n:
            print(f"  [fmask] {n:>35s}: {d:.3e}")
    d_other = max((trained.state_dict()[n] - p).abs().max().item()
                  for n, p in fresh.named_parameters() if "fmask" not in n)
    print(f"\n  max change among NON-fmask params: {d_other:.3e}")

    print("\nBN running_mean diff (fmask.1) — drifts via forward even without grad:")
    for name, mp in mp_names:
        d = (mp.fmask[1].running_mean
             - fresh.state_dict()[f"{name}.fmask.1.running_mean"]).abs().max().item()
        print(f"  {name}.fmask.1.running_mean: {d:.3e}")


if __name__ == "__main__":
    main()
