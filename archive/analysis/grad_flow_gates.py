"""Verify fmask gradient flows under binary vs ste vs soft gating (Phase 2 check).

Loads the 200ep checkpoint, runs 3 training steps on the val loader for each
gate mode, logs grad norm of fmask params per MixPool block.

Usage:
    python analysis/grad_flow_gates.py [--checkpoint checkpoints/checkpoint_200ep_kaggle.pth]
"""
import argparse
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import torch
from torch.utils.data import DataLoader

from fanet.config import load_config
from fanet.data import DATASET, load_data, rle_batch_to_tensor
from fanet.losses import DiceBCELoss
from fanet.models import FANet
from fanet.utils import seeding, init_mask

BATCH = 2
STEPS = 3


def fmask_grad_report(model, name):
    mp = {"e1": model.e1.p1, "e2": model.e2.p1, "e3": model.e3.p1, "e4": model.e4.p1,
          "d1": model.d1.p1, "d2": model.d2.p1, "d3": model.d3.p1, "d4": model.d4.p1}
    norms = {}
    for tag, blk in mp.items():
        p = dict(blk.named_parameters())["fmask.3.weight"]  # last conv of fmask
        g = p.grad
        norms[tag] = float(g.norm().item()) if g is not None else 0.0
    print(f"  [{name}] fmask grad norms: " +
          " | ".join(f"{t}:{norms[t]:.2e}" for t in norms))
    return sum(norms.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kvasir_sessile.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    size = tuple(cfg["dataset"]["image_size"])
    ckpt = args.checkpoint or cfg["paths"]["checkpoint"]

    seeding(cfg["train"]["seed"])
    device = torch.device("cpu")

    (train_x, train_y), _ = load_data(cfg["dataset"]["root"])
    train_mask = init_mask(train_x, size)
    loader = DataLoader(DATASET(train_x, train_y, size),
                        batch_size=BATCH, shuffle=False, num_workers=0)
    loss_fn = DiceBCELoss()

    print(f"Checkpoint: {ckpt}")
    print(f"Running {STEPS} training steps per gate mode\n")

    for gate in ("binary", "ste", "soft"):
        model = FANet(gate=gate).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        model.train()

        total_grad = 0.0
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

            total_grad += fmask_grad_report(model, f"{gate} step {step+1}")
            optimizer.step()

        print(f"  >>> {gate}: total fmask grad over {STEPS} steps = {total_grad:.3e}\n")


if __name__ == "__main__":
    main()
