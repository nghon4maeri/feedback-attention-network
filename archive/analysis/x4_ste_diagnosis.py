"""X4: STE diagnosis using on-disk Phase-4 checkpoints (CPU).

Questions:
  Q1. Gradient variance/norm through the fmask branch under binary vs STE vs
      soft gate, evaluated on the SAME weights (each cell's own checkpoint).
  Q2. Does the trained STE model learn an fmask pattern that differs from the
      feedback mask m_fg (i.e. is fmask doing anything useful), or does it
      collapse to a constant / copy m_fg?
  Q3. BN running statistics: how different are T00 (binary-trained) and T10
      (STE-trained) BN stats — does STE shift the encoder distribution?

Protocol: 8 val images, CPU, batch=1. Gradient variance is measured over
multiple random input batches (seeded) for the 8 MixPool fmask conv weights.

Usage:
    python analysis/x4_ste_diagnosis.py --ckpt-dir checkpoints_phase4
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import cv2
import torch
import torch.nn.functional as F

from fanet.config import load_config
from fanet.data import load_data
from fanet.models import FANet
from fanet.utils import seeding

N_IMAGES = 8
N_BATCHES = 5
SEED = 42


def set_all_gates(model, gate):
    """Set .gate on every MixPool submodule (FANet root gate alone is not used)."""
    for mod in model.modules():
        if type(mod).__name__ == "MixPool":
            mod.gate = gate
    model.gate = gate


def fmask_params(model):
    """Return dict {block_name: [fmask conv weight tensors]} for all MixPools."""
    out = {}
    for name, mod in model.named_modules():
        if type(mod).__name__ == "MixPool":
            params = [p for p in mod.fmask.parameters()]
            out[name] = params
    return out


def collect_fmask_outputs(model, img_t, m):
    """Forward pass capturing fmask maps (B,1,H,W) at each MixPool."""
    model.eval()
    acts = {}
    hooks = []
    for name, mod in model.named_modules():
        if type(mod).__name__ == "MixPool":
            hooks.append((name, mod.register_forward_hook(
                lambda m, i, o, _nm=name: acts.__setitem__(_nm, m.fmask(i[0]).detach()))))
    with torch.no_grad():
        model([img_t, m])
    for _, h in hooks:
        h.remove()
    return acts


def bn_stats_diff(m1, m2):
    diffs = []
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        pass
    out = []
    sd1, sd2 = m1.state_dict(), m2.state_dict()
    for k in sd1:
        if "running_mean" in k or "running_var" in k or "num_batches" in k:
            v1, v2 = sd1[k].float(), sd2[k].float()
            out.append((k, float((v1 - v2).abs().mean()), float((v1 - v2).norm().item())))
    return out


def grad_stats(model, x, m, gate):
    """Forward-backward, return (norm, var) of grads of all fmask params."""
    model.train()
    loss = model([x, m]).sum()
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None
    loss.backward()
    for _, params in fmask_params(model).items():
        for p in params:
            if p.grad is not None:
                grads.append(p.grad.flatten())
    if not grads:
        return 0.0, 0.0
    g = torch.cat(grads)
    return float(g.norm()), float(g.var())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kvasir_sessile.yaml")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_phase4")
    parser.add_argument("--out", type=str, default="results/x4_ste_diagnosis.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    size = tuple(cfg["dataset"]["image_size"])
    device = torch.device("cpu")
    seeding(SEED)

    (train_x, train_y), (valid_x, valid_y) = load_data(cfg["dataset"]["root"])
    valid_x = valid_x[:N_IMAGES]
    valid_y = valid_y[:N_IMAGES]

    ckpts = {"T00": os.path.join(args.ckpt_dir, "ckpt_T00.pth"),
             "T10": os.path.join(args.ckpt_dir, "ckpt_T10.pth")}

    report = {"n_images": N_IMAGES, "n_batches": N_BATCHES}

    models = {}
    for name, ck in ckpts.items():
        if not os.path.exists(ck):
            print(f"SKIP {name}: {ck} not found")
            continue
        gate = "binary" if name == "T00" else "ste"
        m = FANet(gate=gate, dual_path=False)
        m.load_state_dict(torch.load(ck, map_location=device))
        m.to(device)
        models[name] = m
        report[f"{name}_ckpt"] = ck

    # ---- Q1: gradient stats under each gate on each cell's own weights ----
    for cname, m in models.items():
        for gate in ("binary", "ste", "soft"):
            set_all_gates(m, gate)  # swap gate in-place (arch identical)
            norms, vars_ = [], []
            for batch in range(N_BATCHES):
                torch.manual_seed(SEED + batch)
                imgs, msks = [], []
                for i in range(1):
                    img = cv2.imread(valid_x[batch % len(valid_x)], cv2.IMREAD_COLOR)
                    img = cv2.resize(img, size)
                    imgs.append(np.transpose(img, (2, 0, 1)) / 255.0)
                    gt = cv2.imread(valid_y[batch % len(valid_y)], cv2.IMREAD_GRAYSCALE)
                    gt = cv2.resize(gt, size) / 255.0
                    msks.append((gt > 0.5).astype(np.float32)[None])
                x = torch.from_numpy(np.stack(imgs)).float().to(device)
                m_fb = torch.from_numpy(np.stack(msks)).float().to(device)
                n, v = grad_stats(m, x, m_fb, gate)
                norms.append(n); vars_.append(v)
            report[f"Q1_grad_{cname}_{gate}"] = {
                "fmask_grad_norm_mean": float(np.mean(norms)),
                "fmask_grad_norm_std": float(np.std(norms)),
                "fmask_grad_var_mean": float(np.mean(vars_)),
            }
            print(f"Q1 {cname}/{gate}: norm {np.mean(norms):.4f}+-{np.std(norms):.4f}, "
                  f"var {np.mean(vars_):.2e}")

    # ---- Q2: fmask output pattern vs m_fg ----
    q2 = {}
    for cname, m in models.items():
        set_all_gates(m, "binary" if cname == "T00" else "ste")
        m.eval()
        corr_with_mfg = []
        frac_gt_mask = []
        for i in range(N_IMAGES):
            img = cv2.imread(valid_x[i], cv2.IMREAD_COLOR)
            img = cv2.resize(img, size)
            gt = cv2.imread(valid_y[i], cv2.IMREAD_GRAYSCALE)
            gt = cv2.resize(gt, size) / 255.0
            gt = (gt > 0.5).astype(np.float32)
            m_fb = torch.from_numpy(gt[None, None]).float().to(device)
            x = torch.from_numpy(np.transpose(img, (2, 0, 1))[None] / 255.0).float().to(device)
            acts = collect_fmask_outputs(m, x, m_fb)
            for blk, a in acts.items():
                fm = cv2.resize(a[0, 0].cpu().numpy(), size)  # match GT spatial size
                c = np.corrcoef(fm.ravel(), gt.ravel())[0, 1]
                corr_with_mfg.append((blk, c))
                frac_gt_mask.append((blk, float((fm > 0.5).mean())))
        q2[cname] = {
            "corr_with_gt_mfg_mean": {blk: float(np.mean([c for b, c in corr_with_mfg if b == blk]))
                                      for blk in {b for b, _ in corr_with_mfg}},
            "frac_fmask_gt_0.5": {blk: float(np.mean([f for b, f in frac_gt_mask if b == blk]))
                                  for blk in {b for b, _ in frac_gt_mask}},
        }
    report["Q2_fmask_vs_mfg"] = q2

    # ---- Q3: BN running stats T00 vs T10 ----
    if "T00" in models and "T10" in models:
        diffs = bn_stats_diff(models["T00"], models["T10"])
        report["Q3_bn_stats_diff_T00_vs_T10"] = [
            {"layer": k, "mean_abs_diff": round(v1, 6), "norm_diff": round(v2, 4)}
            for k, v1, v2 in diffs
        ]
        print("\nQ3 BN diff (mean_abs) top layers:")
        for k, v1, v2 in sorted(diffs, key=lambda t: -t[1])[:8]:
            print(f"   {k}: {v1:.5f}")

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()