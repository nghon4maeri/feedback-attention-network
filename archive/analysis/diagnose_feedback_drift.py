"""
diagnose_feedback_drift.py
Offline diagnostic: BatchNorm Drift, Feature Collapse & Gradient Saturation
for FANet Feedback Trap analysis.

Supports NumPy 2.x (no seaborn/pandas dependency).

Usage:
    python analysis/diagnose_feedback_drift.py \
        --config configs/kvasir_sessile.yaml \
        --ckpt_m00 checkpoints_phase5/ckpt_T0N.pth \
        --ckpt_m01 checkpoints_phase6/ckpt_TD.pth \
        --ckpt_m10 checkpoints_phase4/ckpt_T00.pth \
        --ckpt_m11 checkpoints_phase6/ckpt_TC.pth \
        --out_dir diagnostics_output
"""

import os
import argparse
import json
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from fanet.models import FANet
from fanet.data import load_data, DATASET
from fanet.config import load_config


# ─────────────────────────────── helpers ────────────────────────────────────

class BottleneckHook:
    """Register a hook on the bottleneck encoder block (e4).

    EncoderBlock.forward returns (pooled_output, skip_feature).
    We capture the skip_feature (index 1) which is the pre-MixPool
    feature map of the deepest encoder — shape [B, 256, H/8, W/8].
    """
    def __init__(self, model):
        self.features = None
        # Hook on the e4 EncoderBlock module directly
        self._hook = model.e4.register_forward_hook(self._fn)

    def _fn(self, module, inp, output):
        # output is tuple (pooled, skip)
        if isinstance(output, (tuple, list)):
            self.features = output[1].detach()   # skip before MixPool
        else:
            self.features = output.detach()

    def remove(self):
        self._hook.remove()


def get_bn_stats(model):
    stats = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            stats[name] = {
                'mean': module.running_mean.clone().cpu().numpy(),
                'var':  module.running_var.clone().cpu().numpy(),
            }
    return stats


def kl_divergence_gaussian(mean1, var1, mean0, var0, eps=1e-5):
    """Per-channel KL(N(mean1,var1) || N(mean0,var0)), averaged over channels."""
    var0 = np.maximum(var0, eps)
    var1 = np.maximum(var1, eps)
    kl = (np.log(np.sqrt(var0) / np.sqrt(var1))
          + (var1 + (mean1 - mean0) ** 2) / (2.0 * var0)
          - 0.5)
    return float(np.mean(kl))


# ─────────────────────────────── main ───────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/kvasir_sessile.yaml")
    parser.add_argument("--ckpt_m00", default="checkpoints_phase5/ckpt_T0N.pth",
                        help="M00: No-Feedback + Base Loss")
    parser.add_argument("--ckpt_m01", default="checkpoints_phase6/ckpt_TD.pth",
                        help="M01: No-Feedback + Asym Loss")
    parser.add_argument("--ckpt_m10", default="checkpoints_phase4/ckpt_T00.pth",
                        help="M10: Feedback + Base Loss")
    parser.add_argument("--ckpt_m11", default="checkpoints_phase6/ckpt_TC.pth",
                        help="M11: Feedback + Asym Loss")
    parser.add_argument("--out_dir",  default="diagnostics_output")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Dataset ──────────────────────────────────────────────────────────────
    cfg  = load_config(args.config)
    size = tuple(cfg["dataset"]["image_size"])   # (256, 256)
    _, (valid_x, valid_y) = load_data(cfg["dataset"]["root"])
    val_loader = DataLoader(
        DATASET(valid_x, valid_y, size, transform=None),
        batch_size=1, shuffle=False)
    print(f"Validation set: {len(valid_x)} images.")

    # ── Model registry ───────────────────────────────────────────────────────
    models_info = {
        "M00": {"path": args.ckpt_m00, "fb": False, "name": "No-FB + Base"},
        "M01": {"path": args.ckpt_m01, "fb": False, "name": "No-FB + Asym"},
        "M10": {"path": args.ckpt_m10, "fb": True,  "name": "FB + Base"},
        "M11": {"path": args.ckpt_m11, "fb": True,  "name": "FB + Asym"},
    }

    loaded_models = {}
    bn_stats_map  = {}
    for key, info in models_info.items():
        if not os.path.exists(info["path"]):
            print(f"[SKIP] {key}: checkpoint not found at {info['path']}")
            continue
        net = FANet(gate="binary", dual_path=False).to(device)
        net.load_state_dict(torch.load(info["path"], map_location=device))
        net.eval()
        loaded_models[key] = net
        bn_stats_map[key]  = get_bn_stats(net)
        print(f"[OK]   Loaded {key} ({info['name']}) ← {info['path']}")

    if len(loaded_models) < 2:
        print("Need ≥2 checkpoints. Exiting."); return

    keys = list(loaded_models.keys())

    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 1 — BatchNorm Parameter Drift
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("1. BATCHNORM PARAMETER DRIFT")
    print("="*60)

    bn_comparison_results = {}
    comparisons = [("M10", "M00"), ("M11", "M01")]
    for fb_key, nofb_key in comparisons:
        if fb_key not in bn_stats_map or nofb_key not in bn_stats_map:
            continue
        layer_kls, layer_l2s = [], []
        for layer in bn_stats_map[fb_key]:
            if layer not in bn_stats_map[nofb_key]:
                continue
            m_fb,   v_fb   = bn_stats_map[fb_key][layer]['mean'],   bn_stats_map[fb_key][layer]['var']
            m_nofb, v_nofb = bn_stats_map[nofb_key][layer]['mean'], bn_stats_map[nofb_key][layer]['var']
            kl = kl_divergence_gaussian(m_fb, v_fb, m_nofb, v_nofb)
            l2 = float(np.linalg.norm(m_fb - m_nofb))
            layer_kls.append((layer, kl))
            layer_l2s.append(l2)

        avg_kl   = float(np.mean([x[1] for x in layer_kls]))
        avg_l2   = float(np.mean(layer_l2s))
        top3_kl  = sorted(layer_kls, key=lambda x: x[1], reverse=True)[:5]

        bn_comparison_results[f"{fb_key}_vs_{nofb_key}"] = {
            "avg_kl": avg_kl, "avg_l2": avg_l2,
            "top_drifted": [(k, round(v, 4)) for k, v in top3_kl]
        }

        print(f"\n  {fb_key} (FB) vs {nofb_key} (No-FB):")
        print(f"    Avg L2 Distance  : {avg_l2:.4f}")
        print(f"    Avg KL Divergence: {avg_kl:.4f}")
        print(f"    Top drifted layers:")
        for lname, lkl in top3_kl:
            print(f"      {lname:<35s}  KL={lkl:.4f}")

    # BN Distribution violin-style plot (pure matplotlib)
    early_layers = ["e1.r1.bn1", "e1.r1.bn3", "d1.p1.conv1.1"]
    fig, axes = plt.subplots(1, len(early_layers), figsize=(14, 4))
    colors = {"M00": "#2196F3", "M01": "#03A9F4", "M10": "#F44336", "M11": "#FF5722"}
    for ax, lname in zip(axes, early_layers):
        for k in keys:
            if lname not in bn_stats_map[k]:
                continue
            data = bn_stats_map[k][lname]['mean']
            ax.violinplot([data], positions=[list(keys).index(k)],
                          showmedians=True, widths=0.7)
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels([models_info[k]['name'] for k in keys], rotation=15, fontsize=8)
        ax.set_title(lname, fontsize=9)
        ax.set_ylabel("Running Mean")
    fig.suptitle("BN running_mean distribution by model & layer", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "bn_drift_violin.png"), dpi=150)
    plt.close()
    print(f"\n  [Saved] bn_drift_violin.png")

    # ═══════════════════════════════════════════════════════════════════════
    # BLOCK 2 + 3 — Feature Collapse, Saturation, Gradient Norm
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("2 & 3. FEATURE COLLAPSE, SATURATION, GRADIENT NORM")
    print("="*60)

    # Register bottleneck hooks for all loaded models
    hooks = {k: BottleneckHook(loaded_models[k]) for k in keys}

    per_model = {k: {"cos_sim": [], "entropy": [], "saturation": [], "grad_norm": []}
                 for k in keys}

    n_valid = 0
    for idx, (x, y) in enumerate(val_loader):
        x = x.to(device, dtype=torch.float32)   # [1,3,256,256]
        y = y.to(device, dtype=torch.float32)   # [1,1,256,256]

        # ── Ground-truth spatial masks ──────────────────────────────────
        y_np = y[0, 0].cpu().numpy()            # (256,256)
        has_fg = (y_np.max() > 0)

        # We'll resize dist_map to match bottleneck feature h×w dynamically
        # (We don't know it until first forward; use 32×32 heuristic for 256→e4)
        if has_fg:
            fg_u8   = (y_np > 0.5).astype(np.uint8)
            bg_u8   = 1 - fg_u8
            dist_bg = cv2.distanceTransform(bg_u8, cv2.DIST_L2, 3)  # dist from BG pixel to nearest FG
            # boundary band: bg pixels within 20px of polyp edge
            boundary_np = (dist_bg > 0) & (dist_bg <= 20)
            farbg_np    = (dist_bg > 30)

        for k in keys:
            model = loaded_models[k]

            # ── Mask input ──────────────────────────────────────────────
            # For FB models: feed zeros (same as T0N diagnostic protocol)
            # Gradient must flow → requires_grad
            mask_input = torch.zeros(1, 1, *size, dtype=torch.float32,
                                     device=device, requires_grad=True)

            # ── Forward ─────────────────────────────────────────────────
            out = model([x, mask_input])          # [1,1,256,256] logits

            # ── Feature Collapse ────────────────────────────────────────
            feat = hooks[k].features              # [1, C, h, w]
            if feat is not None and has_fg:
                h_feat, w_feat = feat.shape[2], feat.shape[3]
                feat_hw = feat[0].permute(1, 2, 0).cpu().numpy()  # [h,w,C]

                # Resize masks to feature spatial size
                bnd_small  = cv2.resize(boundary_np.astype(np.uint8),
                                        (w_feat, h_feat),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)
                far_small   = cv2.resize(farbg_np.astype(np.uint8),
                                        (w_feat, h_feat),
                                        interpolation=cv2.INTER_NEAREST).astype(bool)

                if bnd_small.sum() > 0 and far_small.sum() > 0:
                    vec_bnd = feat_hw[bnd_small].mean(axis=0)
                    vec_far = feat_hw[far_small].mean(axis=0)
                    # Cosine similarity
                    denom = (np.linalg.norm(vec_bnd) * np.linalg.norm(vec_far) + 1e-8)
                    cos_s = float(np.dot(vec_bnd, vec_far) / denom)
                    per_model[k]["cos_sim"].append(cos_s)

            # ── Saturation & Entropy ────────────────────────────────────
            with torch.no_grad():
                prob = torch.sigmoid(out)
            sat   = float(((prob < 0.01) | (prob > 0.99)).float().mean().item())
            ent   = float(-(prob * torch.log(prob + 1e-8)
                            + (1-prob) * torch.log(1-prob + 1e-8)).mean().item())
            per_model[k]["saturation"].append(sat)
            per_model[k]["entropy"].append(ent)

            # ── Gradient Norm on mask input ─────────────────────────────
            out.sum().backward()
            gn = float(mask_input.grad.norm(2).item()) if mask_input.grad is not None else 0.0
            per_model[k]["grad_norm"].append(gn)
            model.zero_grad()

        n_valid += 1

    for k in keys:
        hooks[k].remove()

    print(f"\n  Processed {n_valid} validation images.")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("DIAGNOSTIC SUMMARY (Markdown Table)")
    print("="*60)

    summary = {}
    rows = []
    for k in keys:
        d = per_model[k]
        cos   = float(np.mean(d["cos_sim"]))       if d["cos_sim"]    else float("nan")
        sat   = float(np.mean(d["saturation"])) * 100
        ent   = float(np.mean(d["entropy"]))
        gnorm = float(np.mean(d["grad_norm"]))
        summary[k] = {"name": models_info[k]["name"],
                      "cos_sim": round(cos, 4),
                      "saturation_pct": round(sat, 2),
                      "entropy": round(ent, 4),
                      "grad_norm": round(gnorm, 6)}
        rows.append([models_info[k]["name"],
                     f"{cos:.4f}", f"{sat:.2f}%", f"{ent:.4f}", f"{gnorm:.6f}"])

    headers = ["Model", "CosSim (Bound-Far-BG)", "Saturation (%)", "Pred Entropy", "Mask Grad Norm"]
    col_w   = [max(len(str(r[i])) for r in [headers] + rows) for i in range(len(headers))]
    sep_row  = "|-" + "-|-".join(["-"*w for w in col_w]) + "-|"
    def fmt_row(r):
        return "| " + " | ".join(str(r[i]).ljust(col_w[i]) for i in range(len(r))) + " |"

    print(fmt_row(headers))
    print(sep_row)
    for r in rows:
        print(fmt_row(r))

    # Also save markdown
    md_lines = [fmt_row(headers), sep_row] + [fmt_row(r) for r in rows]
    with open(os.path.join(args.out_dir, "summary_table.md"), "w") as f:
        f.write("\n".join(md_lines) + "\n")

    # Also save BN drift results
    with open(os.path.join(args.out_dir, "bn_drift_results.json"), "w") as f:
        json.dump(bn_comparison_results, f, indent=4)

    # Also save full per-model metrics
    with open(os.path.join(args.out_dir, "diagnostic_metrics.json"), "w") as f:
        json.dump(summary, f, indent=4)

    # ═══════════════════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════════════════

    # 1. Feature Collapse Boxplot
    cos_data   = [per_model[k]["cos_sim"] for k in keys if per_model[k]["cos_sim"]]
    cos_labels = [models_info[k]["name"]  for k in keys if per_model[k]["cos_sim"]]
    if cos_data:
        fig, ax = plt.subplots(figsize=(8, 5))
        bp = ax.boxplot(cos_data, patch_artist=True, notch=False)
        pal = ["#2196F3", "#03A9F4", "#F44336", "#FF5722"]
        for patch, color in zip(bp["boxes"], pal[:len(cos_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(cos_labels)+1))
        ax.set_xticklabels(cos_labels, fontsize=9)
        ax.set_ylabel("Cosine Similarity", fontsize=10)
        ax.set_title("Bottleneck Feature: Boundary vs Far-Background Cosine Similarity\n"
                     "(Higher = Feature Collapse — BG features pulled toward FG space)", fontsize=10)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "feature_collapse_boxplot.png"), dpi=150)
        plt.close()
        print(f"\n  [Saved] feature_collapse_boxplot.png")

    # 2. Grad Norm & Saturation Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    pal = ["#2196F3", "#03A9F4", "#F44336", "#FF5722"]
    names    = [summary[k]["name"]            for k in keys]
    gnorms   = [summary[k]["grad_norm"]       for k in keys]
    sats     = [summary[k]["saturation_pct"]  for k in keys]

    bars0 = axes[0].bar(names, gnorms, color=pal[:len(keys)], alpha=0.8)
    axes[0].set_title("Mask Input Gradient Norm (L2)\n(Lower → more vanishing)", fontsize=10)
    axes[0].set_ylabel("Gradient Norm")
    for b, v in zip(bars0, gnorms):
        axes[0].text(b.get_x() + b.get_width()/2, v + max(gnorms)*0.01,
                     f"{v:.4f}", ha='center', va='bottom', fontsize=8)

    bars1 = axes[1].bar(names, sats, color=pal[:len(keys)], alpha=0.8)
    axes[1].set_title("Prediction Saturation (%)\n(Higher → harder binary decisions)", fontsize=10)
    axes[1].set_ylabel("Saturated pixels (%)")
    for b, v in zip(bars1, sats):
        axes[1].text(b.get_x() + b.get_width()/2, v + max(sats)*0.01,
                     f"{v:.1f}%", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "grad_saturation_bars.png"), dpi=150)
    plt.close()
    print(f"  [Saved] grad_saturation_bars.png")

    print(f"\n  All outputs saved to: {args.out_dir}/")
    print("  Files:", ", ".join(os.listdir(args.out_dir)))


if __name__ == "__main__":
    main()
