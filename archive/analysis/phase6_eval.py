"""Phase 6 eval: TC (FB x Tversky-asym) / TD (no-FB x Tversky-asym) vs baselines.

Protocol (inherits Phase 5):
  - TC : 4-iteration refinement exactly like Phase 4/5 (Otsu init, feedback =
         cell's own binary prediction; keeps the FANet loop). Baseline = T00.
  - TD : single forward pass with zero mask (no-feedback, no Otsu). Baseline = T0N.
  - T00/T0N per-image references are reused from results/phase5_eval.json
    (same split, same 40 val images, same eval protocol).

Metrics per image (iteration 4 for TC): Dice, IoU, precision, recall, FP rate,
FN rate, FP/FN distance-to-GT-boundary. Paired stats vs the matched baseline:
Wilcoxon, Cohen's d, bootstrap 95% CI, n_pos.

Gate (pre-registered, report 09/09, Plan B):
  - TC vs T00: pass if FP < T00 - 0.01 AND Dice >= T00 - 0.02 AND FN <= FN(T00) + 0.02.
  - TD vs T0N: pass if FP < T0N - 0.01 AND Dice >= T0N - 0.02 AND FN <= FN(T0N) + 0.02.

Usage:
    python analysis/phase6_eval.py --ckpt-dir checkpoints_phase6
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import cv2
import torch
from scipy import ndimage
from tqdm import tqdm

from fanet.config import load_config
from fanet.data import load_data
from fanet.models import FANet
from fanet.utils import seeding

NUM_ITER = 4


def dice_np(gt, pred):
    gt = gt.astype(bool); pred = pred.astype(bool)
    inter = (gt & pred).sum()
    return 2 * inter / (gt.sum() + pred.sum() + 1e-15)


def iou_np(gt, pred):
    gt = gt.astype(bool); pred = pred.astype(bool)
    inter = (gt & pred).sum()
    return inter / ((gt | pred).sum() + 1e-15)


def otsu_bin(img, size):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, size)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (th / 255.0 > 0.5).astype(np.float32)


def dist_to_boundary(mask):
    mask = mask.astype(np.uint8)
    edt_in = ndimage.distance_transform_edt(mask)
    edt_out = ndimage.distance_transform_edt(1 - mask)
    return np.where(mask > 0, edt_in, edt_out)


def eval_cell(model, img, otsu, size, device, no_feedback, num_iter=4):
    """Return list of (p_soft_bin) per iteration."""
    img_t = torch.from_numpy(np.transpose(img, (2, 0, 1)) / 255.0).float().unsqueeze(0).to(device)
    outs = []
    prev = None
    for t in range(num_iter):
        if no_feedback:
            m = np.zeros((1, 1, size[0], size[1]), dtype=np.float32)
        elif prev is None:
            m = otsu[None, None]
        else:
            m = (prev > 0.5).astype(np.float32)[None, None]
        m = torch.from_numpy(m).to(device)
        with torch.no_grad():
            p = torch.sigmoid(model([img_t, m]))[0, 0].cpu().numpy()
        prev = p
        outs.append(p)
    return outs


def eval_one_cell(model, test_x, test_y, size, device, no_feedback, n):
    dice = np.zeros((NUM_ITER, n)); iou = np.zeros((NUM_ITER, n))
    fp_rate = np.zeros((NUM_ITER, n)); fn_rate = np.zeros((NUM_ITER, n))
    prec = np.zeros((NUM_ITER, n)); rec = np.zeros((NUM_ITER, n))
    pi = []
    for i, (x_path, y_path) in tqdm(enumerate(zip(test_x, test_y)), total=n):
        img = cv2.imread(x_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, size)
        gt = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, size) / 255.0
        gt = (gt > 0.5).astype(np.float32)
        otsu = otsu_bin(img, size)
        outs = eval_cell(model, img, otsu, size, device, no_feedback)
        for t, p in enumerate(outs):
            pbin = (p > 0.5).astype(np.float32)
            gtb = gt.astype(bool); pbb = pbin.astype(bool)
            fp = (pbb & ~gtb); fn = (~pbb & gtb)
            tp = (pbb & gtb).sum()
            pred_pos = pbb.sum(); gt_pos = gtb.sum()
            dice[t, i] = dice_np(gt, pbin)
            iou[t, i] = iou_np(gt, pbin)
            fp_rate[t, i] = fp.mean(); fn_rate[t, i] = fn.mean()
            prec[t, i] = tp / (pred_pos + 1e-15)
            rec[t, i] = tp / (gt_pos + 1e-15)
            if t == NUM_ITER - 1:
                d = dist_to_boundary(gt)
                pi.append({
                    "image": i, "dice": dice_np(gt, pbin), "iou": iou_np(gt, pbin),
                    "precision": float(prec[t, i]), "recall": float(rec[t, i]),
                    "fp_rate": float(fp.mean()), "fn_rate": float(fn.mean()),
                    "fp_dist": float(d[fp].mean()) if fp.any() else float("nan"),
                    "fn_dist": float(d[fn].mean()) if fn.any() else float("nan"),
                })
    summary = {
        "dice_mean_iter": [float(dice[t].mean()) for t in range(NUM_ITER)],
        "iou_mean_iter": [float(iou[t].mean()) for t in range(NUM_ITER)],
        "final_dice": float(dice[-1].mean()), "final_iou": float(iou[-1].mean()),
        "final_fp_rate": float(fp_rate[-1].mean()),
        "final_fn_rate": float(fn_rate[-1].mean()),
        "final_precision": float(prec[-1].mean()),
        "final_recall": float(rec[-1].mean()),
    }
    return summary, pi


def paired_stats(a_dice, a_fp, a_fn, b_dice, b_fp, b_fn):
    dd = np.array(a_dice) - np.array(b_dice)
    dfp = np.array(a_fp) - np.array(b_fp)
    dfn = np.array(a_fn) - np.array(b_fn)

    def wilcoxon_onesample(x):
        from scipy.stats import wilcoxon
        x = x[~np.isnan(x)]
        if len(x) == 0:
            return float("nan")
        try:
            return float(wilcoxon(x).pvalue)
        except ValueError:
            return 1.0

    rng = np.random.RandomState(42)
    boot = np.array([np.mean(rng.choice(dd, size=len(dd), replace=True))
                     for _ in range(5000)])
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
    sd = dd.std(ddof=1)
    cohend = (dd.mean() / sd) if sd > 0 else float("nan")

    return {
        "delta_dice": float(dd.mean()),
        "delta_fp": float(dfp.mean()),
        "delta_fn": float(dfn.mean()),
        "ci95_dice": list(ci),
        "wilcoxon_p_dice": wilcoxon_onesample(dd),
        "wilcoxon_p_fp": wilcoxon_onesample(dfp),
        "cohens_d_dice": cohend,
        "n_pos_dice": int((dd > 0).sum()),
        "n_pos_fp": int((dfp < 0).sum()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/kvasir_sessile.yaml")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_phase6")
    parser.add_argument("--phase5-json", type=str, default="results/phase5_eval.json")
    parser.add_argument("--out", type=str, default="results/phase6_eval.json")
    parser.add_argument("--stats-out", type=str, default="results/phase6_stats.json")
    args = parser.parse_args()

    cfg = load_config(args.config)
    size = tuple(cfg["dataset"]["image_size"])
    device = torch.device("cpu")

    (train_x, train_y), (test_x, test_y) = load_data(cfg["dataset"]["root"])
    n = len(test_x)

    with open(args.phase5_json) as f:
        p5 = json.load(f)

    baselines = {
        "TC": "T00",
        "TD": "T0N",
    }
    cells = [
        ("TC", "binary", False, False),
        ("TD", "binary", False, True),
    ]

    summary = {}
    per_image = {}
    for name, gate, dual, no_feedback in cells:
        ckpt = os.path.join(args.ckpt_dir, f"ckpt_{name}.pth")
        if not os.path.exists(ckpt):
            print(f"SKIP {name}: {ckpt} not found")
            continue
        model = FANet(gate=gate, dual_path=dual)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device); model.eval()
        print(f"Loaded {name} ({gate}, no_feedback={no_feedback})")
        s, pi = eval_one_cell(model, test_x, test_y, size, device, no_feedback, n)
        summary[name] = s
        per_image[name] = pi

    # carry over the matched baselines from phase5_eval.json
    for name, base in baselines.items():
        summary[f"{name}_base_{base}"] = p5["summary"][base]
        per_image[f"{name}_base_{base}"] = p5["per_image"][base]

    # paired stats vs matched baseline
    stats = {}
    for name, base in baselines.items():
        a = per_image[name]
        b = per_image[f"{name}_base_{base}"]
        a_d = [r["dice"] for r in a]; a_fp = [r["fp_rate"] for r in a]; a_fn = [r["fn_rate"] for r in a]
        b_d = [r["dice"] for r in b]; b_fp = [r["fp_rate"] for r in b]; b_fn = [r["fn_rate"] for r in b]
        stats[name] = paired_stats(a_d, a_fp, a_fn, b_d, b_fp, b_fn)
        stats[name]["baseline"] = base

    # ---- gate verdicts (pre-registered) ----
    for name, base in baselines.items():
        s = summary[name]; b = p5["summary"][base]
        fp_ok = s["final_fp_rate"] < b["final_fp_rate"] - 0.01
        dice_ok = s["final_dice"] >= b["final_dice"] - 0.02
        fn_ok = s["final_fn_rate"] <= b["final_fn_rate"] + 0.02
        gate = {
            "baseline": base, "fp_ok": bool(fp_ok), "dice_ok": bool(dice_ok),
            "fn_ok": bool(fn_ok), "pass": bool(fp_ok and dice_ok and fn_ok),
            "criterion": "FP < base-0.01 AND Dice >= base-0.02 AND FN <= base+0.02",
        }
        stats[name]["gate"] = gate

    print("\n=== Final iteration ===")
    print(f"{'cell':>6} | {'dice':>7} {'iou':>7} {'fp%':>6} {'fn%':>6} {'prec':>7} {'rec':>7}")
    for c in summary:
        s = summary[c]
        print(f"{c:>6} | {s['final_dice']:>7.4f} {s['final_iou']:>7.4f} "
              f"{s['final_fp_rate']*100:>6.2f} {s['final_fn_rate']*100:>6.2f} "
              f"{s['final_precision']:>7.4f} {s['final_recall']:>7.4f}")

    print("\n=== Paired vs matched baseline ===")
    for c in stats:
        if "baseline" not in stats[c]:
            continue
        st = stats[c]
        print(f"{c:>6} vs {st['baseline']:>4} | dDice {st['delta_dice']:+.4f} "
              f"(p={st['wilcoxon_p_dice']:.3f}, CI {st['ci95_dice'][0]:+.3f},{st['ci95_dice'][1]:+.3f}) | "
              f"dFP {st['delta_fp']*100:+.2f}pp (p={st['wilcoxon_p_fp']:.3f}) | "
              f"dFN {st['delta_fn']*100:+.2f}pp | n_pos {st['n_pos_dice']}/40 | "
              f"GATE pass={st['gate']['pass']}")

    os.makedirs(cfg["paths"]["results"], exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"summary": summary, "per_image": per_image}, f, indent=2)
    with open(args.stats_out, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved: {args.out}, {args.stats_out}")


if __name__ == "__main__":
    main()