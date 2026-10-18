"""Phase 4 stats: T00 vs T10 (binary vs STE trained) paired test.

Note: T01/T11 checkpoints are copies of T00/T10 (dual-path bug: m_bg zeros at
train time), and their eval in phase4_eval.json used dual=True mismatch, so they
are EXCLUDED from inference. This script only tests T_A on the valid pair.
"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from scipy import stats

with open("results/phase4_eval.json") as f:
    data = json.load(f)

dice = {}
for c in ("T00", "T10"):
    dice[c] = np.array([r["dice"] for r in data["per_image"][c]])

a, b = dice["T00"], dice["T10"]  # binary vs ste
delta = a - b

w, p = stats.wilcoxon(a, b, alternative="two-sided")
d = delta.mean() / (delta.std(ddof=1) + 1e-15)
rng = np.random.RandomState(42)
boots = np.array([rng.choice(delta, size=len(delta), replace=True).mean()
                  for _ in range(2000)])
lo, hi = np.percentile(boots, [2.5, 97.5])

print(f"n = {len(a)} images (paired, final iteration)")
print(f"T00 binary: {a.mean():.4f} | T10 ste: {b.mean():.4f}")
print(f"delta (binary - ste): {delta.mean():+.4f}  [95% CI: {lo:+.4f}, {hi:+.4f}]")
print(f"Wilcoxon p: {p:.3e} | Cohen's d: {d:+.3f}")
print(f"positive deltas: {(delta > 0).sum()}/40")
sig = "YES" if p < 0.05 else "no"
print(f"\nVerdict T_A (STE >= binary): {'BÁC BỎ' if b.mean() < a.mean() else 'KHÔNG BÁC BỎ'} "
      f"(ste kém hơn {a.mean()-b.mean():+.4f} dice, p={p:.2e})")

with open("results/phase4_stats.json", "w") as f:
    json.dump({
        "T00_dice": float(a.mean()), "T10_dice": float(b.mean()),
        "delta": float(delta.mean()), "ci95": [float(lo), float(hi)],
        "wilcoxon_p": float(p), "cohens_d": float(d),
        "n_pos": int((delta > 0).sum()),
    }, f, indent=2)
print("\nSaved: results/phase4_stats.json")