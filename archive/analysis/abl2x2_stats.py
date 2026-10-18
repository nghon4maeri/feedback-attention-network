"""Statistical analysis of the 2x2 ablation (paired per-image).

Tests (paired by image, n=40):
    C01-t* vs C00, C10 vs C00, C11 vs C00, C11 vs C10, C11 vs C01-t0.1
Metrics: dice at final iteration.
Methods: Wilcoxon signed-rank (2-tailed), Cohen's d (paired: mean diff / SD diff),
bootstrap 95% CI of mean delta (2000 resamples).

Usage:
    python analysis/abl2x2_stats.py
"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from scipy import stats

with open("results/abl2x2_summary.json") as f:
    data = json.load(f)

per = data["per_image"]
cells = list(per.keys())
n = len(per[cells[0]])

dice = {c: np.array([r["dice"] for r in per[c]]) for c in cells}
fp_rate = {c: np.array([r["fp_rate"] for r in per[c]]) for c in cells}
fn_rate = {c: np.array([r["fn_rate"] for r in per[c]]) for c in cells}


def paired_stats(a, b, rng, n_boot=2000):
    delta = a - b
    w, p = stats.wilcoxon(a, b, alternative="two-sided")
    d = delta.mean() / (delta.std(ddof=1) + 1e-15)
    boots = np.array([
        (rng.choice(delta, size=len(delta), replace=True)).mean()
        for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_delta": float(delta.mean()),
            "wilcoxon_p": float(p),
            "cohens_d": float(d),
            "ci95": [float(lo), float(hi)],
            "n_pos": int((delta > 0).sum()),
            "n_neg": int((delta < 0).sum()),
            "n_zero": int((np.abs(delta) < 1e-9).sum())}


rng = np.random.RandomState(42)
comparisons = [
    ("C01-t0.1", "C00"), ("C01-t0.2", "C00"), ("C01-t0.3", "C00"),
    ("C10", "C00"), ("C11", "C00"), ("C11", "C10"), ("C11", "C01-t0.1"),
]

ALPHA_BONF = 0.05 / 6  # 6 primary comparisons

print(f"n = {n} images, paired tests, final-iteration dice")
print(f"Bonferroni threshold for 6 primary comparisons: p < {ALPHA_BONF:.4f}\n")

print(f"{'comparison':>16} | {'mean d':>8} {'CI95 low':>9} {'CI95 hi':>9} "
      f"{'p(wilcox)':>10} {'cohens_d':>8} {'pos/neg':>9} {'sig':>4}")
print("-" * 88)
for a, b in comparisons:
    s = paired_stats(dice[a], dice[b], rng)
    sig = "YES" if s["wilcoxon_p"] < ALPHA_BONF else "no"
    print(f"{a+' vs '+b:>16} | {s['mean_delta']:>+8.4f} {s['ci95'][0]:>9.4f} "
          f"{s['ci95'][1]:>9.4f} {s['wilcoxon_p']:>10.2e} {s['cohens_d']:>+8.3f} "
          f"{str(s['n_pos'])+'/'+str(s['n_neg']):>9} {sig:>4}")

print("\n=== FP rate comparisons (hypothesis P_B1) ===")
for a, b in [("C01-t0.1", "C00"), ("C10", "C00")]:
    s = paired_stats(fp_rate[a], fp_rate[b], rng)
    print(f"  {a} vs {b}: mean delta {s['mean_delta']*100:+.2f}pp "
          f"(p={s['wilcoxon_p']:.2e})")

print("\n=== FN rate comparisons ===")
for a, b in [("C01-t0.1", "C00"), ("C10", "C00")]:
    s = paired_stats(fn_rate[a], fn_rate[b], rng)
    print(f"  {a} vs {b}: mean delta {s['mean_delta']*100:+.2f}pp "
          f"(p={s['wilcoxon_p']:.2e})")

out = {f"{a}_vs_{b}": paired_stats(dice[a], dice[b], rng)
       for a, b in comparisons}
out["fp_rate"] = {f"{a}_vs_{b}": paired_stats(fp_rate[a], fp_rate[b], rng)
                  for a, b in [("C01-t0.1", "C00"), ("C10", "C00")]}
out["fn_rate"] = {f"{a}_vs_{b}": paired_stats(fn_rate[a], fn_rate[b], rng)
                  for a, b in [("C01-t0.1", "C00"), ("C10", "C00")]}
with open("results/abl2x2_stats.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved: results/abl2x2_stats.json")
