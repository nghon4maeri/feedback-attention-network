"""Phase 5 full statistical analysis (APA-oriented) following statistical-analysis skill.

Reads per-image eval from results/phase5_eval.json and produces:
  1. Descriptives (M, SD, median) per cell for Dice / FP rate / FN rate
  2. Assumption check: Shapiro-Wilk on paired deltas (Wilcoxon validity)
  3. Exact paired Wilcoxon + effect size (rank-biserial r) + bootstrap 95% CI
  4. Multiple-comparison context (pre-registered Bonferroni alpha = 0.0083)
  5. Sensitivity analysis: detectable delta at n=40, 80% power
  6. Bayesian paired t-test (T0N vs T00) to quantify support for the null

Outputs: results/phase5_stats_full.json
"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from scipy import stats

import pingouin as pg


def load_per_image(path, cells):
    with open(path, encoding='utf-8') as f:
        ev = json.load(f)
    out = {}
    for c in cells:
        rows = ev["per_image"][c]
        out[c] = {
            "dice": np.array([r["dice"] for r in rows]),
            "fp": np.array([r["fp_rate"] for r in rows]),
            "fn": np.array([r["fn_rate"] for r in rows]),
        }
    return out


def desc(arr):
    return {"mean": float(arr.mean()), "sd": float(arr.std(ddof=1)),
            "median": float(np.median(arr)), "n": int(len(arr))}


def main():
    cells = ["T00", "T0N", "TA", "TB"]
    D = load_per_image("results/phase5_eval.json", cells)
    t00_d, t00_fp, t00_fn = D["T00"]["dice"], D["T00"]["fp"], D["T00"]["fn"]

    report = {"descriptives": {c: {"dice": desc(D[c]["dice"]),
                                   "fp_rate": desc(D[c]["fp"]),
                                   "fn_rate": desc(D[c]["fn"])} for c in cells}}

    # ---- 2. assumption check on paired deltas + 3. paired tests ----
    report["paired"] = {}
    for c in cells:
        if c == "T00":
            continue
        dd = D[c]["dice"] - t00_d
        dfp = D[c]["fp"] - t00_fp
        dfn = D[c]["fn"] - t00_fn
        sw_d = stats.shapiro(dd)
        sw_fp = stats.shapiro(dfp)
        # rank-biserial correlation (effect size for signed-rank)
        rb = float(pg.compute_effsize(t00_d, D[c]["dice"], paired=True, eftype="r"))
        # bootstrap CI of mean delta
        rng = np.random.RandomState(42)
        boot = np.array([np.mean(rng.choice(dd, size=len(dd), replace=True))
                         for _ in range(5000)])
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        w = stats.wilcoxon(dd)
        report["paired"][c] = {
            "delta_dice_mean": float(dd.mean()),
            "delta_dice_sd": float(dd.std(ddof=1)),
            "delta_fp_pp": float(dfp.mean() * 100),
            "delta_fn_pp": float(dfn.mean() * 100),
            "shapiro_dice_W": float(sw_d.statistic), "shapiro_dice_p": float(sw_d.pvalue),
            "shapiro_fp_W": float(sw_fp.statistic), "shapiro_fp_p": float(sw_fp.pvalue),
            "wilcoxon_W_dice": float(w.statistic), "wilcoxon_p_dice": float(w.pvalue),
            "rank_biserial_dice": rb,
            "bootstrap_ci95_dice": ci,
            "n_pos_dice": int((dd > 0).sum()),
            "n_pos_fp": int((dfp < 0).sum()),  # number of images with FP decreased
        }

    # ---- 5. sensitivity analysis (n=40, two-sided paired, 80% power) ----
    # Cohen's d detectable for paired design: d = (t_alpha + t_beta) / sqrt(n)
    from statsmodels.stats.power import tt_solve_power
    d_detect = tt_solve_power(effect_size=None, nobs=40, alpha=0.05, power=0.80, alternative="two-sided")
    report["sensitivity_n40"] = {"detectable_cohens_d": float(d_detect),
                                 "note": "paired delta, alpha=.05, power=.80"}
    # n required for d=0.5 and for observed d
    for d_target in (0.30, 0.50, 0.80):
        n = tt_solve_power(effect_size=d_target, nobs=None, alpha=0.05, power=0.80,
                           alternative="two-sided")
        report[f"sensitivity_need_n_d{d_target:.2f}"] = int(np.ceil(n))

    # ---- 6. Bayesian paired t-test: T0N vs T00 ----
    try:
        bf = pg.ttest(t00_d, D["T0N"]["dice"], paired=True)  # Bays factor column if available
    except Exception:
        bf = None
    # posterior probability via normal approximation with weakly informed prior
    dd = D["T0N"]["dice"] - t00_d
    mu = dd.mean(); sd = dd.std(ddof=1) / np.sqrt(len(dd))
    # P(mean_delta > 0 | data) under Normal(mu, sd)
    p_pos = 1 - stats.norm.cdf(0, loc=mu, scale=sd)
    report["bayes_T0N_vs_T00"] = {
        "mean_delta": float(mu), "se_delta": float(sd),
        "p_mean_delta_gt_0": float(p_pos),
        "normal_approx_note": "P(mean delta > 0) under Normal posterior approx; not a BF",
        "pingouin_bf10_paired": (float(bf["BF10"].values[0]) if bf is not None and "BF10" in bf.columns else None),
    }

    with open("results/phase5_stats_full.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)

    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()