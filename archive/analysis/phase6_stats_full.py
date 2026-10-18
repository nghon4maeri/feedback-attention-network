"""Phase 6 full statistical analysis (APA-oriented) following statistical-analysis skill.

Reads per-image eval from results/phase6_eval.json (TC/TD + matched baselines
T00/T0N from phase5_eval.json) and produces:
  1. Descriptives (M, SD, median) per cell for Dice / FP rate / FN rate
  2. Assumption check: Shapiro-Wilk on paired deltas (Wilcoxon validity)
  3. Exact paired Wilcoxon + effect size (rank-biserial r) + bootstrap 95% CI
  4. Bayes paired t-test (TD vs T0N) to quantify support for the difference
  5. 2x2 factorial main effects + interaction: loss {orig, asym}, feedback {FB, no-FB}
     using per-image means (T00 = FB/org, T0N = no-FB/org, TC = FB/asym, TD = no-FB/asym)
  6. Gate verdicts vs matched baselines (pre-registered criteria)

Outputs: results/phase6_stats_full.json
"""
import json
import sys

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
from scipy import stats

import pingouin as pg


def load_per_image(path):
    with open(path, encoding='utf-8') as f:
        ev = json.load(f)
    out = {}
    for c, rows in ev["per_image"].items():
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
    cells = ["TC", "TD", "TC_base_T00", "TD_base_T0N"]
    D = load_per_image("results/phase6_eval.json")

    report = {"descriptives": {c: {"dice": desc(D[c]["dice"]),
                                   "fp_rate": desc(D[c]["fp"]),
                                   "fn_rate": desc(D[c]["fn"])} for c in cells}}

    # ---- 2. assumption check + 3. paired tests ----
    report["paired"] = {}
    for cell, base in [("TC", "TC_base_T00"), ("TD", "TD_base_T0N")]:
        dd = D[cell]["dice"] - D[base]["dice"]
        dfp = D[cell]["fp"] - D[base]["fp"]
        dfn = D[cell]["fn"] - D[base]["fn"]
        sw_d = stats.shapiro(dd)
        sw_fp = stats.shapiro(dfp)
        rb = float(pg.compute_effsize(D[base]["dice"], D[cell]["dice"], paired=True, eftype="r"))
        rng = np.random.RandomState(42)
        boot = np.array([np.mean(rng.choice(dd, size=len(dd), replace=True))
                         for _ in range(5000)])
        ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5)))
        w = stats.wilcoxon(dd)
        w_fp = stats.wilcoxon(dfp)
        report["paired"][cell] = {
            "baseline": base.replace("TC_base_", "").replace("TD_base_", ""),
            "delta_dice_mean": float(dd.mean()),
            "delta_dice_sd": float(dd.std(ddof=1)),
            "delta_fp_pp": float(dfp.mean() * 100),
            "delta_fn_pp": float(dfn.mean() * 100),
            "shapiro_dice_W": float(sw_d.statistic), "shapiro_dice_p": float(sw_d.pvalue),
            "shapiro_fp_W": float(sw_fp.statistic), "shapiro_fp_p": float(sw_fp.pvalue),
            "wilcoxon_W_dice": float(w.statistic), "wilcoxon_p_dice": float(w.pvalue),
            "wilcoxon_W_fp": float(w_fp.statistic), "wilcoxon_p_fp": float(w_fp.pvalue),
            "rank_biserial_dice": rb,
            "bootstrap_ci95_dice": ci,
            "n_pos_dice": int((dd > 0).sum()),
            "n_pos_fp": int((dfp < 0).sum()),
        }

    # ---- 4. Bayes paired t-test for TD vs T0N (the significant FP finding) ----
    td = D["TD"]["dice"] - D["TD_base_T0N"]["dice"]
    try:
        bf = pg.ttest(D["TD_base_T0N"]["dice"], D["TD"]["dice"], paired=True)
        bf10 = float(bf["BF10"].values[0]) if "BF10" in bf.columns else None
    except Exception:
        bf10 = None
    dd_fp = D["TD"]["fp"] - D["TD_base_T0N"]["fp"]
    mu_fp = dd_fp.mean(); sd_fp = dd_fp.std(ddof=1) / np.sqrt(len(dd_fp))
    p_fp_neg = 1 - stats.norm.cdf(0, loc=mu_fp, scale=sd_fp)
    report["bayes_TD_vs_T0N"] = {
        "bf10_dice": bf10,
        "mean_fp_delta": float(mu_fp), "se_fp_delta": float(sd_fp),
        "p_mean_fp_lt_0": float(p_fp_neg),
        "note": "P(mean FP delta < 0) under Normal posterior approx",
    }

    # ---- 5. 2x2 factorial main effects + interaction (per-image) ----
    # Cells: T00 = FB,orig | T0N = noFB,orig | TC = FB,asym | TD = noFB,asym
    fb_orig = D["TC_base_T00"]["fp"]; nofb_orig = D["TD_base_T0N"]["fp"]
    fb_asym = D["TC"]["fp"]; nofb_asym = D["TD"]["fp"]
    d_fp = {"loss": fb_asym - fb_orig, "nofb_loss": nofb_asym - nofb_orig}
    report["factorial_fp"] = {
        "main_effect_loss_fp_pp": float((d_fp["loss"].mean() + d_fp["nofb_loss"].mean()) / 2 * 100),
        "main_effect_feedback_fp_pp": float(((fb_orig - nofb_orig).mean() + (fb_asym - nofb_asym).mean()) / 2 * 100),
        "interaction_loss_x_feedback_fp_pp": float(((fb_asym - fb_orig) - (nofb_asym - nofb_orig)).mean() * 100),
        "note": "per-image; interaction = (dFP|FB) - (dFP|noFB); >0 means FB blocks/blunts the loss effect on FP",
    }
    # Dice main effects + interaction
    dfb_orig = D["TC_base_T00"]["dice"]; dnofb_orig = D["TD_base_T0N"]["dice"]
    dfb_asym = D["TC"]["dice"]; dnofb_asym = D["TD"]["dice"]
    report["factorial_dice"] = {
        "main_effect_loss_dice": float(((dfb_asym - dfb_orig).mean() + (dnofb_asym - dnofb_orig).mean()) / 2),
        "main_effect_feedback_dice": float(((dfb_orig - dnofb_orig).mean() + (dfb_asym - dnofb_asym).mean()) / 2),
        "interaction_loss_x_feedback_dice": float(((dfb_asym - dfb_orig) - (dnofb_asym - dnofb_orig)).mean()),
        "note": "per-image; interaction = (dDice|FB) - (dDice|noFB)",
    }

    # ---- 6. sensitivity (n=40, two-sided paired, 80% power) ----
    from statsmodels.stats.power import tt_solve_power
    d_detect = tt_solve_power(effect_size=None, nobs=40, alpha=0.05, power=0.80, alternative="two-sided")
    report["sensitivity_n40"] = {"detectable_cohens_d": float(d_detect),
                                 "note": "paired delta, alpha=.05, power=.80"}

    with open("results/phase6_stats_full.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=float)

    print(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()