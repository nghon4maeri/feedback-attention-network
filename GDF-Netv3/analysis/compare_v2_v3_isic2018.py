"""
GDF-Netv2 vs GDF-Netv3 Benchmark Comparison Analysis on ISIC-2018 Blind Test Set
Evaluates all 9 clinical metrics and specifically measures recovery of the 40 outlier failure cases (micro-lesions).
"""

import os
import numpy as np
import pandas as pd

def run_comparison():
    v2_raw_path = "GDF-Netv2/results/isic2018/benchmark_raw_ISIC2018.csv"
    v3_raw_path = "GDF-Netv3/results/isic2018/benchmark_raw_ISIC2018.csv"

    if not os.path.exists(v2_raw_path) or not os.path.exists(v3_raw_path):
        print(f"Error: Either {v2_raw_path} or {v3_raw_path} is missing.")
        return

    v2_df = pd.read_csv(v2_raw_path)
    v3_df = pd.read_csv(v3_raw_path)

    print("=================================================================")
    print("       ISIC-2018 BLIND TEST (1,000 CASES): GDF-NET V2 VS V3       ")
    print("=================================================================")

    metrics = ['dice', 'miou', 'sensitivity', 'specificity', 'precision', 'fpr', 'hd95', 'asd', 'bf2']
    
    summary_rows = []
    for m in metrics:
        v2_mean, v2_med = v2_df[m].mean(), v2_df[m].median()
        v3_mean, v3_med = v3_df[m].mean(), v3_df[m].median()
        delta_mean = v3_mean - v2_mean
        delta_med = v3_med - v2_med
        summary_rows.append({
            'Metric': m.upper(),
            'v2 Mean': f"{v2_mean:.4f}",
            'v3 Mean': f"{v3_mean:.4f}",
            'Delta Mean': f"{delta_mean:+.4f}",
            'v2 Med': f"{v2_med:.4f}",
            'v3 Med': f"{v3_med:.4f}",
            'Delta Med': f"{delta_med:+.4f}"
        })
    
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    print("\n-----------------------------------------------------------------")
    print("                    DICE DISTRIBUTION COMPARISON                 ")
    print("-----------------------------------------------------------------")
    
    total = len(v2_df)
    v2_ge90 = (v2_df['dice'] >= 0.90).sum()
    v3_ge90 = (v3_df['dice'] >= 0.90).sum()
    v2_ge80 = (v2_df['dice'] >= 0.80).sum()
    v3_ge80 = (v3_df['dice'] >= 0.80).sum()
    v2_lt70 = (v2_df['dice'] < 0.70).sum()
    v3_lt70 = (v3_df['dice'] < 0.70).sum()
    v2_eq0  = (v2_df['dice'] == 0.0).sum()
    v3_eq0  = (v3_df['dice'] == 0.0).sum()

    print(f"Dice >= 0.90: v2 = {v2_ge90}/{total} ({v2_ge90/total*100:.1f}%) | v3 = {v3_ge90}/{total} ({v3_ge90/total*100:.1f}%) [Delta: {v3_ge90 - v2_ge90:+d}]")
    print(f"Dice >= 0.80: v2 = {v2_ge80}/{total} ({v2_ge80/total*100:.1f}%) | v3 = {v3_ge80}/{total} ({v3_ge80/total*100:.1f}%) [Delta: {v3_ge80 - v2_ge80:+d}]")
    print(f"Dice <  0.70: v2 = {v2_lt70}/{total} ({v2_lt70/total*100:.1f}%) | v3 = {v3_lt70}/{total} ({v3_lt70/total*100:.1f}%) [Delta: {v3_lt70 - v2_lt70:+d}]")
    print(f"Dice == 0.00: v2 = {v2_eq0}/{total} ({v2_eq0/total*100:.1f}%) | v3 = {v3_eq0}/{total} ({v3_eq0/total*100:.1f}%) [Delta: {v3_eq0 - v2_eq0:+d}]")

    print("\n-----------------------------------------------------------------")
    print("          FAILURE RECOVERY ANALYSIS (v2 Dice < 0.70 cases)       ")
    print("-----------------------------------------------------------------")
    fail_idx = v2_df[v2_df['dice'] < 0.70].index
    v2_fail_dice = v2_df.loc[fail_idx, 'dice']
    v3_fail_dice = v3_df.loc[fail_idx, 'dice']
    
    recovered_to_80 = (v3_fail_dice >= 0.80).sum()
    recovered_to_70 = (v3_fail_dice >= 0.70).sum()
    mean_gain = (v3_fail_dice - v2_fail_dice).mean()
    
    print(f"Total outlier cases in v2: {len(fail_idx)}")
    print(f"Mean Dice of outlier cases in v2: {v2_fail_dice.mean():.4f}")
    print(f"Mean Dice of same cases in v3:   {v3_fail_dice.mean():.4f} (Mean Gain: {mean_gain:+.4f})")
    print(f"Cases recovered to Dice >= 0.70: {recovered_to_70}/{len(fail_idx)} ({recovered_to_70/len(fail_idx)*100:.1f}%)")
    print(f"Cases recovered to Dice >= 0.80: {recovered_to_80}/{len(fail_idx)} ({recovered_to_80/len(fail_idx)*100:.1f}%)")
    print("=================================================================")

if __name__ == "__main__":
    run_comparison()
