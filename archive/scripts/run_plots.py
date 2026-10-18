import os
import csv
import matplotlib.pyplot as plt
import numpy as np

# Set academic style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.dpi': 300
})

os.makedirs('rebuttal_plots', exist_ok=True)

datasets = {
    'Kvasir-SEG': 'kvasir-seg',
    'Kvasir-Sessile': 'kvasir-sessile',
    'CHASE-DB1': 'chase-db1'
}
path_tpl = "kaggle/downloads_rebuttal/outputs/{}_{}_history.csv"

def get_last_row(filename):
    with open(filename, 'r') as f:
        reader = list(csv.DictReader(f))
        last_row = reader[-1]
        all_val_dice = [float(r['val_dice']) for r in reader]
        return float(last_row['val_dice']), float(last_row['val_fpr']), all_val_dice

delta_dice = []
delta_fpr = []
sessile_m11_dice = []
sessile_m12_dice = []

for name, prefix in datasets.items():
    base_dice, base_fpr, all_base_dice = get_last_row(path_tpl.format(prefix, 'Baseline'))
    prop_dice, prop_fpr, all_prop_dice = get_last_row(path_tpl.format(prefix, 'Proposed'))
    
    # Delta (Proposed - Baseline)
    d_dice = (prop_dice - base_dice) * 100 # In percentage
    d_fpr = (prop_fpr - base_fpr) * 100 # In percentage
    
    delta_dice.append(d_dice)
    delta_fpr.append(d_fpr)
    
    if name == 'Kvasir-Sessile':
        sessile_m11_dice = np.array(all_base_dice)
        sessile_m12_dice = np.array(all_prop_dice)

# ==========================================
# Plot 1: Cross-Domain Universality Bar Chart
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6))
x = np.arange(len(datasets))
width = 0.35

rects1 = ax.bar(x - width/2, delta_dice, width, label='Δ Dice Score (%)', color='#2ca02c', edgecolor='black', alpha=0.8)
rects2 = ax.bar(x + width/2, delta_fpr, width, label='Δ False Positive Rate (%)', color='#d62728', edgecolor='black', alpha=0.8)

ax.set_ylabel('Absolute Change (%)')
ax.set_title('Cross-Domain Universality:\nProposed (M12) vs Baseline (M11) at Final Epoch')
ax.set_xticks(x)
ax.set_xticklabels(list(datasets.keys()))
ax.axhline(0, color='black', linewidth=1)
ax.legend()
plt.tight_layout()
plt.savefig('rebuttal_plots/plot1_universality.png', dpi=300)
plt.close()

# ==========================================
# Plot 2: Convergence Stability Line Plot
# ==========================================
fig, ax = plt.subplots(figsize=(8, 6))
epochs = np.arange(1, len(sessile_m11_dice) + 1)

ax.plot(epochs, sessile_m11_dice * 100, marker='o', linestyle='--', color='#d62728', label='M11 (Hard Feedback / Trap)', linewidth=2)
ax.plot(epochs, sessile_m12_dice * 100, marker='s', linestyle='-', color='#1f77b4', label='M12 (Detached Soft-OR / Firewall)', linewidth=2)

ax.set_xlabel('Training Epochs')
ax.set_ylabel('Validation Dice Score (%)')
ax.set_title('Convergence Stability on Kvasir-Sessile')
ax.grid(True, linestyle=':', alpha=0.6)
ax.legend()
plt.tight_layout()
plt.savefig('rebuttal_plots/plot2_convergence.png', dpi=300)
plt.close()

print("Metrics summary:")
for i, name in enumerate(datasets.keys()):
    print(f"{name}: Delta Dice = {delta_dice[i]:.2f}%, Delta FPR = {delta_fpr[i]:.2f}%")
