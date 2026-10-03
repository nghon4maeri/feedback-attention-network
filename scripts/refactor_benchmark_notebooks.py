"""Refactor old fanet_benchmark_*.ipynb notebooks: add section headers + evaluation/results cells.

Data paths, splits, model and training logic are NOT modified.
Usage: python scripts/refactor_benchmark_notebooks.py cvc_clinicdb isic_2018 kvasir_seg
"""
import json, re, sys, ast, uuid, os

EVAL_CELL = r'''# ================================================================
# [Evaluation Loop]  Dice / mIoU / Sensitivity / Specificity / FPR
# Evaluates the best checkpoint on the held-out split (same loader /
# resolution as training -> no change of the baseline protocol).
# Two rows are reported:
#   MFAD (pass-1, no feedback)  : prev_mask = None
#   MFAD (pass-2, feedback)     : prev_mask = sigmoid(pass-1)   [final output]
# ================================================================
import pandas as pd

DEVICE = globals().get('DEVICE', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
OUT_DIR = '/kaggle/working' if os.path.isdir('/kaggle/working') else '.'
EPS = 1e-8

best_path = EVAL_CKPT
assert os.path.exists(best_path), f"Checkpoint not found: {best_path}"
model = FANet_MFAD().to(DEVICE)
model.load_state_dict(torch.load(best_path, map_location=DEVICE))
model.eval()

def conf_metrics(gt, pred):
    """gt, pred: bool tensors (H,W) -> per-image metrics + raw counts"""
    tp = (gt & pred).sum().item(); fp = (~gt & pred).sum().item()
    fn = (gt & ~pred).sum().item(); tn = (~gt & ~pred).sum().item()
    return dict(
        dice=(2*tp)/(2*tp+fp+fn+EPS), iou=tp/(tp+fp+fn+EPS),
        sens=tp/(tp+fn+EPS), spec=tn/(tn+fp+EPS), fpr=fp/(fp+tn+EPS),
        tp=tp, fp=fp, fn=fn, tn=tn)

rows = {'MFAD (pass-1, no feedback)': [], 'MFAD (pass-2, feedback)': []}
preds_cache = []   # (img, gt, p1, p2, fp1, fp2) for qualitative plots
with torch.no_grad():
    for x, y in tqdm(valid_loader, desc='Evaluating'):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out1 = model(x)
        out2 = model(x, torch.sigmoid(out1))
        gt = y[0, 0] > 0.5
        p1 = torch.sigmoid(out1)[0, 0] > 0.5
        p2 = torch.sigmoid(out2)[0, 0] > 0.5
        m1, m2 = conf_metrics(gt, p1), conf_metrics(gt, p2)
        rows['MFAD (pass-1, no feedback)'].append(m1)
        rows['MFAD (pass-2, feedback)'].append(m2)
        if len(preds_cache) < 300:
            preds_cache.append((x[0].cpu(), gt.cpu(), p1.cpu(), p2.cpu(), m1['fp'], m2['fp']))

records = []
for name, ms in rows.items():
    df_ = pd.DataFrame(ms)
    rec = {'Dataset': DATASET_NAME, 'Model': name, 'N_images': len(df_)}
    for k, lab in [('dice', 'Dice'), ('iou', 'mIoU'), ('sens', 'Sensitivity'),
                   ('spec', 'Specificity'), ('fpr', 'FPR')]:
        rec[lab] = round(float(df_[k].mean()), 4)
        rec[lab + '_std'] = round(float(df_[k].std()), 4)
    # pixel-aggregated FPR (dataset level)
    rec['FPR_pixel'] = round(float(df_['fp'].sum() / (df_['fp'].sum() + df_['tn'].sum() + EPS)), 6)
    records.append(rec)
results_df = pd.DataFrame(records)
print(results_df.to_string(index=False))
'''

RESULTS_CELL = r'''# ================================================================
# [Results Output]  benchmark_results.csv + qualitative comparison
# ================================================================
csv_path = os.path.join(OUT_DIR, 'benchmark_results.csv')
results_df.to_csv(csv_path, index=False)
print(f"Saved metrics -> {csv_path}")

# Qualitative: Image | Ground Truth | MFAD pass-1 | MFAD final (feedback)
# Show the samples where feedback changes the false positives the most,
# plus a few random ones, so the Feedback-Trap effect is visible.
rng = np.random.RandomState(42)
delta = np.array([c[4] - c[5] for c in preds_cache])
top = list(np.argsort(-np.abs(delta))[:3])
rest = [i for i in rng.permutation(len(preds_cache)) if i not in top][:3]
sel = top + rest

fig, axes = plt.subplots(len(sel), 4, figsize=(14, 3.4 * len(sel)))
axes = np.atleast_2d(axes)
for r, i in enumerate(sel):
    img, gt, p1, p2, fp1, fp2 = preds_cache[i]
    axes[r, 0].imshow(img.permute(1, 2, 0).numpy().clip(0, 1)); axes[r, 0].set_title('Image')
    axes[r, 1].imshow(gt.numpy(), cmap='gray'); axes[r, 1].set_title('Ground Truth')
    axes[r, 2].imshow(p1.numpy(), cmap='gray'); axes[r, 2].set_title(f'MFAD pass-1 (FP px={fp1})')
    axes[r, 3].imshow(p2.numpy(), cmap='gray'); axes[r, 3].set_title(f'MFAD final (FP px={fp2})')
    for a in axes[r]: a.axis('off')
plt.suptitle(f'{DATASET_NAME}: qualitative comparison', y=1.0)
plt.tight_layout()
png_path = os.path.join(OUT_DIR, 'qualitative_comparison.png')
plt.savefig(png_path, dpi=150, bbox_inches='tight')
plt.show()
print(f"Saved figure -> {png_path}")
'''

CKPT_VAR = {  # variable holding the best-checkpoint path in each notebook's training cell
    'cvc_clinicdb': 'ckpt_name',
    'isic_2018': 'CKPT_BEST',
    'kvasir_seg': 'ckpt',
    'chase_db1': 'ckpt',
    'drive': 'ckpt',
    'dsb2018': 'CKPT_BEST',
    'em_dataset': 'ckpt',
}

def md(text):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {}, "source": text.splitlines(keepends=True)}

def code(text):
    return {"cell_type": "code", "execution_count": None, "id": uuid.uuid4().hex[:8],
            "metadata": {"refactored_eval": True}, "outputs": [], "source": text.splitlines(keepends=True)}

def src(c):
    return ''.join(c['source'])

def refactor(name):
    p = f'notebooks/fanet_benchmark_{name}.ipynb'
    nb = json.load(open(p, encoding='utf-8'))
    # idempotent: drop previously injected cells
    nb['cells'] = [c for c in nb['cells']
                   if not c.get('metadata', {}).get('refactored_eval')
                   and not c.get('metadata', {}).get('refactored_header')]
    cells = nb['cells']

    def first(pred):
        for i, c in enumerate(cells):
            if c['cell_type'] == 'code' and pred(src(c)):
                return i
        return None

    marks = [
        (first(lambda s: s.lstrip().startswith(('import os', '!pip'))), '## [Config & Imports]'),
        (first(lambda s: re.search(r'class \w*Dataset\(Dataset\)', s)), '## [Dataset & Transforms]'),
        (first(lambda s: 'class FANet_MFAD' in s), '## [Model Definition]'),
        (first(lambda s: 'optimizer = torch.optim.Adam' in s), '## [Training Loop]'),
    ]
    for idx, title in sorted([m for m in marks if m[0] is not None], reverse=True):
        h = md(title); h['metadata'] = {"refactored_header": True}
        cells.insert(idx, h)

    ck = CKPT_VAR[name]
    assert any(re.search(rf'\b{ck}\b', src(c)) for c in cells if c['cell_type'] == 'code'), ck
    cells.append(code(f"EVAL_CKPT = {ck}  # best checkpoint written by the training cell\n" + EVAL_CELL))
    cells.append(code(RESULTS_CELL))

    # syntax check every code cell
    for c in cells:
        if c['cell_type'] == 'code':
            s = '\n'.join('pass' if l.strip().startswith(('!', '%')) else l for l in src(c).split('\n'))
            ast.parse(s)
    json.dump(nb, open(p, 'w', encoding='utf-8', newline='\n'), indent=1, ensure_ascii=False)
    print('refactored', p, 'cells =', len(cells))

if __name__ == '__main__':
    for n in sys.argv[1:]:
        refactor(n)
