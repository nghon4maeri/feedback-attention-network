"""Fix: F.binary_cross_entropy is unsafe under torch.autocast (raises RuntimeError on recent PyTorch).
Compute DiceBCELoss in fp32 with autocast disabled. Math is unchanged."""
import json, glob, re, sys

DEF_RE = re.compile(r'^(?P<ind>[ \t]*)def forward\(self, inputs, targets(?:, smooth=1(?:\.0)?)?\):\n', re.M)

def patch_text(s):
    if 'class DiceBCELoss' not in s or 'autocast(device_type' in s.split('class DiceBCELoss')[1][:400]:
        return s, False
    head, tail = s.split('class DiceBCELoss', 1)
    m = DEF_RE.search(tail)
    if not m:
        return s, False
    ind = m.group('ind')
    new = (f"{ind}@torch.autocast(device_type='cuda', enabled=False)  # BCE is unsafe under autocast -> run loss in fp32\n"
           + m.group(0)
           + f"{ind}    inputs, targets = inputs.float(), targets.float()\n")
    tail = tail[:m.start()] + new + tail[m.end():]
    return head + 'class DiceBCELoss' + tail, True

changed = []
for p in glob.glob('notebooks/**/*.ipynb', recursive=True) + glob.glob('kaggle/benchmark_kernels/**/*.ipynb', recursive=True):
    nb = json.load(open(p, encoding='utf-8')); ch = False
    for c in nb['cells']:
        if c['cell_type'] != 'code':
            continue
        s = ''.join(c['source'])
        s2, ok = patch_text(s)
        if ok:
            c['source'] = s2.splitlines(keepends=True); ch = True
    if ch:
        json.dump(nb, open(p, 'w', encoding='utf-8', newline='\n'), indent=1, ensure_ascii=False)
        changed.append(p)
for p in ['rebuild_all_notebooks.py', 'generate_benchmark_v2.py']:
    s = open(p, encoding='utf-8').read()
    s2, ok = patch_text(s)
    if ok:
        open(p, 'w', encoding='utf-8', newline='\n').write(s2); changed.append(p)
print('\n'.join(changed))
