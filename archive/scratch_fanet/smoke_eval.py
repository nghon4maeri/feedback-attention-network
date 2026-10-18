import json, os, sys, tempfile
import numpy as np, torch, torch.nn as nn, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

nb = json.load(open('notebooks/fanet_benchmark_kvasir_seg.ipynb', encoding='utf-8'))
cells = [''.join(c['source']) for c in nb['cells'] if c.get('metadata', {}).get('refactored_eval')]
assert len(cells) == 2

class FANet_MFAD(nn.Module):
    def __init__(self):
        super().__init__(); self.c = nn.Conv2d(3, 1, 3, padding=1)
    def forward(self, x, prev_mask=None):
        o = self.c(x)
        return o if prev_mask is None else o + prev_mask

tmp = tempfile.mkdtemp()
ck = os.path.join(tmp, 'ck.pth'); torch.save(FANet_MFAD().state_dict(), ck)
xs = torch.rand(12, 3, 64, 64); ys = (torch.rand(12, 1, 64, 64) > 0.7).float()
valid_loader = torch.utils.data.DataLoader(list(zip(xs, ys)), batch_size=1)
DATASET_NAME = 'smoke'
g = dict(globals()); g.update(ck=ck)
os.chdir(tmp)
for c in cells:
    exec(c.replace("EVAL_CKPT = ckpt", f"EVAL_CKPT = {ck!r}"), g)
print(open('benchmark_results.csv').read()); print(os.listdir('.'))
