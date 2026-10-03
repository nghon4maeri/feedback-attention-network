import nbformat as nbf
import copy
import os

with open('notebooks/fanet_universal_benchmark.ipynb', 'r', encoding='utf-8') as f:
    base_nb = nbf.read(f, as_version=4)

ds = {
    "name": "Kvasir-SEG",
    "dir": "/kaggle/input/datasets/namnguynnnn/kvasir-seg",
    "img": "images",
    "mask": "masks"
}

nb = copy.deepcopy(base_nb)

config_source = f'''# ==============================================================================
# CẤU HÌNH DATASET CHUYÊN BIỆT: {ds['name']}
# ==============================================================================
DATASET_NAME = "{ds['name']}"
DATA_DIR = "{ds['dir']}"
IMG_SUBDIR = "{ds['img']}"
MASK_SUBDIR = "{ds['mask']}"

# ==============================================================================
# CẤU HÌNH TRAINING
# ==============================================================================
EPOCHS = 30
BATCH_SIZE = 8
VAL_BATCH_SIZE = 1 # Set to 1 for mathematically correct dataset-level Dice calculation
LR = 1e-4
import torch
import os
IMAGE_SIZE = (256, 256)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Checking dataset: {{DATASET_NAME}} at {{DATA_DIR}}...")
if not os.path.exists(os.path.join(DATA_DIR, IMG_SUBDIR)) and 'electron-microscopy' not in DATA_DIR.lower():
    print(f"Warning: {{IMG_SUBDIR}} not found. Initiating auto-search...")
    for root, dirs, files in os.walk(DATA_DIR):
        dirs_lower = [d.lower() for d in dirs]
        if any(x in dirs_lower for x in ['images', 'original', 'input', 'training']) and \\
           any(x in dirs_lower for x in ['masks', 'ground truth', 'labels', 'manual', '1st_manual']):
            IMG_SUBDIR = dirs[dirs_lower.index(next(x for x in ['images', 'original', 'input', 'training'] if x in dirs_lower))]
            MASK_SUBDIR = dirs[dirs_lower.index(next(x for x in ['masks', 'ground truth', 'labels', 'manual', '1st_manual'] if x in dirs_lower))]
            DATA_DIR = root
            break

print(f"✅ FINAL PATHS: Images -> {{os.path.join(DATA_DIR, IMG_SUBDIR)}} | Masks -> {{os.path.join(DATA_DIR, MASK_SUBDIR)}}")
'''

for cell in nb.cells:
    if cell.cell_type == 'code' and 'BƯỚC 1: CẤU HÌNH DATASET' in cell.source:
        cell.source = config_source
        break
        
out_name = f'notebooks/fanet_benchmark_kvasir_seg.ipynb'
with open(out_name, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print(f"Generated {out_name}")
