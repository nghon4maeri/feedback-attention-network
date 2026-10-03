import nbformat as nbf
import os
import copy

datasets = [
    {
        "name": "CVC-ClinicDB",
        "dir": "/kaggle/input/datasets/namnguynnnn/cvc-clinicdb",
        "img": "Original",
        "mask": "Ground Truth"
    },
    {
        "name": "ISIC-2018",
        "dir": "/kaggle/input/datasets/tschandl/isic2018-challenge-task1-data-segmentation",
        "img": "ISIC2018_Task1-2_Training_Input",
        "mask": "ISIC2018_Task1_Training_GroundTruth"
    },
    {
        "name": "DRIVE",
        "dir": "/kaggle/input/datasets/namnguynnnn/drive-vessel",
        "img": "images",
        "mask": "masks"
    },
    {
        "name": "CHASE-DB1",
        "dir": "/kaggle/input/datasets/namnguynnnn/chase-db1",
        "img": "images",
        "mask": "masks"
    },
    {
        "name": "EM-Dataset",
        "dir": "/kaggle/input/datasets/kmader/electron-microscopy-3d-segmentation",
        "img": "",
        "mask": ""
    }
]

with open('notebooks/fanet_universal_benchmark.ipynb', 'r', encoding='utf-8') as f:
    base_nb = nbf.read(f, as_version=4)

os.makedirs('notebooks', exist_ok=True)

for ds in datasets:
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
LR = 1e-4
import torch
IMAGE_SIZE = (256, 256)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import os
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
            
    out_name = f'notebooks/fanet_benchmark_{ds["name"].lower().replace("-", "_")}.ipynb'
    with open(out_name, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
