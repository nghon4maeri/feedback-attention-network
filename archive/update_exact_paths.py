import nbformat as nbf
import os
import re

with open('notebooks/fanet_universal_benchmark.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

new_config_source = '''# ==============================================================================
# BƯỚC 1: CẤU HÌNH DATASET (CHÍNH XÁC THEO ĐƯỜNG DẪN BẠN CUNG CẤP)
# (Hãy uncomment đúng khối Dataset mà bạn muốn chạy)
# ==============================================================================

DATASET_NAME = "Kvasir-SEG"
DATA_DIR = "/kaggle/input/datasets/namnguynnnn/kvasir-seg"
IMG_SUBDIR = "images"
MASK_SUBDIR = "masks"

# DATASET_NAME = "CVC-ClinicDB"
# DATA_DIR = "/kaggle/input/datasets/namnguynnnn/cvc-clinicdb"
# IMG_SUBDIR = "Original" # Hoặc 'images'
# MASK_SUBDIR = "Ground Truth" # Hoặc 'masks'

# DATASET_NAME = "ISIC-2018"
# DATA_DIR = "/kaggle/input/datasets/tschandl/isic2018-challenge-task1-data-segmentation"
# IMG_SUBDIR = "ISIC2018_Task1-2_Training_Input"
# MASK_SUBDIR = "ISIC2018_Task1_Training_GroundTruth"

# DATASET_NAME = "DRIVE"
# DATA_DIR = "/kaggle/input/datasets/namnguynnnn/drive-vessel"
# IMG_SUBDIR = "images"
# MASK_SUBDIR = "masks"

# DATASET_NAME = "CHASE-DB1"
# DATA_DIR = "/kaggle/input/datasets/namnguynnnn/chase-db1"
# IMG_SUBDIR = "images"
# MASK_SUBDIR = "masks"

# DATASET_NAME = "EM-Dataset"
# DATA_DIR = "/kaggle/input/datasets/kmader/electron-microscopy-3d-segmentation"
# IMG_SUBDIR = "" # File TIF 3D không cần thư mục con
# MASK_SUBDIR = ""

# ==============================================================================
# BƯỚC 2: CẤU HÌNH TRAINING
# ==============================================================================
EPOCHS = 30 # Các tập khó như ISIC có thể cần 30-40 epochs
BATCH_SIZE = 8
LR = 1e-4
IMAGE_SIZE = (256, 256)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Auto-fallback search if the exact subdirs aren't found
print(f"Checking dataset: {DATASET_NAME} at {DATA_DIR}...")
if not os.path.exists(os.path.join(DATA_DIR, IMG_SUBDIR)) and 'electron-microscopy' not in DATA_DIR.lower():
    print(f"Warning: {IMG_SUBDIR} not found. Initiating auto-search...")
    for root, dirs, files in os.walk(DATA_DIR):
        dirs_lower = [d.lower() for d in dirs]
        if any(x in dirs_lower for x in ['images', 'original', 'input', 'training']) and \
           any(x in dirs_lower for x in ['masks', 'ground truth', 'labels', 'manual']):
            IMG_SUBDIR = dirs[dirs_lower.index(next(x for x in ['images', 'original', 'input', 'training'] if x in dirs_lower))]
            MASK_SUBDIR = dirs[dirs_lower.index(next(x for x in ['masks', 'ground truth', 'labels', 'manual'] if x in dirs_lower))]
            DATA_DIR = root
            break

print(f"✅ FINAL PATHS: Images -> {os.path.join(DATA_DIR, IMG_SUBDIR)} | Masks -> {os.path.join(DATA_DIR, MASK_SUBDIR)}")
'''

for cell in nb.cells:
    if cell.cell_type == 'code' and 'BƯỚC 1: CẤU HÌNH DATASET' in cell.source:
        cell.source = new_config_source

with open('notebooks/fanet_universal_benchmark.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
