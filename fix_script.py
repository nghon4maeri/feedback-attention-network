with open('generate_notebook.py', 'r', encoding='utf-8') as f:
    text = f.read()

new_config = '''# Configuration
EPOCHS = 20
BATCH_SIZE = 8
LR = 1e-4
IMAGE_SIZE = (256, 256)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- DATASET AUTO-DETECTION ---
# Tự động quét tìm thư mục chứa ảnh và mask (bất chấp tên dataset bạn Add vào Kaggle)
import os
DATA_DIR = None
IMG_SUBDIR = 'images'
MASK_SUBDIR = 'masks'

if os.path.exists('/kaggle/input'):
    for root, dirs, files in os.walk('/kaggle/input'):
        dirs_lower = [d.lower() for d in dirs]
        if 'images' in dirs_lower and 'masks' in dirs_lower:
            IMG_SUBDIR = dirs[dirs_lower.index('images')]
            MASK_SUBDIR = dirs[dirs_lower.index('masks')]
            DATA_DIR = root
            break
else:
    DATA_DIR = '../data/kvasir-seg'

if DATA_DIR is None:
    print("WARNING: KHÔNG TÌM THẤY DATASET! BẠN ĐÃ ADD DATA VÀO KAGGLE CHƯA?")
else:
    print(f"✅ Đã tìm thấy Dataset tại: {DATA_DIR}")
'''

import re
text = re.sub(r'# --- DATASET SELECTION ---.*?MASK_SUBDIR = \'ISIC2018_Task1_Training_GroundTruth\'', new_config, text, flags=re.DOTALL)

with open('generate_notebook.py', 'w', encoding='utf-8') as f:
    f.write(text)
