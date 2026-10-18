import nbformat as nbf
import os
import re

with open('notebooks/fanet_universal_benchmark.ipynb', 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code' and 'class UniversalDataset(Dataset):' in cell.source:
        new_source = '''import tifffile as tiff

class UniversalDataset(Dataset):
    def __init__(self, images_path, masks_path):
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        
        # Check if it's a 3D volume (like EM Dataset)
        self.is_3d_volume = False
        if self.n_samples == 1 and str(images_path[0]).endswith('.tif'):
            img_vol = tiff.imread(images_path[0])
            if len(img_vol.shape) == 3: # (Z, H, W)
                self.is_3d_volume = True
                self.img_vol = img_vol
                self.mask_vol = tiff.imread(masks_path[0])
                self.n_samples = self.img_vol.shape[0]

    def __getitem__(self, index):
        if self.is_3d_volume:
            image = self.img_vol[index]
            mask = self.mask_vol[index]
            # Convert grayscale EM to 3 channel if needed, or keep 1
            if len(image.shape) == 2:
                image = np.stack([image]*3, axis=-1)
        else:
            image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
            mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, IMAGE_SIZE)
        image = np.transpose(image, (2, 0, 1)) / 255.0
        image = image.astype(np.float32)

        mask = cv2.resize(mask, IMAGE_SIZE, interpolation=cv2.INTER_NEAREST)
        mask = np.expand_dims(mask, axis=0)
        mask = (mask > 127).astype(np.float32)

        return image, mask

    def __len__(self):
        return self.n_samples

# Dynamically find extensions
exts = ['*.jpg', '*.png', '*.tif', '*.tiff', '*.jpeg']
images, masks = [], []

# Special case for EM dataset which has training.tif at root
if "electron-microscopy-3d-segmentation" in DATA_DIR.lower():
    images = [os.path.join(DATA_DIR, "training.tif")]
    masks = [os.path.join(DATA_DIR, "training_groundtruth.tif")]
else:
    for ext in exts:
        images.extend(glob(os.path.join(DATA_DIR, IMG_SUBDIR, '**', ext), recursive=True))
        masks.extend(glob(os.path.join(DATA_DIR, MASK_SUBDIR, '**', ext), recursive=True))

images = sorted(images)
masks = sorted(masks)

if len(images) == 0:
    raise ValueError("❌ DATASET EMPTY! Please check Kaggle Data tab and paths.")

if len(images) != len(masks):
    print(f"⚠️ Mismatch: {len(images)} images vs {len(masks)} masks. Truncating to minimum.")
    min_len = min(len(images), len(masks))
    images, masks = images[:min_len], masks[:min_len]

# If it's a 3D volume (1 file), we don't split the file list, we pass the 1 file to both train and valid
# and let the Dataset class handle the indexing. 
# Wait, for a proper train/valid split of a 3D volume, we should split the indices inside the dataset.
# For simplicity in this universal script, we will just use the same file and let it overfit, OR
# better: EM Dataset provides testing.tif for validation.
if "electron-microscopy-3d-segmentation" in DATA_DIR.lower():
    train_x, train_y = images, masks
    valid_x = [os.path.join(DATA_DIR, "testing.tif")]
    valid_y = [os.path.join(DATA_DIR, "testing_groundtruth.tif")]
else:
    train_x, valid_x, train_y, valid_y = train_test_split(images, masks, test_size=0.2, random_state=42)

train_dataset = UniversalDataset(train_x, train_y)
valid_dataset = UniversalDataset(valid_x, valid_y)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f'✅ {DATASET_NAME} loaded! Train size: {len(train_dataset)} | Valid size: {len(valid_dataset)}')
'''
        cell.source = new_source
        
    if cell.cell_type == 'code' and 'Warning: {IMG_SUBDIR} not found' in cell.source:
        # Patch the config cell so it doesn't fail on EM dataset
        new_source = cell.source.replace("if not os.path.exists(os.path.join(DATA_DIR, IMG_SUBDIR)):", "if not os.path.exists(os.path.join(DATA_DIR, IMG_SUBDIR)) and 'electron-microscopy' not in DATA_DIR.lower():")
        cell.source = new_source

with open('notebooks/fanet_universal_benchmark.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
