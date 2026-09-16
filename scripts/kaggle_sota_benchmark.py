"""
KAGGLE EXECUTION INSTRUCTIONS
-----------------------------
1. Upload this script to a Kaggle Notebook with a T4 GPU enabled.
2. Ensure you have added the Kvasir-SEG Sessile dataset to the notebook.
3. Install the required dependency by running the following cell:
   !pip install segmentation-models-pytorch
4. Run this script to compute SOTA baseline metrics (Dice, FPR) for Table 3.
"""
import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from tqdm import tqdm

class SessileDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = sorted(image_paths)
        self.mask_paths = sorted(mask_paths)
        self.transform = transform
        
        # Verify alignment
        for img_p, msk_p in zip(self.image_paths, self.mask_paths):
            assert os.path.basename(img_p) == os.path.basename(msk_p), f"Mismatch: {img_p} vs {msk_p}"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]
        
        # Load image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Resize to 352x352 as standard for polyp segmentation
        image = image.resize((352, 352), Image.BILINEAR)
        mask = mask.resize((352, 352), Image.NEAREST)
        
        # Convert to numpy
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32)
        
        # Normalize image (ImageNet stats)
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        
        # Transpose to CHW
        image = image.transpose(2, 0, 1)
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32).unsqueeze(0)

def compute_metrics(pred, target):
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    fn = ((1 - pred) * target).sum()
    tn = ((1 - pred) * (1 - target)).sum()
    
    dice = (2. * tp + 1e-6) / (2. * tp + fp + fn + 1e-6)
    fpr = fp / (fp + tn + 1e-6)
    
    return dice.item(), fpr.item()

def evaluate_model(model, dataloader, device):
    model.eval()
    dices = []
    fprs = []
    
    with torch.no_grad():
        for images, masks in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            masks = masks.to(device)
            
            # Predict
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            for i in range(preds.size(0)):
                dice, fpr = compute_metrics(preds[i], masks[i])
                dices.append(dice)
                fprs.append(fpr)
                
    return np.mean(dices), np.mean(fprs)

def find_kvasir_val_dirs(base_dir="/kaggle/input/"):
    """Recursively search for the validation images and masks directories."""
    print(f"Scanning {base_dir} for dataset directories...")
    if not os.path.exists(base_dir):
        return None, None
        
    for root, dirs, files in os.walk(base_dir):
        # Many datasets might have 'images/val' or 'val/images'
        # Check if this folder looks like a validation image folder
        lower_root = root.lower()
        if "images" in lower_root and "val" in lower_root:
            # Try to guess the masks folder path
            masks_dir = root.replace("images", "masks")
            if not os.path.exists(masks_dir):
                masks_dir = root.replace("Images", "masks").replace("images", "Masks")
                
            if os.path.exists(masks_dir):
                return root, masks_dir
                
    # Fallback: just look for 'images' and 'masks' if 'val' doesn't exist in path
    for root, dirs, files in os.walk(base_dir):
        lower_root = root.lower()
        if root.endswith("images"):
            masks_dir = root[:-6] + "masks"
            if os.path.exists(masks_dir):
                return root, masks_dir
                
    return None, None

def main():
    # --- Auto-detect Kaggle Environment Paths ---
    VAL_IMG_DIR, VAL_MASK_DIR = find_kvasir_val_dirs("/kaggle/input/")
    
    if VAL_IMG_DIR is None or VAL_MASK_DIR is None:
        print("Error: Could not automatically locate 'images' and 'masks' directories in /kaggle/input/")
        print("Please manually set VAL_IMG_DIR and VAL_MASK_DIR in the script.")
        return
        
    print(f"Found Images Directory: {VAL_IMG_DIR}")
    print(f"Found Masks Directory: {VAL_MASK_DIR}")
    
    val_image_paths = glob.glob(os.path.join(VAL_IMG_DIR, "*.jpg")) + glob.glob(os.path.join(VAL_IMG_DIR, "*.png"))
    val_mask_paths = glob.glob(os.path.join(VAL_MASK_DIR, "*.jpg")) + glob.glob(os.path.join(VAL_MASK_DIR, "*.png"))
    
    if len(val_image_paths) == 0:
        print(f"Warning: No images found in {VAL_IMG_DIR}.")
        print("Skipping evaluation... Provide correct paths to run.")
        return
        
    print(f"Found {len(val_image_paths)} validation images.")
    
    dataset = SessileDataset(val_image_paths, val_mask_paths)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Define Models to Evaluate ---
    models_to_test = {
        "U-Net (ResNet50)": smp.Unet(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1),
        "DeepLabV3+ (ResNet50)": smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1),
        "FPN (ResNet50)": smp.FPN(encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=1)
    }
    
    results = {}
    for name, model in models_to_test.items():
        print(f"\n--- Evaluating {name} ---")
        model = model.to(device)
        dice, fpr = evaluate_model(model, dataloader, device)
        results[name] = {"Dice": dice, "FPR": fpr}
        print(f"{name} - Mean Dice: {dice:.4f}, Mean FPR: {fpr:.4f}")
        
    print("\n--- Summary ---")
    for name, metrics in results.items():
        print(f"{name}: Dice={metrics['Dice']:.4f}, FPR={metrics['FPR']:.4f}")

if __name__ == "__main__":
    main()
