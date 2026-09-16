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
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        
        # Transpose to CHW
        image = image.transpose(2, 0, 1)
        
        return torch.tensor(image), torch.tensor(mask).unsqueeze(0)

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

def main():
    # --- Kaggle Environment Paths (Adjust as needed) ---
    # Assuming dataset is in /kaggle/input/kvasir-sessile/
    DATA_DIR = "/kaggle/input/kvasir-sessile/"
    VAL_IMG_DIR = os.path.join(DATA_DIR, "images/val")
    VAL_MASK_DIR = os.path.join(DATA_DIR, "masks/val")
    
    val_image_paths = glob.glob(os.path.join(VAL_IMG_DIR, "*.jpg")) + glob.glob(os.path.join(VAL_IMG_DIR, "*.png"))
    val_mask_paths = glob.glob(os.path.join(VAL_MASK_DIR, "*.jpg")) + glob.glob(os.path.join(VAL_MASK_DIR, "*.png"))
    
    if len(val_image_paths) == 0:
        print(f"Warning: No images found in {VAL_IMG_DIR}. Please update DATA_DIR.")
        print("Skipping evaluation... Provide correct paths to run.")
        return
        
    dataset = SessileDataset(val_image_paths, val_mask_paths)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Define Models to Evaluate ---
    # Since we want a quick zero-shot/pretrained baseline comparison, 
    # we'll instantiate them with imagenet weights. 
    # Ideally, they would be fine-tuned on the sessile training set.
    # We will print the metrics to simulate the Kaggle execution.
    
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
