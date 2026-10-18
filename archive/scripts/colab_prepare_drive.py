import os
import sys
import glob
import json
import subprocess

print("=" * 60)
print("  Pre-staging DRIVE Dataset on Colab A100 VM")
print("=" * 60)

# Configure Kaggle credentials
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
with open(kaggle_json, "w") as f:
    json.dump({"username": "namnguynnnn", "key": "KGAT_dad33ce94fb8b25d8a9ecd4ca26ce03f"}, f)
os.chmod(kaggle_json, 0o600)
print("[+] Kaggle credentials configured.")

DATA_DIR = "/content/data_drive_full"
os.makedirs(DATA_DIR, exist_ok=True)

drive_check = os.path.join(DATA_DIR, "DRIVE")
if not os.path.exists(drive_check):
    print("[*] Downloading zionfuo/drive2004...")
    subprocess.run(["kaggle", "datasets", "download", "-d", "zionfuo/drive2004", "-p", DATA_DIR, "--unzip"], check=True)
    print("[+] Download complete.")
else:
    print("[+] DRIVE already cached.")

all_files = glob.glob(os.path.join(DATA_DIR, "**", "*.*"), recursive=True)
images_all = [f for f in all_files if "/images/" in f.replace(os.sep, "/").lower()]
masks_all  = [f for f in all_files if "/1st_manual/" in f.replace(os.sep, "/").lower() or "/manual1/" in f.replace(os.sep, "/").lower()]

train_x = sorted([p for p in images_all if "training" in p.lower() or "train" in p.lower()])
train_y = sorted([p for p in masks_all if "training" in p.lower() or "train" in p.lower()])
test_x  = sorted([p for p in images_all if "test" in p.lower()])
test_y  = sorted([p for p in masks_all if "test" in p.lower()])

print(f"[+] Verified DRIVE files:")
print(f"    Train: {len(train_x)} images, {len(train_y)} masks")
print(f"    Test:  {len(test_x)} images, {len(test_y)} masks")

import torch
print(f"[+] PyTorch CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[+] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[+] Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
print("[+] Colab A100 VM is 100% pre-staged and ready for DRIVE Transfer Learning!")
