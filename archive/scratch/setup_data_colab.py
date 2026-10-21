import os, sys, glob, json, zipfile, subprocess

DATA_ROOT = "/content/isic-2018"
os.makedirs(DATA_ROOT, exist_ok=True)

kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
    json.dump({"username": "namnguynnnn", "key": "KGAT_dad33ce94fb8b25d8a9ecd4ca26ce03f"}, f)
os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

print("[*] Downloading test groundtruth...", flush=True)
has_test_gt = any("isic2018-test-groundtruth" in f for f in os.listdir(DATA_ROOT))
if not has_test_gt:
    subprocess.run(["kaggle", "datasets", "download", "-d", "mfahimhossen/isic2018-test-groundtruth", "-p", DATA_ROOT], check=True)
print("[+] Test groundtruth downloaded.", flush=True)

print("[*] Downloading validation set from ISIC S3...", flush=True)
val_img_url = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip"
val_gt_url  = "https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip"
if not os.path.exists(f"{DATA_ROOT}/val_input.zip"):
    subprocess.run(["wget", "-q", "-O", f"{DATA_ROOT}/val_input.zip", val_img_url], check=True)
if not os.path.exists(f"{DATA_ROOT}/val_gt.zip"):
    subprocess.run(["wget", "-q", "-O", f"{DATA_ROOT}/val_gt.zip", val_gt_url], check=True)
print("[+] Validation sets downloaded.", flush=True)

print("[*] Extracting all zip files...", flush=True)
for z in sorted(glob.glob(os.path.join(DATA_ROOT, "*.zip"))):
    print(f"[*] Unzipping {os.path.basename(z)}...", flush=True)
    subprocess.run(["unzip", "-q", "-o", z, "-d", DATA_ROOT], check=True)
    os.remove(z)
    print(f"[+] Done unzipping {os.path.basename(z)}", flush=True)

print("\n[+] Dataset structure after unzip:", flush=True)
for root, dirs, files in os.walk(DATA_ROOT):
    if len(files) > 0:
        print(f"  {root}: {len(files)} files", flush=True)
