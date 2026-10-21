"""
Helper script to resume GDF-Netv4 training on Google Colab Pro A100.
Usage:
    python archive/scratch/resume_gdfnet_v4.py
"""
import subprocess, os, sys, time

SESSION = "gdfnet-v4-a100"
LOCAL_RESULTS = "GDF-Netv4/results/isic2018"
REMOTE_RESULTS = "/content/results_gdfnet_v4"

def run(cmd):
    print(f"[*] Running: {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if res.stdout:
        print(res.stdout.strip())
    if res.stderr and res.returncode != 0:
        print(f"[!] Error: {res.stderr.strip()}", file=sys.stderr)
    return res.returncode == 0

print("=== GDF-NETV4 RESUME LAUNCHER ===", flush=True)

# 1. Create session
print("\n[Step 1] Creating Colab A100 Session...", flush=True)
run(f"colab new -s {SESSION} --gpu A100 --high-mem")

# 2. Install dependencies
print("\n[Step 2] Installing Dependencies...", flush=True)
run(f"colab install -s {SESSION} timm albumentations opencv-python-headless scipy pandas kaggle")

# 3. Setup Dataset
print("\n[Step 3] Setting up ISIC-2018 Dataset...", flush=True)
run(f"colab upload -s {SESSION} archive/scratch/setup_data_colab.py /content/setup_data_colab.py")
run(f"colab exec -s {SESSION} -f archive/scratch/setup_data_colab.py --timeout 300")

# 4. Upload Checkpoint and History to resume
print("\n[Step 4] Uploading Checkpoints for Seamless Resumption...", flush=True)
run(f'colab exec -s {SESSION} "import os; os.makedirs(\'{REMOTE_RESULTS}\', exist_ok=True)"')
run(f"colab upload -s {SESSION} {LOCAL_RESULTS}/GDFNetV4_ISIC2018_latest.pth {REMOTE_RESULTS}/GDFNetV4_ISIC2018_latest.pth")
if os.path.exists(f"{LOCAL_RESULTS}/GDFNetV4_ISIC2018_best.pth"):
    run(f"colab upload -s {SESSION} {LOCAL_RESULTS}/GDFNetV4_ISIC2018_best.pth {REMOTE_RESULTS}/GDFNetV4_ISIC2018_best.pth")
if os.path.exists(f"{LOCAL_RESULTS}/training_history_ISIC2018.csv"):
    run(f"colab upload -s {SESSION} {LOCAL_RESULTS}/training_history_ISIC2018.csv {REMOTE_RESULTS}/training_history_ISIC2018.csv")

# 5. Upload Training Script & Launch
print("\n[Step 5] Uploading and Launching GDF-Netv4 Training Daemon...", flush=True)
run(f"colab upload -s {SESSION} GDF-Netv4/training/colab_run_gdfnet_v4_isic2018.py /content/colab_run_gdfnet_v4_isic2018.py")
run(f"colab upload -s {SESSION} archive/scratch/launch_training.py /content/launch_training.py")
run(f"colab exec -s {SESSION} -f archive/scratch/launch_training.py --timeout 60")

print("\n[+] Resumption successfully initiated! Monitoring will be available via 'archive/scratch/train_v4.log'.")
