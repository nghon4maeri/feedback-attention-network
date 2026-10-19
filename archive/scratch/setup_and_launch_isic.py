import subprocess, sys

print("[*] Installing required packages on Colab...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm", "albumentations", "opencv-python-headless"], check=True)
print("[+] Dependencies installed successfully.")

print("[*] Launching GDF-Netv2 ISIC-2018 Training in background...")
log_file = open('/content/train.log', 'w', buffering=1)
proc = subprocess.Popen(
    ['python3', '-u', '/content/run_gdfnet_v2.py'],
    stdout=log_file,
    stderr=subprocess.STDOUT
)
print(f"[+] Training process started in background with PID: {proc.pid}")
print("[+] Logging output to: /content/train.log")
