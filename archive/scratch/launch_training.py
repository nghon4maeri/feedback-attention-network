import os, subprocess, time

log_file = "/content/train_v4.log"
cmd = "nohup python -u /content/colab_run_gdfnet_v4_isic2018.py > /content/train_v4.log 2>&1 &"
print(f"[*] Launching command: {cmd}", flush=True)
os.system(cmd)
time.sleep(3)

print("[+] Verifying process launch...", flush=True)
res = subprocess.run(["ps", "aux"], capture_output=True, text=True)
for line in res.stdout.splitlines():
    if "colab_run_gdfnet_v4_isic2018.py" in line:
        print(f"[+] Found active training process: {line}", flush=True)

if os.path.exists(log_file):
    with open(log_file) as f:
        print("\n--- Initial Log Output ---")
        print(f.read()[-1000:])
