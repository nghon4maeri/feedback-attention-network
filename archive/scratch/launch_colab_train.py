import subprocess

print("[*] Launching GDF-Netv2 ISIC-2018 Training in background on Colab...")
log_file = open('/content/train.log', 'w', buffering=1)
proc = subprocess.Popen(
    ['python', '-u', '/content/run_gdfnet_v2.py'],
    stdout=log_file,
    stderr=subprocess.STDOUT
)
print(f"[+] Process started successfully with PID: {proc.pid}")
print("[+] Logging output to: /content/train.log")
