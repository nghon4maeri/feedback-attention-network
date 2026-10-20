import subprocess, os

log_file = open("/content/train_gdfnet_v3.log", "w")
p = subprocess.Popen(
    ["python3", "-u", "/content/run_gdfnet_v3.py"],
    stdout=log_file,
    stderr=subprocess.STDOUT,
    start_new_session=True
)

print(f"[+] Launched GDF-Netv3 on Colab A100 with PID {p.pid}")
print("[+] Logging to /content/train_gdfnet_v3.log")
