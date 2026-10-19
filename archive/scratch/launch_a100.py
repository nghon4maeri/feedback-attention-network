import subprocess, os

log_file = open("/content/train_a100.log", "w")
p = subprocess.Popen(
    ["python3", "-u", "/content/colab_run_a100_unified.py"],
    stdout=log_file,
    stderr=subprocess.STDOUT,
    start_new_session=True
)

print(f"[+] Launched colab_run_a100_unified.py with PID {p.pid}")
print("[+] Logging to /content/train_a100.log")
