import os, subprocess, sys

print("[+] Launching GDF-Netv4 training engine in foreground keep-alive mode...", flush=True)
sys.stdout.flush()

# Run directly in foreground: keeps Jupyter kernel BUSY so Colab never disconnects for idle!
process = subprocess.Popen(
    ["python", "-u", "/content/colab_run_gdfnet_v4_isic2018.py"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

with open("/content/train_v4.log", "w", encoding="utf-8") as log_file:
    for line in iter(process.stdout.readline, ''):
        print(line, end='', flush=True)
        log_file.write(line)
        log_file.flush()

process.stdout.close()
return_code = process.wait()
sys.exit(return_code)
