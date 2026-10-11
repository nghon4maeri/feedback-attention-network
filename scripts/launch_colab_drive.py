"""
launch_colab_drive.py
=====================
Launches and monitors scripts/colab_run_mfad_drive.py on the remote Colab T4 GPU instance.
Logs output live to results/colab_drive_train.log.
"""

import os
import sys
import json
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_FILE = os.path.join(ROOT, "results", "colab_drive_train.log")
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

SESSIONS_FILE = os.path.expanduser("~/.config/colab-cli/sessions.json")
if not os.path.exists(SESSIONS_FILE):
    print("[-] Colab sessions file not found!")
    sys.exit(1)

with open(SESSIONS_FILE, "r") as f:
    sessions = json.load(f)

if "fanet-colab" not in sessions:
    print("[-] Session 'fanet-colab' not found!")
    sys.exit(1)

sess = sessions["fanet-colab"]
from colab_cli.runtime import ColabRuntime

runtime = ColabRuntime(
    sess["url"],
    sess["token"],
    kernel_id=sess.get("kernel_id"),
    session_id=sess.get("session_id")
)

script_path = os.path.join(ROOT, "scripts", "colab_run_mfad_drive.py")
with open(script_path, "r", encoding="utf-8") as f:
    code = f.read()

print(f"[*] Starting DRIVE MFAD job on Colab GPU session '{sess['name']}'...")
print(f"[*] Live log output -> {LOG_FILE}")

with open(LOG_FILE, "w", encoding="utf-8") as log_f:
    log_f.write(f"=== Colab DRIVE Job Started at {time.ctime()} ===\n")
    log_f.flush()

    def output_hook(out):
        if out.get("output_type") == "stream":
            text = out.get("text", "")
            sys.stdout.write(text)
            sys.stdout.flush()
            log_f.write(text)
            log_f.flush()
        elif out.get("output_type") == "error":
            tb = "\n".join(out.get("traceback", []))
            sys.stderr.write(f"\n[ERROR] {tb}\n")
            log_f.write(f"\n[ERROR] {tb}\n")
            log_f.flush()

    try:
        runtime.execute_code(code, output_hook=output_hook)
        log_f.write(f"\n=== Colab DRIVE Job Finished at {time.ctime()} ===\n")
        print("\n[+] DRIVE training and evaluation on Colab completed successfully!")
    except Exception as e:
        log_f.write(f"\n[FATAL] Execution failed: {e}\n")
        print(f"\n[-] Execution failed: {e}")
        sys.exit(1)

sys.exit(0)
