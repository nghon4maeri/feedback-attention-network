"""
kaggle_queue_runner.py
======================
Monitors active Kaggle GPU batch sessions (max 2 concurrent)
and automatically pushes pending benchmark kernels as soon as
running jobs complete.

Usage:
    python scripts/kaggle_queue_runner.py
"""

import os
import subprocess
import sys
import time

QUEUE = [
    "drive",
    "dsb2018",
    "em-dataset",
    "isic-2018",
    "kvasir-seg",
]

ACTIVE = [
    "chasedb1",
    "cvc-clinicdb",
]

ROOT = os.getcwd()

def get_kernel_status(slug):
    kernel_id = f"namnguynnnn/fanet-benchmark-{slug}"
    r = subprocess.run(["kaggle", "kernels", "status", kernel_id], capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    for state in ["QUEUED", "RUNNING", "COMPLETE", "CANCELLED", "CANCEL_ACKNOWLEDGED", "FAILED"]:
        if state in out:
            return state
    return "UNKNOWN"

def push_kernel(slug):
    d = os.path.join(ROOT, "kaggle", "benchmark_kernels", slug)
    print(f"[*] Pushing {slug} to Kaggle...")
    r = subprocess.run(["kaggle", "kernels", "push", "-p", d], capture_output=True, text=True)
    res = (r.stdout + r.stderr).strip().replace("\n", " | ")
    print(f"[+] {slug} push result: {res}")
    return "successfully pushed" in res.lower()

def main():
    print("=" * 60)
    print("  Kaggle GPU Queue Runner (Max 2 Concurrent Sessions)")
    print("=" * 60)
    print(f"Initial active: {ACTIVE}")
    print(f"Pending queue:  {QUEUE}")

    active_list = list(ACTIVE)
    queue_list = list(QUEUE)

    while queue_list or active_list:
        print("\n" + "-" * 50)
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Checking statuses...")
        
        still_running = []
        for slug in active_list:
            st = get_kernel_status(slug)
            print(f"  - {slug:15s}: {st}")
            if st in ["QUEUED", "RUNNING", "UNKNOWN"]:
                still_running.append(slug)
            else:
                print(f"    -> Slot freed by {slug} ({st})")

        active_list = still_running

        # Fill available slots up to 2
        while len(active_list) < 2 and queue_list:
            next_slug = queue_list.pop(0)
            ok = push_kernel(next_slug)
            if ok:
                active_list.append(next_slug)
            else:
                print(f"    -> Push failed for {next_slug}, re-queuing...")
                queue_list.insert(0, next_slug)
                break

        if not queue_list and not active_list:
            print("\n[OK] All benchmark kernels have finished execution!")
            break

        print(f"Active ({len(active_list)}/2): {active_list}")
        print(f"Pending: {queue_list}")
        print("Sleeping for 60 seconds...")
        time.sleep(60)

if __name__ == '__main__':
    main()
