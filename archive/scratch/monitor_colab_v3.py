import os

log_path = "/content/train_gdfnet_v3.log"
if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()
        print(f"=== Log Tail ({len(lines)} lines) ===")
        for line in lines[-25:]:
            print(line, end="")
else:
    print("Log file not found yet.")
