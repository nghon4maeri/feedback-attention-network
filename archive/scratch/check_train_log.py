import os, subprocess

# Check if process is still running
res = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
running_lines = [l for l in res.stdout.splitlines() if "run_gdfnet_v2.py" in l]
print("Running processes:", running_lines if running_lines else "None")

# Read last 30 lines of log
if os.path.exists("/content/train.log"):
    with open("/content/train.log", "r") as f:
        lines = f.readlines()
        print("\n--- Last lines of /content/train.log ---")
        for line in lines[-15:]:
            print(line, end="")
else:
    print("/content/train.log does not exist yet.")
