import os, subprocess

res = subprocess.run(["ps", "-ef"], capture_output=True, text=True)
for l in res.stdout.splitlines():
    if "python" in l or "unzip" in l or "kaggle" in l or "wget" in l:
        print(l)

if os.path.exists("/content/isic-2018"):
    print("Files in /content/isic-2018:", os.listdir("/content/isic-2018")[:10])

if os.path.exists("/content/train_a100.log"):
    with open("/content/train_a100.log", "r") as f:
        lines = f.readlines()
        print("\n--- Last lines of /content/train_a100.log ---")
        for line in lines[-20:]:
            print(line, end="")
