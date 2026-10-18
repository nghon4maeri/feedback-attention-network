import os, sys, json, subprocess

kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
with open(kaggle_json, "w") as f:
    json.dump({"username": "namnguynnnn", "key": "KGAT_dad33ce94fb8b25d8a9ecd4ca26ce03f"}, f)
os.chmod(kaggle_json, 0o600)

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle"], check=True)

out_dir = "/content/kaggle_out"
os.makedirs(out_dir, exist_ok=True)

cmd = ["kaggle", "kernels", "output", "namnguynnnn/fanet-benchmark-isic-2018", "-p", out_dir]
print("Running:", " ".join(cmd))
res = subprocess.run(cmd, capture_output=True, text=True)
print("Stdout:", res.stdout)
print("Stderr:", res.stderr)
print("Files in", out_dir, ":", os.listdir(out_dir))
for f in os.listdir(out_dir):
    fp = os.path.join(out_dir, f)
    print(f"  - {f}: {os.path.getsize(fp) / (1024*1024):.2f} MB")
