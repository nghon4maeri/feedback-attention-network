import os, json, subprocess

kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)
with open(os.path.join(kaggle_dir, "kaggle.json"), "w") as f:
    json.dump({"username": "namnguynnnn", "key": "KGAT_dad33ce94fb8b25d8a9ecd4ca26ce03f"}, f)
os.chmod(os.path.join(kaggle_dir, "kaggle.json"), 0o600)

print("Kaggle credentials written.")
res = subprocess.run(["kaggle", "datasets", "list", "-s", "nuclei"], capture_output=True, text=True)
print("Kaggle search nuclei:", res.stdout[:300])
