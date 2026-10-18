import os, sys, time, torch

try:
    if not os.path.exists('/content/drive/MyDrive'):
        from google.colab import drive
        drive.mount('/content/drive', force_remount=False)
except Exception as e:
    print(f"Drive mount note: {e}")

drive_dir = "/content/drive/MyDrive/FANet_runs_MFAD/isic-2018_official_a100"
print(f"Drive Dir: {drive_dir}")
print(f"Exists: {os.path.exists(drive_dir)}")

if os.path.exists(drive_dir):
    files = os.listdir(drive_dir)
    print(f"Files on Drive ({len(files)}): {files}")
    for fname in files:
        fpath = os.path.join(drive_dir, fname)
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        mtime = time.ctime(os.path.getmtime(fpath))
        print(f"  - {fname:<45s} | {size_mb:8.2f} MB | {mtime}")

    latest_path = os.path.join(drive_dir, "GDFNet_ISIC2018_latest.pth")
    if os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", 0)
        best_dice = ckpt.get("best_dice", 0.0)
        history = ckpt.get("history", {})
        val_dices = history.get("val_dice", [])
        train_losses = history.get("train_loss", [])
        print("\n" + "=" * 50)
        print(f"SAVED CHECKPOINT DETAILS ON GOOGLE DRIVE:")
        print(f"Last Saved Epoch: {epoch}")
        print(f"Best Val Dice:    {best_dice:.4f}")
        print(f"Total Epochs:     {len(val_dices)}")
        if val_dices:
            print(f"Latest Train Loss: {train_losses[-1]:.4f}")
            print(f"Latest Val Dice:   {val_dices[-1]:.4f}")
        print("=" * 50)

    csv_path = os.path.join(drive_dir, "benchmark_results_ISIC-2018_official_test_a100.csv")
    if os.path.exists(csv_path):
        print(f"\n[+] FINAL BENCHMARK CSV FOUND:")
        with open(csv_path) as f:
            print(f.read())
else:
    print("Drive directory not found or Drive not mounted.")
