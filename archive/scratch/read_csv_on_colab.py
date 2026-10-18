import os, torch

csv_path = "/content/kaggle_out/benchmark_results_ISIC-2018_official_test.csv"
if os.path.exists(csv_path):
    print("=" * 60)
    print("FINAL BENCHMARK RESULTS ON OFFICIAL 1,000 TEST IMAGES:")
    print("=" * 60)
    with open(csv_path) as f:
        print(f.read())

latest_path = "/content/kaggle_out/GDFNet_ISIC2018_latest.pth"
if os.path.exists(latest_path):
    ckpt = torch.load(latest_path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch", 0)
    best_dice = ckpt.get("best_dice", 0.0)
    history = ckpt.get("history", {})
    val_dices = history.get("val_dice", [])
    print("=" * 60)
    print(f"KAGGLE RUN SUMMARY:")
    print(f"Completed Epochs: {epoch}")
    print(f"Best Val Dice: {best_dice:.4f}")
    print(f"Total Epochs Recorded: {len(val_dices)}")
    print("=" * 60)
