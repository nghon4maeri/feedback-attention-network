"""
scripts/ingest_kaggle.py
Automated ingestion pipeline for Kaggle kernel outputs.

Downloads outputs from a specified Kaggle notebook, extracts archives if any,
and organizes artifacts into:
  - checkpoints_phase7b/       (model weights *.pth)
  - diagnostics_output/phase7b/ (diagnostic metrics, figures, summary json)
  - logs_phase7b/              (training curves *.csv, execution *.log)

Usage:
    python scripts/ingest_kaggle.py [--kernel USERNAME/KERNEL_NAME]
Default:
    python scripts/ingest_kaggle.py --kernel namnguynnnn/fanet-phase7b-training
"""

import os
import sys
import shutil
import zipfile
import argparse
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Ingest Kaggle kernel outputs for FANet Phase 7B.")
    parser.add_argument("--kernel", type=str, default="namnguynnnn/fanet-phase7b-training",
                        help="Kaggle kernel slug (e.g. username/kernel-name)")
    parser.add_argument("--staging-dir", type=str, default="kaggle/downloads_phase7b",
                        help="Temporary directory for downloaded artifacts")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_phase7b",
                        help="Target directory for checkpoints")
    parser.add_argument("--diag-dir", type=str, default="diagnostics_output/phase7b",
                        help="Target directory for diagnostics output")
    parser.add_argument("--log-dir", type=str, default="logs_phase7b",
                        help="Target directory for training logs")
    parser.add_argument("--all-files", action="store_true",
                        help="Download all files including raw dataset copies")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    staging = repo_root / args.staging_dir
    ckpt_dir = repo_root / args.ckpt_dir
    diag_dir = repo_root / args.diag_dir
    log_dir = repo_root / args.log_dir

    for d in [staging, ckpt_dir, diag_dir, log_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("FANET KAGGLE INGESTION PIPELINE")
    print("=" * 65)
    print(f"Kernel target : {args.kernel}")
    print(f"Staging dir   : {staging}")
    print(f"Checkpoints   : {ckpt_dir}")
    print(f"Diagnostics   : {diag_dir}")
    print(f"Logs          : {log_dir}")
    print("=" * 65)

    # 1. Download via Kaggle CLI
    # Filter out dataset images by default to download in seconds
    pattern_arg = "" if args.all_files else ' --file-pattern ".*\\.(pth|csv|json|log|txt)$"'
    cmd = f'kaggle kernels output {args.kernel} -p "{staging}"{pattern_arg}'
    print(f"\n[1/3] Downloading output from {args.kernel} ...")
    print(f"[CMD] {cmd}\n")

    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        print("[FAIL] Failed to download kernel output. Retrying without file-pattern...", file=sys.stderr)
        retry_cmd = f'kaggle kernels output {args.kernel} -p "{staging}"'
        proc2 = subprocess.run(retry_cmd, shell=True)
        if proc2.returncode != 0:
            print("[FAIL] Failed on retry.", file=sys.stderr)
            sys.exit(1)

    print(f"\n[OK] Download completed into {staging}")

    # 2. Extract any zip archives found in staging
    print("\n[2/3] Scanning and extracting archives...")
    for item in staging.glob("*.zip"):
        print(f"  Unzipping archive: {item.name}")
        with zipfile.ZipFile(item, 'r') as zip_ref:
            zip_ref.extractall(staging)

    # 3. Categorize and move files
    print("\n[3/3] Organizing files into workspace directories...")
    moved_summary = {"checkpoints": [], "diagnostics": [], "logs": [], "other": []}

    for file_path in staging.rglob("*"):
        if file_path.is_dir() or file_path.suffix == ".zip":
            continue

        # Skip files from raw Kvasir-SEG copies if any were downloaded
        if "Kvasir-SEG" in str(file_path):
            continue

        fname = file_path.name
        rel = file_path.relative_to(staging)

        # Checkpoints (*.pth)
        if fname.endswith(".pth"):
            dest_file = ckpt_dir / fname
            shutil.copy2(file_path, dest_file)
            moved_summary["checkpoints"].append(str(dest_file.relative_to(repo_root)))

        # Diagnostics (diagnostics*.json, *.png, *.jpg, fig*)
        elif "diag" in fname.lower() or fname.endswith((".png", ".jpg", ".jpeg", ".svg")):
            dest_file = diag_dir / fname
            shutil.copy2(file_path, dest_file)
            moved_summary["diagnostics"].append(str(dest_file.relative_to(repo_root)))

        # Logs & metrics (*.csv, *.log, *summary*.json, eval*.json, stats*.json)
        elif fname.endswith((".csv", ".log", ".txt")) or ("summary" in fname.lower() and fname.endswith(".json")):
            dest_file = log_dir / fname
            shutil.copy2(file_path, dest_file)
            moved_summary["logs"].append(str(dest_file.relative_to(repo_root)))
            # Also mirror summary/metrics json to diagnostics for easy reading
            if fname.endswith(".json"):
                shutil.copy2(file_path, diag_dir / fname)

        else:
            moved_summary["other"].append(str(rel))

    print("\n" + "=" * 65)
    print("INGESTION SUMMARY")
    print("=" * 65)
    print(f"Checkpoints copied ({len(moved_summary['checkpoints'])}):")
    for f in moved_summary["checkpoints"]:
        print(f"  [OK] {f}")

    print(f"\nDiagnostics copied ({len(moved_summary['diagnostics'])}):")
    for f in moved_summary["diagnostics"]:
        print(f"  [OK] {f}")

    print(f"\nLogs copied ({len(moved_summary['logs'])}):")
    for f in moved_summary["logs"]:
        print(f"  [OK] {f}")

    if moved_summary["other"]:
        print(f"\nOther files remaining in staging ({len(moved_summary['other'])}):")
        for f in moved_summary["other"]:
            print(f"  ℹ {f}")

    print("=" * 65)
    print("[SUCCESS] All files ingested successfully!")


if __name__ == "__main__":
    main()
