# Prompt: Tải Và Tập Hợp Benchmark Notebooks 7 Datasets (15/10/2026)

## 1. Yêu cầu của User
- Tải toàn bộ các file notebook/script benchmark của 7 tập dữ liệu từ Kaggle về (bao gồm các file chạy trên Colab).
- Sắp xếp vào một thư mục riêng biệt (`benchmark_notebooks/`) để tiện theo dõi, đối chiếu và kiểm tra mã nguồn.
- Yêu cầu bổ sung:
  1. Chỉ giữ lại đúng 7 file notebook mà người dùng đã chạy thực tế để ra kết quả.
  2. Mỗi tập dataset phải có thêm thư mục `logs/` (chứa toàn bộ log và stdout) và thư mục `outputs/` (chứa model weights `.pth`, metric `.csv`, summary `.json`, và biểu đồ `.png` nếu có).

## 2. Thực thi
- Cấu trúc thư mục chuẩn hóa cho từng bộ dữ liệu:
  1. `01_CVC-ClinicDB`:
     - Notebook: `fanet-benchmark-cvc-clinicdb.ipynb`
     - `logs/`: `kaggle_stdout.log`, `train_and_eval.txt`
     - `outputs/`: Model `FANet_MFAD_CVC-ClinicDB_best.pth`, `benchmark_results_CVC-ClinicDB.csv`, `comparison_CVC-ClinicDB.png`, `training_curves_CVC-ClinicDB.png`
  2. `02_EM-Dataset`:
     - Notebook: `fanet-benchmark-em-dataset.ipynb`
     - `logs/`: `kaggle_stdout.log`, `train_and_eval.txt`
     - `outputs/`: Model `FANet_MFAD_EM-Dataset_best.pth`, `benchmark_results_EM-Dataset.csv`, `benchmark_summary_EM-Dataset.json`, các file đồ thị PNG
  3. `03_Kvasir-SEG`:
     - Notebook: `colab_run_mfad_kvasir.ipynb`
     - `logs/`: `colab_eval.txt`, `colab_eval.log`, `kaggle_stdout.log`, `train_and_eval.txt`
     - `outputs/`: Model `FANet_MFAD_Kvasir-SEG_best.pth`, `benchmark_results_Kvasir-SEG.csv`, `benchmark_summary_Kvasir-SEG.json`
  4. `04_ISIC-2018`:
     - Notebook: `fanet-benchmark-isic-2018.ipynb`
     - `logs/`: `fanet-benchmark-isic-2018.log` (5.7 MB), `kaggle_stdout.log`, `train_and_eval.txt`
     - `outputs/`: Model `FANet_MFAD_ISIC-2018_best.pth`, `benchmark_results_ISIC-2018.csv`, các file đồ thị PNG
  5. `05_CHASE-DB1`:
     - Notebook: `fanet-benchmark-chasedb1.ipynb`
     - `logs/`: `kaggle_stdout.log`, `train_and_eval.txt`
     - `outputs/`: Model `FANet_MFAD_CHASE-DB1_best.pth`, `benchmark_results_CHASE-DB1.csv`, các file đồ thị PNG
  6. `06_DRIVE`:
     - Notebook: `colab_run_mfad_drive_transfer.ipynb`
     - `logs/`: `colab_eval.txt`, `colab_eval.log`, `kaggle_stdout.log`, `train_and_eval.txt`
     - `outputs/`: Model `FANet_MFAD_DRIVE_best.pth`, `benchmark_results_drive_transfer.csv`, `benchmark_summary_drive_transfer.json`
  7. `07_DSB-2018`:
     - Notebook: `colab_run_mfad_dsb2018.ipynb` (Viết hoàn chỉnh từ đầu Epoch 1-100 với In-RAM Caching, bỏ hoàn toàn resume logic)
     - `logs/`: `train_and_eval.txt` (full 100 epochs log), `colab_eval.txt`, `colab_eval.log`, `kaggle_stdout.log`
     - `outputs/`: Model `FANet_MFAD_DSB-2018_best.pth`, `benchmark_results_dsb2018.csv`, `benchmark_summary_dsb2018.json`
- Đồng bộ hóa tài liệu hướng dẫn `benchmark_notebooks/README.md`.
