# Benchmark V2 Notebooks & Execution Code (7/7 Biomedical Datasets)

Thư mục này tổ chức toàn bộ mã nguồn kiểm thử chuẩn hóa cho 7/7 bộ dữ liệu y sinh của mô hình FANet-MFAD. Mỗi bộ dữ liệu được lưu trữ trong một thư mục riêng biệt gồm 3 thành phần cốt lõi:
1. **Tệp Notebook (`.ipynb`)**: Chứa toàn bộ code huấn luyện/đánh giá đã chạy thực tế.
2. **Thư mục `logs/`**: Lưu nhật ký huấn luyện stdout từ Kaggle T4 hoặc Colab A100 và bảng đối chiếu metric.
3. **Thư mục `outputs/`**: Lưu trọng số mô hình đã huấn luyện tốt nhất (`.pth`), file số liệu nghiệm thu (`.csv`, `.json`), cùng các biểu đồ phân tích và so sánh (`.png`).

---

## 1. Danh sách tổng hợp theo từng bộ dữ liệu

| STT | Bộ dữ liệu | Môi trường chạy | Tệp Notebook | Thư mục Log | Thư mục Output & Model | Chỉ số chính (t=2) | Vượt baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | CVC-ClinicDB | Kaggle GPU Tesla T4 | `01_CVC-ClinicDB/fanet-benchmark-cvc-clinicdb.ipynb` | `01_CVC-ClinicDB/logs/` | `01_CVC-ClinicDB/outputs/` | Dice 0.9427 ± 0.0550 | Vượt baseline (+1.04 pp) |
| 02 | EM-Dataset | Kaggle GPU Tesla T4 | `02_EM-Dataset/fanet-benchmark-em-dataset.ipynb` | `02_EM-Dataset/logs/` | `02_EM-Dataset/outputs/` | Dice 0.9518 ± 0.0095 | Vượt baseline (+0.36 pp) |
| 03 | Kvasir-SEG | Colab A100 GPU | `03_Kvasir-SEG/colab_run_mfad_kvasir.ipynb` | `03_Kvasir-SEG/logs/` | `03_Kvasir-SEG/outputs/` | Dice 0.8880 ± 0.1431 | Vượt baseline (+0.28 pp) |
| 04 | ISIC-2018 | Kaggle GPU Tesla T4 | `04_ISIC-2018/fanet-benchmark-isic-2018.ipynb` | `04_ISIC-2018/logs/` | `04_ISIC-2018/outputs/` | Dice 0.9000 ± 0.1246 | Vượt baseline (+1.48 pp) |
| 05 | CHASE-DB1 | Kaggle GPU Tesla T4 | `05_CHASE-DB1/fanet-benchmark-chasedb1.ipynb` | `05_CHASE-DB1/logs/` | `05_CHASE-DB1/outputs/` | Spec 0.9926, Prec 0.8637 | Giảm 65% dương tính giả (FPR 0.74%) |
| 06 | DRIVE | Colab A100 GPU | `06_DRIVE/colab_run_mfad_drive_transfer.ipynb` | `06_DRIVE/logs/` | `06_DRIVE/outputs/` | Dice 0.7932 ± 0.0117 | Vượt baseline (+3.14 pp) |
| 07 | DSB-2018 | Colab A100 GPU | `07_DSB-2018/colab_run_mfad_dsb2018_resume.ipynb` | `07_DSB-2018/logs/` | `07_DSB-2018/outputs/` | Dice 0.8953 ± 0.1201 | Vượt baseline (+2.32 pp) |

---

## 2. Chi tiết cấu trúc từng bộ dữ liệu

### 01_CVC-ClinicDB (Polyp nội soi đại tràng)
- Notebook: `fanet-benchmark-cvc-clinicdb.ipynb` (200 epochs).
- Thư mục `logs/`:
  - `kaggle_stdout.log` & `train_and_eval.txt`: Log đầy đủ quá trình chạy trên Kaggle GPU Tesla T4.
- Thư mục `outputs/`:
  - Model: `FANet_MFAD_CVC-ClinicDB_best.pth` (~93.5 MB).
  - Metrics: `benchmark_results_CVC-ClinicDB.csv`.
  - Figures: `comparison_CVC-ClinicDB.png`, `training_curves_CVC-ClinicDB.png`.

### 02_EM-Dataset (Cấu trúc tế bào kính hiển vi điện tử)
- Notebook: `fanet-benchmark-em-dataset.ipynb` (100 epochs).
- Thư mục `logs/`:
  - `kaggle_stdout.log` & `train_and_eval.txt`: Log quá trình huấn luyện và bảng metric trên Kaggle T4.
- Thư mục `outputs/`:
  - Model: `FANet_MFAD_EM-Dataset_best.pth` (~93.5 MB).
  - Metrics: `benchmark_results_EM-Dataset.csv`, `benchmark_summary_EM-Dataset.json`.
  - Figures: `comparison_EM-Dataset.png`, `training_curves_EM-Dataset.png`.

### 03_Kvasir-SEG (Polyp đường tiêu hóa)
- Notebook: `colab_run_mfad_kvasir.ipynb` (và file script gốc `colab_run_mfad_kvasir.py`).
- Thư mục `logs/`:
  - `colab_eval.txt` & `colab_eval.log`: Nhật ký đánh giá pass t=2 trên GPU Colab A100 (120 ảnh test).
  - `kaggle_stdout.log` & `train_and_eval.txt`: Log chạy đối chiếu trên Kaggle GPU T4.
- Thư mục `outputs/`:
  - Model: `FANet_MFAD_Kvasir-SEG_best.pth` (~93.5 MB).
  - Metrics: `benchmark_results_Kvasir-SEG.csv`, `benchmark_summary_Kvasir-SEG.json`.

### 04_ISIC-2018 (Tổn thương da liễu ung thư hắc tố)
- Notebook: `fanet-benchmark-isic-2018.ipynb` (100 epochs).
- Thư mục `logs/`:
  - `fanet-benchmark-isic-2018.log` (5.7 MB): Log chi tiết toàn bộ từng epoch trên tập 2,594 ảnh.
  - `kaggle_stdout.log` & `train_and_eval.txt`: Tóm tắt huấn luyện và nghiệm thu từ Kaggle T4.
- Thư mục `outputs/`:
  - Model: `FANet_MFAD_ISIC-2018_best.pth` (~93.5 MB) — dùng làm backbone truyền tri thức cho tập DRIVE.
  - Metrics: `benchmark_results_ISIC-2018.csv`.
  - Figures: `comparison_ISIC-2018.png`, `training_curves_ISIC-2018.png`.

### 05_CHASE-DB1 (Vi mạch máu võng mạc đáy mắt)
- Notebook: `fanet-benchmark-chasedb1.ipynb` (100 epochs).
- Thư mục `logs/`:
  - `kaggle_stdout.log` & `train_and_eval.txt`: Log đầy đủ 100 epoch với đánh giá chi tiết tính chọn lọc mạch máu.
- Thư mục `outputs/`:
  - Model: `FANet_MFAD_CHASE-DB1_best.pth` (~93.5 MB).
  - Metrics: `benchmark_results_CHASE-DB1.csv`.
  - Figures: `comparison_CHASE-DB1.png`, `training_curves_CHASE-DB1.png`.

### 06_DRIVE (Vi mạch võng mạc - Transfer Learning)
- Notebook: `colab_run_mfad_drive_transfer.ipynb` (và file script gốc `colab_run_mfad_drive_transfer.py`).
- Thư mục `logs/`:
  - `colab_eval.txt` & `colab_eval.log`: Bảng đánh giá nghiệm thu Transfer Learning trên GPU Colab A100.
  - `kaggle_stdout.log` & `train_and_eval.txt`: Log đối chiếu từ Kaggle.
- Thư mục `outputs/`:
  - Model: `FANet_MFAD_DRIVE_best.pth` (~93.5 MB).
  - Metrics: `benchmark_results_drive_transfer.csv`, `benchmark_summary_drive_transfer.json`.

### 07_DSB-2018 (Phân đoạn nhân tế bào - In-RAM Caching)
- Notebook: `colab_run_mfad_dsb2018_resume.ipynb` (và file script gốc `colab_run_mfad_dsb2018_resume.py`).
- Thư mục `logs/`:
  - `colab_eval.txt` & `colab_eval.log`: Bảng đánh giá nghiệm thu In-RAM Caching trên GPU Colab A100 (134 ảnh test).
  - `kaggle_stdout.log` & `train_and_eval.txt`: Log chạy epoch 1 đến 39 từ Kaggle.
- Thư mục `outputs/`:
  - Model: `FANet_MFAD_DSB-2018_best.pth` (~93.5 MB).
  - Metrics: `benchmark_results_dsb2018.csv`, `benchmark_summary_dsb2018.json`.
