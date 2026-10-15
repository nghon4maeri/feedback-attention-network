# Benchmark V2 Notebooks & Execution Code (7/7 Biomedical Datasets)

Thư mục này chỉ lưu trữ các tệp notebook (.ipynb) và script thực thi (.py) đã trực tiếp chạy thực tế và có đầy đủ cell output kết quả kiểm thử cho 7/7 bộ dữ liệu y sinh của mô hình FANet-MFAD.

Mỗi bộ dữ liệu được tổ chức theo một thư mục riêng biệt kèm đúng tệp đã chạy và sinh ra số liệu báo cáo:

---

## 1. Danh sách tệp thực thi theo từng bộ dữ liệu

| STT | Bộ dữ liệu | Môi trường chạy | Tệp thực thi có kết quả | Chỉ số chính (t=2) | Vượt baseline |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 01 | CVC-ClinicDB | Kaggle GPU Tesla T4 | `01_CVC-ClinicDB/fanet-benchmark-cvc-clinicdb.ipynb` | Dice 0.9427 ± 0.0550 | Vượt baseline (+1.04 pp) |
| 02 | EM-Dataset | Kaggle GPU Tesla T4 | `02_EM-Dataset/fanet-benchmark-em-dataset.ipynb` | Dice 0.9518 ± 0.0095 | Vượt baseline (+0.36 pp) |
| 03 | Kvasir-SEG | Colab A100 GPU | `03_Kvasir-SEG/colab_run_mfad_kvasir.ipynb`<br>`03_Kvasir-SEG/colab_run_mfad_kvasir.py` | Dice 0.8880 ± 0.1431 | Vượt baseline (+0.28 pp) |
| 04 | ISIC-2018 | Kaggle GPU Tesla T4 | `04_ISIC-2018/fanet-benchmark-isic-2018.ipynb` | Dice 0.9000 ± 0.1246 | Vượt baseline (+1.48 pp) |
| 05 | CHASE-DB1 | Kaggle GPU Tesla T4 | `05_CHASE-DB1/fanet-benchmark-chasedb1.ipynb` | Spec 0.9926, Prec 0.8637 | Vượt baseline (Giảm 65% FPR) |
| 06 | DRIVE | Colab A100 GPU | `06_DRIVE/colab_run_mfad_drive_transfer.ipynb`<br>`06_DRIVE/colab_run_mfad_drive_transfer.py` | Dice 0.7932 ± 0.0117 | Vượt baseline (+3.14 pp) |
| 07 | DSB-2018 | Colab A100 GPU | `07_DSB-2018/colab_run_mfad_dsb2018_resume.ipynb`<br>`07_DSB-2018/colab_run_mfad_dsb2018_resume.py` | Dice 0.8953 ± 0.1201 | Vượt baseline (+2.32 pp) |

---

## 2. Chi tiết từng bộ dữ liệu

### 01_CVC-ClinicDB (Polyp nội soi đại tràng)
- Môi trường chạy: Kaggle GPU Tesla T4 (200 epochs).
- Tệp thực thi: `fanet-benchmark-cvc-clinicdb.ipynb` (tải trực tiếp từ kernel Kaggle kèm output stdout đầy đủ 200 epoch và bảng metric nghiệm thu).
- Kết quả: Dice 0.9427 ± 0.0550, mIoU 0.8960 ± 0.0842, Precision 0.9514, Specificity 0.9960, FPR 0.40%.
- Kết quả nguồn: `results/benchmark_v2/cvc-clinicdb/benchmark_results_CVC-ClinicDB.csv`.

### 02_EM-Dataset (Cấu trúc tế bào kính hiển vi điện tử)
- Môi trường chạy: Kaggle GPU Tesla T4 (100 epochs).
- Tệp thực thi: `fanet-benchmark-em-dataset.ipynb` (tải trực tiếp từ kernel Kaggle kèm output stdout 100 epoch và bảng metric nghiệm thu).
- Kết quả: Dice 0.9518 ± 0.0095, mIoU 0.9083 ± 0.0169, Specificity 0.9963, FPR 0.37%, Precision 0.9464.
- Kết quả nguồn: `results/benchmark_v2/em-dataset/benchmark_results_EM-Dataset.csv`.

### 03_Kvasir-SEG (Polyp đường tiêu hóa)
- Môi trường chạy: Google Colab A100 GPU (100 epochs, 120 ảnh test).
- Tệp thực thi: `colab_run_mfad_kvasir.ipynb` và `colab_run_mfad_kvasir.py` (kèm cell output chi tiết kết quả test final feedback t=2).
- Kết quả: Dice 0.8880 ± 0.1431, mIoU 0.8208 ± 0.1765, Sensitivity 0.9138, Specificity 0.9803, Precision 0.8938.
- Kết quả nguồn: `results/benchmark_v2/kvasir-seg/benchmark_results_Kvasir-SEG.csv`.

### 04_ISIC-2018 (Tổn thương da liễu ung thư hắc tố)
- Môi trường chạy: Kaggle GPU Tesla T4 (100 epochs).
- Tệp thực thi: `fanet-benchmark-isic-2018.ipynb` (tải trực tiếp từ kernel Kaggle kèm output stdout 100 epoch và bảng metric nghiệm thu).
- Kết quả: Dice 0.9000 ± 0.1246, mIoU 0.8346 ± 0.1557, Specificity 0.9714, Precision 0.9282.
- Checkpoint: Được sử dụng làm backbone truyền tri thức (transfer learning) sang tập DRIVE.
- Kết quả nguồn: `results/benchmark_v2/isic-2018/benchmark_results_ISIC-2018.csv`.

### 05_CHASE-DB1 (Vi mạch máu võng mạc đáy mắt)
- Môi trường chạy: Kaggle GPU Tesla T4 (100 epochs).
- Tệp thực thi: `fanet-benchmark-chasedb1.ipynb` (tải trực tiếp từ kernel Kaggle kèm output stdout 100 epoch và bảng metric nghiệm thu).
- Kết quả: Specificity 0.9926 ± 0.0020, Precision 0.8637 ± 0.0379, FPR 0.74% (giảm 65% tỷ lệ dương tính giả so với baseline gốc).
- Kết quả nguồn: `results/benchmark_v2/chasedb1/benchmark_results_CHASE-DB1.csv`.

### 06_DRIVE (Vi mạch võng mạc - Transfer Learning)
- Môi trường chạy: Google Colab A100 GPU (100 epochs, fine-tuning từ checkpoint ISIC-2018, repeat factor x32).
- Tệp thực thi: `colab_run_mfad_drive_transfer.ipynb` và `colab_run_mfad_drive_transfer.py` (pipeline transfer learning giải quyết dứt điểm vấn đề suy thoái biểu diễn trên tập chỉ có 20 ảnh huấn luyện, kèm output nghiệm thu vượt toàn bộ metric).
- Kết quả: Dice 0.7932 ± 0.0117, mIoU 0.6580 ± 0.0156, Specificity 0.9773, Precision 0.7755, Sensitivity 0.8122.
- Kết quả nguồn: `results/benchmark_v2/drive_transfer/benchmark_results_drive_transfer.csv`.

### 07_DSB-2018 (Phân đoạn nhân tế bào - In-RAM Caching)
- Môi trường chạy: Google Colab A100 GPU (Resume từ Epoch 39 đến 100 bằng In-RAM Preloading).
- Tệp thực thi: `colab_run_mfad_dsb2018_resume.ipynb` và `colab_run_mfad_dsb2018_resume.py` (tải trước 670 ảnh và hợp nhất masks trực tiếp vào RAM, tăng tốc huấn luyện đạt ~5 giây/epoch, kèm cell output nghiệm thu đầy đủ trên 134 mẫu test).
- Kết quả: Dice 0.8953 ± 0.1201, mIoU 0.8255 ± 0.1481, Specificity 0.9837, Precision 0.8951, Sensitivity 0.9009.
- Kết quả nguồn: `results/benchmark_v2/dsb2018/benchmark_results_dsb2018.csv`.
