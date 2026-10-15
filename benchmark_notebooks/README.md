# Benchmark V2 Notebooks & Execution Code (7/7 Biomedical Datasets)

Thư mục này chứa đầy đủ toàn bộ mã nguồn huấn luyện và kiểm thử (notebooks `.ipynb` và scripts thực thi `.py`) cho toàn bộ 7 bộ dữ liệu y sinh của nghiên cứu FANet-MFAD.

Mã nguồn được tải trực tiếp từ các kernel trên Kaggle (`namnguynnnn/fanet-benchmark-*`) kết hợp các script và notebook thực thi trên Google Colab A100.

---

## Cấu Trúc Thư Mục

### 01_CVC-ClinicDB (Polyp nội soi đại tràng)
- Môi trường chạy chính: Kaggle GPU Tesla T4 (200 epochs).
- Tệp Kaggle: `fanet-benchmark-cvc-clinicdb.ipynb` (tải trực tiếp từ Kaggle kernel).
- Tệp Colab: `fanet_mfad_cvc_clinicdb_colab.ipynb`.
- Kết quả nghiệm thu: Dice 0.9427 ± 0.0550, mIoU 0.8960, Precision 0.9514, FPR 0.40%.
- Trọng số và kết quả: `results/benchmark_v2/cvc-clinicdb/`.

### 02_EM-Dataset (Kính hiển vi điện tử tế bào)
- Môi trường chạy chính: Kaggle GPU Tesla T4 (100 epochs).
- Tệp Kaggle: `fanet-benchmark-em-dataset.ipynb` (tải trực tiếp từ Kaggle kernel).
- Tệp Colab: `fanet_mfad_em_dataset_colab.ipynb`.
- Kết quả nghiệm thu: Dice 0.9518 ± 0.0095, mIoU 0.9083, Specificity 0.9963, FPR 0.37%.
- Trọng số và kết quả: `results/benchmark_v2/em-dataset/`.

### 03_Kvasir-SEG (Polyp nội soi đường tiêu hóa)
- Môi trường chạy chính: Google Colab A100-SXM4-40GB GPU (100 epochs).
- Tệp Colab Script: `colab_run_mfad_kvasir.py` (script thực thi trực tiếp trên GPU A100).
- Tệp Colab Notebook: `fanet_mfad_kvasir_seg_colab.ipynb`.
- Tệp Kaggle đối chiếu: `fanet-benchmark-kvasir-seg.ipynb`.
- Kết quả nghiệm thu: Dice 0.8880 ± 0.1431, mIoU 0.8208, Recall 0.9138, Precision 0.8938.
- Trọng số và kết quả: `results/benchmark_v2/kvasir-seg/`.

### 04_ISIC-2018 (Tổn thương sắc tố da liễu)
- Môi trường chạy chính: Kaggle GPU Tesla T4 (100 epochs).
- Tệp Kaggle: `fanet-benchmark-isic-2018.ipynb` (tải trực tiếp từ Kaggle kernel).
- Tệp Colab: `fanet_mfad_isic_2018_colab.ipynb`.
- Kết quả nghiệm thu: Dice 0.9000 ± 0.1246, mIoU 0.8346, Specificity 0.9714, Precision 0.9282.
- Trọng số và kết quả: `results/benchmark_v2/isic-2018/` (checkpoint dùng làm backbone transfer cho DRIVE).

### 05_CHASE-DB1 (Vi mạch võng mạc đáy mắt)
- Môi trường chạy chính: Kaggle GPU Tesla T4 (100 epochs).
- Tệp Kaggle: `fanet-benchmark-chasedb1.ipynb` (tải trực tiếp từ Kaggle kernel).
- Tệp Colab: `fanet_mfad_chase_db1_colab.ipynb`.
- Kết quả nghiệm thu: Specificity 0.9926, Precision 0.8637, FPR 0.74% (giảm 65% báo động giả).
- Trọng số và kết quả: `results/benchmark_v2/chasedb1/`.

### 06_DRIVE (Vi mạch võng mạc - Transfer Learning từ ISIC-2018)
- Môi trường chạy chính: Google Colab A100 GPU (100 epochs, kế thừa trọng số ISIC-2018, repeat factor x32).
- Tệp Colab Script: `colab_run_mfad_drive_transfer.py` (pipeline transfer learning giải quyết sụp đổ biểu diễn trên tập 20 ảnh).
- Tệp Colab Notebook: `fanet_mfad_drive_colab.ipynb`.
- Tệp Kaggle đối chiếu: `fanet-benchmark-drive.ipynb` (bản train-from-scratch Dice 0.2837).
- Kết quả nghiệm thu: Dice 0.7932 ± 0.0117 (vượt baseline 0.7618 trên 100% chỉ số).
- Trọng số và kết quả: `results/benchmark_v2/drive_transfer/`.

### 07_DSB-2018 (Phân đoạn nhân tế bào - In-RAM Caching)
- Môi trường chạy chính: Google Colab A100 GPU (Resume từ Epoch 39 đến 100 bằng In-RAM Preloading).
- Tệp Colab Script: `colab_run_mfad_dsb2018_resume.py` (tải 100% 670 ảnh và hợp nhất masks vào RAM trong 22s, tốc độ ~5s/epoch).
- Tệp Colab Notebook: `fanet_mfad_dsb2018_colab.ipynb`.
- Tệp Kaggle đối chiếu: `fanet-benchmark-dsb2018.ipynb` (bản Kaggle chạy đến epoch 39).
- Kết quả nghiệm thu: Dice 0.8953 ± 0.1201, mIoU 0.8255, Precision 0.8951, FPR 1.63%.
- Trọng số và kết quả: `results/benchmark_v2/dsb2018/`.
