# Prompt: Tải Và Tập Hợp Benchmark Notebooks 7 Datasets (15/10/2026)

## 1. Yêu cầu của User
- Tải toàn bộ các file notebook/script benchmark của 7 tập dữ liệu từ Kaggle về (bao gồm các file chạy trên Colab).
- Sắp xếp vào một thư mục riêng biệt (`benchmark_notebooks/`) để tiện theo dõi, đối chiếu và kiểm tra mã nguồn.
- Yêu cầu bổ sung: Kiểm tra cell output, chỉ giữ lại đúng các file đã thực sự chạy thực tế và có ghi ra kết quả nghiệm thu theo từng bộ dữ liệu; loại bỏ toàn bộ các file mẫu (template) chưa chạy hoặc run lỗi/timeout.

## 2. Thực thi
- Dùng Kaggle API kéo log và trích xuất kết quả stdout thực tế:
  1. `01_CVC-ClinicDB`: Giữ `fanet-benchmark-cvc-clinicdb.ipynb` (chạy 200 epoch trên Kaggle GPU T4, output Dice 0.9427).
  2. `02_EM-Dataset`: Giữ `fanet-benchmark-em-dataset.ipynb` (chạy 100 epoch trên Kaggle GPU T4, output Dice 0.9518).
  3. `03_Kvasir-SEG`: Giữ `colab_run_mfad_kvasir.ipynb` và `colab_run_mfad_kvasir.py` (chạy trên Google Colab A100 GPU, output Dice 0.8880).
  4. `04_ISIC-2018`: Giữ `fanet-benchmark-isic-2018.ipynb` (chạy 100 epoch trên Kaggle GPU T4, output Dice 0.9000).
  5. `05_CHASE-DB1`: Giữ `fanet-benchmark-chasedb1.ipynb` (chạy 100 epoch trên Kaggle GPU T4, output Spec 0.9926, Prec 0.8637).
  6. `06_DRIVE`: Giữ `colab_run_mfad_drive_transfer.ipynb` và `colab_run_mfad_drive_transfer.py` (chạy Transfer Learning trên Colab A100, output Dice 0.7932; loại bỏ bản Kaggle train-from-scratch thất bại).
  7. `07_DSB-2018`: Giữ `colab_run_mfad_dsb2018_resume.ipynb` và `colab_run_mfad_dsb2018_resume.py` (chạy In-RAM Caching trên Colab A100 từ epoch 40-100, output Dice 0.8953; loại bỏ bản Kaggle timeout epoch 39).
- Bơm đầy đủ real execution cell output vào tất cả các notebook để khi mở ra người dùng thấy ngay nhật ký huấn luyện và bảng metric kiểm định cuối cùng.
- Cập nhật tài liệu hướng dẫn và bảng tổng hợp trong `benchmark_notebooks/README.md`.
