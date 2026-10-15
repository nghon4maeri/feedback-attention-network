# Prompt: Tải Và Tập Hợp Benchmark Notebooks 7 Datasets (15/10/2026)

## 1. Yêu cầu của User
- Tải toàn bộ các file notebook/script benchmark của 7 tập dữ liệu từ Kaggle về (bao gồm các file chạy trên Colab).
- Sắp xếp vào một thư mục riêng biệt (`benchmark_notebooks/`) để tiện theo dõi, đối chiếu và kiểm tra mã nguồn.

## 2. Thực thi
- Dùng Kaggle API kéo trực tiếp 7 kernel chính thức:
  1. `namnguynnnn/fanet-benchmark-cvc-clinicdb`
  2. `namnguynnnn/fanet-benchmark-em-dataset`
  3. `namnguynnnn/fanet-benchmark-kvasir-seg`
  4. `namnguynnnn/fanet-benchmark-isic-2018`
  5. `namnguynnnn/fanet-benchmark-chasedb1`
  6. `namnguynnnn/fanet-benchmark-drive`
  7. `namnguynnnn/fanet-benchmark-dsb2018`
- Tập hợp các file thực thi và notebook Colab tương ứng:
  - `colab_run_mfad_kvasir.py` (Kvasir-SEG)
  - `colab_run_mfad_drive_transfer.py` (DRIVE Transfer Learning)
  - `colab_run_mfad_dsb2018_resume.py` (DSB-2018 In-RAM Caching)
  - 7 notebook Colab `fanet_mfad_*_colab.ipynb`
- Biên soạn tài liệu hướng dẫn `benchmark_notebooks/README.md`.
