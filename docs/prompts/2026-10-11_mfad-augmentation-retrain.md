# Prompt (11/10/2026): Retrain MFAD-Only with Enhanced Paper-Grade Data Augmentation (DRIVE 32x, CHASE 32x, EM 16x) across Kaggle & Colab

**Mục tiêu**:
1. Phân tích paper gốc FANet (Tomar et al., 2022) và mã nguồn chính thức (`nikhilroxtomar/FANet`):
   - Làm rõ cơ chế Data Augmentation và các tham số mở rộng dữ liệu (Table I & Section IV-A-2).
   - Xác định nguyên nhân gây mode collapse trên tập DRIVE: bộ train chỉ có 20 ảnh, nếu train batch size 2 mà không có repeat factor thì mỗi epoch chỉ có 10 gradient steps (quá ít để hội tụ trên vi mạch máu võng mạc ~8% foreground). Paper giải quyết điều này bằng hệ số mở rộng 32x (Train after aug = 640).
2. Xây dựng bộ training script và notebook mới huấn luyện **DUY NHẤT model FANet_MFAD đề xuất** (bỏ qua huấn luyện baseline FANet Orig để tiết kiệm 50% thời gian tính toán và tránh timeout).
3. Tích hợp bộ data augmentation hoàn chỉnh chuẩn paper vào `UniversalDataset` và `EMVolumeDataset`:
   - `repeat_factor=32` cho DRIVE (20 ảnh -> 640 ảnh/epoch) và CHASE-DB1 (20 ảnh -> 640 ảnh/epoch).
   - `repeat_factor=16` cho EM-Dataset (24 lát cắt -> 384 lát cắt/epoch).
   - `repeat_factor=2` cho DSB-2018 và CVC-ClinicDB.
   - Thêm bộ lọc CLAHE chuyên biệt cho ảnh mạch máu võng mạc (tăng cường độ tương phản vi mạch).
   - Dynamic Albumentations: Horizontal/Vertical Flip, ShiftScaleRotate, ElasticTransform, GridDistortion, OpticalDistortion, RandomBrightnessContrast, HueSaturationValue, CoarseDropout.
4. Đóng gói 2 phiên bản độc lập:
   - **Kaggle version:** 7 notebooks trong `notebooks/benchmark_v2/` và `kaggle/benchmark_kernels/`.
   - **Colab version:** 7 notebooks trong `notebooks/colab/` (hỗ trợ A100/T4, tự động tải dữ liệu từ Kaggle qua Colab Secrets `KAGGLE_USERNAME`/`KAGGLE_KEY`, lưu checkpoint và kết quả trực tiếp về Google Drive `/content/drive/MyDrive/FANet_runs_MFAD/<slug>`).
5. Đẩy và kích hoạt các kernel lên Kaggle, quản lý giới hạn 2 batch GPU đồng thời của Kaggle và cung cấp notebook Colab để người dùng chạy song song các tập nặng (ISIC-2018, DSB-2018, Kvasir-SEG, DRIVE).

**Phạm vi & Yêu cầu**:
- 7 tập datasets: Kvasir-SEG, CHASE-DB1, DRIVE, DSB-2018, ISIC-2018, CVC-ClinicDB, EM-Dataset.
- Đánh giá đầy đủ các độ đo: Dice, mIoU, Sensitivity, Specificity, FPR, Precision (cho cả Pass-1 không feedback và Pass-2 có feedback).
- Xuất đầy đủ `benchmark_results_<DATASET>.csv`, `comparison_<DATASET>.png`, `training_curves_<DATASET>.png`.
- Tuân thủ git workflow: commit và push ngay sau khi cập nhật file.
