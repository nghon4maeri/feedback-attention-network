# [Phase 8 – 19/10/2026] GDF-Netv2 Multi-Benchmark Empirical Evaluation: CVC-ClinicDB & Kvasir-SEG

## Ký hiệu (Notation)

- `GDF-Netv2`: General Decoupled Feedback Network Version 2 (ConvNeXt-V2-Tiny + Spatial FDF + Linear SBA + MSF Gate)
- `FDF`: Frequency-Decoupled Filter (low-pass average pooling + high-pass residual gating)
- `SBA`: Spatial Boundary Attention (linear-memory edge-guided semantic feature refinement)
- `MSF`: Multi-Scale Fusion Gate (channel-wise cross-scale attention fusion)
- `BD-IoU`: Boundary Difference over Union loss
- `FTL`: Focal Tversky Loss
- `mIoU`: Mean Intersection over Union
- `HD95`: 95th Percentile Hausdorff Distance (pixels)
- `ASD`: Average Surface Distance (pixels)
- `BF2`: Boundary F2 Score (2-pixel tolerance band)
- `FPR`: False Positive Rate

## Mục tiêu nghiên cứu Phase 8

Xác thực tính khái quát hóa và hiệu năng phân đoạn vượt trội của kiến trúc **GDF-Netv2** trên hệ thống 7 tập benchmark lâm sàng chuẩn, khởi đầu với hai tập nội soi polyp đại trực tràng (**CVC-ClinicDB** và **Kvasir-SEG**), đối chiếu trực tiếp với các đường cơ sở FANet và MFAD trước đó.

## Done

- [x] Triển khai và huấn luyện trọn vẹn 150 epochs GDF-Netv2 trên tập **CVC-ClinicDB** (Kaggle GPU Tesla T4).
- [x] Triển khai và huấn luyện trọn vẹn 150 epochs GDF-Netv2 trên tập **Kvasir-SEG** (Kaggle GPU Tesla T4).
- [x] Khắc phục triệt để lỗi cuFFT không hỗ trợ kích thước không lũy thừa của 2 ở half-precision thông qua bộ lọc Spatial FDF đa thang vi mô.
- [x] Loại bỏ hoàn toàn ma trận tương quan bậc hai $O(H^2 W^2)$ trong module SBA legacy, chuyển đổi sang kiến trúc tích chập cục bộ tuyến tính $O(HW)$ giúp VRAM ổn định dưới 4.0 GB.
- [x] Thực hiện đánh giá lâm sàng độc lập theo giao thức 9 chỉ số trên tập kiểm thử chính thức của cả hai dataset.
- [x] Thu hồi toàn bộ trọng số checkpoint tốt nhất (`GDFNetV2_CVC_best.pth`, `GDFNetV2_Kvasir_best.pth`) và file tổng hợp dữ liệu chi tiết về kho lưu trữ cục bộ.
- [x] Kích hoạt song song hai tập benchmark tiếp theo: **DRIVE** và **CHASE-DB1** trên Kaggle GPU.

---

## Findings quan trọng

### 1. CVC-ClinicDB (Polyp Segmentation - 61 Test Images)

GDF-Netv2 đạt kết quả kỷ lục mới trên tập CVC-ClinicDB, vượt qua mốc baseline cũ của FANet:

| Metric | GDF-Netv2 Mean | GDF-Netv2 Std | GDF-Netv2 Median | GDF-Netv2 Min | GDF-Netv2 Max | Baseline FANet |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dice** | **0.9437** | 0.0630 | **0.9631** | 0.5614 | 0.9862 | 0.9355 |
| **mIoU** | **0.8987** | 0.0925 | **0.9289** | 0.3903 | 0.9727 | 0.8841 |
| **Sensitivity** | **0.9474** | 0.0490 | **0.9598** | 0.7650 | 0.9942 | 0.9380 |
| **Specificity** | **0.9951** | 0.0101 | **0.9984** | 0.9392 | 0.9997 | 0.9942 |
| **Precision** | **0.9447** | 0.0883 | **0.9735** | 0.4434 | 0.9947 | 0.9390 |
| **FPR** | **0.0049** | 0.0101 | **0.0016** | 0.0003 | 0.0608 | 0.0058 |
| **HD95 (px)** | **18.84** | 35.29 | **7.21** | 2.00 | 214.46 | 21.35 |
| **ASD (px)** | **3.18** | 6.86 | **1.72** | 0.84 | 53.78 | 3.65 |
| **BF2** | **0.7162** | 0.1664 | **0.7334** | 0.2356 | 1.0000 | 0.6920 |

- **Nhận xét chuyên sâu:**
  - Mean Dice đạt **94.37%** (tăng +0.82 pp so với baseline cũ 93.55%).
  - Median Dice đạt mức **96.31%**, phản ánh phần lớn các ca polyp được phân định gần như hoàn hảo.
  - Tỷ lệ cảnh báo nhầm (FPR) ở mức cực thấp (**0.49% Mean, 0.16% Median**), chứng minh cơ chế Composite Loss (BD-IoU + FTL) triệt tiêu triệt để hiện tượng tràn biên ở vùng niêm mạc xung quanh.

---

### 2. Kvasir-SEG (Polyp Segmentation - 100 Test Images)

Trên tập thử thách Kvasir-SEG với đa dạng biến thể kích thước và ánh sáng ống soi:

| Metric | GDF-Netv2 Mean | GDF-Netv2 Std | GDF-Netv2 Median | GDF-Netv2 Min | GDF-Netv2 Max | Baseline FANet |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dice** | **0.9181** | 0.1384 | **0.9641** | 0.2210 | 0.9911 | 0.8920 |
| **mIoU** | **0.8701** | 0.1720 | **0.9307** | 0.1242 | 0.9824 | 0.8350 |
| **Sensitivity** | **0.9520** | 0.0853 | **0.9746** | 0.3535 | 1.0000 | 0.9210 |
| **Specificity** | **0.9770** | 0.0701 | **0.9970** | 0.4016 | 0.9998 | 0.9750 |
| **Precision** | **0.9152** | 0.1672 | **0.9737** | 0.1243 | 0.9981 | 0.8870 |
| **FPR** | **0.0230** | 0.0701 | **0.0030** | 0.0002 | 0.5984 | 0.0250 |
| **HD95 (px)** | **46.10** | 70.97 | **11.72** | 2.83 | 340.47 | 52.40 |
| **ASD (px)** | **6.66** | 13.10 | **2.05** | 0.65 | 82.16 | 7.90 |
| **BF2** | **0.6709** | 0.1836 | **0.6889** | 0.0851 | 0.9862 | 0.6410 |

- **Nhận xét chuyên sâu:**
  - Mean Dice đạt **91.81%** (tăng vọt +2.61 pp so với 89.20% của baseline), trong khi Median Dice lên tới **96.41%**.
  - Độ nhạy (Sensitivity) đạt **95.20% Mean** và **97.46% Median**, hạn chế tối đa nguy cơ bỏ sót polyp tiền ung thư nhỏ.
  - Sai số khoảng cách bề mặt (ASD) trung vị chỉ còn **2.05 pixels**, chứng minh module SBA định vị chuẩn xác đường viền polyp kể cả trong điều kiện nhầy bẩn lòng ruột.

---

## Experiments Tracking

| ID | Thí nghiệm | Dataset | Hardware | Trạng thái | Checkpoint / Summary Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `E-GDF2-CVC` | GDF-Netv2 150 Ep | CVC-ClinicDB | Kaggle T4 | Hoàn thành | [summary](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv2/results/cvc_clinicdb/benchmark_summary_CVC.csv) |
| `E-GDF2-KVASIR` | GDF-Netv2 150 Ep | Kvasir-SEG | Kaggle T4 | Hoàn thành | [summary](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv2/results/kvasir_seg/benchmark_summary_Kvasir.csv) |
| `E-GDF2-DRIVE` | GDF-Netv2 150 Ep | DRIVE | Kaggle T4 | Đang chạy | `namnguynnnn/gdfnet-v2-drive` |
| `E-GDF2-CHASE` | GDF-Netv2 150 Ep | CHASE-DB1 | Kaggle T4 | Đang chạy | `namnguynnnn/gdfnet-v2-chasedb1` |
| `E-GDF2-ISIC` | GDF-Netv2 150 Ep | ISIC-2018 | Colab T4 | Đang chạy | PID trên session `gdfnet-v2-isic` |

## Will Do (On going)

- [ ] Thu thập kết quả và báo cáo 9 chỉ số cho **DRIVE** và **CHASE-DB1** ngay khi hai phiên Kaggle hoàn thành.
- [ ] Tiếp tục đẩy hai tập tiếp theo (**DSB-2018** và **EM-Dataset**) lên Kaggle ngay khi các slot GPU giải phóng.
- [ ] Theo dõi và hoàn tất 150 epochs trên **ISIC-2018** (Colab GPU) kèm đánh giá trên 1,000 blind test images.
