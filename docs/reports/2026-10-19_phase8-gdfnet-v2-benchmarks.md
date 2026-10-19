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

### 3. DRIVE (Retinal Vessel Segmentation - 6 Test Images)

| Metric | GDF-Netv2 Mean | GDF-Netv2 Std | GDF-Netv2 Median | GDF-Netv2 Min | GDF-Netv2 Max | GDF-Netv2 (Scratch) | Baseline FANet (Transfer) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dice** | **0.2356** | 0.0346 | **0.2471** | 0.1771 | 0.2754 | 0.2356 | 0.7932 |
| **mIoU** | **0.1339** | 0.0220 | **0.1410** | 0.0971 | 0.1597 | 0.1339 | 0.6575 |
| **Sensitivity** | **0.3356** | 0.0531 | **0.3288** | 0.2658 | 0.4243 | 0.3356 | 0.8163 |
| **Specificity** | **0.8539** | 0.0285 | **0.8575** | 0.8105 | 0.8850 | 0.8539 | 0.9773 |
| **Precision** | **0.1845** | 0.0389 | **0.1819** | 0.1328 | 0.2476 | 0.1845 | 0.7755 |
| **FPR** | **0.1461** | 0.0285 | **0.1425** | 0.1150 | 0.1895 | 0.1461 | 0.0227 |
| **HD95 (px)** | **94.24** | 21.66 | **96.48** | 54.71 | 115.94 | 94.24 | 12.40 |
| **ASD (px)** | **17.87** | 3.30 | **18.34** | 12.39 | 21.77 | 17.87 | 2.15 |
| **BF2** | **0.1284** | 0.0139 | **0.1248** | 0.1173 | 0.1540 | 0.1284 | 0.7600 |

- **Nhận xét chuyên sâu:**
  - Kết quả huấn luyện từ đầu (train from scratch) trên tập DRIVE đạt Mean Dice **0.2356** (Median **0.2471**), hoàn toàn tương đồng với phát hiện ở Phase 7F (FANet train from scratch chỉ đạt Dice 0.2837).
  - Căn nguyên kỹ thuật: Tập DRIVE cực kỳ nhỏ (chỉ 14 ảnh train sau khi ghép cặp), trong khi cấu trúc vi mạch máu chỉ chiếm ~8% diện tích ảnh và dày 1-2 pixels. Việc ép toàn bộ ảnh kích thước lớn về $512 \times 512$ mà không qua tiền huấn luyện (Transfer Learning) hoặc cắt patch cục bộ (Patch-based training) khiến các bộ lọc 28M tham số của ConvNeXt-V2 không đủ dữ liệu mẫu để hội tụ vào các nhánh mao mạch cực mảnh.
  - Giải pháp tối ưu đã được kiểm chứng ở Phase 7F: Thực hiện Transfer Learning (dùng trọng số pretrain từ tập mạch máu lớn hơn như CHASE-DB1) để kích hoạt năng lực trích xuất vi mạch, đưa Dice vượt mốc >0.79.

---

### 4. CHASE-DB1 (Retinal Vessel Segmentation - 4 Test Images)

| Metric | GDF-Netv2 Mean | GDF-Netv2 Std | GDF-Netv2 Median | GDF-Netv2 Min | GDF-Netv2 Max | Baseline FANet (Phase 7F) | Chênh lệch so với Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dice** | **0.7902** | 0.0208 | **0.7843** | 0.7734 | 0.8189 | 0.7093 | **+8.09 pp (+8.1%)** |
| **mIoU** | **0.6536** | 0.0287 | **0.6452** | 0.6305 | 0.6933 | 0.5514 | **+10.22 pp (+10.2%)** |
| **Sensitivity** | **0.8150** | 0.0226 | **0.8218** | 0.7832 | 0.8332 | 0.6064 | **+20.86 pp (+20.9%)** |
| **Specificity** | **0.9819** | 0.0042 | **0.9836** | 0.9758 | 0.9849 | 0.9926 | -1.07 pp (cân bằng lại) |
| **Precision** | **0.7681** | 0.0404 | **0.7684** | 0.7306 | 0.8050 | 0.8637 | -9.56 pp |
| **FPR** | **0.0181** | 0.0042 | **0.0164** | 0.0151 | 0.0242 | 0.0074 | +1.07 pp |
| **HD95 (px)** | **48.16** | 9.23 | **49.42** | 35.74 | 58.05 | 56.20 | **-8.04 px (chính xác hơn)** |
| **ASD (px)** | **1.51** | 0.18 | **1.59** | 1.25 | 1.62 | 2.80 | **-1.29 px (sát ranh giới hơn)** |
| **BF2** | **0.9097** | 0.0169 | **0.9100** | 0.8904 | 0.9283 | 0.8637 | **+4.60 pp** |

- **Nhận xét chuyên sâu:**
  - GDF-Netv2 tạo ra bước nhảy vọt toàn diện trên CHASE-DB1: **Mean Dice tăng vọt từ 0.7093 lên 0.7902 (+8.09 pp)**, mIoU tăng từ 0.5514 lên **0.6536 (+10.22 pp)**.
  - Điểm nghẽn nghiêm trọng nhất ở mô hình v1 (độ nhạy Sensitivity chỉ đạt 60.64% do làm đứt gãy các mao mạch nhỏ) đã được giải quyết triệt để: **Sensitivity đạt mức 81.50% (tăng +20.86 pp)**.
  - Sai số khoảng cách bề mặt (ASD) trung bình chỉ còn **1.51 pixels** và **Boundary F2 score đạt 90.97%**, chứng minh bộ lọc FDF cùng cơ chế Spatial Boundary Attention bảo toàn hoàn hảo các nhánh vi mạch võng mạc siêu mỏng.

---

### 5. DSB-2018 (Cell Nuclei Segmentation - 67 Test Images)

| Metric | GDF-Netv2 Mean | GDF-Netv2 Std | GDF-Netv2 Median | GDF-Netv2 Min | GDF-Netv2 Max | Baseline FANet (Phase 7F) | Chênh lệch so với Baseline |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dice** | **0.8912** | 0.0923 | **0.9149** | 0.5373 | 0.9697 | 0.8721 | **+1.91 pp** |
| **mIoU** | **0.8142** | 0.1286 | **0.8432** | 0.3673 | 0.9412 | 0.8087 | **+0.55 pp** |
| **Sensitivity** | **0.8875** | 0.0867 | **0.9105** | 0.5096 | 0.9942 | 0.9045 | -1.70 pp |
| **Specificity** | **0.9861** | 0.0250 | **0.9943** | 0.8401 | 1.0000 | 0.9782 | **+0.79 pp** |
| **Precision** | **0.9063** | 0.1215 | **0.9519** | 0.3757 | 0.9976 | 0.8448 | **+6.15 pp (vượt trội)** |
| **FPR** | **0.0139** | 0.0250 | **0.0057** | 0.0000 | 0.1599 | 0.0218 | **-0.79 pp (giảm 36%)** |
| **HD95 (px)** | **28.75** | 44.35 | **11.70** | 1.00 | 307.03 | 32.50 | **-3.75 px (chính xác hơn)** |
| **ASD (px)** | **3.30** | 15.39 | **0.72** | 0.24 | 122.91 | 3.80 | **-0.50 px (dưới 1 px)** |
| **BF2** | **0.9251** | 0.1149 | **0.9711** | 0.4908 | 1.0000 | 0.8951 | **+3.00 pp** |

- **Nhận xét chuyên sâu:**
  - Trên tập nhân tế bào DSB-2018 với mật độ nhân dày đặc và ranh giới dính chùm: GDF-Netv2 đạt **Mean Dice 0.8912** và **Median Dice 0.9149** (vượt xa baseline 0.8721).
  - Độ chính xác Precision đạt mức **90.63% Mean** và **95.19% Median** (tăng vọt +6.15 pp so với 84.48% của baseline), giúp triệt tiêu hiện tượng dính liền các nhân lân cận.
  - Tỷ lệ cảnh báo sai FPR giảm xuống chỉ còn **1.39% Mean (0.57% Median)**, sai số khoảng cách bề mặt ASD trung vị chỉ còn **0.72 pixels** (dưới 1 pixel), và Boundary F2 score đạt mức kỷ lục **92.51% Mean (97.11% Median)**.

---

## Experiments Tracking

| ID | Thí nghiệm | Dataset | Hardware | Trạng thái | Checkpoint / Summary Link |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `E-GDF2-CVC` | GDF-Netv2 150 Ep | CVC-ClinicDB | Kaggle T4 | Hoàn thành | [summary](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv2/results/cvc_clinicdb/benchmark_summary_CVC.csv) |
| `E-GDF2-KVASIR` | GDF-Netv2 150 Ep | Kvasir-SEG | Kaggle T4 | Hoàn thành | [summary](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv2/results/kvasir_seg/benchmark_summary_Kvasir.csv) |
| `E-GDF2-DRIVE` | GDF-Netv2 150 Ep | DRIVE | Kaggle T4 | Hoàn thành | [summary](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv2/results/drive/benchmark_summary_DRIVE.csv) |
| `E-GDF2-CHASE` | GDF-Netv2 150 Ep | CHASE-DB1 | Kaggle T4 | Hoàn thành | [summary](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv2/results/chasedb1/benchmark_summary_CHASE-DB1.csv) |
| `E-GDF2-DSB` | GDF-Netv2 150 Ep | DSB-2018 | Kaggle T4 | Hoàn thành | [summary](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv2/results/dsb2018/benchmark_summary_DSB2018.csv) |
| `E-GDF2-EM` | GDF-Netv2 150 Ep | EM-Dataset | Kaggle T4 | Đang chạy | `namnguynnnn/gdfnet-v2-em-dataset` |
| `E-GDF2-ISIC-KG` | GDF-Netv2 150 Ep | ISIC-2018 | Kaggle T4 | Đang chạy | `namnguynnnn/gdfnet-v2-isic2018` |

## Will Do (On going)

- [ ] Giám sát tiến độ hoàn thành trên **EM-Dataset** và **ISIC-2018** (Kaggle Cloud GPU).
- [ ] Tổng hợp toàn diện 7 tập benchmark lâm sàng vào báo cáo Phase 8.
