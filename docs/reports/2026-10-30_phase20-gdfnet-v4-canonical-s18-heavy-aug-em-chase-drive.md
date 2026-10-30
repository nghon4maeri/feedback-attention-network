# Phase 20 — Huấn luyện Toàn diện Canonical GDF-Netv4 (ConvFormer-S18) với Heavy Data Augmentation trên EM-Dataset, CHASE-DB1, DRIVE

**Ngày:** 30/10/2026 · **Kernel:** `namnguynnnn/gdfnetv4-s18-heavy-aug-em-chase-drive` (Kaggle T4, ~85m) · **Script:** [`run_gdfnet_v4_canonical_s18_heavy_aug_em_chase_drive.py`](../../GDF-Netv4/kaggle/canonical_s18_heavy_aug_em_chase_drive/run_gdfnet_v4_canonical_s18_heavy_aug_em_chase_drive.py)

---

## 1. Mục tiêu Thực nghiệm
Triển khai huấn luyện mô hình chủ lực đề xuất cho bài báo khoa học (**`GDF-Netv4-ConvFormerS18`** — 29.5M params Pure-CNN) với kiến trúc chuẩn **Canonical GDF-Netv4** (`MaskGuidedLGAG` + `CascadedMSCAMDecoderBlock`) kết hợp bộ tăng cường dữ liệu mở rộng (**Clinical & Biological Heavy Augmentation**) trên 3 tập dữ liệu cấu trúc vi mô và vi mạch:
1. **EM-Dataset (Màng tế bào thần kinh 3D EM):**
   - Áp dụng giao thức chia tập chuẩn hóa **153 Train / 6 Val / 6 Test** trên toàn bộ thể tích 165 lát cắt 3D TIFF 512×512 (khắc phục dứt điểm lỗi runner cũ chia sai 20 train / 141 test), cô lập 6 lát validation để hiệu chỉnh ngưỡng $T_{\text{opt}}$ tránh rò rỉ dữ liệu.
2. **CHASE-DB1 (Mạch máu võng mạc độ phân giải cao):**
   - Độ phân giải nâng cấp 768×768, hàm mất mát cấu trúc vi mạch mỏng (Loss kernel=7, trọng số Dice=0.6, weight_factor=4.0), 60 epochs.
3. **DRIVE (Mạch máu võng mạc tiệm cận kích thước gốc):**
   - Độ phân giải tiệm cận gốc 576×576 (12 Train / 2 Val / 6 Test từ official set ID 21–40), Loss kernel=7, 50 epochs.
   - **Nghiên cứu đối đầu chuyển giao tri thức (Transfer Learning Ablation):** Đánh giá so sánh trực tiếp giữa **Intra-domain warm-start** (khởi tạo từ checkpoint mạch võng mạc CHASE-DB1) và **Inter-domain warm-start** (khởi tạo từ checkpoint tổn thương da ISIC-2018).

---

## 2. Phân loại Bằng chứng
- `TRAIN`: Toàn bộ 4 cấu hình huấn luyện mới 100% từ checkpoint trên GPU Kaggle Tesla T4.
- `EVAL`: Tuyển chọn mô hình tối ưu theo Val Macro Dice thông qua ModelEMA ($decay=0.995$), tự động quét tìm ngưỡng quyết định tối ưu $T_{\text{opt}}$ trên tập Validation (51 mốc từ 0.25 đến 0.75), và nghiệm thu độc lập trên tập Test theo giao thức **Calibrated 4-View Test-Time Augmentation (TTA)** kèm surface metrics toàn diện (HD95, ASD, BF2).

---

## 3. Kết quả Nghiệm thu Chi tiết (Calibrated 4-View TTA, Test Set)

| Dataset | Backbone | Warm-Start Mode | Split | Best Ep | $T_{\text{opt}}$ | SV Mean Dice | TTA Mean Dice | **Cal Mean Dice** | **Cal Median Dice** | mIoU | Sensitivity | Specificity | Precision | HD95 (px) | ASD (px) | BF2 | Global Micro Dice |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **EM-Dataset** | ConvFormer-S18 | ISIC-2018 | 153/6/6 | 47 | 0.740 | 0.9476 | 0.9510 | **0.9525** | **0.9508** | **0.9093** | 0.9466 | **0.9972** | **0.9590** | **2.15** | **1.22** | **0.9556** | **0.9515** |
| **CHASE-DB1** | ConvFormer-S18 | ISIC-2018 | 20/4/4 | 56 | 0.580 | 0.8109 | 0.8181 | **0.8210** | **0.8158** | **0.6966** | 0.8432 | **0.9847** | **0.8019** | **5.65** | **1.64** | **0.8986** | **0.8213** |
| **DRIVE (Intra)** | ConvFormer-S18 | **CHASE-DB1** | 12/2/6 | 44 | 0.550 | 0.7853 | 0.7968 | **0.7982** | **0.7994** | **0.6646** | 0.8247 | **0.9779** | **0.7765** | **6.24** | **1.35** | **0.9165** | **0.8001** |
| **DRIVE (Inter)** | ConvFormer-S18 | ISIC-2018 | 12/2/6 | 49 | 0.590 | 0.7685 | 0.7809 | 0.7851 | 0.7907 | 0.6469 | 0.8146 | 0.9764 | 0.7623 | 7.39 | 1.48 | 0.9050 | 0.7874 |

---

## 4. Phát hiện Khoa học Trọng tâm

### 4.1. Hiệu năng Xuất sắc của EM-Dataset theo Giao thức Chuẩn hóa
- Việc chuẩn hóa thể tích về **153 Train / 6 Val / 6 Test** đã giải phóng tối đa năng lực học tập của `ConvFormer-S18`.
- Với 153 lát cắt huấn luyện thực chất (thay vì 20 lát như runner lỗi cũ), mô hình đạt độ hội tụ vượt bậc tại Epoch 47:
  - **Calibrated Macro Dice đạt 0.9525 (95.25%)**, Median Dice đạt **0.9508**, Global Micro Dice đạt **0.9515**.
  - Sai số biên viền tiệm cận mức hoàn hảo: **HD95 chỉ 2.15 px**, **ASD 1.22 px**, và **Boundary F2 score (BF2) đạt 0.9556**.
  - Độ đặc hiệu đạt **99.72%**, xác nhận hiện tượng dương tính giả trên chất nền soma tế bào thần kinh gần như bị triệt tiêu hoàn toàn.

### 4.2. Khẳng định Vai trò Chuyển giao Tri thức Cùng Miền (Intra-Domain Transfer) trên DRIVE
- Thực nghiệm đối đầu kiểm chứng chuyển giao tri thức cho thấy kết quả rõ ràng:
  - Chuyển giao cùng miền lâm sàng (Retinal-to-Retinal từ CHASE-DB1) vượt trội chuyển giao khác miền (Skin Lesion-to-Retinal từ ISIC-2018) trên toàn bộ hệ thống chỉ số:
    + **Mean Dice:** **0.7982** vs 0.7851 (**+1.31 pp**)
    + **Global Micro Dice:** **0.8001** vs 0.7874 (**+1.27 pp**)
    + **mIoU:** **0.6646** vs 0.6469 (**+1.77 pp**)
    + **HD95:** **6.24 px** vs 7.39 px (**giảm 1.15 px sai số biên**)
    + **BF2:** **0.9165** vs 0.9050 (**+1.15 pp**)
- Kết luận: Khi dữ liệu mục tiêu cực kỳ khan hiếm (DRIVE chỉ có 12 ảnh huấn luyện), việc khởi tạo trọng số từ một tập cùng miền giải phẫu (CHASE-DB1) cung cấp đặc trưng định hướng cấu trúc hình học mạch máu (vessel tree geometry) tốt hơn đáng kể so với chỉ học từ tổn thương khối u phẳng (dermoscopic lesion).

### 4.3. CHASE-DB1 Phân giải Cao 768×768
- Mô hình đạt **0.8210 Calibrated Dice** (Global Micro Dice 0.8213), độ nhạy bắt vi mạch đạt **84.32%** (Sensitivity = 0.8432), và Boundary F2 đạt **0.8986**.
- Việc kết hợp Heavy Augmentation đa trục cùng kernel mất mát co nhỏ (kernel=7) giúp duy trì tính liên tục của các mao mạch võng mạc mảnh, chống đứt gãy phân đoạn.

---

## 5. Artifacts
Toàn bộ kết quả, log và metric raw/summary được lưu tại:
`GDF-Netv4/results/s18_heavy_aug_em_chase_drive/`
- `master_retrain_summary.csv`
- `benchmark_summary_*` (SingleView, TTA, TTA_Calibrated)
- `benchmark_raw_*`
- `training_history_*`
- `gdfnetv4-s18-heavy-aug-em-chase-drive.log`
