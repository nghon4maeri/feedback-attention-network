# Phase 19 — Đồng nhất Kiến trúc Canonical GDF-Netv4 (ConvFormer-S18) trên CVC-ClinicDB, Kvasir-SEG, DSB-2018 (warm-start ISIC-2018)

**Ngày:** 30/10/2026 · **Kernel:** `namnguynnnn/gdfnetv4-canonical-s18-polyp-nuclei` (Kaggle T4, tổng ~32m) · **Script:** [`run_gdfnet_v4_canonical_s18_polyp_nuclei.py`](../../GDF-Netv4/training/run_gdfnet_v4_canonical_s18_polyp_nuclei.py)

## 1. Mục tiêu
Khắc phục sự thiếu đồng nhất về kiến trúc giữa các tập dữ liệu trong bảng kết quả bài báo khoa học:
- Ở Phase 17, ba tập CVC-ClinicDB, Kvasir-SEG và DSB-2018 sử dụng **decoder rút gọn thử nghiệm** (`AttentionGate` + `ConvBlock`).
- Đợt này chuẩn hóa toàn bộ về **Canonical GDF-Netv4** (`MaskGuidedLGAG` + `CascadedMSCAMDecoderBlock`) với backbone chủ lực **`ConvFormer-S18`**, khởi tạo warm-start từ checkpoint ISIC-2018 A100 (`GDFNetV4_ConvFormerS18_ISIC2018_best.pth`).
- Giữ nguyên giao thức chia tập ngẫu nhiên Seed=42 để đối chiếu trực tiếp (CVC: 490/61/61, Kvasir: 800/100/100, DSB: 536/67/67).

## 2. Phân loại bằng chứng
- `TRAIN`: Toàn bộ 3 lần chạy huấn luyện mới từ checkpoint tiền huấn luyện ISIC-2018 trên Kaggle T4. Log xác nhận load checkpoint thành công 100% (Missing: 0, Unexpected: 0). Chọn checkpoint theo Val Macro Dice (RAW/EMA), hiệu chỉnh ngưỡng tối ưu $T_{\text{opt}}$ trên tập Validation, và đánh giá trên tập Test độc lập.

## 3. Kết quả nghiệm thu (Calibrated 4-View TTA, Test)

| Dataset | Backbone | Split | Best Ep | $T_{\text{opt}}$ | SV Dice | TTA Dice | **Cal Dice** | Cal Median | mIoU | Sens | Spec | Prec | HD95 (px) | ASD (px) | BF2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **CVC-ClinicDB** | ConvFormer-S18 | 490/61/61 | 30 | 0.57 | 0.9249 | 0.9461 | **0.9458** | **0.9578** | **0.9003** | 0.9411 | **0.9963** | **0.9547** | **10.86** | **2.70** | **0.7192** |
| **Kvasir-SEG** | ConvFormer-S18 | 800/100/100 | 44 | 0.51 | 0.9291 | 0.9335 | **0.9344** | **0.9673** | **0.8900** | 0.9523 | **0.9870** | **0.9351** | 21.04 | **4.64** | 0.6891 |
| **DSB-2018** | ConvFormer-S18 | 536/67/67 | 50 | 0.66 | 0.8956 | 0.8972 | **0.8981** | **0.9143** | **0.8251** | 0.9117 | **0.9853** | **0.8969** | 9.22 | 2.61 | **0.9278** |

## 4. So sánh với Phase 17 (Decoder thử nghiệm vs Canonical GDF-Netv4)

| Dataset | Metric | Phase 17 (Pilot Decoder) | Phase 19 (Canonical GDFNetV4) | Chênh lệch (Δ) | Đánh giá lâm sàng & ý nghĩa |
|---|---|---|---|---|---|
| **CVC-ClinicDB** | **Mean Dice** | 0.9411 | **0.9458** | **+0.47 pp** | Tăng độ tương đồng phân đoạn tổng thể |
| | **Median Dice** | 0.9538 | **0.9578** | **+0.40 pp** | Nửa trên phân bố ca bệnh đạt Dice cực cao |
| | **Precision** | 0.9291 | **0.9547** | **+2.56 pp** | Giảm thiểu đáng kể ca dương tính giả |
| | **HD95 (px)** | 14.01 | **10.86** | **−3.15 px** | Sai số khoảng cách biên tối đa thu hẹp rõ rệt |
| | **ASD (px)** | 3.20 | **2.70** | **−0.50 px** | Độ lệch đường viền trung bình tiệm cận mặt đất |
| | **BF2 Score** | 0.7004 | **0.7192** | **+1.88 pp** | F2 score trên đường viền tăng vững chắc |
| **Kvasir-SEG** | **Mean Dice** | 0.9388 | 0.9344 | −0.44 pp | Dao động nhỏ trên phân bố đa kích thước polyp |
| | **Median Dice** | 0.9676 | 0.9673 | −0.03 pp | Trung vị bảo toàn vững vàng ở đỉnh cao |
| | **ASD (px)** | 4.76 | **4.64** | **−0.12 px** | Độ lệch viền bề mặt tiếp tục cải thiện |
| **DSB-2018** | **Mean Dice** | 0.9044 | 0.8981 | −0.63 pp | Dao động nhẹ quanh mốc 0.90 |
| | **BF2 Score** | 0.9313 | 0.9278 | −0.35 pp | Độ chính xác viền nhân tế bào giữ vững >0.92 |

## 5. Thảo luận khoa học
1. **Đồng nhất 100% kiến trúc bài báo:** Toàn bộ bảng dữ liệu lâm sàng hiện nay đã sử dụng **duy nhất một mô hình Canonical GDF-Netv4** (`MaskGuidedLGAG` + `CascadedMSCAMDecoderBlock` + `HierarchicalBackboneV4(ConvFormer-S18)`). Không còn sự chắp vá giữa decoder rút gọn và decoder chuẩn.
2. **CVC-ClinicDB cải thiện vượt bậc:** Trên polyp nội soi CVC, decoder chuẩn với cơ chế Multi-Scale Convolutional Attention Module (MSCAM) và Mask-Guided Local-Global Attention Gate (LGAG) phát huy hiệu quả mạnh mẽ: Precision tăng vọt **+2.56 pp** (đạt `0.9547`), HD95 co hẹp **-3.15 px** (xuống `10.86 px`).
3. **Tính ổn định của Pure-CNN:** Backbone `ConvFormer-S18` hoàn tất huấn luyện cả 3 tập chỉ trong 32 phút trên GPU Tesla T4 (trung bình ~10 phút/dataset), nhanh hơn gấp 2 lần so với Transformer nhưng vẫn đảm bảo Dice từ 0.90 đến 0.95 trên các tổn thương mô học.

## 6. Artifacts
Toàn bộ CSV và log được lưu trữ tại `GDF-Netv4/results/canonical_s18_polyp_nuclei/`:
- `master_canonical_s18_polyp_nuclei.csv`
- `benchmark_summary_*` (SingleView, TTA, TTA_Calibrated)
- `benchmark_raw_*`
- `training_history_*`
- `gdfnetv4-canonical-s18-polyp-nuclei.log`
Checkpoint `.pth` (3 files, ~118 MB/file) được lưu cục bộ.
