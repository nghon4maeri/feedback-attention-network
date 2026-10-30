# Phase 18 — Huấn luyện lại Canonical GDF-Netv4 trên EM-Dataset, CHASE-DB1, DRIVE (warm-start ISIC-2018)

**Ngày:** 30/10/2026 · **Kernel:** `namnguynnnn/gdfnetv4-retrain-em-chase-drive` (Kaggle T4, tổng ~2h52m) · **Script:** [`run_gdfnet_v4_retrain_em_chase_drive.py`](../../GDF-Netv4/training/run_gdfnet_v4_retrain_em_chase_drive.py)

## 1. Mục tiêu
Khắc phục 3 tập có kết quả yếu/sai protocol ở Phase 17:
- **EM-Dataset:** sửa split lỗi 20/4/141 → **153 / 6 / 6** (volume 165 lát).
- **CHASE-DB1:** 512² → **768²**, boundary-loss `kernel_size=7`.
- **DRIVE:** 512² → **576²**, warm-start **ISIC-2018** (thay vì CHASE-DB1), `kernel_size=7`.
- Dùng decoder canonical (`MaskGuidedLGAG` + `CascadedMSCAMDecoderBlock`) để khớp checkpoint ISIC-2018.

## 2. Phân loại bằng chứng
- `TRAIN`: toàn bộ 6 lần chạy (3 dataset × 2 backbone). Log xác nhận load ISIC-2018 **Missing: 0, Unexpected: 0** cho cả 6 run. Chọn checkpoint theo Val Macro Dice, hiệu chỉnh $T_{\text{opt}}$ trên Val, đánh giá trên Test độc lập.
- Không có kết quả `INVALID` trong phase này. Kết quả DRIVE Phase 8 (0.2356) vẫn `INVALID`.

## 3. Kết quả (Calibrated 4-View TTA, Test)

| Dataset | Backbone | Split | Best Ep | $T_{\text{opt}}$ | SV Dice | TTA Dice | **Cal Dice** | Cal Median | mIoU | Sens | Prec | HD95 | BF2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| EM | ConvFormer-S18 | 153/6/6 | 46 | 0.52 | 0.9504 | 0.9517 | 0.9516 | 0.9504 | 0.9078 | 0.9442 | 0.9596 | 2.45 | 0.955 |
| EM | PVTv2-B2 | 153/6/6 | 60 | 0.74 | 0.9530 | 0.9535 | **0.9542** | 0.9525 | 0.9126 | 0.9498 | 0.9592 | 2.05 | 0.958 |
| CHASE-DB1 | ConvFormer-S18 | 20/4/4 | 41 | 0.46 | 0.8246 | 0.8239 | 0.8238 | 0.8179 | 0.7006 | 0.8545 | 0.7973 | 4.12 | 0.908 |
| CHASE-DB1 | PVTv2-B2 | 20/4/4 | 46 | 0.59 | 0.8183 | 0.8261 | **0.8269** | 0.8213 | 0.7052 | 0.8426 | 0.8142 | 4.22 | 0.911 |
| DRIVE | ConvFormer-S18 | 12/2/6 | 38 | 0.53 | 0.7816 | 0.7817 | 0.7833 | 0.7854 | 0.6443 | 0.8337 | 0.7432 | 6.72 | 0.913 |
| DRIVE | PVTv2-B2 | 12/2/6 | 49 | 0.54 | 0.7749 | 0.7851 | **0.7855** | 0.7938 | 0.6480 | 0.8022 | 0.7745 | 7.25 | 0.909 |

## 4. So sánh với Phase 17 (Cal Dice)

| Dataset | Backbone | Phase 17 | Phase 18 | Δ | Ghi chú |
|---|---|---|---|---|---|
| EM | ConvFormer-S18 | 0.9531 (20/4/141) | 0.9516 | — | Khác protocol, không so trực tiếp |
| EM | PVTv2-B2 | 0.9518 (20/4/141) | 0.9542 | — | Khác protocol |
| CHASE-DB1 | ConvFormer-S18 | 0.8023 | 0.8238 | **+2.15 pp** | |
| CHASE-DB1 | PVTv2-B2 | 0.8125 | 0.8269 | **+1.44 pp** | Vượt v2 (0.7902) +3.67 pp |
| DRIVE | ConvFormer-S18 | 0.7932 | 0.7833 | **−0.99 pp** | Giảm |
| DRIVE | PVTv2-B2 | 0.7937 | 0.7855 | **−0.82 pp** | Giảm |

## 5. Thảo luận
1. **CHASE-DB1 cải thiện rõ** (+1.4 đến +2.2 pp ở cả 2 backbone) — phù hợp giả thuyết độ phân giải cao giữ được mạch 1–2 px. Tuy nhiên đã đổi đồng thời 4 yếu tố (độ phân giải, kernel loss, decoder canonical, warm-start ISIC) nên **chưa thể quy công cho riêng yếu tố nào**; cần ablation.
2. **DRIVE không cải thiện, còn giảm ~0.8–1.0 pp.** Warm-start ISIC-2018 (ảnh da liễu, vùng tổn thương lớn) **không tốt hơn** warm-start CHASE-DB1 (cùng miền võng mạc) ở Phase 17. Giả thuyết: ưu thế transfer cùng miền > transfer từ checkpoint tốt nhưng khác miền. Mức chênh nhỏ trên N=6 test, 1 seed → không có ý nghĩa thống kê.
3. **EM:** đã về protocol 6 lát test; Dice 0.952–0.954, vẫn **thấp hơn v2 Phase 8 (0.9660)**. Lưu ý chưa xác minh 6 lát test trùng hệ thống với Phase 8 (v2 dùng 159/–/6, không có val).
4. **Cảnh báo protocol DRIVE:** dataset Kaggle chỉ có mask cho ID 21–40 (tập *training* chính thức) → split 12/2/6 lấy từ training set, **không phải official test split (ID 01–20)**. Không so trực tiếp với số liệu DRIVE trong literature.
5. Precision vi mạch vẫn thấp (0.74–0.81) so với Sensitivity (0.80–0.85): mô hình có xu hướng over-segment biên mạch.

## 6. Hạn chế
1 seed; test rất nhỏ (EM 6, CHASE 4, DRIVE 6); val nhỏ (DRIVE 2 ảnh) khiến $T_{\text{opt}}$ dễ overfit; thay đổi nhiều yếu tố cùng lúc.

## 7. Bước tiếp theo
- DRIVE: chạy 576² + kernel 7 + **warm-start từ checkpoint CHASE-DB1 768² mới** (cùng miền) để tách hiệu ứng warm-start.
- Ablation CHASE-DB1: 768² vs 512² với cùng warm-start để định lượng đóng góp độ phân giải.
- Patch-based training (vd 256² patches native resolution) — chuẩn phổ biến cho DRIVE/CHASE trong literature.
- Nếu cần so với SOTA: dùng official DRIVE test (cần nguồn có mask ID 01–20), và 3 seed.

## 8. Artifacts
`GDF-Netv4/results/retrain_em_chase_drive/` — `master_retrain_summary.csv`, `benchmark_summary_*`, `benchmark_raw_*`, `training_history_*`, kernel log. Checkpoint `.pth` (6 file, ~120 MB/file) giữ local, không commit.
