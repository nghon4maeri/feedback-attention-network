# [Phase 17 – 29/10/2026] GDF-Netv4 Unified Multi-Dataset Benchmark (ConvFormer-S18 vs PVTv2-B2 trên 6 tập dữ liệu ngoài ISIC)

## 1. Phân loại bằng chứng

- `TRAIN`: Toàn bộ 12 lần chạy (6 dataset × 2 backbone) huấn luyện mới từ ImageNet-pretrained trên Kaggle T4, chọn checkpoint theo Val Macro Dice (RAW/EMA), hiệu chỉnh $T_{\text{opt}}$ trên Val, đánh giá trên Test độc lập.
- `INVALID` (Phase 8): Kết quả DRIVE của GDF-Netv2 (`0.2356`) do lỗi ghép cặp ảnh–nhãn theo thứ tự chữ cái (`01_test.tif` ↔ `21_manual1.gif`). Không dùng để so sánh.
- **Không so sánh trực tiếp được** (khác protocol): EM-Dataset (xem mục 4), HD95 giữa v2 và v4 (v2 dùng Hausdorff max, v4 dùng percentile 95 — đã ghi nhận ở Phase 10).

## 2. Thiết lập

| Dataset | Kích thước | Split (Train/Val/Test) | Epochs | Batch |
|---|---|---|---|---|
| CVC-ClinicDB | 384² | 490 / 61 / 61 (seed 42, giống Phase 8) | 60 | 8 |
| Kvasir-SEG | 384² | 800 / 100 / 100 (giống Phase 8) | 60 | 8 |
| DSB-2018 | 256² | 536 / 67 / 67 (giống Phase 8) | 60 | 16 |
| EM-Dataset | 512² | **20 / 4 / 141** (KHÁC Phase 8: 159 / – / 6) | 60 | 4 (accum 2) |
| CHASE-DB1 | 512² | 20 / 4 / 4 (giống Phase 8) | 60 | 4 (accum 2) |
| DRIVE | 512² | 12 / 2 / 6 (ghép cặp theo ID 21–40, warm-start từ CHASE-DB1) | 50 | 4 (accum 2) |

Kiến trúc: GDF-Netv4 đầy đủ (MS-FDF [1,3,5], FeedbackModulator, AttentionGate, Deep Supervision, StructureLoss + SoftDice, WarmupEMA 0.995, ImageNet Normalize), CosineAnnealingWarmRestarts (T0=20, Tmult=2), AdamW (backbone 1e-4 / head 3e-4).

Script: [`run_gdfnet_v4_multibench_group1_polyp_nuclei.py`](../../GDF-Netv4/training/run_gdfnet_v4_multibench_group1_polyp_nuclei.py), [`run_gdfnet_v4_multibench_group2_em_retinal.py`](../../GDF-Netv4/training/run_gdfnet_v4_multibench_group2_em_retinal.py).

## 3. Kết quả (Test, Mean Dice; Calibrated = 4-View TTA @ $T_{\text{opt}}$ từ Val)

| Dataset | Backbone | Best Ep | $T_{\text{opt}}$ | SV | TTA | **Calibrated** | Median | mIoU | Sens | Prec | ASD (px) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CVC-ClinicDB | ConvFormer-S18 | 32 | 0.49 | 0.9382 | 0.9418 | 0.9411 | 0.9538 | 0.8956 | 0.9591 | 0.9291 | 3.20 |
| CVC-ClinicDB | PVTv2-B2 | 52 | 0.48 | 0.9426 | **0.9462** | 0.9432 | 0.9576 | 0.8963 | 0.9595 | 0.9310 | 2.96 |
| Kvasir-SEG | ConvFormer-S18 | 53 | 0.45 | 0.9393 | 0.9392 | **0.9388** | 0.9676 | 0.8964 | 0.9545 | 0.9355 | 4.76 |
| Kvasir-SEG | PVTv2-B2 | 21 | 0.45 | 0.9343 | 0.9388 | 0.9358 | 0.9629 | 0.8883 | 0.9606 | 0.9249 | 4.74 |
| DSB-2018 | ConvFormer-S18 | 55 | 0.75 | 0.8959 | 0.8980 | **0.9044** | 0.9157 | 0.8345 | 0.9177 | 0.9016 | 2.43 |
| DSB-2018 | PVTv2-B2 | 52 | 0.75 | 0.8919 | 0.8916 | 0.9007 | 0.9226 | 0.8296 | 0.9255 | 0.8896 | 2.71 |
| EM-Dataset* | ConvFormer-S18 | 50 | 0.63 | 0.9520 | 0.9529 | **0.9531** | 0.9546 | 0.9107 | 0.9498 | 0.9571 | 1.25 |
| EM-Dataset* | PVTv2-B2 | 52 | 0.69 | 0.9495 | 0.9492 | 0.9518 | 0.9533 | 0.9084 | 0.9519 | 0.9523 | 1.31 |
| CHASE-DB1 | ConvFormer-S18 | 41 | 0.55 | 0.7987 | 0.8002 | 0.8023 | 0.8009 | 0.6702 | 0.8385 | 0.7712 | 1.07 |
| CHASE-DB1 | PVTv2-B2 | 60 | 0.73 | 0.7932 | 0.7983 | **0.8125** | 0.8092 | 0.6845 | 0.8348 | 0.7937 | 1.02 |
| DRIVE | ConvFormer-S18 | 44 | 0.65 | 0.7856 | 0.7886 | 0.7932 | 0.7910 | 0.6576 | 0.8065 | 0.7832 | 1.14 |
| DRIVE | PVTv2-B2 | 39 | 0.68 | 0.7827 | 0.7847 | **0.7937** | 0.7928 | 0.6584 | 0.8153 | 0.7767 | 1.19 |

\* EM-Dataset dùng protocol khác Phase 8 (xem mục 4).

## 4. Lỗi phát hiện trong Phase 17 (EM-Dataset split)

Script Group 2 giả định volume EM có 30 lát cắt, nhưng thực tế có **165 lát**. Vì vậy `indices[:20]` / `[20:24]` / `[24:]` cho ra **20 train / 4 val / 141 test**, thay vì giữ nguyên 6 lát test của Phase 8 như comment trong code ghi. Hệ quả:
- Kết quả EM của v4 (`0.9531`) **không so sánh trực tiếp** được với v2 (`0.9660`, train 159 lát / test 6 lát).
- Protocol mới khó hơn nhiều (ít dữ liệu train hơn ~8 lần) nhưng test lớn hơn ~23 lần → ước lượng ổn định hơn. Có thể trình bày như thiết lập "low-data" nếu ghi rõ.
- Nếu cần so sánh công bằng với v2: chạy lại EM với split 159/6 (hoặc chạy lại v2 với split 20/4/141).

## 5. Đối chiếu với GDF-Netv2 (Phase 8) trên cùng split

| Dataset | v2 (150 ep, SV) | v4 tốt nhất (60 ep, Calibrated) | Δ | Ghi chú |
|---|---|---|---|---|
| CVC-ClinicDB | 0.9437 | 0.9432 (PVTv2-B2) | −0.05 pp | **Không cải thiện** (v4 TTA @0.5 = 0.9462) |
| Kvasir-SEG | 0.9181 | 0.9388 (ConvFormer-S18) | +2.07 pp | Cải thiện rõ |
| DSB-2018 | 0.8912 | 0.9044 (ConvFormer-S18) | +1.32 pp | Cải thiện |
| CHASE-DB1 | 0.7902 | 0.8125 (PVTv2-B2) | +2.23 pp | N=4 test, rất nhỏ |
| DRIVE | 0.2356 (`INVALID`) | 0.7937 (PVTv2-B2) | – | v2 bị lỗi ghép cặp |
| EM-Dataset | 0.9660 | 0.9531 | – | Khác protocol |

## 6. Nhận định

1. **Không có backbone thắng tuyệt đối.** ConvFormer-S18 nhỉnh hơn ở Kvasir, DSB-2018, EM; PVTv2-B2 nhỉnh hơn ở CVC, CHASE-DB1, DRIVE. Chênh lệch đa số < 1 pp, trên test nhỏ và 1 seed → **chưa đủ để kết luận có ý nghĩa thống kê**. Kết luận an toàn: GDF-Netv4 hoạt động tốt với cả CNN lẫn Transformer backbone (backbone-agnostic).
2. **Calibration không phải lúc nào cũng có lợi.** Có ích rõ ở DSB-2018 (+0.6–0.9 pp, $T=0.75$) và mạch máu (+0.5–1.4 pp); nhưng giảm nhẹ ở CVC/Kvasir so với TTA @0.5. Với tập Val nhỏ (DRIVE 2 ảnh, CHASE 4 ảnh), $T_{\text{opt}}$ dễ overfit.
3. **ConvFormer-S18 có Precision cao hơn** ở 4/6 tập — nhất quán với quan sát trên ISIC-2018 (Phase 16).
4. **Hạn chế**: 1 seed; test CHASE (4) và DRIVE (6) quá nhỏ; split random (không phải official split cho Kvasir/CVC); 60 epoch (v2 dùng 150).

## 7. Việc tiếp theo đề xuất

- Chạy lại EM-Dataset với split 159/6 để so sánh công bằng với v2 (hoặc giữ protocol mới, ghi rõ trong paper).
- Kiểm định Wilcoxon theo từng ảnh ConvFormer vs PVTv2 trên CVC/Kvasir/DSB/EM (dữ liệu raw đã có).
- Nếu nhắm Q1: thêm 3 seed hoặc 5-fold cho các tập nhỏ; đối chiếu SOTA đã công bố (PraNet, Polyp-PVT, EMCAD…) theo đúng split chuẩn.

## Artifacts

- `GDF-Netv4/results/multibench_group1/` (CVC, Kvasir, DSB-2018)
- `GDF-Netv4/results/multibench_group2/` (EM, CHASE-DB1, DRIVE)
- `master_benchmark_group1.csv`, `master_benchmark_group2.csv`
