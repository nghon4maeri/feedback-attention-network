# Prompt — 2026-10-28: Phase 17 Unified Multi-Dataset GDF-Netv4 Benchmark (6 External Clinical Datasets)

## Mục tiêu (Objective)
Triển khai **Phương án B**: Xây dựng bảng thực nghiệm đa phương thức (Multi-Dataset Benchmark) **đồng nhất 100% kiến trúc `GDF-Netv4`** trên toàn bộ 6 tập dữ liệu lâm sàng bên ngoài (bổ sung cho tập chính ISIC-2018 đã hoàn tất ở Phase 15–16):
1. **CVC-ClinicDB** (Polyp nội soi đại trực tràng, $N=612$: 490 Train / 61 Val / 61 Test, $384 \times 384$)
2. **Kvasir-SEG** (Polyp tiêu hóa đa kích thước, $N=1,000$: 800 Train / 100 Val / 100 Test, $384 \times 384$)
3. **2018 Data Science Bowl (DSB-2018)** (Hiển vi nhân tế bào, $N=670$: 536 Train / 67 Val / 67 Test, $256 \times 256$)
4. **EM-Dataset** (Hiển vi điện tử màng tế bào thần kinh, 30 lát cắt 3D TIFF: 24 Train / 6 Test, $512 \times 512$)
5. **CHASE-DB1** (Vi mạch máu võng mạc mắt, $N=28$: 20 Train / 4 Val / 4 Test, $512 \times 512$)
6. **DRIVE** (Vi mạch máu võng mạc mắt, $512 \times 512$, huấn luyện kết hợp **Transfer Learning Warm-Start từ CHASE-DB1** để khắc phục hiện tượng suy thoái học từ scratch trên tập mẫu nhỏ).

## Phạm vi & Thiết kế Thực nghiệm (Scope & Experimental Design)
- **Kiến trúc đồng nhất `GDF-Netv4`**:
  - Đối chiếu song song 2 họ Backbone chủ lực của dự án trên từng tập dữ liệu:
    1. **Pure-CNN Flagship:** `convformer_s18.sail_in22k_ft_in1k_384` (`GDF-Netv4-ConvFormerS18`)
    2. **Transformer Flagship:** `pvt_v2_b2` (`GDF-Netv4-PVTv2B2`)
  - Giữ nguyên 100% các module lõi của GDF-Netv4: `MultiScaleFrequencyFilter [1, 3, 5]`, `FeedbackModulator` (100% differentiable), `AttentionGate`, `DeepSupervisionDecoder`, `StructureAndMacroDiceLoss`, `WarmupModelEMA(0.998)`, và `ImageNet Normalization`.
- **Chế độ Đánh giá (3 Modes / 9 Clinical Metrics)**:
  - Đánh giá trên tập Test độc lập của từng dataset với đầy đủ 9 chỉ số lâm sàng (`Dice`, `mIoU`, `Sensitivity`, `Specificity`, `Precision`, `FPR`, `HD95`, `ASD`, `BF2`) cùng Global Micro Dice/IoU ở 3 chế độ:
    1. **Single-View ($T = 0.50$)**
    2. **4-View Flip TTA ($T = 0.50$)**
    3. **Val-Calibrated 4-View TTA ($T_{\text{opt}}$)** (tối ưu ngưỡng trên tập Validation của chính bộ dữ liệu đó).

## Ràng buộc (Constraints)
- Tuân thủ tuyệt đối quy trình Git trong `AGENTS.md`: `git status` -> `git diff` -> `git add` -> `git commit` -> `git push` sau mỗi lần tạo/sửa file code hoặc báo cáo.
- Thực thi trên hạ tầng Kaggle GPU Tesla T4, thu hồi đầy đủ file CSV (`benchmark_summary_*`, `benchmark_raw_*`, `training_history_*`) và viết báo cáo nghiệm thu Phase 17 theo mẫu `docs/reports/_TEMPLATE.md`.
