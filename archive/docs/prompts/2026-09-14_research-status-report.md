# PROMPT — Báo cáo tổng quan tình hình nghiên cứu FANet (Senior AI Research Engineer & Tech Lead Review)

## Ngày: 2026-09-14

## Mục tiêu phiên làm việc
Rà soát toàn bộ codebase và các tài liệu (README, docs, configs, logs, research notes) trong workspace `FANet` để lập báo cáo kỹ thuật toàn diện từ góc nhìn Senior AI Research Engineer và Technical Lead chuyên về Computer Vision / Deep Learning.

## Phạm vi rà soát
1. **Bối cảnh & Mục tiêu Nghiên cứu (Research Context):**
   - Bài toán y sinh (Polyp segmentation trên Kvasir-SEG / sessile subset).
   - Giả thuyết cốt lõi (Cross-epoch feedback mask, hard attention vs soft attention).
2. **Phân tích Kiến trúc Mạng (Network Architecture Analysis):**
   - Định nghĩa model (`src/fanet/models/fanet.py`, `src/fanet/models/blocks.py`).
   - Module `MixPool`: cơ chế gating (binary, STE, soft), dual-path, split-transform-merge (conv1/conv2), input/output shape và flow.
   - Tích hợp với U-Net encoder-decoder (4 stages, skip connections, output head).
   - Các biến thể / ablations đã cài đặt.
3. **Quy trình Huấn luyện & Dữ liệu (Pipeline & Data Flow):**
   - Dataset & Augmentation (`src/fanet/data/dataset.py`, Albumentations).
   - Loss functions (`src/fanet/losses.py`: DiceBCELoss, NegativeAreaDiceBCELoss, WSDice, FarWeightedIoUBCELoss, TverskyLoss, Phase6AsymmetricBCELoss).
   - Optimizer & LR scheduler (Adam, ReduceLROnPlateau), hyperparams.
   - Metrics đánh giá (`src/fanet/metrics.py`: Dice/F1, Jaccard/IoU, Precision, Recall, FPR/per-image, Specificity).
4. **Trạng thái Thí nghiệm Hiện tại (Current Status & Benchmarks):**
   - Đã qua Phase 1 đến Phase 6: từ baseline reproduce ban đầu, phát hiện 2 điểm yếu cấu trúc, ablation STE/dual-path, root cause oracle FP, loss-side Phase 5, đến lưới factorial 2×2 Phase 6.
   - Các kết quả định lượng chính (T00, T0N, TC, TD, TA, TB) và phân loại bằng chứng (TRAIN vs FROZEN EVAL vs INVALID).
   - Kết luận khoa học chính: feedback mask-at-input chặn (blunt) hiệu ứng giảm FP của asymmetric loss (interaction +4.17pp).
5. **Rào cản Kỹ thuật & Bước đi tiếp theo (Bottlenecks & Next Steps):**
   - Nhận diện các điểm nghẽn (gradient cutoff ở binary threshold, over-segmentation/FP xa biên GT 40.7px, feedback loop trapping model trong vùng false positive cũ, disparity giữa soft-dice và binary evaluation, dataset scale).
   - Đề xuất 3 hành động kỹ thuật ưu tiên cần làm ngay.

## Ràng buộc & Quy chuẩn
- Trích dẫn đường dẫn file cụ thể (`path/to/file.py:line`) cho mọi luận điểm.
- Tuân thủ quy định phân biệt loại bằng chứng theo `AGENTS.md` (TRAIN / FROZEN EVAL / INVALID).
- Không tự chạy GPU khi chưa có yêu cầu; tôn trọng kết quả thực nghiệm Kaggle đã hoàn tất.
