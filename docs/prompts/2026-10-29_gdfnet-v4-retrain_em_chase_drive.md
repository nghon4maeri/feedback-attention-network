# Prompt — 2026-10-29: Retraining GDF-Netv4 on EM-Dataset, CHASE-DB1, and DRIVE

## Mục tiêu (Objective)
Giải quyết triệt để vấn đề sụt giảm hiệu năng trên 3 tập dữ liệu vi cấu trúc và vi mạch máu (EM-Dataset, CHASE-DB1, DRIVE) bằng cách huấn luyện lại với kiến trúc chuẩn **Canonical GDF-Netv4** (`MaskGuidedLGAG` + `CascadedMSCAMDecoderBlock`), đồng thời áp dụng các cải tiến tối ưu hóa đặc thù cho từng miền dữ liệu:
1. **EM-Dataset (Hiển vi điện tử màng tế bào)**:
   - Sửa lỗi phân chia tập dữ liệu nghiêm trọng từ runner trước (`test_idx = indices[24:]` khiến tập train chỉ có 20 lát cắt và test lên đến 141 lát cắt trong volume 165 lát).
   - Thiết lập lại phân chia chuẩn tương thích Phase 8: **153 Train / 6 Val / 6 Test**.
   - Khởi tạo warm-start từ trọng số pretrain ISIC-2018 A100.
2. **CHASE-DB1 (Vi mạch võng mạc mắt)**:
   - Nâng độ phân giải huấn luyện và đánh giá từ $512 \times 512$ lên độ phân giải cao **$768 \times 768$** (bảo toàn các mao mạch 1–2 pixel vốn bị xóa nhòa khi downsample).
   - Điều chỉnh kernel của boundary loss (`kernel_size=7` thay vì 15) nhằm tối ưu hóa đường viền vi mạch siêu mảnh.
   - Khởi tạo warm-start từ ISIC-2018.
3. **DRIVE (Vi mạch võng mạc mắt)**:
   - Nâng độ phân giải lên **$576 \times 576$** (gần như độ phân giải gốc $565 \times 584$).
   - Sửa triệt để lỗi ghép cặp theo thứ tự từ điển (`INVALID` pairing của Phase 8 ghép `01_test.tif` với `21_manual1.gif`) bằng thuật toán ghép cặp ID số nguyên 2 chữ số.
   - Sử dụng trực tiếp trọng số pretrain từ **ISIC-2018** (`GDFNetV4_ConvFormerS18_ISIC2018_best.pth` và `GDFNetV4_PVTv2B2_ISIC2018_best.pth`) làm warm-start theo đúng chỉ đạo của user.

## Thiết kế Thực nghiệm (Experimental Design)
- **Kiến trúc đồng nhất Canonical GDF-Netv4**:
  - `HierarchicalBackboneV4` (ConvFormer-S18 vs PVTv2-B2)
  - `MultiScaleFrequencyFilter` [1, 3, 5]
  - `FeedbackModulator` (100% differentiable)
  - `MaskGuidedLGAG` (Local-Global Attention Gate)
  - `CascadedMSCAMDecoderBlock` (Multi-Scale Channel Attention Module)
  - `DeepSupervisionDecoder` + `BoundaryAwareFrequencyLoss`
- **Bộ trọng số Pretrained ISIC-2018**:
  - `GDFNetV4_ConvFormerS18_ISIC2018_best.pth` (118.3 MB, Colab A100 300 epochs)
  - `GDFNetV4_PVTv2B2_ISIC2018_best.pth` (123.1 MB, Colab A100 300 epochs)
- **Đánh giá**:
  - 9 chỉ số lâm sàng (`Dice`, `mIoU`, `Sensitivity`, `Specificity`, `Precision`, `FPR`, `HD95`, `ASD`, `BF2`) + Global Micro Dice/IoU.
  - 3 chế độ: Single-View, 4-View TTA, Val-Calibrated 4-View TTA.

## Ràng buộc (Constraints)
- Tuân thủ nghiêm ngặt Git workflow trong `AGENTS.md`: commit và git push ngay lập tức.
- Phân định rõ ràng bản chất bằng chứng: TRAIN vs FROZEN EVAL vs INVALID.
