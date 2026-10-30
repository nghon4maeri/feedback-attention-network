# Prompt — 2026-10-30: Canonical GDF-Netv4 (ConvFormer-S18) on CVC-ClinicDB, Kvasir-SEG, DSB-2018

## Mục tiêu (Objective)
Đồng nhất 100% kiến trúc của toàn bộ bảng dữ liệu trong paper xung quanh mô hình đề xuất **`GDF-Netv4-ConvFormerS18`**:
- Ở Phase 17, 3 tập dữ liệu CVC-ClinicDB, Kvasir-SEG, và DSB-2018 được huấn luyện với decoder rút gọn pilot (`AttentionGate` + `ConvBlock`).
- Tiến hành huấn luyện lại 3 tập này với kiến trúc chuẩn **Canonical GDF-Netv4** (`MaskGuidedLGAG` + `CascadedMSCAMDecoderBlock`) và khởi tạo warm-start từ checkpoint ISIC-2018 A100 (`GDFNetV4_ConvFormerS18_ISIC2018_best.pth`).
- Tập trung duy nhất vào backbone chủ lực **`ConvFormer-S18`** để tối ưu hóa tài nguyên tính toán và chuẩn bị bảng thực nghiệm nhất quán cho bài báo khoa học.

## Thiết kế Thực nghiệm (Experimental Design)
- **Tập dữ liệu & Giao thức chia tập (Seed=42 giống hệt Phase 8 và Phase 17)**:
  1. **CVC-ClinicDB**: 490 Train / 61 Val / 61 Test, độ phân giải $384 \times 384$, batch size 8, 60 epochs.
  2. **Kvasir-SEG**: 800 Train / 100 Val / 100 Test, độ phân giải $384 \times 384$, batch size 8, 60 epochs.
  3. **DSB-2018**: 536 Train / 67 Val / 67 Test, độ phân giải $256 \times 256$, batch size 16, 60 epochs.
- **Kiến trúc**:
  - `HierarchicalBackboneV4` (`convformer_s18.sail_in22k_ft_in1k_384`)
  - `MultiScaleFrequencyFilter` [1, 3, 5]
  - `FeedbackModulator` (100% khả vi)
  - `MaskGuidedLGAG` (Local-Global Attention Gate)
  - `CascadedMSCAMDecoderBlock` (Multi-Scale Channel Attention Module)
  - `DeepSupervisionDecoder` + `BoundaryAwareFrequencyLoss` (`kernel_size=15`)
- **Khởi tạo**: Warm-start từ trọng số ISIC-2018 A100 (`namnguynnnn/gdfnetv4-isic-pretrained`).
- **Đánh giá**: 9 chỉ số lâm sàng ở 3 chế độ (Single-View, 4-View TTA, Val-Calibrated 4-View TTA).

## Ràng buộc (Constraints)
- Tuân thủ quy tắc Git trong `AGENTS.md`: commit và git push ngay lập tức.
- Phân biệt bằng chứng rõ ràng (TRAIN).
