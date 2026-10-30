# Prompt — 2026-10-30: Canonical GDF-Netv4 (ConvFormer-S18) with Heavy Data Augmentation on EM-Dataset, CHASE-DB1, and DRIVE

## Mục tiêu (Objective)
Áp dụng gói **Heavy Data Augmentation** chuyên sâu cho 3 tập vi cấu trúc và vi mạch máu (EM-Dataset, CHASE-DB1, DRIVE) xoay quanh mô hình đề xuất **`GDF-Netv4-ConvFormerS18`**:
1. Khắc phục hiện tượng hạn chế khái quát hóa trên tập mẫu nhỏ (DRIVE chỉ 12 ảnh train, CHASE-DB1 chỉ 20 ảnh train).
2. Tích hợp các phép biến đổi tăng cường mô phỏng lâm sàng:
   - `RandomResizedCrop` đa tỷ lệ (scale 0.70–1.0) mô phỏng phóng đại tiêu cự vi mạch / màng tế bào.
   - `CoarseDropout` (Cutout) chống phụ thuộc nhánh mạch lớn, kích thích mô hình học kết nối mạng mao quản mảnh.
   - `GaussianBlur` / `MotionBlur` mô phỏng rung giật chuyển động mắt và nhòe quang học.
   - `ColorJitter` + `RandomGamma` + `CLAHE` tăng cường độ phân giải tương phản biên.
   - `ElasticTransform` + `GridDistortion` tăng tính co giãn tự nhiên của cấu trúc sinh học.
3. Thử nghiệm so sánh đối đầu trên DRIVE giữa 2 chiến lược khởi tạo:
   - **Intra-domain warm-start:** Khởi tạo từ checkpoint CHASE-DB1 $768^2$ vừa huấn luyện (cùng miền vi mạch võng mạc mắt).
   - **Inter-domain warm-start:** Khởi tạo từ checkpoint ISIC-2018 (tổn thương da liễu A100).

## Thiết kế Thực nghiệm (Experimental Design)
- **Kiến trúc đồng nhất 100%:** Canonical GDFNetV4 (`MaskGuidedLGAG` + `CascadedMSCAMDecoderBlock`).
- **Backbone duy nhất:** `convformer_s18.sail_in22k_ft_in1k_384` (Pure-CNN Flagship).
- **Tập dữ liệu & Cấu hình:**
  - **EM-Dataset:** 153 Train / 6 Val / 6 Test ($512 \times 512$, 60 epochs).
  - **CHASE-DB1:** 20 Train / 4 Val / 4 Test ($768 \times 768$, loss `kernel_size=7`, 60 epochs).
  - **DRIVE:** 12 Train / 2 Val / 6 Test ($576 \times 576$, loss `kernel_size=7`, 50 epochs).
- **Đánh giá:** 9 chỉ số lâm sàng (Single-View, 4-View TTA, Val-Calibrated 4-View TTA).

## Ràng buộc (Constraints)
- Tuân thủ nghiêm ngặt Git workflow trong `AGENTS.md`.
