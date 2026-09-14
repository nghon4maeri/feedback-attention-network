# [Tuần 4 – 14/09/2026] Báo cáo Cơ chế "Feedback Trap" (Phase 6)

## Ký hiệu (Notation)

> Mục BẮT BUỘC cho mọi report. Khai báo các ký hiệu/viết tắt dùng trong report này, kể cả ký hiệu kế thừa từ report trước (không bắt người đọc đọc lại chuỗi report cũ). Có thể sao chép các mục quen thuộc bên dưới rồi bổ sung mục mới.

- `M00`: No-Feedback + Base Loss (`T0N`)
- `M01`: No-Feedback + Asymmetric Loss (`TD`)
- `M10`: Feedback + Base Loss (`T00`)
- `M11`: Feedback + Asymmetric Loss (`TC`)
- `STE`: Straight-Through Estimator · `BN`: BatchNorm · `fmask`: nhánh mask attention của MixPool
- `FP` / `FN`: False positive / False negative · `pp`: điểm phần trăm
- `Asymmetric Loss`: Hàm mất mát bất đối xứng (dùng Tversky để phạt nặng FP)
- `Saturation rate`: Tỷ lệ pixel có xác suất dự đoán cực đoan ($p < 0.01$ hoặc $p > 0.99$)

## Mục tiêu tuần này

Tổng hợp và phân tích bản chất của hiện tượng "Feedback Trap" trong FANet gốc, làm rõ bằng thực nghiệm lý do tại sao vòng lặp mask-at-input lại chặn đứng khả năng sửa lỗi vẽ thừa của hàm Asymmetric Loss.

## Done

- [x] Chạy thực nghiệm lưới 2x2 (Phase 6) để phát hiện sự tương tác giữa Feedback Loop và Loss (interaction effect +4.17pp).
- [x] Khởi tạo và chạy script chẩn đoán offline `analysis/diagnose_feedback_drift.py`.
- [x] Phân tích BatchNorm Drift, Feature Collapse, và Gradient Attenuation trên tập Validation (40 ảnh).
- [x] Lập báo cáo kỹ thuật "Anatomy of a Failure" chuẩn bị xuất bản bài báo khoa học.

## Findings quan trọng

**1. Vấn đề cốt lõi của FANet nguyên bản**
- **Nút thắt Gradient (Zero Gradient):** Nhánh MixPool do sử dụng hàm ngưỡng cứng (hard thresholding) để tách nền và vật thể nên bị chặn toàn bộ gradient truyền ngược. Nhánh tự học mask (`fmask`) hoàn toàn vô tác dụng.
- **Bệnh "Vẽ thừa" (Over-segmentation):** Phân tích Oracle cho thấy 84.6% tổng lỗi của mô hình nằm ở FP (lan sang vùng background xa tới 40.7 pixel).

**2. Quá trình kiểm định & Bước ngoặt khoa học (Phase 6)**
- Đã thử nghiệm **6 hướng can thiệp** (STE, Dual-path, Far-weighted loss, WSDice...) nhưng đa phần thất bại (lỗi nặng hơn hoặc mô hình bị collapse đoán toàn bộ nền).
- **Phát hiện lớn nhất (Thực nghiệm lưới 2x2):** Hàm Tversky (Asymmetric Loss) giúp giảm mạnh **3.25pp lỗi FP** (p=0.005), NHƯNG tác dụng này **BỊ TRIỆT TIÊU HOÀN TOÀN** (interaction effect +4.17pp) khi BẬT vòng lặp Feedback.

**3. Cơ chế của "Feedback Trap" (từ chẩn đoán offline)**
Dưới đây là bằng chứng số liệu trực tiếp lý giải hiện tượng trên:

| Hiện tượng | Bằng chứng (từ script offline) | Hệ quả |
|-----------|-----------|--------|
| **BatchNorm Drift** | Lệch KL cực đại **$D_{KL} = 33.63$** tại `e1.r1.bn3`. Trung bình KL = 1.82 (M10 vs M00) | Vòng lặp ép phân phối feature trôi dạt mạnh ngay tại encoder đầu tiên, đóng đinh prior trước khi loss kịp sửa đổi. |
| **Prediction Saturation** | Tỷ lệ saturation tăng vọt từ **0.00% (M00)** lên **4.63% (M10)**. Bottleneck cosine sim giảm về 0.8045. | Mạng không bị "feature collapse" đơn giản, mà chuyển sang over-commitment: quá tự tin sai lệch ở vùng nền. |
| **Gradient Attenuation** | Gradient Norm truyền về mask giảm **50.5%** (từ 3247.49 của M00 xuống 1607.60 của M10). | Loss gradient bị bóp nghẹt tại vòng lặp, khiến Asymmetric Loss bất lực không thể dọn dẹp FP. |

## Experiments

| ID | Giả thuyết | Config | Kết quả | Link log |
|----|-----------|--------|---------|----------|
| P6_T00 | Baseline FB | FB + BCE | Over-segmentation nặng, gradient mask bị bóp | `checkpoints_phase4/ckpt_T00.pth` |
| P6_T0N | Baseline No-FB | No-FB + BCE | Gradient flow tốt, mask không gây bão hòa | `checkpoints_phase5/ckpt_T0N.pth` |
| P6_TD | Asym Loss | No-FB + Tversky | Giảm FP rất tốt (-3.25pp, p=0.005) | `checkpoints_phase6/ckpt_TD.pth` |
| P6_TC | FB Trap | FB + Tversky | Tác dụng của Tversky bị chặn đứng (Tương tác +4.17pp) | `checkpoints_phase6/ckpt_TC.pth` |

## Will Do (On going)

- [ ] Chuẩn bị kịch bản chạy Multi-seed (5 seeds) cho các cell M00, M01, M10, M11 để chuẩn hóa thống kê và triệt tiêu nghi ngờ về nhiễu dataset.
- [ ] Không cố hack SOTA: Viết outline paper khoa học dạng Negative-Result / Anatomy of a Failure (Tạm gọi: "The Feedback Trap").
- [ ] Cập nhật bảng kết quả cuối cùng vào repo.

## Any Stuck / Open Questions

- Không. Cơ chế thất bại đã được chứng minh tường tận về mặt toán học và thực nghiệm.

## Đính kèm link chi tiết

- Script chẩn đoán: `analysis/diagnose_feedback_drift.py`
- Kết quả raw: `diagnostics_output/`
- Report Phase 6 chi tiết: `docs/reports/2026-09-09_phase6-results.md`
