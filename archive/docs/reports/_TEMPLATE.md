# [Tuần N – dd/mm/yyyy]

> File mới: copy template này, đặt tên `yyyy-mm-dd_<slug>.md`, rồi cập nhật
> Timeline + Experiment Tracking trong [README.md](README.md).

## Ký hiệu (Notation)

> Mục BẮT BUỘC cho mọi report. Khai báo các ký hiệu/viết tắt dùng trong report
> này, kể cả ký hiệu kế thừa từ report trước (không bắt người đọc đọc lại chuỗi
> report cũ). Có thể sao chép các mục quen thuộc bên dưới rồi bổ sung mục mới.

- `X1`–`X6`: 6 hướng nghiên cứu (X2 = phạt FP ở loss, X4 = chẩn đoán STE, ...)
- `T00` / `T10` / `T01` / `T11` / `T0N`: cell thí nghiệm — `T` + gate
  (0 = binary, 1 = STE) + dual-path (0 = không, 1 = có) + `N` = no-feedback
- `TA` / `TB`: cell A (loss WSDice) / cell B (loss far-weighted wIoU+wBCE)
- `Gate 0` / `Gate 1` / `Gate 2`: cổng quyết định — no-feedback baseline /
  screening 1 seed / multi-seed
- `1a` / `1b`: nhiệm vụ verify công thức loss (1a = CFA-Net, 1b = WSDice)
- `STE`: straight-through estimator · `BN`: BatchNorm · `fmask`: nhánh mask attention của MixPool
- `wIoU` / `wBCE`: weighted IoU / weighted BCE · `WSDice`: weighted soft dice
- `CFA-Net`: Cross-level Feature Aggregation Network (PR 2023)
- `FP` / `FN`: false positive / false negative · `pp`: điểm phần trăm
- `CBL`: Conditional Boundary Loss (TIP 2023)

## Mục tiêu tuần này

(1–2 câu: đang giải quyết vấn đề gì, gắn với mục tiêu lớn của dự án)

## Done

- [x] Việc đã làm 1 — kèm bằng chứng: `logs/...`, `results/...`, script nào
- [x] Việc đã làm 2
- [ ] Việc dở dang (nếu có)

## Findings quan trọng

(Kết quả có số liệu cụ thể — bảng, metric, hiện tượng quan sát được)

| Hiện tượng | Bằng chứng | Hệ quả |
|-----------|-----------|--------|
| ... | ... | ... |

## Experiments

| ID | Giả thuyết | Config | Kết quả | Link log |
|----|-----------|--------|---------|----------|
| E… | … | … | … | `logs/...` |

## Will Do (On going)

- [ ] Công việc tiếp theo 1
- [ ] Công việc tiếp theo 2

## Any Stuck / Open Questions

- Không / Vướng mắc cụ thể + đã thử hướng nào

## Đính kèm link chi tiết

- Paper: …
- File/script: …
- Kết quả: …
