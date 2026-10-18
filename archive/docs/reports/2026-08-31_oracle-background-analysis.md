# [CN – 31/08/2026]

## Mục tiêu tuần này

Hoàn thành Phase 1 của hướng nghiên cứu A+B: phân tích pixel-wise giữa oracle feedback và prediction feedback để xác nhận giả thuyết "dư địa của feedback nằm ở thông tin background" — tiền đề cho việc thiết kế dual-path feedback (hướng B) và hồi sinh fmask (hướng A).

## Done

- [x] Viết script `analysis/oracle_analysis.py` — so sánh 2 loại feedback (prediction vs ground truth) pixel-wise qua 3 iterations trên 40 ảnh val
- [x] Chạy phân tích trên checkpoint 200 epochs Kaggle — `logs/oracle_analysis_log.txt`
- [x] Phân tích 4 khía cạnh: (A) error decomposition FP/FN, (B) vị trí oracle khác prediction, (C) khoảng cách lỗi tới biên ground truth, (D) dice gap qua iterations
- [x] Lưu kết quả chi tiết — `results/oracle_analysis.csv`, `results/oracle_summary.json`

## Findings quan trọng

### 1. Over-segmentation chiếm ưu thế rõ rệt (FP/FN ratio = 2.48)

| Chỉ số | Giá trị |
|---|---|
| FP rate (dự đoán thừa) | **0.1173** (11.7% diện tích ảnh) |
| FN rate (dự đoán thiếu) | 0.0473 (4.7%) |
| Tỷ lệ FP/FN | **2.48** — thừa gấp 2.5 lần thiếu |

Prediction feedback của model dự đoán foreground thừa gần 12% diện tích ảnh — phù hợp với quan sát precision thấp (0.22-0.26) từ tuần trước. Đây là xác nhận định lượng đầu tiên rằng vấn đề chính của feedback là **over-segmentation**, không phải under-segmentation.

### 2. Oracle khác prediction chủ yếu ở vùng prediction ĐANG THỪA (84.6%)

Trong 6.98% pixel mà output của oracle khác output của prediction feedback:
- **84.6% nằm ở vùng prediction-FP** (model dự đoán foreground nhưng GT là background)
- 6.5% nằm ở vùng prediction-FN (model bỏ sót)
- 6.4% ở vùng cả hai đã đúng

Điều này trực tiếp trả lời câu hỏi "oracle cung cấp thông tin gì mà prediction feedback thiếu": **thông tin background chính xác**. Oracle "dạy" model chỗ nào không phải polyp — đúng cái mà hard binary feedback hiện tại không làm được (feedback chỉ chứa foreground, background bị zero hoá).

### 3. Lỗi FP nằm XA biên (40.7px), lỗi FN nằm SÁT biên (8.6px)

- FP errors: trung bình cách biên ground truth **40.72 px** — over-segmentation trải sâu ra nền, không phải chỉ sai nhẹ ở rìa
- FN errors: trung bình **8.58 px** — thiếu chủ yếu ở vùng biên polyp

Hệ quả thiết kế: cơ chế mới cần "dọn" cả vùng nền xa polyp, không chỉ tinh chỉnh biên. Background feedback phải bao phủ vùng nền rộng, không chỉ vùng lân cận polyp.

### 4. Oracle gap tái xác nhận (0.093 sau 3 iterations)

| Iter | Pred feedback | Oracle feedback | Gap |
|---:|---:|---:|---:|
| 1 | 0.2430 | 0.2430 | 0.0000 |
| 2 | 0.2365 | 0.3312 | **+0.0947** |
| 3 | 0.2379 | 0.3312 | +0.0933 |

Gap mở ra ngay từ iteration 2 và giữ nguyên — dư địa ổn định, có thể khai thác được nếu feedback mang thông tin background.

## Experiments

| Giả thuyết | Config | Kết quả | Link log |
|-----------|--------|---------|----------|
| Oracle feedback tốt hơn prediction feedback chủ yếu nhờ thông tin background (vùng prediction-FP) | 2 feedback types x 3 iters x 40 ảnh, checkpoint 200ep | XÁC NHẬN: 84.6% khác biệt nằm ở vùng FP; FP/FN = 2.48; FP cách biên 40.7px | `logs/oracle_analysis_log.txt`, `results/oracle_summary.json` |

## Will Do (On going)

- [ ] Phase 2 — Hướng A: cài 2 variants hồi sinh fmask (soft gating `max(fmask, m)` và straight-through estimator) vào `blocks.py`, chạy lại grad flow để xác nhận fmask có gradient
- [ ] Phase 3 — Hướng B: cài dual-path feedback `[m_fg, m_bg]` dựa trên kết quả Phase 1 (m_bg = vùng background tin cậy, gate = fg_attention OR NOT bg_attention)
- [ ] Phase 4 — Train end-to-end ablation 2x2 (gốc / +A / +B / +A+B) trên cùng split
- [ ] Phase 0 (song song) — xác minh checkpoint 200ep: align batch/split/augmentation
- [ ] Phase 5 — Mở rộng đa dataset (full Kvasir-SEG, CVC-ClinicDB, DRIVE, CHASE)
- [ ] Chưa mở rộng sang deep supervision — giữ trọng tâm A+B

## Any Stuck / Open Questions

- Không có blocker mới. Lưu ý kỹ thuật: script oracle analysis cần fix 2 lỗi nhỏ trước khi chạy (UnicodeEncodeError khi print tiếng Việt trên Windows console, và FP/FN ratio tính mean của per-image ratios gây giá trị bùng nổ ~1e18 — đã sửa thành tổng FP/tổng FN)
- Câu hỏi mở cho Phase 3: ngưỡng tin cậy cho m_bg là bao nhiêu (đối xứng với m_fg = p > 0.5)? Có thể cần sweep tương tự như gated-tau tuần trước

## Đính kèm link chi tiết

- Script Phase 1: `analysis/oracle_analysis.py` — log: `logs/oracle_analysis_log.txt`
- Kết quả: `results/oracle_analysis.csv`, `results/oracle_summary.json`
- Checkpoint: `checkpoints/checkpoint_200ep_kaggle.pth`
- Nền tảng: `results/confgated_summary.json` (tuần trước), `results/test_results.csv`
- Template: `docs/reports/_TEMPLATE.md` — Index: `docs/reports/README.md`
