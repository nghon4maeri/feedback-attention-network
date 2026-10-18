# Thiết kế thí nghiệm Phase 2+3 (31/08/2026)

## Mô hình thí nghiệm: ablation 2x2

| | fmask-dead (hiện tại) | fmask-live (STE) |
|---|---|---|
| **feedback 1 kênh (m_fg)** | C00 = baseline FANet gốc | C10 = +A |
| **feedback 2 kênh (m_fg, m_bg)** | C01 = +B | C11 = +A+B |

Đánh giá tại inference: KHÔNG train lại ở phase này (đánh giá feedback mechanism qua iterative refinement trên checkpoint 200ep hiện có — cùng model weight cho mọi variants, chỉ đổi cơ chế feedback/mask đầu vào). Train end-to-end để riêng Will Do.

## Đơn vị thí nghiệm & dữ liệu

- 40 ảnh val (toàn bộ val set sessile Kvasir-SEG), kích thước 256x256
- Cùng checkpoint `checkpoints/checkpoint_200ep_kaggle.pth` cho mọi cell
- Cùng Otsu init ở iteration 1; feedback cập nhật từ prediction của chính variant đó

## Metric

**Chính**: Dice (F1) tại iteration cuối (4 iterations) + mean qua iterations
**Phụ**:
- IoU (Jaccard)
- Precision, Recall
- FP rate / FN rate (pixel-wise vs GT) — trực tiếp kiểm tra P_B1
- Mean error distance tới biên GT (FP, FN riêng)
- Oracle gap (Dice_oracle − Dice_variant) — đo độ "đóng" khoảng cách
- Help/hurt count vs none-feedback

## Ngưỡng quyết định (đã khai báo trong hypotheses)

- Bác bỏ P_A2/P_B2 nếu ΔDice < +0.005
- Bác bỏ P_B1 nếu FP rate giảm < 0.01
- P_B3 (negative control): m_bg ngẫu nhiên không được cải thiện

## Phân tích thống kê

- Paired so sánh theo ảnh: Dice per-image của variant A vs variant B trên cùng 40 ảnh → Wilcoxon signed-rank test (không giả định phân phối chuẩn) + Cohen's d effect size (mean diff / pooled SD)
- Báo cáo: median delta, mean delta, p-value (2-tailed), 95% CI bootstrap (n=2000 resamples)
- Bội số: 6 so sánh chính (4 cells vs baseline + A+B vs từng thành phần) → điều chỉnh Bonferroni (ngưỡng α = 0.05/6 ≈ 0.0083) khi diễn giải

## Kiểm soát confounding

- Cùng split, checkpoint, seed(42), Otsu init, num_iter=4
- Đơn vị phân tích = ảnh (paired) — không dùng mean pool gộp
- Không chạy train ở phase này → không có confounding từ training dynamics

## Giới hạn đã biết (ghi nhận, không claim)

- 1 checkpoint duy nhất (200ep Kaggle) — kết quả gắn với checkpoint này
- 1 split (40 ảnh) — ngoại suy sang tập khác chưa được kiểm tra
- Model weight không được train với cơ chế mới — đây là đo "khả năng khai thác feedback" của model hiện có, không phải "giá trị đầy đủ" của cơ chế mới (train end-to-end là bước sau)

## Quy trình chạy

1. `analysis/abl2x2_eval.py` — chạy 5 cells (C00, C10, C01, C11, negative control) qua 4 iterations
2. Output: `results/abl2x2_summary.json`, `results/abl2x2_per_image.csv`
3. `analysis/abl2x2_stats.py` — Wilcoxon + Cohen's d + bootstrap CI
4. Figures: Dice curves, FP/FN bar, oracle gap per variant
5. Report theo template
