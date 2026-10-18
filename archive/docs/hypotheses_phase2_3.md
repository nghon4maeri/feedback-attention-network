# Giả thuyết Phase 2+3 — FANet (31/08/2026)

## Quan sát đã đóng băng (Phase 1 + trước đó)

- O1: fmask nhận 0 gradient (10 steps, 0/160 params có grad) — `logs/analysis_grad_log.txt`
- O2: FP/FN ratio = 2.48; FP rate 11.7% diện tích; FP cách biên GT 40.7px — `results/oracle_summary.json`
- O3: 84.6% chỗ oracle khác prediction nằm ở vùng prediction-FP
- O4: Oracle gap +0.093 ổn định từ iteration 2
- O5: Confidence-gating bị bác bỏ (binary 0.2390 vẫn tốt nhất) — `results/confgated_summary.json`

## Câu hỏi nghiên cứu

RQ1: Nếu gradient chảy được vào fmask (qua STE/soft gating), nhánh "learned attention" có học và cải thiện Dice so với fmask chết không?

RQ2: Nếu feedback mang thêm kênh background tin cậy [m_fg, m_bg], FP rate và Dice có cải thiện so với feedback foreground-only không?

RQ3: Hai cơ chế kết hợp có tương tác (synergy) hay độc lập?

## Hypotheses (candidate — không phải kết luận)

### H_A (fmask revival)
**Mechanism**: hard threshold chặn gradient → fmask chưa bao giờ học. STE/soft gating cho gradient chảy → fmask học được vùng đáng chú ý từ feature → gate chính xác hơn → Dice tăng.

**Prediction bác bỏ được**:
- P_A1 (điều kiện cần): với STE/soft, grad norm fmask > 0 ở mọi MixPool (so với 0 hiện tại). **Bác bỏ H_A nếu grad vẫn = 0.**
- P_A2 (điều kiện đủ): Dice(+fmask-live) − Dice(fmask-dead) > +0.005 trên 40 ảnh val sau 4 iterations refinement. **Ngưỡng đo: +0.005 (khoảng 1/2 oracle gap 0.093 là dư địa lớn; +0.005 là cải thiện tối thiểu có ý nghĩa thực tiễn).**

**Rival**: R_A1 — fmask không giúp vì feedback mask m đã đủ thông tin (gate OR che lấp đóng góp fmask); R_A2 — STE gradient qua soft value không đủ ổn định để học có ích (nhiễu gradient qua ngưỡng).

### H_B (dual-path background feedback)
**Mechanism**: feedback foreground-only zero hoá background → model không nhận tín hiệu "đâu không phải polyp" → over-segmentation (O2). Thêm kênh m_bg (background tin cậy) → model dọn vùng nền → FP giảm, Dice tăng.

**Prediction bác bỏ được**:
- P_B1: FP rate(+dual-path) < FP rate(binary) − 0.01 (tức giảm ít nhất 1% diện tích so với 11.7% baseline). **Bác bỏ nếu FP không giảm.**
- P_B2: Dice(+dual-path) − Dice(binary) > +0.005 sau 4 iterations. **Ngưỡng: +0.005.**
- P_B3 (âm tính — negative control): thêm kênh m_bg NGẪU NHIÊN (nhiễu) không được cải thiện Dice → đảm bảo hiệu ứng đến từ thông tin background thật, không phải việc "thêm kênh input".

**Rival**: R_B1 — background không giúp vì model đã tự học background qua nhánh conv2 nguyên bản (50% feature không gate); R_B2 — tăng tham số/input làm model overfit val 40 ảnh (cần kiểm tra trên nhiều seeds).

### H_AB (interaction)
**Mechanism**: fmask sống (chọn vùng đáng chú ý) + m_bg (chặn vùng nền) bổ trợ nhau: fmask học cách kết hợp 2 kênh.

**Prediction**: Dice(A+B) − Dice(B-only) > +0.005 (synergy) **và** Dice(A+B) > Dice(A-only) + Dice(B-only) − Dice(baseline) (tính siêu tuyến tính). **Bác bỏ nếu hiệu ứng thuần cộng hoặc triệt tiêu.**

## Loại claim

- P_A1, P_B1, P_B3: **descriptive/associational** (grad norm, FP rate trước-sau)
- P_A2, P_B2, H_AB: **associational + predictive** (so sánh variants trên cùng test set, không claim causal vì chưa có thiết kế can thiệp chéo hoàn chỉnh; model ablation 2x2 trên cùng split giảm confounding từ split/seed)

## Điều chưa kiểm soát

- Confounders: batch size, split, augmentation (CoarseDropout args) — kiểm soát bằng cách dùng CÙNG split + checkpoint + eval pipeline cho mọi variants trong thí nghiệm này.
- Chưa kiểm soát: seed training (chỉ 1 seed ở phase này — ghi nhận, không claim mạnh).
