# Giả thuyết Phase 4 — FANet train end-to-end (31/08/2026)

## Quan sát đóng băng (từ Phase 1-3, không khảo sát lại)

- O1: oracle gap +0.093; FP/FN = 2.48; FP xa biên 40.7px — `results/oracle_summary.json`
- O2: 84.6% oracle khác prediction nằm ở vùng prediction-FP → feedback thiếu background info
- O3: STE hồi sinh gradient fmask (0 → 0.125); soft (0.106) — `logs/grad_flow_gates_log.txt`
- O4: Frozen ablation: C10 soft 0.2760 > C01 dual 0.2691 > C00 0.2390; dual tăng FP +2.92pp (p=1.27e-7)
- O5: Nghi ngờ BN mismatch — model train hard-gate, BN fit phân phối prune cứng; inference với gate mới lệch phân phối
- O6: C11 (soft+dual) 0.2713 < C10 (soft only) 0.2760 → không synergy trên frozen

## Câu hỏi nghiên cứu Phase 4

RQ4: Khi train end-to-end với từng cơ chế (loại bỏ BN mismatch), giá trị thật của fmask-STE (T_A) và dual-path (T_B1/T_B2) là gì?

## Hypotheses (candidate)

### T_A (train: fmask-STE/soft ≥ binary)
**Mechanism**: gradient chảy vào fmask (O3) → fmask học vùng đáng chú ý → gate chính xác → Dice tăng. Đây là test thật của H_A (Phase 2 chỉ chứng minh điều kiện cần gradient).

**Prediction bác bỏ được**:
- P_T_A: mean Dice(STE-trained) − mean Dice(binary-trained) > **+0.005** trên 40 ảnh val, final iteration (4 iters). **Bác bỏ nếu delta ≤ +0.005 hoặc p ≥ α.**

**Rival**: STE gradient qua soft value nhiễu → fmask không hội tụ có ích; hoặc gate OR đã đủ (fmask không thêm thông tin).

### T_B1 (train: dual-path GIẢM FP)
**Mechanism**: Phase 1 (O1/O2) chứng minh vấn đề là FP (over-segmentation) + oracle dạy vùng background. Sau train với m_bg suppression, model học "đâu không phải polyp" → FP giảm. Frozen (O4) thất bại do BN mismatch, không phải do cơ chế sai.

**Prediction bác bỏ được**:
- P_T_B1: FP rate(dual-trained) < FP rate(binary-trained) − **0.01** (tức giảm ≥1pp so với ~11.6% baseline). **Bác bỏ nếu FP không giảm đủ hoặc còn tăng.**

**Rival**: dual-path làm model "nhút nhát" hơn — giảm FP nhưng tăng FN mạnh (đánh đổi); hoặc suppression không hiệu quả vì m_bg từ prediction cũng chứa lỗi.

### T_B2 (train: dual-path tăng Dice)
**Mechanism**: tương tự T_B1 + recall giữ nguyên (không bị suppression đánh vào vùng polyp thật).

**Prediction bác bỏ được**:
- P_T_B2: mean Dice(dual-trained) − mean Dice(binary-trained) > **+0.005**, final iter. **Bác bỏ nếu delta ≤ +0.005.**

### T_BN (train: dual-path ổn định BN)
**Mechanism**: khi train với dual-path ngay từ đầu, BN running stats fit phân phối mới → hết mismatch. Đây là cơ chế giải thích tại sao frozen thất bại (O5) còn train có thể thành công.

**Prediction bác bỏ được** (test trực tiếp BN mismatch):
- P_T_BN: với dual-path, |BN running_mean của conv1 (MixPool e1) − BN running_mean sau khi đổi gate| nhỏ hơn 0.05 (khớp) SAU train; ở checkpoint cũ (frozen) độ lệch > 0.2. So sánh activation distribution trước/sau gate đổi trên cùng ảnh.
- **Bác bỏ nếu sau train vẫn lệch lớn hoặc model không ổn định (loss dao động).**

### T_AB (train: synergy)
**Mechanism**: fmask sống + m_bg bổ trợ nhau sau khi cả hai được train.

**Prediction**: Dice(ste+dual) > Dice(ste-only) **và** Dice(ste+dual) > Dice(dual-only) với ngưỡng +0.005; **và** hiệu ứng siêu tuyến tính (delta_AB > delta_A + delta_B). **Bác bỏ nếu thuần cộng hoặc triệt tiêu.**

### Fallback (nếu T_B1 bác bỏ lần nữa)
- F1: **negative-area Dice loss** (phạt trực tiếp vùng sai-fg trong loss) — từ paper IEEE Access 2020.
- F2: **decoder-only suppression** — gate m_bg chỉ ở decoder, bảo toàn BN encoder (theo hướng EMCAD).
- Quyết định sau khi có kết quả train.

## Loại claim

- P_T_A, P_T_B1, P_T_B2: **associational/predictive** (so sánh mô hình trained trên cùng split, paired per-image). Không claim causal trước khi multi-seed.
- P_T_BN: **descriptive** (đo BN stats/activation distribution trước-sau train).
- Chỉ 1 seed trong phase này; multi-seed ≥3 là điều kiện để claim mạnh trong paper.

## Điều chưa kiểm soát

- Confounders: khớp chính xác split/seed/batch/augmentation giữa 4 cell (đã thiết kế); training dynamics (lr schedule, BN momentum) giống nhau.
- Multi-seed chưa chạy (ghi Will Do); kết quả phase này là pilot 1-seed có giá trị chỉ báo.
