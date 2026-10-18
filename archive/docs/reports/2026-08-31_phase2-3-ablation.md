# [CN – 31/08/2026]

## Mục tiêu tuần này

Hoàn thành Phase 2 + Phase 3 theo đúng kế hoạch: literature grounding cho novelty, tinh chỉnh giả thuyết A/B/A+B thành các prediction bác bỏ được, thiết kế ablation 2x2 có negative control, cài đặt 2 cơ chế mới (fmask revival + dual-path feedback), chạy thí nghiệm với phân tích thống kê đầy đủ, và trả lời rõ giả thuyết nào được ủng hộ/bác bỏ.

## Done

- [x] Literature grounding: 2 rounds, 11 queries OpenAlex, 10 papers liên quan (PraNet, UACANet, STAR-Caps, negative-area Dice loss, HRINet...) — `docs/literature_grounding_phase2.md`
- [x] Tinh chỉnh giả thuyết: H_A / H_B / H_AB với 7 prediction bác bỏ được (P_A1, P_A2, P_B1, P_B2, P_B3, H_AB) + rivals + ngưỡng đo — `docs/hypotheses_phase2_3.md`
- [x] Thiết kế ablation 2x2 (C00/C10/C01/C11 + neg-ctrl + oracle), metric chính/phụ, Wilcoxon + Cohen's d + bootstrap CI, Bonferroni α=0.0083 — `docs/experiment_design_phase2_3.md`
- [x] Cài đặt Phase 2: MixPool với 3 gate modes (binary/ste/soft) + FANet configurable — `src/fanet/models/blocks.py`, `src/fanet/models/fanet.py`
- [x] Cài đặt Phase 3: dual-path feedback [m_fg, m_bg] với bg suppression `keep * (1 - m_bg)` — cùng 2 file trên
- [x] Kiểm chứng gradient: binary = 0 gradient (tái xác nhận), **STE = 0.125, soft = 0.106** tổng grad qua 8 MixPool — `logs/grad_flow_gates_log.txt`
- [x] Chạy ablation 2x2 (8 cells x 4 iters x 40 ảnh, frozen checkpoint 200ep) — `logs/abl2x2_log.txt`
- [x] Phân tích thống kê paired — `results/abl2x2_stats.json`
- [x] 4 figures publication-ready — `kaggle/figures/fig_abl_*.png`

## Findings quan trọng

### 1. P_A1 ĐƯỢC XÁC NHẬN — STE/soft hồi sinh gradient fmask

| Gate | Tổng fmask grad (3 steps, 8 MixPool) |
|---|---|
| binary (gốc) | 0.000e+00 |
| **ste** | **1.247e-01** |
| soft | 1.061e-01 |

Điều kiện cần của H_A được đáp ứng: straight-through estimator làm gradient chảy vào nhánh fmask ở tất cả 8 MixPool blocks. Binary vẫn chết hoàn toàn (0 gradient).

### 2. Dual-path feedback tăng Dice +0.030 nhưng KHÔNG giảm FP — P_B1 BỊ BÁC BỎ

| Cell | Dice (iter 4) | FP rate | FN rate | Recall | Oracle gap |
|---|---:|---:|---:|---:|---:|
| C00 (baseline) | 0.2390 | 11.57% | 4.74% | 0.472 | 0.092 |
| C01-t0.1 (dual) | **0.2691** | **14.49%** | 3.98% | 0.569 | 0.062 |
| C10 (soft gate) | **0.2760** | 14.62% | 4.24% | 0.583 | 0.055 |
| C11 (soft+dual) | 0.2713 | 15.13% | 3.96% | 0.583 | 0.060 |
| neg-ctrl | 0.0829 | 0.97% | 8.33% | 0.071 | 0.248 |
| oracle | 0.3312 | 5.33% | 4.67% | 0.480 | — |

**P_B1 bác bỏ**: kỳ vọng dual-path GIẢM FP (over-segmentation), thực tế FP TĂNG +2.92pp (p=1.27e-07, rất significant). Thay vào đó, cải thiện Dice đến từ **Recall tăng mạnh** (+0.10) và FN giảm (-0.76pp, p=1.09e-04).

**Giải thích khả dĩ (cần kiểm chứng)**: checkpoint 200ep được train với hard gate nên BN của conv1 đã fit phân phối feature bị prune cứng. Dual-path làm mềm phân phối → model phản ứng khác với lúc train — recall phình lên. Cơ chế "bg suppression" ở inference trên model chưa được train với nó là **confounder thật sự** — cần train end-to-end mới kết luận được.

### 3. Negative control hoạt động đúng (P_B3 xác nhận)

Random m_bg làm Dice sụp còn 0.0829 (từ 0.2390) — hiệu ứng của dual-path là do thông tin background THẬT, không phải do "thêm kênh input". P_B3 không bị vi phạm.

### 4. Ý nghĩa thống kê: hiệu ứng dương nhưng KHÔNG qua Bonferroni

| So sánh | Mean delta | CI95 | p (Wilcoxon) | Sig (α=0.0083) |
|---|---:|---:|---:|---:|
| C01-t0.1 vs C00 | +0.030 | [0.004, 0.059] | 0.035 | no |
| C10 vs C00 | +0.037 | [0.003, 0.080] | 0.091 | no |
| C11 vs C00 | +0.032 | [0.003, 0.066] | 0.050 | no |

- CI95 đều loại trừ 0 (hiệu ứng dương nhất quán), nhưng p > 0.0083 → chưa đủ mạnh về mặt thống kê với n=40.
- P_B2 (ngưỡng +0.005) **đạt về magnitude** (+0.030) nhưng không significant → hỗ trợ yếu.

### 5. H_AB KHÔNG có synergy

C11 (soft+dual, 0.2713) < C10 (soft only, 0.2760) — hai cơ chế không cộng hưởng, thậm chí gây nhiễu nhau trên frozen checkpoint. Không ủng hộ H_AB.

## Experiments

| Giả thuyết | Config | Kết quả | Link log |
|-----------|--------|---------|----------|
| Gradient fmask hồi sinh qua STE/soft (P_A1) | 3 gate modes x 3 steps, checkpoint 200ep | XÁC NHẬN: binary=0, STE=0.125, soft=0.106 | `logs/grad_flow_gates_log.txt` |
| Dual-path giảm FP + tăng Dice (P_B1, P_B2) | 8 cells x 4 iters x 40 ảnh frozen 200ep | BÁC BỎ P_B1 (FP +2.92pp, p=1e-7); P_B2 magnitude đạt (+0.030) nhưng p=0.035 | `logs/abl2x2_log.txt`, `results/abl2x2_stats.json` |
| Negative control m_bg ngẫu nhiên (P_B3) | neg-ctrl cell | XÁC NHẬN: Dice sụp 0.239→0.083 | `results/abl2x2_summary.json` |
| Synergy A+B (H_AB) | C11 vs C10/C01 | BÁC BỎ: C11 (0.271) < C10 (0.276) | `results/abl2x2_stats.json` |

## Kết luận giả thuyết tuần này

| Giả thuyết | Phán quyết | Bằng chứng |
|---|---|---|
| H_A: STE hồi sinh gradient fmask | **ỦNG HỘ (điều kiện cần)** | grad 0 → 0.125 |
| H_A: fmask sống cải thiện Dice | Chưa kết luận — cần train end-to-end | soft gate +0.037 nhưng p=0.091 |
| H_B: dual-path giảm over-segmentation | **BÁC BỎ** — FP tăng, không giảm | +2.92pp FP, p=1.27e-07 |
| H_B: dual-path tăng Dice | Hỗ trợ yếu — magnitude đạt, không significant | +0.030, p=0.035 |
| H_AB: synergy | **BÁC BỎ** | C11 < C10 |

## Will Do (On going)

- [ ] **Train end-to-end với từng cơ chế** (bắt buộc, trên Kaggle T4): BN mismatch ở inference trên frozen checkpoint là confounder chính — chỉ train lại mới kết luận được giá trị thật của dual-path và STE
- [ ] Sửa notebook train: batch=2 (khớp local), sửa CoarseDropout args theo API mới, cùng split
- [ ] Ablation train 4 model: binary-1kênh / binary-dual / ste-1kênh / ste-dual
- [ ] Đánh giá lại trên cùng 40 ảnh val + full test_results (10 iters)
- [ ] Nếu dual-path sau train vẫn tăng FP: chuyển hướng sang phạt FP qua loss (negative-area Dice, Lovász) hoặc chỉ bật bg-suppression ở decoder (bảo toàn encoder distribution)
- [ ] Multi-seed (>=3) trước khi claim trong paper
- [ ] Chưa mở rộng deep supervision / multi-dataset ở phase này

## Any Stuck / Open Questions

- Confounder đã xác định: đánh giá cơ chế mới trên model train với cơ chế cũ làm lệch phân phối BN → mọi con số inference tuần này chỉ là "khả năng khai thác feedback của model hiện có", không phải giá trị đầy đủ của cơ chế. Đã ghi rõ trong experiment design.
- Open question: vì sao dual-path tăng FP? Giả thuyết BN mismatch cần test trực tiếp (so sánh BN running stats trước/sau khi đổi gate).
- Novelty claim "background-aware feedback loop" cần search bổ sung (Semantic Scholar, Google Scholar) trước khi viết paper — OpenAlex mới là 1 DB.

## Đính kèm link chi tiết

- Literature: `docs/literature_grounding_phase2.md` — Hypotheses: `docs/hypotheses_phase2_3.md` — Design: `docs/experiment_design_phase2_3.md`
- Code: `src/fanet/models/blocks.py` (MixPool gate/dual_path), `src/fanet/models/fanet.py`
- Scripts: `analysis/grad_flow_gates.py`, `analysis/abl2x2_eval.py`, `analysis/abl2x2_stats.py`, `analysis/abl2x2_figures.py`
- Logs: `logs/grad_flow_gates_log.txt`, `logs/abl2x2_log.txt`
- Kết quả: `results/abl2x2_summary.json`, `results/abl2x2_stats.json`
- Figures: `kaggle/figures/fig_abl_dice_curves.png`, `fig_abl_final_metrics.png`, `fig_abl_oracle_gap.png`, `fig_abl_stats.png`
- Template: `docs/reports/_TEMPLATE.md` — Index: `docs/reports/README.md`
