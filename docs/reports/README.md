# Research Reports — FANet

Index tổng hợp các báo cáo nghiên cứu của dự án FANet (Feedback Attention Network).

## Timeline

| # | File | Ngày | Nội dung chính |
|---|------|------|----------------|
| 1 | [2026-08-10_week2_setup.md](2026-08-10_week2_setup.md) | 10/08/2026 | Setup env, chạy code FANet, đọc sơ paper |
| 2 | [2026-08-20_uacanet-mixpool-gap.md](2026-08-20_uacanet-mixpool-gap.md) | 20/08/2026 | Đào sâu UACANet + research gap MixPool + kế hoạch analysis |
| 3 | [2026-08-24_feedback-help-hurt-analysis.md](2026-08-24_feedback-help-hurt-analysis.md) | 24/08/2026 | Phân tích feedback help/hurt + uncertainty correlation + 2 điểm yếu xác nhận |
| 4 | [2026-08-27_confidence-gated-feedback.md](2026-08-27_confidence-gated-feedback.md) | 27/08/2026 | Checkpoint 200ep Kaggle + kiểm định confidence-gating (bác bỏ) + hướng negative feedback |
| 5 | [2026-08-31_oracle-background-analysis.md](2026-08-31_oracle-background-analysis.md) | 31/08/2026 | Phase 1: oracle pixel analysis xác nhận giả thuyết background (84.6% khác biệt ở vùng FP) |
| 6 | [2026-08-31_phase2-3-ablation.md](2026-08-31_phase2-3-ablation.md) | 31/08/2026 | Phase 2+3: literature + hypotheses + ablation 2x2 (STE hồi sinh fmask, dual-path); P_B1 bác bỏ, P_A1 xác nhận |
| 7 | [2026-08-31_phase4-train-end-to-end.md](2026-08-31_phase4-train-end-to-end.md) | 31/08/2026 | Phase 4: code train 4 cells sẵn sàng + smoke-test CPU + fallback loss; chờ chạy Kaggle T4 |
| 8 | [2026-08-31_phase4-results.md](2026-08-31_phase4-results.md) | 31/08/2026 | Phase 4 kết quả: T_A bác bỏ (binary 0.2806 > STE 0.1951); bug dual-path phát hiện + sửa; cần chạy lại T01/T11 |
| 9 | [2026-09-04_next-directions.md](2026-09-04_next-directions.md) | 04/09/2026 | Research strategy: audit bằng chứng (train/frozen/invalid), literature 25 paper đa nguồn + 29 PDF, novelty assessment (FANetv2 + predictive-coding 2026), xếp hạng X1–X6 → ưu tiên loss-side FP penalty + multi-seed; STE/dual-path bế tắc |
| 10 | [2026-09-05_new-papers-review.md](2026-09-05_new-papers-review.md) | 05/09/2026 | Đọc 7 paper tải tay (FEGNet, RefineU-Net, CBL, BCNet, CTNet, CFA-Net, BUNet): feedback của họ là feature-level nội mạng ≠ mask-at-input của FANet; Gate 1 X2 đổi sang wIoU+wBCE (+neg-area / boundary-weight); audit PDF — sửa 6/8 file sai, xóa 2 |
| 11 | [2026-09-07_phase5-x2-gate01.md](2026-09-07_phase5-x2-gate01.md) | 07/09/2026 | Phase 5: verify 1a/1b (μ nặng biên → pivot far-weighted; WSDice công thức gốc), code + smoke 3 cells (T0N/TA/TB), notebook phase5 sẵn sàng, X4 verdict ĐÓNG STE; Gate 0/1 chờ chạy Kaggle |
| 12 | [2026-09-08_research-overview.md](2026-09-08_research-overview.md) | 08/09/2026 | **Báo cáo tổng quan nghiên cứu** (research overview) — bản dễ hiểu cho người mới: giải thích khái niệm nền (polyp, mask, attention, feedback, FP/FN...), kể lại hành trình nghiên cứu theo trình tự (reproduce → phát hiện điểm yếu → 3 hướng thất bại → định vị root cause → pivot loss-side → đang chờ Kaggle), bảng kết quả chính, định hướng novelty + dàn ý paper + future work + từ điển thuật ngữ |
| 13 | [2026-09-08_phase5-results.md](2026-09-08_phase5-results.md) | 08/09/2026 | **Phase 5 KẾT QUẢ (Gate 0/1)**: Kaggle T4 đã chạy xong (T0N/TA/TB × 200ep, seed 43). Gate 0 → feedback loop là liability trên Dice (T0N 0.3034 ≥ T00 0.2806) nhưng KHÔNG giải quyết FP (FP +1.91pp); Gate 1 **FAIL cả 2 cell** (TA WSDice không giảm FP; TB far-weighted Dice COLLAPSE 0.0086 — soft-dice train che giấu collapse) → **KHÔNG multi-seed, dừng GPU, pivot negative-result paper**. Stats chuẩn: Wilcoxon + rank-biserial + bootstrap CI + BF10 + sensitivity |
| 14 | [2026-09-09_phase6-preregistration.md](2026-09-09_phase6-preregistration.md) | 09/09/2026 | **Phase 6 PRE-REGISTRATION (trả lời advisor)**: audit 2 điểm feedback vào bằng chứng Phase 5 (bảng 2×2: no-FB×orig=T0N đã có; gap THIẾU = no-FB×new-loss + asymmetric/class-frequency loss chưa implement); literature grounding (ASL/Tversky/Unified Focal/ENet/Cui/Kvasir/Metrics Reloaded); 2 phương án lấp gap tối thiểu (A: lưới 2×2 sạch 14h; B: tái dùng + thêm cell mới, khuyến nghị B); stopping rules + metrics (PRIMARY=FP per-image) + power (δ≈0.13) + stats plan + memo trả lời advisor. **CHƯA chạy GPU, chờ user duyệt** |
| 15 | [2026-09-09_phase6-results.md](2026-09-09_phase6-results.md) | 09/09/2026 | **Phase 6 KẾT QUẢ (2×2 feedback×loss)**: Kaggle T4 chạy xong TC (FB×Tversky-asym) + TD (no-FB×Tversky-asym), seed 43, 200ep. **TD PASS**: FP −3.25pp (p=.005) so T0N, Dice giữ nguyên (BF10=0.172→null) → asymmetric loss giảm FP khi KHÔNG có feedback. **TC FAIL**: FP +0.92pp (không giảm) khi CÓ feedback → feedback loop CHẶN hiệu ứng loss. Factorial 2×2: interaction FP = +4.17pp. Câu chuyện paper negative-result được tinh chỉnh: "feedback mask-at-input blocks loss-side FP suppression" |
| 16 | [2026-09-14_phase6-feedback-trap-analysis.md](2026-09-14_phase6-feedback-trap-analysis.md) | 14/09/2026 | **Báo cáo Chẩn đoán Offline Feedback Trap**: Phát hiện BatchNorm Drift cực đoan ($D_{KL}=33.63$ tại `e1.r1.bn3`), Prediction Saturation tăng (4.63%) và Gradient Norm nhánh mask giảm 50.5% (gradient attenuation) khiến loss bị bóp nghẹt. Khẳng định root cause của interaction 2x2. |

## Experiment Tracking

| Giả thuyết / Mục đích | Trạng thái | Kết quả chính | Artifacts |
|----------------------|------------|---------------|-----------|
| Gate binary `(fmask > 0.5)` cắt gradient → nhánh fmask không học | Done | fmask: 0/… params có gradient sau 10 steps; conv1/conv2: 160 params có grad; weight fmask không đổi sau optimizer step | `logs/analysis_grad_log.txt`, `analysis/grad_flow.py` |
| Baseline: test-time refinement trên checkpoint 53 epochs | Done | Iter 1→2: Jaccard 0.2166→0.2251, F1 0.3147→0.3268 (feedback giúp nhẹ) | `results/test_results.csv`, `scripts/evaluate.py` |
| Hard binary feedback: khi nào giúp / khi nào hại; uncertainty trước threshold có correlate với lỗi không | Done | 65 giúp / 46 hại; mean delta +0.031; corr(prev_dice)=+0.355, corr(unc)=−0.297; err 48.7% (conf<0.1) vs 2.7% (conf>0.35); binary 0.334 vs soft 0.311 vs none 0.301 vs oracle 0.370 | `logs/feedback_analysis_log.txt`, `results/feedback_summary.json`, `analysis/feedback_analysis.py` |
| Đề xuất cơ chế feedback mới nhắm đúng điểm yếu đã xác nhận (confidence-gated + hồi sinh fmask) | Planned | — | — |
| Kiểm định confidence-gated feedback: 9 variants (binary, conf-weighted, gated tau 0.05-0.30, soft, none, oracle) | Done | Bác bỏ: binary 0.2390 tốt nhất, mọi biến thể confidence-based thấp hơn; hurt không giảm; oracle gap 0.092 | `logs/confgated_log.txt`, `results/confgated_summary.json`, `analysis/confgated_feedback.py` |
| Baseline refinement 10 iterations trên checkpoint 200ep Kaggle | Done | F1 0.2430 → 0.2414 (plateau, feedback gần như không giúp) | `results/test_results.csv` |
| Oracle vs prediction feedback: oracle tốt hơn nhờ thông tin gì (background hay foreground)? | Done | 84.6% khác biệt ở vùng prediction-FP; FP/FN = 2.48; FP cách biên GT 40.7px → xác nhận thiếu thông tin background | `logs/oracle_analysis_log.txt`, `results/oracle_summary.json`, `analysis/oracle_analysis.py` |
| STE/soft hồi sinh gradient fmask (P_A1) | Done | binary=0 grad; STE=0.125, soft=0.106 tổng grad 8 MixPool | `logs/grad_flow_gates_log.txt`, `analysis/grad_flow_gates.py` |
| Ablation 2x2 dual-path + gate variants (P_B1/P_B2/P_B3/H_AB) | Done | P_B1 BÁC BỎ (FP +2.92pp, p=1e-7); Dice +0.030 nhưng p=0.035; neg-ctrl sụp 0.083 (P_B3 ok); không synergy (C11<C10) | `logs/abl2x2_log.txt`, `results/abl2x2_summary.json`, `results/abl2x2_stats.json` |
| Phase 4: train end-to-end 4 cells (T00/T10/T01/T11) loại BN mismatch | ⏳ Chờ chạy Kaggle T4 | Code + smoke-test CPU OK (4 cells 1 epoch, loss ~0.79); fallback negdice OK (loss ~1.25) | `notebooks/fanet_kaggle_phase4.py`, `scripts/train.py`, `src/fanet/losses.py` |
| Phase 4 kết quả: T_A (STE train vs binary train) | Done | T_A KHÔNG ỦNG HỘ: binary 0.2806 > STE 0.1951 (delta -0.0855, p=0.277); train lại loại BN mismatch (frozen 0.239 → trained 0.281) | `results/phase4_eval.json`, `results/phase4_stats.json` |
| Phase 4 bug: T01/T11 == T00/T10 (m_bg zeros khi train) | Fixed | Phát hiện bằng weight diff = 0; sửa m_bg = 1 - m_fg; cần chạy lại T01/T11 trên Kaggle | `notebooks/fanet_kaggle_phase4.py` |
| Phase 4 run mới (papermill 9/4): T00/T10/T01/T11 + bug complement m_bg=1−m_fg | Done (chẩn đoán) | T00 0.5590 / T10 0.5611 / T01 0.6478 / T11 collapse (1.11–1.13 từ ep 40) — T01/T11 INVALID (complement triệt tiêu fmask); STE ≈ binary ở run này | `kaggle/fanet-phase4.ipynb` (cell 17) |
| Literature grounding đa nguồn cho next directions (OpenAlex/arXiv/Europe PMC) | Done | 25 paper curated + novelty assessment (FANetv2, predictive-coding 2026); 29 PDF tải về docs/ | `docs/literature_grounding_next.md`, `docs/papers_not_accessible.md` |
| Xếp hạng hướng tiếp theo X1–X6 | Planned | Ưu tiên X2+X5 (loss-side FP penalty, multi-seed ≈21+7 GPU-h); dự phòng X4 (0 GPU); X1 chỉ nếu X2 thất bại; pivot rule: FP không giảm ≥1pp ở seed đầu → reframe negative-result | `docs/reports/2026-09-04_next-directions.md` |
| Review 7 paper mới (FEGNet, RefineU-Net, CBL, BCNet, CTNet, CFA-Net, BUNet) | Done | 5/7 dùng wIoU+wBCE; feedback thành công là feature-level nội mạng (soft gate, deep sup, ở skip) ≠ mask-at-input; Gate 1 X2 → wIoU+wBCE+neg-area & boundary-weight (1+5μ); novelty verdict giữ nguyên "mỏng" | `docs/reports/2026-09-05_new-papers-review.md`, `docs/literature_grounding_next.md` |
| Audit PDF docs/ (38 file) | Done | 8 file sai nội dung (arXiv ID sai); sửa 6 (verify trang đầu), xóa 2 (STARCaps, DistanceTransform — OpenReview 403, chờ tải tay) | `papers_not_accessible.md` |
| Phase 5 verify 1a/1b: μ (CFA/F³Net) + WSDice gốc | Done | μ nặng BIÊN → pivot far-weighted (1+5(1−μ)); WSDice = 1−[2ΣĜG]/[ΣĜ²+ΣG²], w=y(v2−v1)+v1, v1=0.3 heuristic | `src/fanet/losses.py`, `docs/reports/2026-09-07_phase5-x2-gate01.md` |
| Phase 5 X4: chẩn đoán STE (CPU, ckpt disk) | Done | STE grad chảy (13107) nhưng var < soft; fmask encoder corr ~0.2 (yếu); BN shift lớn (d1.r1.bn1 mean_abs 18.8) → ĐÓNG STE vĩnh viễn | `results/x4_ste_diagnosis.json`, `kaggle/figures/fig_x4_*.png` |
| Phase 5 X2 Gate 0 (T0N) + Gate 1 (TA/TB) | ✅ Done (Kaggle 08/09) | **Gate 0**: feedback loop = liability trên Dice (T0N 0.3034 ≥ T00 0.2806, p=.36, BF10=0.21→null) nhưng FP TĂNG +1.91pp → không phải nguồn gốc FP. **Gate 1 FAIL**: TA (WSDice) FP 5.19% không giảm; TB (far-weighted) Dice 0.0086 COLLAPSE (recall 0.0047; soft-dice train 0.19 cheপদে che giấu). → **KHÔNG multi-seed; pivot negative-result paper** | `results/phase5_eval.json`, `results/phase5_stats.json`, `results/phase5_stats_full.json`, `checkpoints_phase5/`, `kaggle/figures/fig_p5_*.png` |
| Phase 6: trả lời advisor — no-FB baseline + asymmetric/class-frequency loss | ⏳ **Pre-registered (chờ user duyệt, chưa GPU)** | Audit: no-FB×orig=T0N đã có; gap THIẾU = no-FB×new-loss + asymmetric/class-frequency loss (chưa implement). Literature: khuyến nghị Tversky α=0.7/β=0.3 (asymmetric) + ENet-bounded (class-freq, tránh inverse thô/collapse). 2 phương án (A sạch 14h / B tái dùng, khuyến nghị B). Stopping: FP<base−0.01 & Dice≥base−0.02 & FN≤base+0.02; diverge ep40 theo binary val-Dice | `docs/literature_grounding_phase6.md`, `docs/reports/2026-09-09_phase6-preregistration.md` |
| Phase 6 KẾT QUẢ: TC (FB×Tversky) / TD (no-FB×Tversky) | ✅ Done (Kaggle 09/09) | **TD PASS** (no-FB×asym): FP −3.25pp (p=.005) vs T0N, Dice giữ nguyên (BF10=0.172→null). **TC FAIL** (FB×asym): FP +0.92pp (p=.097) không giảm → feedback loop CHẶN hiệu ứng giảm FP của loss (interaction 2×2 = +4.17pp). Câu chuyện paper: "feedback mask-at-input blocks loss-side FP suppression". Không collapse (binary val-metrics log mỗi epoch) | `results/phase6_eval.json`, `results/phase6_stats.json`, `results/phase6_stats_full.json`, `checkpoints_phase6/`, `kaggle/figures/fig_p6_*.png` |
| Phase 6 Phân tích "Feedback Trap" | ✅ Done (Offline 14/09) | Viết script offline chẩn đoán BatchNorm drift ($D_{KL}=33.63$ tại e1), Saturation (4.63%) và Gradient Norm truyền về mask (giảm 50.5%). Giải thích trực tiếp sự thất bại của TC. | `analysis/diagnose_feedback_drift.py`, `diagnostics_output/`, `docs/reports/2026-09-14_phase6-feedback-trap-analysis.md` |

## Quy trình cập nhật

1. **Cuối mỗi phiên làm việc**: cập nhật report hiện tại (Done, Findings, Experiments).
2. **Mỗi milestone** (xong 1 analysis / 1 experiment): tạo file report mới theo `_TEMPLATE.md` + cập nhật bảng Timeline và Experiment Tracking ở file này.
3. **Số liệu luôn trỏ về artifacts trong repo** (`logs/`, `results/`, `checkpoints/`) thay vì chép tay.
4. Khi đủ các mảnh (reproduce → gap → analysis → đề xuất), viết 1 bản **research summary** nối mạch toàn bộ quá trình.

## Artifacts chính

- Model checkpoint: `checkpoints/checkpoint.pth` (53 epochs, best val loss 0.505)
- Training log: `logs/train_log.txt`
- Papers: `docs/2103.17235v3.pdf` (FANet), `docs/3474085.3475375.pdf` (UACANet)
- Guide kiến trúc: `docs/FANet_Complete_Guide.md`
