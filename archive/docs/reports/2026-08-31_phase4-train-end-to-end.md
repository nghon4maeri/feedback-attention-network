# [CN – 31/08/2026] — Phase 4: Train end-to-end (A/B verdicts)

## Mục tiêu tuần này

Loại bỏ confounder BN mismatch đã xác định ở Phase 2/3 bằng cách train end-to-end 4 mô hình (T00 baseline / T10 ste / T01 dual / T11 ste+dual), và trả lời dứt khoát giá trị thật của fmask-STE (T_A) và dual-path feedback (T_B1/T_B2) — thay vì các con số frozen thiên lệch.

## Done

- [x] Literature grounding Phase 4: fallback losses (negative-area Dice, Lovász, background-aware), gating scope, novelty check — `docs/literature_grounding_phase4.md`
- [x] Tinh chỉnh giả thuyết train end-to-end: T_A / T_B1 / T_B2 / T_BN / T_AB với prediction bác bỏ được + ngưỡng — `docs/hypotheses_phase4.md`
- [x] Thiết kế ablation train 4 cells, điều kiện khớp chính xác (batch=2, seed 42, CoarseDropout mới, cùng split) — `docs/experiment_design_phase4.md`
- [x] Cài đặt fallback loss `NegativeAreaDiceBCELoss` (phạt FP theo negative-area) — `src/fanet/losses.py`
- [x] Viết notebook Kaggle Phase 4 (batch=2, CoarseDropout num_holes, gate/dual_path, checkpoint per cell, train log CSV) — `notebooks/fanet_kaggle_phase4.py`
- [x] Mở rộng `scripts/train.py`: `--gate binary|ste|soft`, `--dual-path`, `--loss dicebce|negdice`, tên checkpoint/log theo cell
- [x] Viết eval script cho checkpoint trained — `analysis/abl2x2_eval_trained.py`
- [x] **CPU sanity run**: cả 4 cells train được 1 epoch (loss ~0.79, không crash), checkpoint load đúng vào từng variant; fallback negdice chạy được (loss ~1.25)

## Findings quan trọng

### 1. Code Phase 4 sẵn sàng và đã smoke-test trên CPU

| Cell | Lệnh (CPU test) | Kết quả 1 epoch |
|---|---|---|
| T00 | `--gate binary` | train 0.793 / val 0.763 |
| T10 | `--gate ste` | train 0.795 / val 0.830 |
| T01 | `--gate binary --dual-path` | train 0.794 / val 0.759 |
| T11 | `--gate ste --dual-path` | train 0.793 / val 0.757 |
| F1 | `--loss negdice` | train 1.253 (penalty term nâng loss, đúng thiết kế) |

Tất cả checkpoint smoke-test load lại đúng vào variant tương ứng → kiến trúc và đường dữ liệu hợp lệ. Đây là điều kiện tiên quyết trước khi đốt GPU.

### 2. Fallback đã sẵn sàng (quyết định sau khi có kết quả train)

- **F1 negative-area Dice** (`--loss negdice`): phạt FP trực tiếp trong loss, giữ nguyên feedback mechanism — paper gốc IEEE Access 2020.
- **F2 decoder-only suppression**: gate m_bg chỉ ở decoder, bảo toàn BN encoder (hướng EMCAD CVPR 2024).

### 3. Novelty check: pending trên Semantic Scholar (429) + Google Scholar

OpenAlex không có tiền lệ "background-aware feedback loop". S2 bị rate-limit (5/6 queries 429), Google Scholar chưa truy cập — novelty claim cần xác nhận thêm trước khi viết paper.

## Experiments

| Giả thuyết | Config | Kết quả | Link log |
|-----------|--------|---------|----------|
| Smoke test 4 cells train được trên CPU | 1 epoch/cell, seed 42, batch 2 | OK: loss ~0.79, checkpoint load đúng 4 variants | `scripts/train.py` |
| Fallback negdice chạy được | 1 epoch, `--loss negdice` | OK: loss 1.253 (có penalty) | `src/fanet/losses.py` |
| T_A/T_B1/T_B2/T_BN/T_AB (train đủ 200ep) | Kaggle T4, notebook phase4 | **CHỜ CHẠY KAGGLE** | `notebooks/fanet_kaggle_phase4.py` |

## Will Do (On going)

- [ ] **Chạy notebook Phase 4 trên Kaggle T4** (~40 phút/cell, 4 cells ≈ 2.5-3h): đảm bảo data kvasir-sessile được Add Data, check `DATASET_PATH` detect đúng
- [ ] Sau khi có 4 checkpoint trained: chạy `analysis/abl2x2_eval_trained.py` → so sánh paired (Wilcoxon + Cohen's d + bootstrap CI + Bonferroni)
- [ ] Kiểm tra BN mismatch (T_BN): đo BN running stats / activation distribution dual-path vs binary
- [ ] Figures: dice curves khi train (từ train_log CSV), final metrics, oracle gap, stats forest plot
- [ ] Quyết định fallback: nếu T_B1 bác bỏ → chạy F1 (negdice) hoặc F2 (decoder-only)
- [ ] Multi-seed ≥3 cho cell thắng (điều kiện claim paper)
- [ ] Novelty: retry Semantic Scholar + Google Scholar cho "background-aware feedback loop"
- [ ] Multi-dataset / deep supervision chỉ ghi Will Do (giữ trọng tâm A+B)

## Any Stuck / Open Questions

- **Không thể train đủ 200ep trên CPU local** (1.5-2 phút/epoch → ~5-6h/cell, 4 cells = 1 ngày). Bắt buộc dùng Kaggle T4. Cần user thực hiện bước: upload `notebooks/fanet_kaggle_phase4.py`, Add Data kvasir-sessile, bật GPU, Run All.
- Semantic Scholar rate-limited (HTTP 429) — cần retry sau hoặc dùng key.
- Lưu ý: `DATASET_PATH` trong notebook detect tự động nhưng cần xác nhận split khớp local (156/40) trước khi train.

## Đính kèm link chi tiết

- Literature: `docs/literature_grounding_phase4.md` — Hypotheses: `docs/hypotheses_phase4.md` — Design: `docs/experiment_design_phase4.md`
- Notebook: `notebooks/fanet_kaggle_phase4.py` — Train script: `scripts/train.py` (gate/dual/loss flags)
- Loss fallback: `src/fanet/losses.py` (NegativeAreaDiceBCELoss)
- Eval: `analysis/abl2x2_eval_trained.py` — Stats: `analysis/abl2x2_stats.py` (reuse)
- Baseline frozen: `checkpoints/checkpoint_200ep_kaggle.pth`
- Template: `docs/reports/_TEMPLATE.md` — Index: `docs/reports/README.md`