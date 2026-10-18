# [CN – 09/09/2026] — Phase 6: Pre-registration — trả lời advisor (no-FB baseline + asymmetric / class-frequency loss)

## Ký hiệu (Notation)

- `FB` / `no-FB`: có vòng feedback mask-at-input / không feedback (mask = 0).
- `T00`: baseline FB × orig loss (dicebce), seed 42, 200ep — reference Phase 4 disk.
- `T0N`: no-FB × orig loss (dicebce), seed 43, 200ep — Phase 5.
- `TA` / `TB`: FB × loss mới — cell A (WSDice v1=0.3) / cell B (far-weighted γ=5), seed 43 — Phase 5. TB **COLLAPSE**.
- `TC` / `TD`: **cell mới Phase 6** — asymmetric loss (Tversky α>β) / class-frequency-weighted loss (ENet-bounded), seed 43.
- `Gate 0/1/2`: cổng quyết định (no-FB vs FB / screening 1 seed / multi-seed).
- `pp`: điểm phần trăm. `FPR`: false positive rate (per-image). `rank-biserial r`: effect size Wilcoxon.
- `ASL`: asymmetric loss (Ridnik 2021) · `UF`: Unified Focal (Yeung 2022) · `MFB`: median frequency balancing.
- Bằng chứng: `TRAIN` (đáng tin) / `FROZEN EVAL` (khai thác feedback model cũ) / `INVALID` (bug).

## Mục tiêu tuần này

Phản biện + trả lời 2 điểm feedback của advisor bằng bằng chứng có sẵn, ground
literature (asymmetric loss + class-frequency weight), và **pre-register** thí nghiệm
Phase 6 lấp gap tối thiểu (không chạy lại cell đã chạy). KHÔNG chạy GPU trước khi
user duyệt plan.

## Done

- [x] Bước 1: đọc toàn bộ context bắt buộc (overview, phase5-results, phase5-x2-gate01, next-directions, losses.py, train.py, config, analysis phase5, notebook phase5, phase5_eval.json, phase5_stats_full.json)
- [x] Bước 2: audit — map feedback vào bằng chứng Phase 5 (bảng 2×2 + memo trả lời advisor bên dưới)
- [x] Bước 3: literature grounding đa nguồn → `docs/literature_grounding_phase6.md` (ASL, Tversky, Focal Tversky, Unified Focal, ENet, Cui 2019, Kvasir-SEG, Metrics Reloaded — DOI verify)
- [x] Bước 4: pre-registration plan (report này, CHƯA chạy GPU)
- [x] **User chốt thiết kế:** Phương án B (tái dùng T0N/T00) · Tversky α=0.7/β=0.3 · chỉ asymmetric trước (không chạy class-frequency song song)
- [x] Bước 5: implement `TverskyLoss` + `Phase6AsymmetricBCELoss` (losses.py) + CLI `--loss tversky` + binary val-metric logging (train.py) → `f14107a` (commit `e85ce3b`)
- [x] Bước 5: smoke-test CPU 1 epoch 2 cells (TC: 0.851/0.827, binDice 0.2222; TD: 0.842/1.141, binDice 0.2294) + sanity gradient (finite-diff FP/FN penalty ratio 2.308 ≈ α/β)
- [x] Bước 5: tạo `notebooks/fanet_kaggle_phase6.py` + `kaggle/fanet-phase6.ipynb` (18 cells, log binary val-metrics mỗi epoch, diverge rule ep40) → commit `f14107a`
- [x] **User submit Kaggle** `kaggle/fanet-phase6.ipynb` (2 cells TC/TD × 200ep, seed 43) → kernel COMPLETE (09/09 07:20)
- [x] Bước 6: download → eval/stats/figures → report **`docs/reports/2026-09-09_phase6-results.md`** (TD PASS −3.25pp FP; TC FAIL; interaction +4.17pp)

---

## Findings quan trọng

### F1. AUDIT — map feedback điểm 1 vào bằng chứng Phase 5 (lưới 2×2)

| Cell | Đã chạy? | ID | Seed/protocol | Kết quả chính | Bằng chứng |
|---|---|---|---|---|---|
| no-FB × orig loss | **Có** | T0N | seed 43, 200ep | Dice 0.3034, FP 6.53%, FN 5.98% | `results/phase5_eval.json` (TRAIN) |
| FB × orig loss | **Có** | T00 | **seed 42**, 200ep (disk) | Dice 0.2806, FP 4.62%, FN 6.30% | `results/phase4_eval.json` (TRAIN) |
| FB × new loss #1 (WSDice cell A) | **Có** | TA | seed 43, 200ep | Dice 0.2853, FP 5.19% — **FAIL** (FP không giảm) | `results/phase5_eval.json` (TRAIN) |
| FB × new loss #2 (far-weighted γ=5) | **Có** | TB | seed 43, 200ep | **COLLAPSE** Dice 0.0086 (recall 0.0047) — **FAIL** | `results/phase5_eval.json` (TRAIN) |
| **no-FB × new loss** | **THIẾU** | — | — | — | — |
| **asymmetric / class-frequency loss** | **CHƯA CÓ** (chưa implement) | — | — | — | — |

### F2. AUDIT — trả lời thẳng advisor bằng bằng chứng

1. **Feedback FANet có "prune/suppress unwanted regions" không, và đã đo riêng chưa?**
   CÓ đúng thiết kế: MixPool `keep=max(fmask_g, m_fg)` + mask-at-input chỉ giữ vùng
   từng được cho là polyp → có tác dụng prune. **ĐÃ ĐƯỢC ĐO RIÊNG ở Gate 0 Phase 5**
   (T0N vs T00): bỏ feedback làm Dice nhích lên **không significant** (Δ+0.023,
   p=.36, BF10=0.21 → ủng hộ null) NHƯNG **FP TĂNG +1.91pp** (p=.155). Kết luận an toàn
   (đã ghi report 09/08): feedback có tác dụng "prune" nhẹ FP ở inference (T00 FP
   thấp hơn T0N) nhưng **không giúp Dice**, và **không phải nguồn gốc FP**. ⇒ Advisor
   đúng khi lo ngại nhầm lẫn hiệu ứng loss vs feedback; repo **đã có nhánh no-FB
   (orig loss)** để làm mốc.

2. **Confound seed 42 vs 43:** "FB × orig loss" = T00 dùng **seed 42**, còn mọi cell
   Phase 5 (T0N/TA/TB) dùng **seed 43**. So sánh T00 với các cell seed 43 trộn lẫn
   hiệu ứng seed vào hiệu ứng loss/feedback. Gate 0 (T0N vs T00) và Gate 1 (TA/TB vs
   T00) đều so chéo seed. Đây là điểm yếu thiết kế Phase 5 — Bước 4 xử lý bằng 2
   phương án (F4).

3. **Có lặp lại cell không?** Chạy LẠI đủ 4-cell 2×2 như advisor mô tả sẽ **lặp lại
   2 cell đã chạy sạch** (T0N, T00) và **2 cell FAIL đã đóng** (TA, TB). Thay vào đó
   lấp gap tối thiểu (F4). **Ghi nhận feedback đúng:** nếu muốn claim "loss mới giúp
   FP" thì BẮT BUỘC có nhánh **no-FB × new-loss** để tách hiệu ứng loss khỏi hiệu ứng
   prune của feedback — đây chính là gap thiếu ở Phase 5 (chỉ có FB × new-loss).

4. **Feedback điểm 2:** advisor đúng là `(1+5μ)` (CFA-Net/F³Net) là do tác giả adapt
   riêng (boundary-importance), và đề xuất class-frequency weight từ **training set**
   là hướng khác bản chất, hợp root cause FP. Chi tiết: `docs/literature_grounding_phase6.md` §3.

### F3. GAP THẬT cần lấp (chốt)

- **Gap 1 (bắt buộc để tách loss vs feedback):** no-FB × new-loss. Chưa có.
- **Gap 2 (hướng mới advisor gợi ý):** asymmetric loss (Tversky α>β) và class-frequency
  weight (ENet-bounded) — CHƯA implement, CHƯA test. Phase 5 chỉ thử WSDice (chênh lệch
  quá nhỏ) và far-weighted (collapse).

### F4. THIẾT KẾ LẤP GAP TỐI THIỂU — 2 phương án để user chọn

> Nguyên tắc: không chạy lại cell sạch (T0N/T00), không chạy lại cell FAIL (TA/TB).

**Phương án A — lưới 2×2 sạch (nghiêm ngặt, ~14 GPU-h T4):**
Chạy LẠI 4 arm cùng protocol + cùng seed 43: {FB, no-FB} × {orig, new} với **loss mới
duy nhất** (asymmetric Tversky α=0.7/β=0.3, hoặc cell gộp). Lợi: hết confound seed,
ước lượng được interaction FB×loss đúng nghĩa. Chi phí cao, và TB/TA đã chứng minh
loss-side khó; rủi ro chạy lại "sạch" mà vẫn fail → tốn GPU vô ích.

**Phương án B — tái sử dụng + chỉ thêm cell MỚI (rẻ hơn, khuyến nghị):**
- Giữ T0N (no-FB × orig) + T00 (FB × orig) làm 2 arm orig-loss, **chấp nhận caveat
  seed 42 vs 43 (ghi rõ trong report)**.
- CHỈ train thêm, seed 43: **TC** = FB × asymmetric, **TD** = no-FB × asymmetric
  (hoặc {TC, TD} cho class-frequency nếu muốn — đề xuất asymmetric trước).
- Screening 1 seed (43) trước; **chỉ multi-seed nếu screening pass**.
- Lợi: tận dụng bằng chứng đã có, tiết kiệm GPU, lấp đúng 2 gap (no-FB × new-loss +
  asymmetric loss). Nhược: interaction không sạch hoàn toàn (2 arm orig loss là seed
  42/43 khác nhau) → nếu kết quả dương tính mới quay lại làm phương án A.

> **Khuyến nghị: Phương án B** — rẻ, lấp đúng gap, theo đúng tinh thần "lấp gap tối
> thiểu" của advisor feedback. Chỉ nâng lên A nếu Phase 6 cho tín hiệu dương tính
> đáng tin (cần interaction sạch cho paper).

### F5. CỔNG QUYẾT ĐỊNH (stopping rules) — giữ nguyên convention repo

- **Screening pass** (mỗi cell TC/TD vs baseline tương ứng): `FP < baseline − 0.01`
  VÀ `Dice ≥ baseline − 0.02` VÀ `FN ≤ baseline + 0.02`.
  - Baseline cho TC (FB×asym) = **T00**; baseline cho TD (no-FB×asym) = **T0N**.
- **Diverge rule ở epoch 40:** theo **binary val-Dice** (KHÔNG theo loss-scale — bài
  học TB: soft-dice 0.19 che giấu collapse 0.0086). Nếu binary val-Dice tại ep40 tụt
  mạnh so với best-so-far đồng thời Recall ~0 và FPR ~0 → collapse → dừng cell.
- **Multi-seed (Gate 2) CHỈ khi screening pass.** Không pass → report null + pivot
  negative-result (đúng tinh thần 09/04, 09/08).

### F6. METRICS

- **PRIMARY** = **FP rate/FPR per-image** ở ngưỡng nhị phân 0.5 (FP variance nhỏ hơn
  Dice nhiều — đo Phase 5: FP SD≈0.057 vs Dice SD≈0.27 ⇒ FP là metric chính để phát
  hiện hiệu ứng).
- **SECONDARY** = Dice, IoU, Precision, Recall, FN%. Báo cáo **per-image distribution**
  (không chỉ mean) — đúng Metrics Reloaded (khuyến nghị Fβ khi có preference FP/FN).
- **Mở rộng log train:** ghi **binary val-Dice / Precision / Recall / FPR mỗi epoch**
  (không chỉ soft-dice) để áp diverge rule ep40.

### F7. POWER / SENSITIVITY (giữ nguyên Phase 5)

- n=40 val, SD(ΔDice)≈0.30 → δ phát hiện được ≈ **0.13** (80% power, 1 seed, α=.05).
- FP variance nhỏ hơn Dice ⇒ FP là metric chính; δ FP nhỏ hơn có thể phát hiện được.
- **Giới hạn rõ ràng:** không hứa hẹn claim nhỏ hơn ~0.1 Dice; nếu target effect nhỏ,
  phải multi-seed hoặc val set lớn hơn (full Kvasir-SEG). Bất kỳ kết luận nào cũng
  report effect size + CI, không đào p.

### F8. KẾ HOẠCH THỐNG KÊ (nhất quán Phase 5 để so sánh được)

- **Per-image paired Wilcoxon** + **rank-biserial r** + **bootstrap 95% CI** + **Bayes
  BF10** (pingouin). Giữ nguyên `analysis/phase5_stats_full.py` làm chuẩn.
- **Caveat so sánh chéo model:** cùng val set nhưng train độc lập → paired là hợp lý
  cho per-image (cùng 40 ảnh), nhưng khác seed (42 vs 43) là confound còn lại ở
  phương án B — ghi rõ.
- **Alpha pre-registered:** α = 0.05 per test; **Bonferroni** nếu multi-comparison
  (Phase 5 dùng α=0.0083 cho nhiều cell — tái áp).
- **Main effects + interaction (lưới 2×2):** ước lượng bằng chênh lệch trung bình
  per-image:
  - Main effect `loss` = mean[(new−orig) averaged over FB & no-FB].
  - Main effect `feedback` = mean[(FB−no-FB) averaged over orig & new].
  - Interaction `FB×loss` = (Δloss|FB) − (Δloss|no-FB); nếu ≈0 → hiệu ứng loss không
    phụ thuộc feedback (additive); khác 0 → synergy/antagonism. Diễn giải cho paper
    negative-result: improvement đến từ loss, feedback, cả hai, hay không đâu.

### F9. MEMO TRẢ LỜI ADVISOR (5–10 dòng — gửi kèm report)

> Cảm ơn anh. 2 điểm em đối chiếu với bằng chứng Phase 5:
> 1) Về no-feedback baseline: em ĐÃ chạy no-FB×orig loss (T0N, seed43) — bỏ feedback
>    làm Dice nhích lên nhưng không significant (p=.36, BF=0.21) và FP tăng +1.9pp, tức
>    feedback chỉ prune nhẹ FP ở inference, không phải nguồn gốc FP. Em đồng ý phải có
>    nhánh no-FB×new-loss để tách hiệu ứng loss khỏi feedback — gap này em đang lấp
>    (chạy thêm no-FB×asymmetric loss). Không chạy lại 2 cell FAIL (WSDice, far-weighted
>    collapse) đã đóng.
> 2) Về pixel-weight: em xác nhận (1+5μ) (CFA-Net/F³Net) là boundary-importance do tác
>    giả adapt riêng, và đã pivot far-weighted làm Dice collapse (γ=5) ở Phase 5. Em
>    chuyển sang class-frequency weight tính từ TRAIN split (ENet-bounded / inverse
>    log-frequency, tránh inverse thô để không collapse), kèm theo dõi binary val-Dice.
>    Sẽ báo anh kết quả sau khi chạy.

---

## Experiments (KẾ HOẠCH pre-registered — CHƯA chạy GPU, chờ user duyệt)

| ID | Giả thuyết | Config | Kết quả (dự kiến) | Link |
|----|-----------|--------|-------------------|------|
| Smoke TC | FB × asymmetric (Tversky α>β) chạy được | seed 43, 1 epoch CPU | ✅ 0.851/0.827, binDice 0.2222 | `scripts/train.py` |
| Smoke TD | no-FB × asymmetric chạy được | seed 43, 1 epoch CPU | ✅ 0.842/1.141, binDice 0.2294 | `scripts/train.py` |
| Sanity grad | Tversky đẩy FP xuống + FP/FN asymmetric | finite-difference | ✅ FP down / FN up; FP/FN ratio 2.308 ≈ α/β | `src/fanet/losses.py` |
| TC | FB × asymmetric giảm FP, giữ Dice (vs T00) | seed 43, 200ep | Chờ user submit Kaggle | `kaggle/fanet-phase6.ipynb` |
| TD | no-FB × asymmetric giảm FP, giữ Dice (vs T0N) | seed 43, 200ep | Chờ user submit Kaggle | ditto |
| (tùy chọn) TE/TF | FB/no-FB × class-frequency (ENet-bounded) | seed 43 | **Hoãn** (user chốt chỉ asymmetric trước) | — |

> **CHỜ USER SUBMIT KAGGLE** `kaggle/fanet-phase6.ipynb` (2 cells TC/TD × 200ep, seed 43).

## Will Do (On going)

- [x] **User duyệt plan** (phương án B + Tversky α=0.7/β=0.3 + chỉ asymmetric) — ✅ ĐÃ CHỐT
- [x] Implement losses (asymmetric Tversky) vào `src/fanet/losses.py`; CLI + eval-log binary metrics
- [x] Smoke test CPU 1 epoch mọi cell mới (loss-scale + sanity gradient đẩy FP xuống)
- [x] Tạo `notebooks/fanet_kaggle_phase6.py` + `kaggle/fanet-phase6.ipynb` (log binary val-metrics mỗi epoch)
- [ ] **User submit Kaggle** `kaggle/fanet-phase6.ipynb` (2 cells TC/TD, seed 43, ~3.5h×2)
- [ ] Sau khi user submit: download → `analysis/phase6_eval.py` + `phase6_stats_full.py` + `phase6_figures.py` → report Phase 6 (Bước 6)

## Any Stuck / Open Questions

- **Cần user quyết:** Phương án A (14h, sạch 2×2) hay B (rẻ, chấp nhận caveat seed)?
  Khuyến nghị B.
- **Cần user quyết:** asymmetric loss — Tversky `α=0.7/β=0.3` (đơn giản, verify được)
  hay Unified Focal asymmetric (mạnh hơn, phức tạp)? Khuyến nghị Tversky trước.
- **Cần user quyết:** có chạy cell class-frequency (ENet-bounded) song song không, hay
  chỉ asymmetric trước rồi mới tính đến?
- Loss mới thay thế hay combine với DiceBCE? (Đề xuất: `λ·DiceBCE + (1−λ)·Tversky`,
  λ=0.5 như pattern cell A / Unified Focal.)

## Đính kèm link chi tiết

- Literature grounding (asymmetric + class-frequency + F³Net/CFA-Net + metrics):
  `docs/literature_grounding_phase6.md`
- Prompt phiên: `docs/prompts/2026-09-09_advisor-feedback-prompt.md`
- Bằng chứng Phase 5: `docs/reports/2026-09-08_phase5-results.md`, `2026-09-07_phase5-x2-gate01.md`, `results/phase5_eval.json`, `results/phase5_stats_full.json`
- Plan trước: `docs/reports/2026-09-04_next-directions.md`
- Template: `docs/reports/_TEMPLATE.md` — Index: `docs/reports/README.md`
