# PROMPT — Phản hồi advisor (2 điểm) cho bài research FANet

## Vai trò & luật làm việc

Bạn là research assistant khoa học làm việc trong repo `D:\Giselle_\My Project\FANet`.
Đọc và tuân thủ `AGENTS.md` NGAY TỪ ĐẦU:

- Sau MỖI file code/tài liệu tạo hoặc sửa: `git status` → `git diff` → `git add <đúng file>` → `git commit -m "..."` (message tiếng Anh, phong cách hiện có) → `git push`. KHÔNG stage file không liên quan.
- Mọi phase nghiên cứu phải viết report theo `docs/reports/_TEMPLATE.md` và cập nhật index `docs/reports/README.md`.
- Khi tra literature/papers BẮT BUỘC dùng scientific skills (load bằng skill tool): paper-lookup, literature-review, citation-management, bgpt-paper-search, exa-search, research-lookup, pdf-translate (nếu cần dịch PDF), hypothesis-generation, experimental-design, statistical-analysis, scientific-visualization, scientific-writing, markdown-mermaid-writing. Lưu citation đầy đủ (title/venue/năm/DOI) vào `docs/literature_grounding_*.md`.
- Mọi claim phải gán loại bằng chứng: TRAIN (đáng tin) vs FROZEN EVAL (chỉ là khả năng khai thác feedback của model cũ) vs INVALID (bug). KHÔNG claim significance khi chưa đủ power.
- KHÔNG tự chạy GPU. Mọi training chạy trên Kaggle notebook do USER submit — bạn phải dừng lại và xin user xác nhận trước khi soạn notebook tốn GPU.

## Feedback của advisor (trích nguyên văn)

"1) Nên chạy thử no-feedback baseline. Hiện tại nếu em đang thay loss để xử lý over-prediction, trong khi feedback của FANet cũng được thiết kế để prune/suppress unwanted regions. Nếu giữ cả hai ngay từ đầu thì khó biết improvement đến từ loss hay feedback. Em có thể chạy 4 ablation: no-FB/original loss, FB/original loss, no-FB/new loss, FB/new loss.
2) Nếu mục tiêu của em specifically là giảm FP thì nên theo dõi Precision–Recall/FPR và cân nhắc asymmetric loss. Còn anh nghĩ công thức pixel-weight này có khi chỉ authors đưa ra adapt với bài toán của họ thôi, thực tế em nên tính pixel weight dựa trên training set (dựa trên tần suất xuất hiện của các class) thì thực tế hơn, btw em cứ thử thôi."

## Nhiệm vụ tổng quát

Phân tích phản biện 2 điểm feedback đó dựa trên BẰNG CHỨNG HIỆN CÓ của repo, xác định chính xác phần nào đã được Phase 5 trả lời (tránh chạy lại), phần nào là hướng MỚI đáng làm, rồi thiết kế + thực thi thí nghiệm LẤP GAP TỐI THIỂU theo đúng chuẩn khoa học của repo. Có checkpoint xin user xác nhận trước mỗi bước tốn GPU.

## Bước 1 — Đọc context bắt buộc (trước khi làm gì khác)

- `docs/reports/2026-09-08_research-overview.md` (toàn cảnh + bản đồ hướng nghiên cứu)
- `docs/reports/2026-09-08_phase5-results.md`, `docs/reports/2026-09-07_phase5-x2-gate01.md`, `docs/reports/2026-09-04_next-directions.md`
- `src/fanet/losses.py`, `scripts/train.py`, `configs/kvasir_sessile.yaml`
- `analysis/phase5_eval.py`, `analysis/phase5_stats_full.py`, `analysis/phase5_figures.py`
- `notebooks/fanet_kaggle_phase5.py`, `kaggle/fanet-phase5.ipynb`
- `results/phase5_eval.json`, `results/phase5_stats_full.json`

## Bước 2 — Audit: map feedback vào bằng chứng Phase 5 (giao bảng rõ ràng)

Feedback điểm 1 đề nghị lưới 2×2 {no-FB, FB} × {original loss, new loss}. Hãy lập bảng đối chiếu và nêu rõ:

| Cell | Đã chạy? | ID | Seed/protocol | Kết quả chính |
|---|---|---|---|---|
| no-FB × orig loss | Có | T0N (Phase 5) | seed 43 | Dice 0.8234, FP 6.53% |
| FB × orig loss | Có | T00 (Phase 4 disk) | seed 42 | Dice 0.8150, FP 4.62% |
| FB × new loss #1 (WSDice cell A) | Có | TA | seed 43 | Dice 0.2853, FP 5.19% — FAIL |
| FB × new loss #2 (far-weighted γ=5) | Có | TB | seed 43 | COLLAPSE Dice 0.0086 — FAIL |
| no-FB × new loss | **THIẾU** | — | — | — |
| new asymmetric loss / class-frequency-weight loss | **CHƯA CÓ** (chưa implement) | — | — | — |

Phân tích phản biện — trả lời thẳng advisor bằng bằng chứng:

1. Feedback của FANet có đúng là "prune/suppress unwanted regions" không, và phần đó ĐÃ ĐƯỢC ĐO riêng chưa? (Gate 0 Phase 5: bỏ feedback làm Dice nhích lên không significant nhưng FP TĂNG +1.91pp → feedback có tác dụng "prune" nhẹ FP ở inference nhưng không giúp Dice; kết luận an toàn đã ghi trong report).
2. Vì sao "FB × orig loss" dùng T00 seed 42 làm baseline là confound về seed so với các cell Phase 5 seed 43 — nêu rõ để thiết kế Bước 4 xử lý.
3. Việc chạy LẠI đủ 4-cell 2×2 như advisor mô tả có lặp lại cell đã chạy không → đề xuất phương án lấp gap tối thiểu (Bước 4) thay vì chạy lại toàn bộ. Ghi nhận feedback đúng ở chỗ: nếu định claim "loss mới giúp FP" thì bắt buộc phải có nhánh no-FB × new-loss để tách hiệu ứng loss khỏi hiệu ứng prune của feedback.
4. Sản phẩm trung gian: soạn "memo trả lời advisor" ngắn (5–10 dòng) tóm tắt: điểm nào feedback đã được repo trả lời (kèm evidence), điểm nào là mới và sẽ làm. Đưa memo này vào report Bước 6 để user gửi advisor.

## Bước 3 — Literature grounding (BẮT BUỘC dùng skills, đa nguồn, verify full-text)

Tra và ground đầy đủ (OpenAlex + Semantic Scholar + arXiv + Europe PMC; dùng bgpt-paper-search để lấy chi tiết thực nghiệm) các chủ đề sau, lưu vào `docs/literature_grounding_phase6.md`:

1. **Asymmetric loss cho FP suppression trong segmentation nhị phân** (chú ý nguồn gốc multi-label của ASL — Ridnik 2020): asymmetric focal, asymmetric dice, Tversky β<0.5, Unified Focal loss, weighted BCE/FP-weighted; bằng chứng áp dụng trong polyp segmentation & class-imbalance.
2. **Class-frequency pixel weighting tính từ training set**: median-frequency balancing (ENet), inverse / sqrt-inverse frequency, class-balanced loss (Cui 2019); chuẩn "tính weight trên TRAIN split, không dùng val" và vì sao (tránh leakage); cách normalize.
3. **Công thức pixel-weight mà advisor gọi là "authors adapt"**: đối chiếu F³Net Eq.4 / CFA-Net (1+5μ) — thứ đã được đảo thành far-weighted cell B và COLLAPSE ở Phase 5 — với class-frequency weighting; phân biệt rõ "pixel-importance theo hình học biên" (μ) vs "class-frequency weight" (advisor đề xuất). Verify lại full-text để xác định advisor đang trỏ vào công thức nào.
4. **Tiền lệ báo cáo factorial/ablation** tách hiệu ứng cơ chế vs loss (2×2 + interaction) trong segmentation; và chuẩn báo cáo Precision/Recall/FPR cho polyp (Kvasir-SEG, Metrics Reloaded).

Kết luận literature phải đưa ra: (a) công thức asymmetric loss được khuyến nghị + hyper-parameter + citation DOI đã verify; (b) công thức class-frequency weight cụ thể (kèm normalization); (c) cảnh báo rủi ro collapse kiểu TB (weight nền quá mạnh → model đoán toàn background) và biện pháp bắt buộc (theo dõi binary val-Dice/P/R mỗi epoch, không tin soft-dice).

## Bước 4 — Thiết kế thí nghiệm + pre-registration (skills: experimental-design, hypothesis-generation, statistical-analysis)

Soạn plan pre-registered (ghi vào report Bước 6, KHÔNG chạy GPU trước khi user duyệt). Plan phải xử lý:

1. **Thiết kế lấp gap tối thiểu** — 2 phương án để user chọn:
   - Phương án A (nghiêm ngặt, tốn GPU ~14h T4): chạy LẠI 4 arm cùng protocol + cùng seed (vd 43) để lưới 2×2 sạch, hết confound seed 42 vs 43, tính được interaction FB×loss đúng nghĩa.
   - Phương án B (tái sử dụng, rẻ hơn): dùng T0N/T00 (orig-loss 2 arm, chấp nhận caveat seed 42 vs 43 — ghi rõ), CHỈ train thêm các cell MỚI: {FB, no-FB} × {asymmetric loss} và {FB, no-FB} × {class-frequency-weighted loss} (hoặc cell gộp) cùng seed 43; screening 1 seed trước, chỉ multi-seed nếu qua cổng.
   Trình bày trade-off (sạch về thống kê vs tiết kiệm GPU + tái dùng bằng chứng đã có) và khuyến nghị.

2. **Cổng quyết định (stopping rules)**, giữ nguyên convention repo: cell qua screening nếu `FP < baseline − 0.01` VÀ `Dice ≥ baseline − 0.02` VÀ `FN ≤ baseline + 0.02`; diverge rule ở epoch 40 theo binary val-Dice (KHÔNG theo loss-scale — bài học TB soft-dice che giấu collapse). Chỉ qua multi-seed khi screening pass.

3. **Metrics**: PRIMARY = FP%/FPR tính per-image ở ngưỡng nhị phân 0.5; SECONDARY = Dice, IoU, Precision, Recall, FN%. Báo cáo per-image distribution (không chỉ mean). Mở rộng log train để ghi binary val-Dice/Precision/Recall/FPR mỗi epoch.

4. **Power/sensitivity**: n=40 val, SD(ΔDice)≈0.30 → δ phát hiện được ≈0.13 (80% power, 1 seed); FP có variance nhỏ hơn Dice → FP là metric chính. Nêu rõ giới hạn, đừng hứa hẹn claim nhỏ hơn.

5. **Kế hoạch thống kê** (nhất quán Phase 5 để so sánh được): per-image paired Paired T-Test + rank-biserial r + bootstrap 95% CI + Bayes BF10; nêu caveat so sánh chéo model (cùng val set nhưng train độc lập); alpha pre-registered; Bonferroni nếu multi-comparison. Cách ước lượng main effect (FB, loss) + interaction từ lưới 2×2 và diễn giải cho paper.

## Bước 5 — Implementation (chỉ code + smoke test CPU, chưa GPU)

- Thêm loss classes vào `src/fanet/losses.py`: asymmetric loss và/hoặc class-frequency-weighted loss theo chốt ở Bước 3. Weight tần suất class tính từ **training split** một lần (script nhỏ + lưu JSON vào `data/` hoặc `configs/`), không bao giờ từ val.
- Mở rộng CLI `scripts/train.py` (`--loss`, `--no-feedback`, `--seed`, path class-weights, v.v.) và mở rộng eval để log binary Precision/Recall/FPR.
- Smoke test CPU 1 epoch mọi cell mới: không crash, loss-scale hợp lý, và sanity gradient (gradient đẩy FP background đi xuống đúng hướng). So sánh loss-scale với các cell Phase 5 (T0N 0.785/0.755, TA 2.66/2.53, TB 5.29/5.53) để phát hiện sớm diverge.
- Tạo `notebooks/fanet_kaggle_phase6.py` + `kaggle/fanet-phase6.ipynb` (log binary val-metrics mỗi epoch). Commit/push từng bước. DỪNG và xin user submit Kaggle.

## Bước 6 — Sau khi user submit xong (download → eval → stats → report)

- Download checkpoint/log về `checkpoints_phase6/`; viết `analysis/phase6_eval.py` + `phase6_stats_full.py` + `phase6_figures.py` (kế thừa protocol Phase 5); output `results/phase6_*.json` + figures vào `kaggle/figures/`.
- Chạy toàn bộ stats plan Bước 4; phán quyết từng cell theo cổng; ước lượng main effects + interaction của lưới 2×2; kết luận: improvement (nếu có) đến từ loss, từ feedback, từ cả hai (interaction) hay không từ đâu → cập nhật câu chuyện negative-result paper.
- Viết report phase theo `_TEMPLATE.md` với đủ Notation/Done/Findings/Experiments/Will Do/Open Questions + mermaid verdict flow. Đính kèm "memo trả lời advisor". Cập nhật `docs/reports/README.md` và `docs/reports/2026-09-08_research-overview.md` (bản đồ hướng + PHẦN E). Commit/push.

## Constraints

- KHÔNG chạy GPU local; KHÔNG submit Kaggle thay user; KHÔNG multi-seed khi screening chưa qua.
- KHÔNG claim significance thiếu power; report null đúng cách (effect size + CI); phân loại bằng chứng TRAIN/FROZEN/INVALID ở mọi kết luận.
- Mỗi lần tạo/sửa file → commit + push ngay (message tiếng Anh, chỉ stage file liên quan).
- Trước khi tốn GPU: dừng, trình plan cho user duyệt. Nếu có điểm mơ hồ về thiết kế thí nghiệm hoặc hướng loss → hỏi user, đừng tự đoán.
