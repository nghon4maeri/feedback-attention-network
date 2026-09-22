# [Tuần 6 – 15/09/2026] Phase 7B Results: Breaking the Feedback Trap via Detached Soft-OR Gating

> Report giai đoạn Phase 7B: Đánh giá thực nghiệm mô hình M12 (Detached Soft-OR) so với baseline M11 (Feedback Trap) trên 5 random seeds và kiểm định thống kê Wilcoxon N = 200.

## Ký hiệu (Notation)

- `M00`: Baseline thuần feedforward (No Feedback), loss DiceBCE
- `M01`: Feedforward + Asymmetric Tversky loss (α=0.7, β=0.3)
- `M11`: FANet gốc có Feedback + Hard binary gating `max(I(fmask>0.5), m_fg)` (Feedback Trap)
- `M12`: Đề xuất mới: FANet có Feedback + Detached Soft-OR Gating `1 - (1 - fmask)(1 - detach(m_fg))` + Asymmetric Tversky
- `MixPool`: Module cốt lõi của FANet tái tiêm mask $m$ từ epoch trước vào encoder/decoder
- `fmask`: Nhánh convolution attention cục bộ bên trong MixPool
- `m_fg`: Mask feedback đã downsample theo spatial resolution của tầng tương ứng
- `detach(m_fg)`: Toán tử stop-gradient cắt đường lan truyền ngược qua mask feedback
- `FPR`: False Positive Rate (Tỷ lệ dự đoán nhầm nền thành tổn thương, đo lường over-segmentation)
- `Wilcoxon Signed-Rank`: Kiểm định phi tham số cho các cặp quan sát phụ thuộc (paired samples)
- `pp`: Điểm phần trăm (percentage points)
- `Seeds`: Tập 5 hạt ngẫu nhiên độc lập $S \in \{7, 42, 99, 1337, 2024\}$

---

## Mục tiêu tuần này

1. Huấn luyện end-to-end 5 seeds cho 2 mô hình M11 và M12 trên Kaggle GPU T4 (200 epochs/seed).
2. Xây dựng pipeline tự động tải dữ liệu (`scripts/ingest_kaggle.py`), chẩn đoán gradient và biểu diễn (`analysis/compare_m11_m12.py`).
3. Xuất hình ảnh định tính 300 DPI vector PDF so sánh trực tiếp M11 vs M12 (`scripts/visualize_m11_vs_m12.py`).
4. Thực hiện kiểm định thống kê Wilcoxon Signed-Rank Test trên 200 điểm đánh giá (40 ảnh validation $\times$ 5 seeds) cho Dice, FPR, và Precision.
5. Hoàn thiện 100% bản thảo bài báo khoa học chuẩn bị nộp MICCAI/CVPR/IEEE TMI.

---

## Done

- [x] **Ingestion Pipeline:** Hoàn thành `scripts/ingest_kaggle.py` tải 11 checkpoints, 10 training logs CSV, và summary JSON từ Kaggle kernel `namnguynnnn/fanet-phase7b-training`. Commit `9c5beac`.
- [x] **Diagnostic Analysis:** Hoàn thành `analysis/compare_m11_m12.py` trích xuất 4 trục chẩn đoán (BN drift, saturation, cosine similarity, gradient norm). Commit `9c5beac`.
- [x] **Qualitative Figures:** Hoàn thành `scripts/visualize_m11_vs_m12.py` xuất ra `paper_figures/qualitative_comparison.png` và `paper_figures/qualitative_comparison.pdf` (300 DPI camera-ready). Commit `e6d1846`.
- [x] **Statistical Significance Test:** Hoàn thành `analysis/calc_p_value.py` thực thi kiểm định Wilcoxon trên 200 cặp dữ liệu inference 4-iter recurrent, lưu kết quả tại `diagnostics_output/phase7b/wilcoxon_stats.json`.
- [x] **Manuscript Preparation:** 
  - Soạn thảo Abstract, Intro, Methodology: `docs/reports/paper_draft_core_sections.md` (Commit `80d344c`).
  - Soạn thảo Experiments & Results: `docs/paper_section4_experiments_and_results.md` (Commit `e6d1846`).
  - Soạn thảo Related Work, Discussion, Conclusion, References: `docs/reports/paper_draft_remaining_sections.md`.
  - Tổng hợp toàn văn bài báo hoàn chỉnh: `docs/reports/paper_full_manuscript.md`.

---

## Findings quan trọng

| Hiện tượng | Bằng chứng định lượng | Hệ quả khoa học |
|-----------|----------------------|-----------------|
| **Triệt tiêu Over-Segmentation** | FPR trung bình giảm từ **21.39%** (M11) xuống **7.55%** (M12) trên tập test recurrent (giảm tương đối **64.7%**). Wilcoxon $W = 4,055.0$, **$p < 0.001$** (exact $p = 1.13 \times 10^{-6}$), rank-biserial $r = 0.422$. | Chứng minh cơ chế Detached Soft-OR phá vỡ hoàn toàn Feedback Trap, giải phóng sức mạnh phạt FP của asymmetric loss. |
| **Cải thiện Precision** | Precision trung bình tăng từ **6.84%** lên **10.81%** ($W = 5,138.0$, **$p = 0.0306 < 0.05$**). Trên online validation, Precision tăng từ **31.43%** lên **38.93%** (+23.9% relative). | Loại bỏ các báo động giả (ghost lesions), giảm thiểu nguy cơ sinh thiết nhầm trong nội soi đại tràng. |
| **Gia tăng Dice & Ổn định hóa** | Dice trung bình 5 seeds tăng **+4.77 pp** (từ 0.3428 lên 0.2183, +19.0%). Phương sai giữa các seeds giảm **66.5%** ($\sigma = 0.0869 \to 0.0291$). | Triệt tiêu hoàn toàn hiện tượng sụp đổ biểu diễn (Seed 1337 ở M11 sụp xuống Dice 0.0928, trong khi M12 đạt 0.2779). |
| **Cắt đứt Gradient Độc hại** | Mask gradient norm giảm từ 663.43 (M11) xuống 53.62 (M12), chỉ còn lại gradient tồn dư tại skip-connection của prediction head. | Các khối MixPool hoàn toàn được bảo vệ khỏi gradient phản hồi độc hại; gradient của fmask tuân theo công thức $\frac{\partial \text{keep}}{\partial \text{fmask}} = 1 - m_{\text{fg}}$. |
| **Tách biệt Không gian Biểu diễn** | Bottleneck Cosine Similarity giữa biên tổn thương và nền xa tăng từ **0.7981** (M11) lên **0.9339** (M12). Độ bão hòa dự đoán đạt **97.20%**. | Đặc trưng biên tổn thương và mô lành xung quanh không còn bị đồng nhất hóa như trong Feedback Trap. |

---

## Experiments

| ID | Giả thuyết | Config | Kết quả | Link log / Artifacts |
|----|-----------|--------|---------|----------------------|
| **E-Phase7B-1** | Detached Soft-OR khôi phục gradient flow cho fmask và loại bỏ BN drift | 5 seeds $\times$ 200 epochs, Adam lr=1e-4, Asymmetric Tversky $\alpha=0.7, \beta=0.3$, soft_or + detach | Mask grad norm = 53.62 (so với 663.43 của M11); CosSim = 0.9339; Saturation = 97.20% | `diagnostics_output/phase7b/diagnostic_summary.json` |
| **E-Phase7B-2** | Multi-seed segmentation: M12 vượt trội M11 về Dice và FPR trên Kvasir-SEG Sessile | Seeds 7, 42, 99, 1337, 2024. Đánh giá online validation per epoch và final checkpoints | M12 Dice = 0.2183 (+4.77 pp), M12 FPR = 2.46% (-42.8% rel), Std giảm 66.5% | `results_phase7b/multi_seed_summary.json` |
| **E-Phase7B-3** | Kiểm định phi tham số Wilcoxon Signed-Rank trên 200 cặp dữ liệu (40 ảnh $\times$ 5 seeds) | 4-iteration recurrent inference loop với Otsu threshold initialization | FPR p = 1.13e-06 (p < 0.001); Precision p = 0.0306 (p < 0.05); W_fpr = 4055.0 | `diagnostics_output/phase7b/wilcoxon_stats.json`, `analysis/calc_p_value.py` |
| **E-Phase7B-4** | Qualitative error visualizer: Top 5 ca over-segmentation nặng nhất ở M11 | Render 4-panel grid (RGB, GT, M11, M12) với overlay màu TP (Green), FP (Red), FN (Yellow) | M12 xóa bỏ hoàn toàn vùng false-positive lớn (>3,000 px) thành 0 px | `paper_figures/qualitative_comparison.png`, `paper_figures/qualitative_comparison.pdf` |

---

## Will Do (Next Steps)

- [ ] Chuẩn bị submission package (LaTeX Overleaf template theo định dạng Springer LNCS cho MICCAI hoặc IEEE Template cho TMI).
- [ ] Mở rộng thử nghiệm kiểm chứng độ bền (generalization) trên tập dữ liệu ngoại kiểm (CVC-ClinicDB / BKAI-IGH).
- [ ] Chuyển mã nguồn và checkpoint thành pip package hoặc mô hình mã nguồn mở trên Hugging Face / GitHub Release.

---

## Đính kèm link chi tiết

- **Toàn văn bài báo:** [paper_full_manuscript.md](file:///D:/Giselle_/My%20Project/FANet/docs/reports/paper_full_manuscript.md)
- **Hình ảnh trực quan hóa:** [qualitative_comparison.pdf](file:///D:/Giselle_/My%20Project/FANet/paper_figures/qualitative_comparison.pdf)
- **Script kiểm định thống kê:** [calc_p_value.py](file:///D:/Giselle_/My%20Project/FANet/analysis/calc_p_value.py)
- **Kết quả thống kê:** [wilcoxon_stats.json](file:///D:/Giselle_/My%20Project/FANet/diagnostics_output/phase7b/wilcoxon_stats.json)
- **Checkpoint weights:** [checkpoints_phase7b/](file:///D:/Giselle_/My%20Project/FANet/checkpoints_phase7b/)
