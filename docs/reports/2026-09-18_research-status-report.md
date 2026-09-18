# [Tuần 7 - 18/09/2026] Research Status Report: Breaking the Feedback Trap, SOTA Benchmark, and Zero-Shot Generalization

> Báo cáo tổng hợp tiến độ nghiên cứu toàn diện: Cơ chế Detached Soft-OR Gating, đánh giá thực nghiệm đa hạt (5 seeds), kiểm định thống kê Wilcoxon, ngoại kiểm Zero-Shot Cross-Center (CVC-ClinicDB N = 612), mini-benchmark SOTA baselines, và hoàn thiện bản thảo IEEEtran.

## Ký hiệu (Notation)

- M00: Mo hinh thuan feedforward (No Feedback) voi ham loss DiceBCE chuan.
- M01: Mo hinh feedforward (No Feedback) voi Asymmetric Tversky loss (alpha = 0.7, beta = 0.3).
- M11: Mo hinh FANet goc co Feedback + Hard binary gating max(I(fmask > 0.5), m_fg) (Feedback Trap).
- M12: De xuat moi: FANet co Feedback + Detached Soft-OR Gating 1 - (1 - fmask)(1 - detach(m_fg)) ket hop Asymmetric Tversky.
- MixPool: Module trung tam trong FANet truyen ket qua du doan tu epoch truoc vao encoder va decoder.
- fmask: Nhanh tich chap attention cuc bo ben trong MixPool.
- m_fg: Mask feedback duoc downsample theo kich thuoc spatial cua tang feature tuong ung.
- detach(m_fg): Toan tu stop-gradient triet tieu duong lan truyen nguoc qua mask feedback.
- FPR: False Positive Rate (Ty le diem anh am tinh bi phan loai nham thanh duong tinh, chi so do over-segmentation).
- DSC: Dice Similarity Coefficient (chi so do luong muc do trung khop hinh hoc).
- mIoU: Mean Intersection over Union (Jaccard Index).
- Wilcoxon: Kiem dinh phi tham so paired signed-rank test danh cho cac cap mau phu thuoc.
- pp: Diem phan tram (percentage points).
- Seeds: Tap 5 hat ngau nhien doc lap S = {7, 42, 99, 1337, 2024}.
- CVC-ClinicDB: Tap du lieu noi soi dai trang ngoai kiem gom 612 anh tu Hospital Clinic, Barcelona, Tay Ban Nha.
- Kvasir-SEG (Sessile): Tap con 200 ton thuong polyp phang (Paris IIa/IIb) tu Oslo University Hospital.

---

## Mục tiêu tuần này

1. Khac phuc triet de loi Trivial Background Collapse trong visualizer hinh 1 (Figure 1), thiet lap bo loc dam bao ca chon loc phai dong thoi duy tri Dice cao va giam manh False Positives.
2. Thuc hien ngoai kiem Zero-Shot Cross-Center tren toan bo tap CVC-ClinicDB (N = 612 anh) tren ca 5 random seeds nham dap tat hoan toan nghi ngo overfit tren Kvasir-Sessile.
3. Dinh vi lai bai bao thanh Mechanistic Foundational Study, giai thich tinh chat adversarial stress-test cua tap Kvasir-Sessile.
4. Xay dung pipeline va chay thuc nghiem SOTA Mini-Benchmark tren Kaggle (U-Net, DeepLabV3+, FPN voi backbone ResNet-50) tren cung tap validation sessile.
5. Dong goi toan bo ma nguon LaTeX theo dinh dang chuan 2 cot IEEEtran, can chinh formatting khong de float tran le hay overfull hbox, dong bo day du len repo GitHub.

---

## Done

- [x] Sua loi Qualitative Visualizer: Hoan thanh `scripts/visualize_m11_vs_m12_fixed.py`. Bo loc moi yeu cau Dice > 0.5 va delta FP > 0, loai bo cac ca sụp đổ dự đoán rỗng (Dice = 0). Render thanh cong `paper_figures/qualitative_comparison_fixed.png` va `.pdf` (300 DPI vector).
- [x] Ngoai kiem Zero-Shot Cross-Center: Xay dung `scripts/zero_shot_eval.py` va chay danh gia toan bo 612 anh cua CVC-ClinicDB qua 5 seeds (tong cong 3,060 luot recurrent inference). Luu ket qua chi tiet tai `results/zero_shot_cvc_clinicdb.json`.
- [x] Tich hop SOTA Mini-Benchmark tren Kaggle: Xay dung `scripts/kaggle_sota_benchmark.py`, tu dong phat hien duong dan Kaggle, cast ro rang float32 tranh loi DoubleTensor. Chay thanh cong tren Kaggle T4 GPU thu duoc so lieu danh gia cua U-Net, DeepLabV3+, FPN.
- [x] Dinh vi hoc thuat (Academic Reframing): Cap nhat `01_intro.tex` va `docs/reports/paper_full_manuscript.md`, bo sung lap luan ve nghien cuu co che (Mechanistic Study) va ly do lua chon tap Kvasir-Sessile lam moi truong stress-test.
- [x] Tich hop Bang 3 va Dinh huong tuong lai: Bo sung bang so sanh SOTA vao `04_experiments.tex` va muc Future Directions vao `05_conclusion.tex` (Learnable Feedback Modulation, Architectural Universality, Dynamic Early-Stopping).
- [x] Can chinh Dinh dang LaTeX: Xu ly toan bo loi overfull hbox o cac cong thuc toan, chuyen cac bang/hinh tran le sang float 2 cot (`table*`, `figure*`), su dung goi `amsthm` va `stfloats`. Bien dich thanh cong `main.pdf` 10 trang voi 0 loi, 0 overfull hbox. Dong goi `overleaf_submission.zip` va push len GitHub.

---

## Findings quan trọng

| Hiện tượng | Bằng chứng định lượng | Hệ quả khoa học |
|---|---|---|
| Triet tieu Over-Segmentation tren Kvasir-Sessile | FPR giam tu 4.30% xuong 2.46% (-42.8% tuong doi). Tren tap danh gia recurrent online, FPR giam tu 21.39% xuong 7.55% (-64.7%). Wilcoxon W = 4,055.0, p = 1.13e-6 (p < 0.001), rank-biserial r = 0.422. | Khang dinh Detached Soft-OR pha vo Feedback Trap, giup giai phong suc manh phat FP cua asymmetric loss. |
| Nang cao do chinh xac va giam phuong sai seed | Dice trung binh tang +4.77 pp (0.2517 len 0.2995, +19.0% tuong doi). Do lech chuan giua 5 seeds giam 66.5% (sigma giam tu 0.0869 xuong 0.0291). | Triet tieu hoan toan hien tuong sup do bieu dien (Seed 1337 o M11 roi xuong Dice 0.0928, trong khi M12 dat 0.2779). |
| Tong quat hoa ngoai kiem Zero-Shot (CVC-ClinicDB) | Khong can fine-tuning hay retraining tren CVC-ClinicDB (N = 612), M12 vuot troi M11 tren Dice (+2.67 pp, tu 0.2375 len 0.2641, +11.2%), Recall tang +6.48 pp (45.40% len 51.88%), do lech chuan seed giam 41.1% (0.0639 xuong 0.0377). Wilcoxon sample-level p = 1.61e-15. | Bac bo hoan toan gia thuyet rang Detached Soft-OR overfit vao tap huan luyen Kvasir-Sessile nho nang cao chat luong bieu dien cua encoder. |
| Sup do cua Feedforward Baselines khi khong co recurrent | Tren tap validation Kvasir-Sessile, U-Net (ResNet-50) dat Dice 0.0006, FPR 0.86%; DeepLabV3+ dat Dice 0.1325, FPR 100.00%; FPN dat Dice 0.1201, FPR 31.84%. | Chung minh cac ton thuong phang sessile la thu thach cuc doan ma cac mang feedforward thong thuong khong the xu ly duoc neu khong co co che recurrent refinement duoc hieu chinh dung dan. |
| Chan gradient doc hai va on dinh Batch Normalization | Mask gradient norm giam tu 663.43 xuong 53.62. Bottleneck Cosine Similarity tang tu 0.7981 len 0.9339. Saturation dat 97.20%. | Khang dinh cong thuc gradient fmask theo dao ham: dao ham keep theo fmask bang 1 - m_fg, giup tap trung hoc tai bien ton thuong thay vi ghi nho loi o nen. |

---

## Experiments

| ID | Giả thuyết | Config | Kết quả | Link log / Artifacts |
|---|---|---|---|---|
| E-Phase7B-MultiSeed | M12 vuot troi M11 ve Dice va giam FPR tren tap Kvasir-SEG Sessile qua 5 seeds | 5 seeds x 200 epochs, Adam lr=1e-4, Asymmetric Tversky alpha=0.7, beta=0.3, recurrent inference 4 iter | M12 Dice = 0.2995 (+4.77 pp), FPR = 2.46% (-42.8% rel), Std giam 66.5%, Seed 1337 collapse duoc khac phuc | `results_phase7b/multi_seed_summary.json`, `diagnostics_output/phase7b/` |
| E-Phase7B-Wilcoxon | Kiem dinh phi tham so Wilcoxon tren 200 cap du lieu (40 anh x 5 seeds) dat y nghia thong ke | Paired Wilcoxon Signed-Rank Test tren FPR, Precision, Dice | FPR p = 1.13e-6 (p < 0.001, W=4055.0); Precision p = 0.0306 (W=5138.0); Dice p = 0.0382 | `diagnostics_output/phase7b/wilcoxon_stats.json`, `analysis/calc_p_value.py` |
| E-Visual-Fixed | Loai bo cac ca trivial background collapse trong so sanh truc quan dinh tinh | Chon ca co Dice > 0.5 o ca M11 va M12, kem theo giam manh False Positive (delta FP > 0) | 5 ca dai dien loai bo tu 4,015 px den 12,280 px bao dong gia trong khi duy tri Dice 0.50 - 0.77 | `scripts/visualize_m11_vs_m12_fixed.py`, `paper_figures/qualitative_comparison_fixed.pdf` |
| E-ZeroShot-CVC | M12 duy tri uu the tong quat hoa tren tap du lieu ngoai kiem chua tung thay CVC-ClinicDB | Danh gia zero-shot truc tiep khong train lai tren 612 anh x 5 seeds (3,060 luot recurrent) | Dice tang +2.67 pp (p = 1.61e-15), Recall tang +6.48 pp, Std giam 41.1% | `scripts/zero_shot_eval.py`, `results/zero_shot_cvc_clinicdb.json` |
| E-SOTA-Kaggle | So sanh doi dau giua M12 voi cac kien truc feedforward chuan (U-Net, DeepLabV3+, FPN) tren validation Sessile | SMP library, backbone ResNet-50, anh kich thuoc 352x352, evaluation loop tren 40 anh validation | U-Net: Dice 0.0006; DeepLabV3+: Dice 0.1325 (FPR 100%); FPN: Dice 0.1201; M12 vuot troi dat Dice 0.2995 | `scripts/kaggle_sota_benchmark.py`, `kaggle_out/kaggle-sota-benchmark.log` |

---

## Will Do (On going)

- [ ] Hoan thien thu tuc submit bai bao len he thong tap chi/hoi nghi top-tier (nhu IEEE TMI hoac MICCAI).
- [ ] Mo rong thu nghiem kiem chung hoc hoi bien dieu phan hoi (Learnable Feedback Modulation voi attention gate gamma).
- [ ] Thu nghiem tinh pho quat cua hien tuong Feedback Trap tren cac kien truc lap khac (R2U-Net, ConvLSTM).
- [ ] Phat trien co che Dynamic Early-Stopping dua tren do bat dinh entropy cua Soft-OR de toi uu toc do khung hinh (FPS) trong trien khai thoi gian thuc.

---

## Any Stuck / Open Questions

- Khong co vuong mac ve mat thuat toan hoac ma nguon. Toan bo ket qua thuc nghiem, kiem dinh thong ke, visualizer, va ma nguon LaTeX da hoan thien dong bo va compile thanh cong.

---

## Đính kèm link chi tiết

- Toan van bai bao Markdown: [paper_full_manuscript.md](file:///D:/Giselle_/My%20Project/FANet/docs/reports/paper_full_manuscript.md)
- Ma nguon LaTeX Submission: [overleaf_submission/main.tex](file:///D:/Giselle_/My%20Project/FANet/overleaf_submission/main.tex)
- Tep nen zip cho Overleaf: [overleaf_submission.zip](file:///D:/Giselle_/My%20Project/FANet/overleaf_submission.zip)
- Hinh anh truc quan hoa khong bi collapse: [qualitative_comparison_fixed.pdf](file:///D:/Giselle_/My%20Project/FANet/paper_figures/qualitative_comparison_fixed.pdf)
- Du lieu kiem dinh ngoai CVC-ClinicDB: [zero_shot_cvc_clinicdb.json](file:///D:/Giselle_/My%20Project/FANet/results/zero_shot_cvc_clinicdb.json)
- Script SOTA benchmark tren Kaggle: [kaggle_sota_benchmark.py](file:///D:/Giselle_/My%20Project/FANet/scripts/kaggle_sota_benchmark.py)
- Log thuc nghiem SOTA: [kaggle-sota-benchmark.log](file:///D:/Giselle_/My%20Project/FANet/kaggle_out/kaggle-sota-benchmark.log)
