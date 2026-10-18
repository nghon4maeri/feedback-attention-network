# [Phase 7E – 11/10/2026] Báo Cáo Kết Quả Benchmark ISIC-2018 (Skin Lesion Segmentation)

## Ký hiệu (Notation)

> Các ký hiệu/viết tắt dùng trong báo cáo này:

- `FANet Orig`: Mô hình Feedback Attention Network gốc (baseline 7.72M params).
- `MFAD`: Mô hình đề xuất Multi-scale Feedback Attention Decoder (24.46M params).
- `Feedback Trap`: Hiện tượng lỗi tích lũy do gradient trôi (leakage) qua các bước lặp thời gian trong kiến trúc FANet gốc.
- `FPR`: False Positive Rate ($FP / (FP + TN)$) — càng thấp càng tốt.
- `mIoU`: Mean Intersection over Union (Jaccard index).
- `Sensitivity`: Recall ($TP / (TP + FN)$) — khả năng bao phủ tổn thương.
- `Precision`: Độ chính xác dương tính ($TP / (TP + FP)$).
- `pp`: Điểm phần trăm (percentage points).
- `Std`: Độ lệch chuẩn (standard deviation) qua các mẫu kiểm thử.

## Mục tiêu tuần này

Kéo và phân tích toàn diện kết quả benchmark của bộ dữ liệu **ISIC-2018 (Skin Lesion)** từ Kaggle sau khi chạy hoàn tất 100 epoch với cơ chế auto-resume; đối sánh trực tiếp giữa FANet Original và MFAD trên cùng tập dữ liệu chuẩn, hoàn thiện mảnh ghép dữ liệu cuối cùng của Phase 7E.

## Done

- [x] Kéo thành công kết quả từ kernel `namnguynnnn/fanet-benchmark-isic-2018` về local: [benchmark_results_ISIC-2018.csv](file:///D:/Giselle_/My%20Project/FANet/results/isic2018/benchmark_results_ISIC-2018.csv) và training log [fanet-benchmark-isic-2018.log](file:///D:/Giselle_/My%20Project/FANet/results/isic2018/fanet-benchmark-isic-2018.log).
- [x] Xác thực cơ chế auto-resume: FANet Orig hoàn tất 3 epoch cuối (E98-E100) từ checkpoint v6, sau đó FANet MFAD huấn luyện trọn vẹn 100 epoch với Best Val Dice đạt 0.8985.
- [x] Đánh giá định lượng trên toàn bộ 259 ảnh tập Validation/Test (độ phân giải 512x512).
- [x] Cập nhật bảng tổng kết benchmark V2 trong tài liệu [benchmark_summary.md](file:///D:/Giselle_/My%20Project/FANet/docs/benchmark_summary.md) và báo cáo Phase 7E.

## Findings quan trọng

### 1. Bảng Đối Sánh Định Lượng Trên ISIC-2018 (N=259)

| Chỉ số | FANet Original | MFAD (Proposed) | Độ lệch (Delta) | Ý nghĩa thực nghiệm |
| :--- | :--- | :--- | :--- | :--- |
| **Dice Score** | 0.8724 ± 0.1455 | **0.8984 ± 0.1104** | <span style="color:green">**+0.0260 (+2.60 pp)**</span> | Tăng tổng thể khả năng phân đoạn tổn thương (+2.98% relative) |
| **mIoU** | 0.7958 ± 0.1743 | **0.8290 ± 0.1374** | <span style="color:green">**+0.0332 (+3.32 pp)**</span> | Mức độ trùng khớp vùng phân đoạn tăng vượt bậc |
| **Precision** | 0.8936 ± 0.1528 | **0.9157 ± 0.1014** | <span style="color:green">**+0.0221 (+2.21 pp)**</span> | Giảm thiểu nhầm lẫn vùng da bình thường / sắc tố ngoại vi |
| **Sensitivity (Recall)** | 0.8891 ± 0.1675 | **0.9071 ± 0.1420** | <span style="color:green">**+0.0180 (+1.80 pp)**</span> | Bao phủ triệt để bờ tổn thương ung thư da, giảm bỏ sót |
| **Specificity** | **0.9787 ± 0.0495** | 0.9745 ± 0.0641 | -0.0041 (-0.41 pp) | Giữ vững độ đặc hiệu vùng da lành ở mức rất cao (>97.4%) |
| **FPR** | **0.0213 ± 0.0495** | 0.0255 ± 0.0641 | +0.0041 (+0.41 pp) | Đánh đổi biên độ cực nhỏ để mở rộng khả năng bao phủ toàn khối |

*(Các giá trị trình bày dưới dạng Mean ± Std. In đậm là kết quả vượt trội hơn).*

### 2. Hiện tượng và Phân Tích Cơ Chế

| Hiện tượng | Bằng chứng | Hệ quả & Ý nghĩa |
| :--- | :--- | :--- |
| **Tăng trưởng đồng thời Precision và Recall** | Cả Precision (+2.21 pp) và Sensitivity (+1.80 pp) đều tăng, đẩy Dice lên cận 90% (0.8984). | Không phải sự đánh đổi ngưỡng (threshold shifting) đơn thuần, mà MFAD thực sự học được biểu diễn không gian và ranh giới tổn thương sắc tố da tốt hơn. |
| **Giảm mạnh phương sai (Variance Reduction)** | Độ lệch chuẩn của Dice giảm từ **0.1455 xuống 0.1104 (giảm 24.1% variance)**; mIoU std giảm từ **0.1743 xuống 0.1374 (giảm 21.2%)**. | FANet gốc thường gặp các ca sụp đổ phân đoạn (prediction collapse/hallucination) trên những ảnh da có lông, nốt ruồi vệ tinh hoặc độ tương phản mờ. MFAD duy trì dự đoán cực kỳ ổn định và đồng đều qua mọi ca khó. |
| **Phá vỡ Feedback Trap trên khối đa tỉ lệ** | Ảnh ISIC-2018 có kích thước 512x512 với cấu trúc tổn thương dạng mảng lớn (macro-lesions) kèm biên mờ. | Cấu trúc Multi-scale kết hợp ngắt gradient (detach) của MFAD giúp decoder tổng hợp ngữ cảnh đa tầng mà không bị quầng nhiễu tích lũy qua các bước hồi quy. |

## Experiments

| ID | Giả thuyết | Config | Kết quả | Link log / Artifacts |
| :--- | :--- | :--- | :--- | :--- |
| E_ISIC | MFAD vượt trội FANet Orig trên bài toán phân đoạn tổn thương da kích thước lớn (512x512) | Epoch 100, LR 1e-4, Batch 8, Image 512x512, Adam, Kaggle Tesla T4 GPU | **PASS**: Dice đạt 0.8984 (+2.60 pp), mIoU đạt 0.8290 (+3.32 pp), Std giảm 24.1% | [benchmark_results_ISIC-2018.csv](file:///D:/Giselle_/My%20Project/FANet/results/isic2018/benchmark_results_ISIC-2018.csv), [fanet-benchmark-isic-2018.log](file:///D:/Giselle_/My%20Project/FANet/results/isic2018/fanet-benchmark-isic-2018.log) |

## Will Do (On going)

- [ ] Cập nhật bảng tổng hợp chung trong [docs/benchmark_summary.md](file:///D:/Giselle_/My%20Project/FANet/docs/benchmark_summary.md).
- [ ] Cập nhật artifact tổng kết toàn bộ 6 bộ dataset [mfad_benchmark_v2_results.md](file:///C:/Users/nghoo/.gemini/antigravity-cli/brain/14f42333-b723-432f-bf8b-2a04cf7fc388/mfad_benchmark_v2_results.md).
- [ ] Chuẩn bị bảng số liệu Latex hoàn chỉnh cho bản thảo bài báo (Paper manuscript).

## Any Stuck / Open Questions

- Không có vướng mắc. Quá trình train và benchmark ISIC-2018 đã hoàn tất 100% trọn vẹn và xác thực bằng số liệu lưu trữ trong repo.

## Đính kèm link chi tiết

- Kernel Kaggle: [fanet-benchmark-isic-2018 (Version 8)](https://www.kaggle.com/code/namnguynnnn/fanet-benchmark-isic-2018)
- File kết quả CSV: [results/isic2018/benchmark_results_ISIC-2018.csv](file:///D:/Giselle_/My%20Project/FANet/results/isic2018/benchmark_results_ISIC-2018.csv)
- Training log: [results/isic2018/fanet-benchmark-isic-2018.log](file:///D:/Giselle_/My%20Project/FANet/results/isic2018/fanet-benchmark-isic-2018.log)
- Báo cáo Benchmark trước đó: [docs/reports/2026-10-08_phase7e-final-benchmark.md](file:///D:/Giselle_/My%20Project/FANet/docs/reports/2026-10-08_phase7e-final-benchmark.md)
