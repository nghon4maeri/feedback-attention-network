# [Phase 7E - 08/10/2026] Báo cáo Final Benchmark MFAD

## Ký hiệu (Notation)

> Các ký hiệu/viết tắt dùng trong báo cáo này:

- FANet Orig: Mô hình Feedback Attention Network gốc (baseline).
- MFAD: Mô hình đề xuất Multi-scale Feedback Attention Decoder.
- BCE / IoU: Binary Cross Entropy / Intersection over Union.
- FPR / FN: False Positive Rate / False Negative.
- DSB / ISIC / CVC / Kvasir: Các tập dữ liệu y tế/sinh học chuẩn.
- Feedback Trap: Hiện tượng lỗi tích lũy do gradient trôi (leakage) qua các bước lặp thời gian trong kiến trúc FANet gốc, gây ra nhiễu "halo".
- pp: Điểm phần trăm (percentage points).

## Mục tiêu tuần này

Đánh giá Benchmark toàn diện trên quy mô lớn 6 tập dữ liệu (CVC-ClinicDB, Kvasir-SEG, CHASE-DB1, DRIVE, EM-Dataset, DSB-2018) để chứng minh hiệu quả thực nghiệm của cấu trúc MFAD so với FANet gốc, hoàn thiện số liệu cho bài báo chuẩn bị nộp.

## Done

- [x] Chạy lại toàn bộ Benchmark V2 với code tự động chống timeout và auto-resume.
- [x] Hoàn thiện lấy kết quả DSB-2018 (Nuclei) trên Google Colab A100.
- [x] Tổng hợp và phân tích 5 bộ Dataset chính (CVC-ClinicDB, Kvasir-SEG, CHASE-DB1, EM-Dataset, DSB-2018).
- [ ] Chờ kết quả chạy trọn vẹn 100 epoch của ISIC-2018 trên Colab (Kaggle đã chạy đến giới hạn timeout và tự lưu latest.pth).

## Findings quan trọng

Thực nghiệm đã chứng minh MFAD giải quyết xuất sắc vấn đề "Feedback Trap" thông qua cơ chế ngắt gradient (detach) kết hợp Gating:

| Hiện tượng | Bằng chứng | Hệ quả |
|-----------|-----------|--------|
| **Triệt tiêu nhiễu Halo (False Positives)** | Trên CVC-ClinicDB và EM-Dataset, FPR giảm mạnh từ 2-3 lần (ví dụ: CVC giảm từ 0.96% xuống 0.30%). | Precision tăng vọt (+6.84% trên CVC, +4.22% trên EM), cải thiện tổng thể Dice mà không hi sinh quá nhiều Recall. |
| **Gỡ rối ranh giới dày đặc** | Trên tập DSB-2018 (Nuclei), MFAD tăng đồng thời cả Precision (+2.38%) và Sensitivity (+0.43%). | Chứng minh MFAD học được biểu diễn hình thái học tốt hơn thay vì chỉ đơn thuần thắt chặt ngưỡng (threshold). Cải thiện Dice +1.34%. |
| **Nhược điểm với cấu trúc vi mạch mỏng** | Trên CHASE-DB1 (Vessels), Dice giảm nhẹ -2.2%. | Việc ngắt gradient (detach) của MFAD làm mất tín hiệu liên kết liền mạch ở các mao mạch siêu mỏng. FANet gốc với gradient liên tục đóng vai trò như một regularizer tốt hơn cho dạng topological này. |

## Experiments

| ID | Giả thuyết | Config | Kết quả | Link log |
|----|-----------|--------|---------|----------|
| 1 | Benchmark DSB-2018 | Epoch 100, LR 1e-4, Batch 16 (Colab A100) | Dice: +1.34%, mIoU: +1.68%, Prec: +2.38% | 
otebooks/colab/benchmark_v2_fanet_vs_mfad_dsb2018_colab.ipynb |
| 2 | Benchmark CVC-ClinicDB | Kaggle (như cũ) | Dice: +4.11%, FPR giảm từ 0.96% xuống 0.30% | kaggle/outputs/cvc-clinicdb/ |
| 3 | Benchmark EM-Dataset | Kaggle (như cũ) | Dice: +0.63%, Precision: +4.20%, FPR giảm >2x | kaggle/outputs/em-dataset/ |
| 4 | Benchmark Kvasir-SEG | Kaggle (như cũ) | Dice: +1.46%, Sensitivity: +6.53% | kaggle/outputs/Kvasir-SEG/ |
| 5 | Benchmark CHASE-DB1 | Kaggle (như cũ) | Dice: -2.24%, mIoU: -3.18% | kaggle/outputs/chasedb1/ |

## Will Do (On going)

- [ ] Lấy nốt kết quả ISIC-2018 từ Google Colab.
- [ ] Bổ sung ISIC-2018 vào bảng tổng hợp.
- [ ] Đóng gói bảng biểu đưa vào Overleaf cho bài báo (IEEE T-MI).

## Any Stuck / Open Questions

- Kernel ISIC-2018 trên Kaggle bị vướng timeout sau ~8-12 tiếng chạy, code đã kích hoạt cơ chế sinh tồn sys.exit() để giữ lại file latest.pth và thoát hoàn toàn thay vì chết đứng. Tạm thời giải quyết bằng cách user chủ động đem lên Google Colab A100 chạy cho nhanh (đang đợi kết quả CSV).

## Đính kèm link chi tiết

- File Artifact tổng hợp chi tiết: rain/14f42333-b723-432f-bf8b-2a04cf7fc388/mfad_benchmark_v2_results.md
- Code Benchmark: 
otebooks/benchmark_v2/
