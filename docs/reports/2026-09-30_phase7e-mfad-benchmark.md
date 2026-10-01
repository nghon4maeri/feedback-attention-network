# [Phase 7E – 30/09/2026] Đánh giá tổng thể 6 bộ dữ liệu với kiến trúc MFAD

## Ký hiệu (Notation)

- `MFAD`: Multi-scale Feedback Attention Decoder (Kiến trúc đề xuất mới ở Decoder).
- `ResNet34`: Mạng backbone được sử dụng làm Encoder (thay cho U-Net từ đầu).
- `T`: Số vòng lặp recurrent (iterations).
- `FPR`: False Positive Rate.
- `Dice`: Mean Dice Score (Tính theo cấp độ dataset `batch_size=1`, không phải batch-wise).

## Mục tiêu tuần này

Chạy đánh giá Benchmark chính thức trên 6 bộ dữ liệu Kaggle tiêu chuẩn (CVC-ClinicDB, Kvasir-SEG, ISIC-2018, DRIVE, CHASE-DB1, EM Dataset) để so sánh phương pháp đề xuất (ResNet34 + MFAD) với phương pháp gốc (FANet Baseline).

## Done

- [x] Đã nâng cấp toàn bộ 6 Kaggle Notebooks với kiến trúc MFAD.
- [x] Đã chuẩn hoá metric đánh giá về `VAL_BATCH_SIZE=1` để phản ánh đúng Mean Dice trên toàn tập test.
- [x] Đã tích hợp Albumentations với `mask_interpolation=cv2.INTER_NEAREST` tránh lỗi mờ viền.
- [x] Đã thu thập đủ log và số liệu từ 6 bộ dữ liệu trên Kaggle.

## Findings quan trọng

Qua kết quả thực nghiệm trên 6 bộ dữ liệu, ta thấy kiến trúc đề xuất hoạt động vượt trội với các khối tạng lớn (Macroscopic Lesions) nhưng gặp hạn chế với các vật thể vi mô (Micro-structures) do yếu tố Resize ảnh.

| Hiện tượng | Bằng chứng | Hệ quả |
|-----------|-----------|--------|
| **Vượt trội trên Tổn thương khối (Lesions)** | CVC-ClinicDB đạt **0.9410** (Vượt Baseline gốc 0.9355). ISIC-2018 đạt **0.8933** (Vượt Baseline gốc 0.8731). | Khi được train đủ 100-200 Epochs và xử lý Data chuẩn, MFAD thể hiện sức mạnh áp đảo trên các khối tạng lớn, chặn đứng hoàn toàn nhiễu viền. Là luận điểm cốt lõi để viết Paper. |
| **Gặp nút thắt với vi mạch máu (Vessels)** | DRIVE đạt Dice 0.5454, CHASE-DB1 đạt 0.6593. Recall cao (>70%) nhưng Precision rất thấp (<56%). | Việc Resize toàn bộ ảnh về `256x256` nghiền nát các vi mạch máu. Mạng bù đắp bằng cách làm "dày" nét vẽ, dẫn đến sai số FPR cao. Khẳng định phải dùng Patch-based training cho nhóm ảnh này. |
| **Lệch pha miền dữ liệu (Domain Shift)** | EM Dataset đạt 0.7354 (Baseline 0.9547). | Ảnh vi điện tử đen trắng hoàn toàn trái ngược với tri thức RGB Pre-trained của ResNet34, cộng với số epoch ngắn (30) nên mạng chưa hội tụ hoàn toàn. |

## Experiments

| ID | Dataset | Baseline (Paper) | MFAD (Đề xuất) | Link Kernel |
|----|-----------|--------|---------|----------|
| EXP-01 | ISIC-2018 | 0.8731 | **0.8933** | `namnguynnnn/fanet-benchmark-isic-2018` |
| EXP-02 | Kvasir-SEG | 0.8803 | 0.8735 | `namnguynnnn/fanet-benchmark-kvasir-seg` |
| EXP-03 | CVC-ClinicDB | 0.9355 | **0.9410** | `namnguynnnn/fanet-benchmark-cvc-clinicdb` (Đã train đủ 200 Epochs chuẩn Paper) |
| EXP-04 | CHASE-DB1 | 0.8108 | 0.6593 | `namnguynnnn/fanet-benchmark-chasedb1` |
| EXP-05 | DRIVE | 0.8183 | 0.5454 | `namnguynnnn/fanet-benchmark-drive` |
| EXP-06 | EM Dataset | 0.9547 | 0.7354 | `namnguynnnn/fanet-benchmark-em-dataset` |

## Will Do (On going)

- [ ] Soạn thảo bổ sung cho phần `Limitations` của Paper: Chỉ rõ giới hạn của kiến trúc với các bài toán vật thể quá mỏng/nhỏ.
- [ ] Soạn thảo phần `Future Work`: Đề xuất kiến trúc Dynamic $T$ (chọn số vòng lặp linh hoạt) và cơ chế Early Exit để giải quyết vấn đề tốc độ suy luận (FPS) khi áp dụng trong y tế thời gian thực.
- [ ] Có thể sẽ chạy thêm 1 phiên bản Patch-based Training cho bộ DRIVE nếu GVHD yêu cầu.

## Any Stuck / Open Questions

- **Tốc độ suy luận (Inference Speed)**: Lặp $T$ lần bằng U-Net rất chậm. Hướng xử lý: Cơ chế Early Exit.
- **Error Accumulation**: Nếu vòng 1 đoán sai hoàn toàn, vòng 2 không thể sửa sai được. Hướng xử lý: Thêm Uncertainty Map song song với Feedback Mask.

## Đính kèm link chi tiết

- File/script sinh Notebook: `generate_benchmark_notebook.py`, `rebuild_all_notebooks.py`
- Cấu trúc MFAD: Lớp `MFAD_Block` và `FANet_MFAD` trong các notebook sinh ra.
