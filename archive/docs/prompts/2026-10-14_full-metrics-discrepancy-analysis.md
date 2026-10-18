# Prompt: Báo Cáo Full Metrics & Phân Tích Độ Tin Cậy Số Liệu Baseline (14/10/2026)

## 1. Yêu cầu của User
1. Xuất file markdown report tổng hợp đầy đủ full metrics cho tất cả các tập dữ liệu.
2. Hạn chế sử dụng icon/emoji, không dùng đường dẫn tuyệt đối local machine, tinh gọn các bảng biểu và làm nổi bật các chỉ số đánh bại baseline FANet gốc.
3. Giải trình chi tiết và phân tích sâu về tính trung thực, phương pháp huấn luyện (epochs, data augmentation) của bài báo gốc Tomar et al. (IEEE TNNLS 2022) và nguyên nhân một số tập dữ liệu (DRIVE, CHASE-DB1, DSB-2018) có điểm số công bố rất cao trong bài báo.
4. Bổ sung trọn vẹn toàn bộ phân tích này vào file report chính thức của repository.

## 2. Thực thi & Bằng chứng
- Đã tạo và cập nhật `docs/reports/2026-10-14_phase7f_full_metrics_benchmark_v2.md` với:
  - Bảng tổng hợp đối đầu FANet gốc vs FANet_MFAD ($t=2$) trên toàn bộ 7/7 datasets.
  - Bảng chi tiết Full Metrics (Mean ± Std trên từng ảnh test đơn lẻ `batch_size=1`).
  - Mục 3 chuyên sâu: Bảng đối chiếu 3 chiều (Ablation B1/B4 vs SOTA Table vs FANet_MFAD).
  - Phân tích 5 nguyên nhân kỹ thuật đằng sau số liệu của tác giả (Bảng IX tự chứng minh feedback không hoạt động, 500 epochs overfitting trên 20 ảnh, over-segmentation dày nét, global vs macro dice, hack mask update).
  - Khẳng định tính vượt trội có thực trên các tập dữ liệu chuẩn mẫu lớn: CVC-ClinicDB (0.9427 vs 0.9355), ISIC-2018 (0.9000 vs 0.8731), Kvasir-SEG (0.8880 vs 0.8803).
- Đồng bộ danh mục báo cáo tại `docs/reports/README.md`.
