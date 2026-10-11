# Prompt: Khảo sát Transfer Learning từ ISIC-2018 sang DRIVE (12/10/2026)

## 1. Bối cảnh & Mục tiêu
- **Quan sát thực nghiệm:** Tập `DRIVE` (mạch máu võng mạc) chỉ có 20 ảnh huấn luyện. Khi train từ đầu (scratch) toàn khung hình 512×512, mô hình đạt Dice ~0.284 do mất cân bằng dữ liệu cực đoan (<8% điểm ảnh là mạch máu) và thiếu hụt dữ liệu biểu diễn trầm trọng (sample scarcity).
- **Ý tưởng chỉ đạo từ User:** Sử dụng trọng số (weights) của mô hình `FANet_MFAD` đã huấn luyện hoàn chỉnh trên tập đa dạng lớn nhất là **`ISIC-2018`** (2,594 ảnh da liễu) làm điểm khởi tạo (transfer learning / pre-training) để fine-tune sang tập ít ảnh `DRIVE`.

## 2. Kế hoạch Thực nghiệm
1. **Giai đoạn 1:** Chờ tiến trình huấn luyện Benchmark V2 của `ISIC-2018` hoàn tất.
2. **Giai đoạn 2:** Tải trọng số tốt nhất `FANet_MFAD_ISIC-2018_best.pth`.
3. **Giai đoạn 3:** Khởi tạo `FANet_MFAD` với trọng số này, đóng băng nhẹ hoặc fine-tune toàn mạng với Learning Rate nhỏ (ví dụ $5 \times 10^{-5}$ hoặc $1 \times 10^{-4}$) trên tập `DRIVE` (kèm độ lặp $\times 32$ data augmentation).
4. **Giai đoạn 4:** Đo lường trực tiếp pha Final Feedback ($t=2$) và so sánh với baseline train-from-scratch (0.2837).

## 3. Giả thuyết Khoa học (Hypothesis)
- $H_1$: Các bộ lọc Attention Gate đa tỷ lệ (Multi-scale Feedback Attention Gates) được tôi luyện trên 2,594 ảnh ISIC-2018 đã hình thành khả năng định vị ranh giới không gian tổng quát (generalized boundary awareness). Khi chuyển sang DRIVE, mạng không phải học lại các filter chú ý từ đầu, giúp khắc phục tình trạng sụp đổ biểu diễn trên tập 20 ảnh.
