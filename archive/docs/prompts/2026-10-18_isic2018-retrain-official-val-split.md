# Prompt: Huấn luyện lại GDF-Net trên toàn bộ 2.594 ảnh và thẩm định trên Tập Official Validation (18/10/2026)

## 1. Yêu cầu của User & Bối cảnh
- **Bối cảnh:**
  - Ở pha trước (Phase 7G), mô hình được huấn luyện theo phân chia BGDNet (trích 2.334 ảnh train và 260 ảnh thẩm định nội bộ từ Training Pool). Kết quả trên 1.000 ảnh test chính thức đạt 0.8906 Dice và trên 100 ảnh validation chính thức đạt 0.8904 Dice.
  - Người dùng nhận định điểm số này thấp hơn kỳ vọng (>0.90) và đặt câu hỏi về hiện tượng overclaim của các bài báo trong y văn.
  - Sau khi làm rõ bằng chứng rằng các bài báo thường dùng 5-fold CV nội bộ trên 2.594 ảnh, người dùng yêu cầu:
- **Yêu cầu cụ thể:**
  1. Huấn luyện lại mô hình từ đầu (hoặc đầy đủ chu kỳ 200 epochs).
  2. Sử dụng **tập validation chính thức (Official Validation Set, 100 ảnh)** làm tập thẩm định (validation set) để điều khiển learning rate scheduler và lựa chọn checkpoint tốt nhất (`best_dice`).
  3. Sử dụng toàn bộ **2.594 ảnh của Training Pool** để huấn luyện (tăng thêm 260 ảnh so với trước).
  4. Đánh giá kiểm định không rò rỉ trên **1.000 ảnh test chính thức (Official Test Set)** để đo lường năng lực khái quát hóa ngoại kiểm.

## 2. Thiết kế Thí nghiệm & Giả thuyết Khoa học
- **Giả thuyết (Hypothesis):**
  - Khi tập thẩm định là 100 ảnh validation chính thức (vốn thu nhận từ các trung tâm y khoa bên ngoài, cùng phân phối đa trung tâm với 1.000 ảnh test chính thức), mô hình sẽ được dẫn hướng lựa chọn checkpoint tối ưu cho năng lực ngoại kiểm ("out-of-distribution generalization"), thay vì overfit vào phân phối nội bộ của Training Pool.
  - Việc bổ sung thêm 260 ảnh vào tập train (2.594 ảnh thay vì 2.334 ảnh) cung cấp thêm độ đa dạng tổn thương da liễu.
- **Cấu hình Thực nghiệm:**
  - Tập huấn luyện (Train): 2.594 cặp ảnh-mặt nạ (`ISIC2018_Task1-2_Training_Input` + `ISIC2018_Task1_Training_GroundTruth`) có Data Augmentation đầy đủ.
  - Tập thẩm định (Val): 100 cặp ảnh-mặt nạ chính thức (`ISIC2018_Task1-2_Validation_Input` + `ISIC2018_Task1_Validation_GroundTruth`) không Data Augmentation.
  - Tập kiểm thử (Test): 1.000 cặp ảnh-mặt nạ chính thức (`ISIC2018_Task1-2_Test_Input` + `ISIC2018_Task1_Test_GroundTruth`).
  - Kiến trúc: GDF-Net (ResNet-34 ImageNet pretrained + Feedback Firewall + MFAD Decoder).
  - Tối ưu hóa: AdamW (lr=1e-4, weight_decay=1e-4), ReduceLROnPlateau (patience=8), Mixed Precision AMP, 200 epochs.
  - Môi trường: Kaggle GPU (`namnguynnnn/fanet-isic2018-retrain-official-val`).
