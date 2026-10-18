# Prompt: Đánh giá GDF-Net trên Tập Validation Chính Thức (Official Validation Set) ISIC-2018 (18/10/2026)

## 1. Yêu cầu của User & Bối cảnh
- **Bối cảnh:** Sau khi hoàn tất 200 epochs huấn luyện mô hình GDF-Net (FANet-MFAD) theo phân chia dữ liệu của paper BGDNet (*AIMS Bioengineering* 2026) và đánh giá trên 1.000 ảnh test chính thức (`ISIC2018_Task1-2_Test_Input`), kết quả đạt được là **0.8906 Dice** (độ nhạy rất cao 0.9436, song Dice chưa đạt mốc 0.90+ như trên tập thẩm định nội bộ 260 ảnh vốn đạt **0.9026 Dice**).
- **Yêu cầu cụ thể:**
  1. Người dùng nhận định kết quả test 0.8906 là tương đối thấp so với kỳ vọng (vốn kỳ vọng > 0.90).
  2. Người dùng yêu cầu chạy thẩm định thử nghiệm (evaluate) mô hình trên **tập validation chính thức (Official Validation Set, 100 ảnh)** của ISIC-2018 (`ISIC2018_Task1-2_Validation_Input` + `ISIC2018_Task1_Validation_GroundTruth`).
  3. Đảm bảo phương pháp đo lường, tiền xử lý và tham số đánh giá hoàn toàn thống nhất với các thí nghiệm trước đó (kích thước ảnh 512x512, không data augmentation trong quá trình validation/eval, tính toán đầy đủ các chỉ số vùng và chỉ số đường biên HD95, ASD, BF@2).

## 2. Nguồn Dữ liệu & Phương pháp Triển khai
- **Dữ liệu Validation Chính thức ISIC-2018 (Task 1):**
  - Ảnh đầu vào (100 ảnh JPG): `https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1-2_Validation_Input.zip` (228 MB).
  - Mặt nạ nhãn chuẩn (100 ảnh PNG binary ground truth): `https://isic-archive.s3.amazonaws.com/challenges/2018/ISIC2018_Task1_Validation_GroundTruth.zip` (742 KB).
  - Tải trực tiếp từ AWS S3 chính thức của ISIC Archive về thư mục `data/isic2018_validation_official/`.
- **Checkpoint Đánh giá:**
  - `results/isic-2018_official_kaggle_resumed/GDFNet_ISIC2018_best.pth` (98 MB, lưu lại tại epoch có điểm Val Dice cao nhất 0.9026 trong pha huấn luyện 200 epochs).
- **Môi trường Thực thi:**
  - GPU NVIDIA GeForce RTX 4060 Laptop (PyTorch 2.6.0 + CUDA 12.4) tại máy local.
  - Script điều khiển: `scripts/eval_isic2018_official_val.py`.
- **Tập chỉ số đo lường:**
  - Region-based metrics: Dice Score, mIoU, Sensitivity (Recall), Specificity, Precision, False Positive Rate (FPR).
  - Boundary-based metrics (tại độ phân giải chuẩn 352x352): 95th-percentile Hausdorff Distance (HD95), Average Surface Distance (ASD), Boundary F1 Score (BF@2).
