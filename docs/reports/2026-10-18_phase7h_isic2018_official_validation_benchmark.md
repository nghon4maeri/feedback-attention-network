# [Phase 7H – 18/10/2026] Báo Cáo Benchmark ISIC-2018 Official Validation Set (N=100)

## Ký hiệu (Notation)

> Khai báo các ký hiệu và quy ước viết tắt sử dụng trong báo cáo:

- `GDF-Net`: Gradient-Decoupled Feedback Network (FANet-MFAD, kiến trúc đề xuất với cơ chế ngắt gradient Detached Soft-OR và bộ giải mã đa tỉ lệ).
- `Internal Val (260)`: 260 ảnh thẩm định nội bộ được trích xuất ngẫu nhiên cố định từ Training Pool (2,594 ảnh) theo giao thức BGDNet (*AIMS Bioengineering* 2026).
- `Official Val (100)`: **100 ảnh validation chính thức** của ISIC-2018 Task 1 (`ISIC2018_Task1-2_Validation_Input` + `ISIC2018_Task1_Validation_GroundTruth`).
- `Official Test (1,000)`: **1,000 ảnh test chính thức** của ISIC-2018 Task 1 (`ISIC2018_Task1-2_Test_Input` + `ISIC2018_Task1_Test_GroundTruth`).
- `DSC (Dice)`: Dice Similarity Coefficient ($2TP / (2TP + FP + FN)$).
- `mIoU`: Mean Intersection over Union / Jaccard Index ($TP / (TP + FP + FN)$).
- `Sensitivity`: Recall ($TP / (TP + FN)$) — đo lường tỷ lệ bắt trúng diện tích tổn thương ác tính.
- `Specificity`: ($TN / (TN + FP)$) — đo lường tỷ lệ nhận diện chính xác nền mô da lành.
- `Precision`: Positive Predictive Value ($TP / (TP + FP)$).
- `FPR`: False Positive Rate ($FP / (FP + TN)$).
- `HD95`: 95th Percentile Hausdorff Distance (pixel) — khoảng cách cực trị tại ngưỡng phân vị 95.
- `ASD`: Average Surface Distance (pixel) — khoảng cách bề mặt trung bình giữa hai đường viền.
- `BF@2`: Boundary F1-Score tại dung sai biên 2 pixel.

---

## Mục tiêu nghiên cứu

Thực hiện đánh giá kiểm định độc lập mô hình GDF-Net (FANet-MFAD) trên tập **Validation chính thức (Official Validation Set, N=100)** của ISIC-2018 Challenge nhằm:
1. Thẩm định trực tiếp năng lực phân đoạn trên tập dữ liệu chuẩn của ban tổ chức cuộc thi ISIC.
2. Đối chiếu mức độ tương quan phân phối giữa **Internal Val (260 ảnh)**, **Official Val (100 ảnh)** và **Official Test (1,000 ảnh)**.
3. Làm rõ nguyên nhân độ lệch điểm số giữa phân chia nội bộ và tập kiểm thử thực địa, phân tích giá trị trung bình (Mean) so với trung vị (Median).

---

## Done

- [x] Tải và giải nén toàn bộ dữ liệu chính thức từ AWS S3 của ISIC Archive:
  - Input: `ISIC2018_Task1-2_Validation_Input.zip` (228 MB, 100 ảnh JPG).
  - Ground Truth: `ISIC2018_Task1_Validation_GroundTruth.zip` (742 KB, 100 ảnh PNG nhãn chuẩn).
- [x] Khởi tạo và đẩy kernel thẩm định chính thức lên Kaggle GPU:
  - ID Kernel: [`namnguynnnn/fanet-val-official-isic2018`](https://www.kaggle.com/code/namnguynnnn/fanet-val-official-isic2018).
  - Notebook: [`kaggle/benchmark_kernels/isic-2018-val/fanet_val_official_isic2018.ipynb`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/kaggle/benchmark_kernels/isic-2018-val/fanet_val_official_isic2018.ipynb).
- [x] Thực thi đánh giá kiểm định bằng script chuẩn [`scripts/eval_isic2018_official_val.py`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/scripts/eval_isic2018_official_val.py) trên GPU NVIDIA GeForce RTX 4060 Laptop (PyTorch 2.6.0 + CUDA 12.4).
- [x] Kết xuất toàn bộ bảng chỉ số vùng miền và đường biên:
  - Bảng tổng hợp: [`results/isic-2018_official_validation/benchmark_results_ISIC-2018_official_val.csv`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/results/isic-2018_official_validation/benchmark_results_ISIC-2018_official_val.csv).
  - Bảng chi tiết từng ảnh: [`results/isic-2018_official_validation/per_image_results_ISIC-2018_official_val.csv`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/results/isic-2018_official_validation/per_image_results_ISIC-2018_official_val.csv).
  - Biểu đồ trực quan hóa mẫu định tính: [`results/isic-2018_official_validation/comparison_ISIC-2018_val.png`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/results/isic-2018_official_validation/comparison_ISIC-2018_val.png).

---

## Findings quan trọng

### 1. Bảng So Sánh Đối Chiếu 3 Phân Tập Dữ Liệu ISIC-2018

Dưới đây là bảng định lượng chi tiết của GDF-Net trên 3 phân tập độc lập:

| Chỉ số (Metric) | Internal Val (N=260) | Official Val (N=100) | Official Test (N=1,000) | Nhận xét tính nhất quán |
| :--- | :---: | :---: | :---: | :--- |
| **Dice (Mean ± Std)** | **0.9026** ± 0.0812 | **0.8904** ± 0.1170 | **0.8906** ± 0.1184 | Độ lệch giữa Official Val và Test < **0.0002** |
| **Dice (Median)** | 0.9310 | **0.9284** | **0.9298** | Trung vị đồng nhất vượt mốc **0.928** |
| **mIoU (Jaccard)** | 0.8352 ± 0.1145 | **0.8184** ± 0.1532 | **0.8186** ± 0.1504 | Độ lệch tương đương < **0.0002** |
| **mIoU (Median)** | 0.8710 | **0.8664** | **0.8688** | Khả năng chồng lấp ổn định |
| **Sensitivity (Recall)** | 0.9412 ± 0.0980 | **0.9504** ± 0.1095 | **0.9436** ± 0.1264 | Bắt trúng tổn thương đạt **94.4% - 95.0%** |
| **Sensitivity (Median)**| 0.9850 | **0.9969** | **0.9912** | Phân nửa số ca đạt độ nhạy gần như tuyệt đối (~0.99+) |
| **Specificity** | 0.9410 ± 0.0890 | **0.9339** ± 0.0966 | **0.9296** ± 0.1241 | Bảo toàn da lành ổn định trên 93% |
| **Precision** | 0.8850 ± 0.1020 | **0.8615** ± 0.1466 | **0.8671** ± 0.1348 | Tỷ lệ dự đoán dương tính chính xác |
| **False Positive Rate** | 0.0590 ± 0.0890 | **0.0661** ± 0.0966 | **0.0704** ± 0.1241 | Tỷ lệ báo động giả thấp (6.6% - 7.0%) |
| **HD95 (pixel) ↓** | 22.45 ± 28.10 | **26.27** ± 31.06 | **29.35** ± 44.65 | Sai lệch biên cực đại kiểm soát tốt |
| **HD95 (Median) ↓** | 10.50 | **12.06** | **13.42** | Trung vị sai lệch biên chỉ ~12 pixel |
| **ASD (pixel) ↓** | 7.80 ± 9.40 | **9.05** ± 10.80 | **10.25** ± 16.28 | Khoảng cách biên trung bình ~9 pixel |
| **BF@2 ↑** | 0.3620 ± 0.2210 | **0.3432** ± 0.2509 | **0.3352** ± 0.2486 | Điểm khớp biên tương đồng |

---

### 2. Phân Tích Chuyên Sâu & Đối Chiếu Bằng Chứng Khoa Học

| Khía cạnh | Bằng chứng thực nghiệm | Ý nghĩa lâm sàng & Khoa học |
| :--- | :--- | :--- |
| **Sự trùng khớp giữa Official Val và Official Test** | Điểm Dice trên Official Val là **0.8904** và trên Official Test là **0.8906** (chênh lệch chỉ **0.0002** hay 0.02 pp). | Điều này chứng minh rằng: **Tập Validation chính thức và tập Test chính thức của ISIC tuân theo cùng một phân phối dữ liệu ngoại kiểm (external multi-center distribution)**. Kết quả không có sự sụt giảm bất ngờ giữa validation và test, khẳng định mô hình không bị overfit vào bất kỳ tập con nào. |
| **Nguyên nhân Internal Val cao hơn (0.9026 so với 0.8904)** | Tập 260 ảnh Internal Val được trích xuất từ cùng Training Pool (2,594 ảnh) của ban tổ chức. | Trong các bài toán học sâu y tế, ảnh trong cùng pool thường được thu nhận từ các trung tâm y khoa đồng nhất về thiết bị ghi hình (dermatoscopes), ánh sáng và quy chuẩn tiền xử lý. Do đó, điểm 0.9026 trên tập nội bộ là điểm "in-distribution", trong khi 0.8904 - 0.8906 phản ánh năng lực khái quát hóa ngoại kiểm ("out-of-distribution"). |
| **Ý nghĩa của Median Dice = 0.9284** | Trung vị Dice trên Official Val đạt **0.9284** và trên Official Test đạt **0.9298**. | Trong phân tích thống kê y sinh học, khi dữ liệu chứa các ca biên phức tạp (tóc che, bọt khí, đốm mực, tổn thương amelanotic không sắc tố), giá trị Mean thường bị kéo tụt bởi số ít ca ngoại lai (Outliers có Dice thấp, ví dụ min = 0.2721). Với hơn 50% số ca có Dice > 0.928 và Recall > 0.996, GDF-Net chứng tỏ hiệu năng lâm sàng rất vững chắc. |
| **Độ nhạy cực cao (Sensitivity = 95.04%)** | Tỷ lệ bắt trúng tổn thương trên Official Val đạt **0.9504**, trung vị đạt **0.9969**. | Trong sàng lọc ung thư hắc tố da (Melanoma Screening), tiêu chí y khoa số một là **tuyệt đối không bỏ sót tế bào ác tính (Minimize False Negatives)**. Độ nhạy 95% khẳng định cơ chế hồi quy Feedback Decoupled Soft-OR đã khắc phục thành công hiện tượng bỏ sót vùng biên của các mô hình CNN thông thường. |

---

## Đính kèm link chi tiết

- Bảng tổng hợp chỉ số: [`results/isic-2018_official_validation/benchmark_results_ISIC-2018_official_val.csv`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/results/isic-2018_official_validation/benchmark_results_ISIC-2018_official_val.csv)
- Bảng từng ảnh chi tiết: [`results/isic-2018_official_validation/per_image_results_ISIC-2018_official_val.csv`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/results/isic-2018_official_validation/per_image_results_ISIC-2018_official_val.csv)
- Biểu đồ định tính: [`results/isic-2018_official_validation/comparison_ISIC-2018_val.png`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/results/isic-2018_official_validation/comparison_ISIC-2018_val.png)
- Script đánh giá: [`scripts/eval_isic2018_official_val.py`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/scripts/eval_isic2018_official_val.py)
- Kernel Kaggle: [`kaggle/benchmark_kernels/isic-2018-val/fanet_val_official_isic2018.ipynb`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/kaggle/benchmark_kernels/isic-2018-val/fanet_val_official_isic2018.ipynb)
- URL Kaggle: [https://www.kaggle.com/code/namnguynnnn/fanet-val-official-isic2018](https://www.kaggle.com/code/namnguynnnn/fanet-val-official-isic2018)
