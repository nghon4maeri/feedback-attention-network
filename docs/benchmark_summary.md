# Báo Cáo Tổng Hợp Kết Quả Benchmark V2 (FANet Orig vs. MFAD)

Tài liệu này tổng hợp chi tiết kết quả thực nghiệm đánh giá mô hình **Multi-scale Feedback Attention Decoder (MFAD)** do nhóm nghiên cứu đề xuất, so sánh trực tiếp với kiến trúc **FANet gốc (Baseline)**. 

Để đảm bảo tính công bằng khoa học tuyệt đối, cả hai mô hình đều được huấn luyện lại từ đầu (train from scratch) trên cùng một phân chia dữ liệu (Train/Val split), cùng seed ngẫu nhiên (seed=42), cùng siêu tham số (Learning Rate = 1e-4) và cùng môi trường đánh giá.

---

## 1. Bảng Tổng Sắp Chỉ Số (Quantitative Results)

> **Quy ước:** Các chỉ số được in đậm thể hiện hiệu năng tốt hơn. (Lưu ý: FPR càng thấp càng tốt).
> *(Bộ dữ liệu ISIC-2018 đang trong quá trình chạy và sẽ được cập nhật sau).*

| Tập dữ liệu (Domain) | Mô hình | Dice Score | mIoU | Precision | Sensitivity (Recall) | FPR (False Positive Rate) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CVC-ClinicDB**<br>*(Polyp nội soi)* | FANet Orig | 0.8958 | 0.8306 | 0.8968 | **0.9250** | 0.96% |
| | MFAD (Ours) | **0.9369** | **0.8846** | **0.9652** | 0.9131 | **0.30%** |
| **Kvasir-SEG**<br>*(Polyp nội soi)* | FANet Orig | 0.8852 | 0.8185 | **0.9140** | 0.8916 | **1.12%** |
| | MFAD (Ours) | **0.9100** | **0.8461** | 0.8825 | **0.9569** | 2.61% |
| **DSB-2018**<br>*(Tế bào nhân - Nuclei)* | FANet Orig | 0.8721 | 0.7938 | 0.8448 | 0.9137 | 2.18% |
| | MFAD (Ours) | **0.8855** | **0.8106** | **0.8686** | **0.9180** | **2.13%** |
| **EM-Dataset**<br>*(Kính hiển vi điện tử)* | FANet Orig | 0.9379 | 0.8832 | 0.9150 | **0.9627** | 0.60% |
| | MFAD (Ours) | **0.9437** | **0.8936** | **0.9572** | 0.9311 | **0.27%** |
| **CHASE-DB1**<br>*(Vi mạch máu võng mạc)*| FANet Orig | **0.8250** | **0.7028** | **0.8044** | **0.8492** | 1.61% |
| | MFAD (Ours) | 0.8026 | 0.6710 | 0.8027 | 0.8053 | **1.55%** |

*(Ghi chú: Bộ dữ liệu DRIVE bị loại khỏi báo cáo do hiện tượng Mode Collapse trên cả 2 mô hình - Dice < 0.2 - xuất phát từ kích thước tập dữ liệu quá nhỏ (16 ảnh) kết hợp với Learning Rate chưa phù hợp để huấn luyện từ đầu).*

---

## 2. Phân Tích Chuyên Sâu (Qualitative Analysis)

Dựa trên bảng số liệu trên, kiến trúc MFAD đã chứng minh được sự ưu việt trong việc xử lý các nhược điểm cốt lõi của mạng FANet gốc, cụ thể:

### 2.1. Triệt tiêu hiệu ứng "Feedback Trap" và nhiễu Halo
Trong FANet gốc, việc dòng gradient lan truyền liên tục qua các vòng lặp phản hồi (feedback loop) vô tình tạo ra sự tích lũy lỗi. Điều này khiến mạng có xu hướng phình to dự đoán (over-segmentation) và tạo ra quầng nhiễu "halo" xung quanh vật thể. 

Bằng cách ngắt gradient (detach) của mask phản hồi kết hợp với cổng Gating, MFAD ép mạng phải tinh chỉnh lại ranh giới một cách chính xác thay vì mở rộng vô tội vạ. 
- Trên **CVC-ClinicDB**, điều này giúp tỷ lệ sai dương (FPR) giảm mạnh từ 0.96% xuống 0.30% (giảm hơn 3 lần). Hệ quả trực tiếp là **Precision tăng vọt +6.84%** và đẩy tổng thể **Dice Score tăng +4.11%**.
- Tương tự trên **EM-Dataset**, FPR giảm hơn một nửa (0.60% xuống 0.27%), mang lại mức tăng **+4.22% cho Precision**.

### 2.2. Gỡ rối ranh giới tế bào dày đặc (Dataset: DSB-2018)
Đặc thù của tập DSB-2018 là các cụm tế bào nằm san sát nhau, dính chùm và ranh giới cực kỳ mờ nhạt. Ở các thuật toán thông thường, việc cố gắng tách ranh giới sẽ làm giảm Sensitivity (nhận diện sót).
Tuy nhiên, MFAD đã phá vỡ sự đánh đổi này. Trên DSB-2018, MFAD tăng đồng thời cả **Precision (+2.38%)** lẫn **Sensitivity (+0.43%)**. Điều này là minh chứng đắt giá cho thấy MFAD thực sự học được biểu diễn hình thái học (morphological representation) sâu sắc hơn, giúp cắt nghĩa chính xác các cụm vật thể độc lập.

### 2.3. Khả năng thích ứng mở rộng (Dataset: Kvasir-SEG)
Trên tập Kvasir-SEG, các polyp thường bị che lấp bởi dịch nhầy hoặc phản chiếu ánh sáng, làm ranh giới trở nên ảo. Ở đây, MFAD tự động chấp nhận đánh đổi một lượng nhỏ FPR (+1.49%) để nới rộng vùng chú ý, từ đó bao phủ trọn vẹn tổn thương và mang về mức tăng khổng lồ **+6.53% cho Sensitivity**. Trong y tế thực hành, việc tránh bỏ sót bệnh (high Sensitivity) mang ý nghĩa sống còn. Do đó, mức tăng **+2.48% Dice** tổng thể là một thành công lớn.

### 2.4. Nhược điểm (Limitation) trên cấu trúc vi mạch mỏng
Trên tập **CHASE-DB1** (vi mạch máu võng mạc), ranh giới không phải là khối (blob) mà là các dạng ống siêu mảnh, kéo dài và dễ đứt khúc. 
Cơ chế ngắt gradient (detach) của MFAD đã vô tình làm mất đi tín hiệu liên kết không gian từ các bước thời gian trước, khiến các vi mạch bị đứt gãy nhiều hơn so với baseline. Điều này lý giải mức giảm nhẹ **-2.24% Dice**. Mặc dù vậy, MFAD vẫn kiểm soát FPR tốt hơn (giảm xuống 1.55%), chứng tỏ khả năng ép nhiễu của nó vẫn hoạt động, chỉ là nó không phù hợp với cấu trúc hình học (topological) dạng mạng lưới mỏng.
