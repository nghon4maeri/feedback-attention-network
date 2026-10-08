# 🏆 Tổng hợp Kết quả Benchmark MFAD vs FANet Gốc

> Bảng tổng sắp hiệu năng của kiến trúc MFAD (đề xuất) so với FANet (baseline) trên 5 tập dữ liệu đa dạng. Cả 2 mô hình được huấn luyện lại từ đầu (from scratch) trong môi trường đối chứng tuyệt đối (chung cấu hình, chung data split, seed=42).
> *(Đang chờ cập nhật bộ ISIC-2018)*

## Bảng so sánh các chỉ số (Metrics)

| Tập dữ liệu (Domain) | Mô hình | Dice Score | mIoU | Precision | Sensitivity (Recall) | FPR (False Positive Rate) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **CVC-ClinicDB**<br>*(Polyp dạ dày)* | FANet Orig | 0.8958 | 0.8306 | 0.8968 | **0.9250** | 0.96% |
| | MFAD (Ours) | **0.9369** | **0.8846** | **0.9652** | 0.9131 | **0.30%** |
| | *Chênh lệch* | *+0.0411* | *+0.0540* | *+0.0684* | *-0.0119* | *-0.66%* (Giảm 3 lần) |
| | | | | | | |
| **Kvasir-SEG**<br>*(Polyp dạ dày)* | FANet Orig | 0.8852 | 0.8185 | **0.9140** | 0.8916 | **1.12%** |
| | MFAD (Ours) | **0.9100** | **0.8461** | 0.8825 | **0.9569** | 2.61% |
| | *Chênh lệch* | *+0.0248* | *+0.0276* | *-0.0315* | *+0.0653* | *+1.49%* |
| | | | | | | |
| **DSB-2018**<br>*(Tế bào nhân dính chùm)* | FANet Orig | 0.8721 | 0.7938 | 0.8448 | 0.9137 | 2.18% |
| | MFAD (Ours) | **0.8855** | **0.8106** | **0.8686** | **0.9180** | **2.13%** |
| | *Chênh lệch* | *+0.0134* | *+0.0168* | *+0.0238* | *+0.0043* | *-0.05%* |
| | | | | | | |
| **EM-Dataset**<br>*(Kính hiển vi điện tử)* | FANet Orig | 0.9379 | 0.8832 | 0.9150 | **0.9627** | 0.60% |
| | MFAD (Ours) | **0.9437** | **0.8936** | **0.9572** | 0.9311 | **0.27%** |
| | *Chênh lệch* | *+0.0058* | *+0.0104* | *+0.0422* | *-0.0316* | *-0.33%* (Giảm >2 lần)|
| | | | | | | |
| **CHASE-DB1**<br>*(Vi mạch máu võng mạc)*| FANet Orig | **0.8250** | **0.7028** | **0.8044** | **0.8492** | 1.61% |
| | MFAD (Ours) | 0.8026 | 0.6710 | 0.8027 | 0.8053 | **1.55%** |
| | *Chênh lệch* | *-0.0224* | *-0.0318* | *-0.0017* | *-0.0439* | *-0.06%* |

## Tóm lược đánh giá
- **Thế mạnh tuyệt đối:** MFAD vượt trội hoàn toàn trên các cấu trúc có ranh giới nhiễu hoặc tế bào đóng cụm (CVC, EM, DSB). Nó đóng vai trò như một bộ nén nhiễu "halo" cực mạnh, làm giảm tỷ lệ sai dương (FPR) xuống mức rất thấp, đẩy Precision tăng vọt (lên tới +6.84% trên CVC).
- **Trường hợp ngoại lệ (CHASE-DB1):** Khi áp dụng cho các cấu trúc dạng ống, liên kết mỏng như vi mạch máu võng mạc, cơ chế ngắt gradient (detach) của MFAD có xu hướng làm đứt gãy một số nhánh nhỏ (giảm Sensitivity). Nguyên nhân là do các cấu trúc siêu mỏng cần dòng gradient liên tục từ các bước thời gian trước để duy trì tính liên kết không gian. 
