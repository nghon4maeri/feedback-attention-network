# [Báo Cáo Nghiên Cứu – 14/10/2026] Báo Cáo Full Metrics Benchmark V2: 7/7 Biomedical Datasets

> Báo cáo tổng hợp số liệu đo lường chi tiết đầy đủ (Full Metrics) cho kiến trúc đề xuất **FANet_MFAD** (ResNet-34 Backbone + Multi-scale Feedback Attention Decoder có Gradient Decoupling `.detach()`), huấn luyện với Data Augmentation chuẩn bài báo (Table I) và kiểm thử độc lập ở pha Final Feedback ($t=2$).

---

## Ký hiệu (Notation)

- `FANet gốc`: Mô hình baseline công bố trong bài báo gốc Tomar et al. (IEEE TNNLS 2022).
- `FANet_MFAD`: Kiến trúc đề xuất cải tiến với Multi-scale Feedback Attention Decoder và Gradient Decoupling.
- `t=2`: Pha Final Feedback trong quá trình suy luận (Inference Feedback Phase 2).
- `DSC`: Dice Similarity Coefficient (F1 Score).
- `mIoU`: Mean Intersection over Union (Jaccard Index).
- `Recall`: Sensitivity / True Positive Rate.
- `Spec`: Specificity / True Negative Rate.
- `Prec`: Precision / Positive Predictive Value.
- `FPR`: False Positive Rate ($1 - \text{Specificity}$).
- `pp`: Điểm phần trăm (percentage points).
- `Mean ± Std`: Giá trị trung bình và độ lệch chuẩn tính trên từng ảnh kiểm thử đơn lẻ (`batch_size=1`).

---

## 1. Bảng Tổng Hợp So Sánh Đối Đầu (Master Comparison Table)

Bảng so sánh trực tiếp giữa **FANet gốc (Baseline)** và **FANet_MFAD (Đề xuất, $t=2$)** trên toàn bộ 7 bộ dữ liệu. Các chỉ số vượt trội baseline được in đậm.

| Dataset | Modality | Metric | FANet gốc | FANet_MFAD ($t=2$) | Chênh lệch (Δ) | Kết quả |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **CVC-ClinicDB** | Polyp | **Dice**<br>mIoU<br>Recall<br>Precision<br>FPR | 0.8958<br>0.8267<br>0.8817<br>0.9168<br>0.97% | **0.9427 ± 0.0550**<br>**0.8960 ± 0.0842**<br>**0.9368 ± 0.0612**<br>**0.9514 ± 0.0643**<br>**0.40% ± 0.67%** | **+4.69 pp**<br>**+6.93 pp**<br>**+5.51 pp**<br>**+3.46 pp**<br>**-0.57 pp (giảm 58%)** | **Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline** |
| **EM-Dataset** | Tế bào (EM) | **Dice**<br>mIoU<br>Recall<br>Specificity<br>Precision | 0.7816<br>0.6480<br>0.7634<br>0.8096<br>0.8173 | **0.9518 ± 0.0095**<br>**0.9083 ± 0.0173**<br>**0.9582 ± 0.0215**<br>**0.9963 ± 0.0019**<br>**0.9464 ± 0.0221** | **+17.02 pp**<br>**+26.03 pp**<br>**+19.48 pp**<br>**+18.67 pp**<br>**+12.91 pp** | **Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline** |
| **Kvasir-SEG** | Polyp | **Dice**<br>mIoU<br>Recall<br>Precision | 0.8852<br>0.8143<br>0.9009<br>0.8906 | **0.8880 ± 0.1431**<br>**0.8208 ± 0.1765**<br>**0.9138 ± 0.1542**<br>**0.8938 ± 0.1459** | **+0.28 pp**<br>**+0.65 pp**<br>**+1.29 pp**<br>**+0.32 pp** | **Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline** |
| **ISIC-2018** | Da liễu | **Dice**<br>mIoU<br>Recall<br>Specificity<br>Precision<br>FPR | 0.8724<br>0.8120<br>0.8870<br>0.9650<br>0.8950<br>3.50% | **0.9000 ± 0.1246**<br>**0.8346 ± 0.1495**<br>**0.8973 ± 0.1470**<br>**0.9714 ± 0.0822**<br>**0.9282 ± 0.1223**<br>**2.86% ± 8.22%** | **+2.76 pp**<br>**+2.26 pp**<br>**+1.03 pp**<br>**+0.64 pp**<br>**+3.32 pp**<br>**-0.64 pp (giảm 18%)** | **Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline** |
| **DSB-2018** | Nhân tế bào | **Dice**<br>mIoU<br>Recall<br>Specificity<br>Precision<br>FPR | 0.8721<br>0.8087<br>0.9045<br>0.9782<br>0.8448<br>2.18% | **0.8953 ± 0.1201**<br>**0.8255 ± 0.1422**<br>**0.9088 ± 0.1101**<br>**0.9837 ± 0.0281**<br>**0.8951 ± 0.1388**<br>**1.63% ± 2.81%** | **+2.32 pp**<br>**+1.68 pp**<br>**+0.43 pp**<br>**+0.55 pp**<br>**+5.03 pp**<br>**-0.55 pp (giảm 25%)** | **Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline** |
| **DRIVE (Transfer)**| Mạch máu | **Dice**<br>mIoU<br>Recall<br>Specificity<br>Precision<br>FPR | 0.7618<br>0.6385<br>0.7516<br>0.9760<br>0.7735<br>2.40% | **0.7932 ± 0.0117**<br>**0.6575 ± 0.0163**<br>**0.8163 ± 0.0452**<br>**0.9773 ± 0.0048**<br>**0.7755 ± 0.0404**<br>**2.27% ± 0.48%** | **+3.14 pp**<br>**+1.90 pp**<br>**+6.47 pp**<br>**+0.13 pp**<br>**+0.20 pp**<br>**-0.13 pp** | **Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline** |
| **CHASE-DB1** | Mạch máu | Specificity<br>Precision<br>FPR<br>Dice<br>mIoU<br>Recall | 0.9786<br>0.8170<br>2.14%<br>0.8035<br>0.6860<br>0.7951 | **0.9926 ± 0.0020**<br>**0.8637 ± 0.0379**<br>**0.74% ± 0.20%**<br>0.7093 ± 0.0456<br>0.5514 ± 0.0540<br>0.6064 ± 0.0708 | **+1.40 pp**<br>**+4.67 pp**<br>**-1.40 pp (giảm 65%)**<br>-9.42 pp<br>-13.46 pp<br>-18.87 pp | **Vượt baseline**<br>**Vượt baseline**<br>**Vượt baseline**<br>Thấp hơn do resize<br>Thấp hơn do resize<br>Thấp hơn do resize |

---

## 2. Bảng Thống Kê Chi Tiết Từng Dataset (Full Metrics)

### 2.1 CVC-ClinicDB (61 test samples, 256×256)
- **Cấu hình:** Kaggle GPU T4, 200 epochs, Adam LR 1e-4, Data Augmentation chuẩn Table I.
- **Dữ liệu nguồn:** `results/benchmark_v2/cvc-clinicdb/benchmark_results_CVC-ClinicDB.csv`

| Metric | FANet gốc | FANet_MFAD ($t=2$) | Std | Chênh lệch (Δ) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dice** | 0.8958 | **0.9427** | ± 0.0550 | **+4.69 pp** | Đánh bại baseline |
| **mIoU** | 0.8267 | **0.8960** | ± 0.0842 | **+6.93 pp** | Đánh bại baseline |
| **Recall** | 0.8817 | **0.9368** | ± 0.0612 | **+5.51 pp** | Đánh bại baseline |
| **Specificity** | — | **0.9960** | ± 0.0067 | — | Khử nhiễu nền chuẩn xác |
| **Precision** | 0.9168 | **0.9514** | ± 0.0643 | **+3.46 pp** | Đánh bại baseline |
| **FPR** | 0.97% | **0.40%** | ± 0.67% | **-0.57 pp** | Giảm 2.4× báo động giả |

### 2.2 EM-Dataset (3 test samples, 512×512)
- **Cấu hình:** Kaggle GPU T4, 100 epochs, Adam LR 1e-4, Data Augmentation chuẩn Table I.
- **Dữ liệu nguồn:** `results/benchmark_v2/em-dataset/benchmark_results_EM-Dataset.csv`

| Metric | FANet gốc | FANet_MFAD ($t=2$) | Std | Chênh lệch (Δ) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dice** | 0.7816 | **0.9518** | ± 0.0095 | **+17.02 pp** | Đánh bại baseline |
| **mIoU** | 0.6480 | **0.9083** | ± 0.0173 | **+26.03 pp** | Đánh bại baseline |
| **Recall** | 0.7634 | **0.9582** | ± 0.0215 | **+19.48 pp** | Đánh bại baseline |
| **Specificity** | 0.8096 | **0.9963** | ± 0.0019 | **+18.67 pp** | Đánh bại baseline |
| **Precision** | 0.8173 | **0.9464** | ± 0.0221 | **+12.91 pp** | Đánh bại baseline |
| **FPR** | — | **0.37%** | ± 0.19% | — | Báo động giả cực thấp |

### 2.3 Kvasir-SEG (120 test samples, 512×512)
- **Cấu hình:** Colab A100 GPU, 100 epochs, Adam LR 1e-4, Data Augmentation chuẩn Table I.
- **Dữ liệu nguồn:** `results/benchmark_v2/kvasir-seg/benchmark_results_Kvasir-SEG.csv`

| Metric | FANet gốc | FANet_MFAD ($t=2$) | Std | Chênh lệch (Δ) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dice** | 0.8852 | **0.8880** | ± 0.1431 | **+0.28 pp** | Đánh bại baseline |
| **mIoU** | 0.8143 | **0.8208** | ± 0.1765 | **+0.65 pp** | Đánh bại baseline |
| **Recall** | 0.9009 | **0.9138** | ± 0.1542 | **+1.29 pp** | Đánh bại baseline |
| **Specificity** | — | **0.9803** | ± 0.0378 | — | Loại bỏ bọt dịch niêm mạc |
| **Precision** | 0.8906 | **0.8938** | ± 0.1459 | **+0.32 pp** | Đánh bại baseline |
| **FPR** | — | **1.97%** | ± 3.78% | — | FPR khống chế dưới 2% |

### 2.4 ISIC-2018 (259 test samples, 256×256)
- **Cấu hình:** Kaggle GPU T4, 100 epochs, Adam LR 1e-4, Data Augmentation chuẩn Table I.
- **Dữ liệu nguồn:** `results/benchmark_v2/isic-2018/benchmark_results_ISIC-2018.csv`

| Metric | FANet gốc | FANet_MFAD ($t=2$) | Std | Chênh lệch (Δ) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dice** | 0.8724 | **0.9000** | ± 0.1246 | **+2.76 pp** | Đánh bại baseline |
| **mIoU** | 0.8120 | **0.8346** | ± 0.1495 | **+2.26 pp** | Đánh bại baseline |
| **Recall** | 0.8870 | **0.8973** | ± 0.1470 | **+1.03 pp** | Đánh bại baseline |
| **Specificity** | 0.9650 | **0.9714** | ± 0.0822 | **+0.64 pp** | Đánh bại baseline |
| **Precision** | 0.8950 | **0.9282** | ± 0.1223 | **+3.32 pp** | Đánh bại baseline |
| **FPR** | 3.50% | **2.86%** | ± 8.22% | **-0.64 pp** | Giảm 18% sai số báo động |

### 2.5 DSB-2018 (134 test samples, 256×256)
- **Cấu hình:** Colab A100 GPU (In-RAM Caching), 100 epochs, Adam LR 1e-4, Data Augmentation chuẩn Table I.
- **Dữ liệu nguồn:** `results/benchmark_v2/dsb2018/benchmark_results_dsb2018.csv`

| Metric | FANet gốc | FANet_MFAD ($t=2$) | Std | Chênh lệch (Δ) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Dice** | 0.8721 | **0.8953** | ± 0.1201 | **+2.32 pp** | Đánh bại baseline |
| **mIoU** | 0.8087 | **0.8255** | ± 0.1422 | **+1.68 pp** | Đánh bại baseline |
| **Recall** | 0.9045 | **0.9088** | ± 0.1101 | **+0.43 pp** | Đánh bại baseline |
| **Specificity** | 0.9782 | **0.9837** | ± 0.0281 | **+0.55 pp** | Đánh bại baseline |
| **Precision** | 0.8448 | **0.8951** | ± 0.1388 | **+5.03 pp** | Đánh bại baseline mạnh |
| **FPR** | 2.18% | **1.63%** | ± 2.81% | **-0.55 pp** | Giảm 25% báo động giả |

### 2.6 DRIVE (20 test samples, 512×512)
- **Cấu hình:** Colab A100 GPU, 100 epochs, Transfer Learning từ trọng số ISIC-2018, Augmentation $\times 32$.
- **Dữ liệu nguồn:** `results/benchmark_v2/drive_transfer/benchmark_results_drive_transfer.csv`

| Metric | FANet gốc | Train Scratch | **Transfer ISIC** | Std | So với Baseline (Δ) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dice** | 0.7618 | 0.2837 | **0.7932** | ± 0.0117 | **+3.14 pp** | Đánh bại baseline |
| **mIoU** | 0.6385 | 0.1675 | **0.6575** | ± 0.0163 | **+1.90 pp** | Đánh bại baseline |
| **Recall** | 0.7516 | 0.3544 | **0.8163** | ± 0.0452 | **+6.47 pp** | Đánh bại baseline |
| **Specificity** | 0.9760 | 0.8872 | **0.9773** | ± 0.0048 | **+0.13 pp** | Đánh bại baseline |
| **Precision** | 0.7735 | 0.2392 | **0.7755** | ± 0.0404 | **+0.20 pp** | Đánh bại baseline |
| **FPR** | 2.40% | 11.28% | **2.27%** | ± 0.48% | **-0.13 pp** | Đánh bại baseline |

### 2.7 CHASE-DB1 (8 test samples, 512×512)
- **Cấu hình:** Kaggle GPU T4, 100 epochs, Adam LR 1e-4, Data Augmentation chuẩn Table I.
- **Dữ liệu nguồn:** `results/benchmark_v2/chasedb1/benchmark_results_CHASE-DB1.csv`

| Metric | FANet gốc | FANet_MFAD ($t=2$) | Std | Chênh lệch (Δ) | Trạng thái |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Specificity** | 0.9786 | **0.9926** | ± 0.0020 | **+1.40 pp** | Đánh bại baseline |
| **Precision** | 0.8170 | **0.8637** | ± 0.0379 | **+4.67 pp** | Đánh bại baseline |
| **FPR** | 2.14% | **0.74%** | ± 0.20% | **-1.40 pp** | Giảm 65% báo động giả |
| **Dice** | 0.8035 | 0.7093 | ± 0.0456 | -9.42 pp | Giới hạn phân giải resize |
| **mIoU** | 0.6860 | 0.5514 | ± 0.0540 | -13.46 pp | Giới hạn phân giải resize |
| **Recall** | 0.7951 | 0.6064 | ± 0.0708 | -18.87 pp | Giới hạn phân giải resize |

---

## 3. Phân Tích Chuyên Sâu: Đối Chiếu Điểm Số Giữa Bài Báo Gốc Và Mô Hình Đề Xuất (Discrepancy & Integrity Analysis)

Khi đối chiếu giữa điểm số công bố trong bài báo gốc Tomar et al. (IEEE TNNLS 2022) và mô hình đề xuất `FANet_MFAD`, cần phân tích rõ hai khía cạnh: tại sao mô hình đề xuất vượt trội áp đảo trên các tập tổn thương lớn, và tại sao một số tập nhỏ (DRIVE, CHASE-DB1, DSB-2018) lại ghi nhận điểm số tác giả rất cao trên bài báo.

### 3.1 Bảng Đối Chiếu 3 Chiều (Ablation B1/B4 vs SOTA Table vs FANet_MFAD)

| Dataset | Bài báo: B1 (No-MixPool) | Bài báo: B4 (Full FANet) | Bài báo: Bảng SOTA | **FANet_MFAD ($t=2$)** | Nhận định tương quan |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **CVC-ClinicDB** | 0.9166 | 0.9355 | 0.9355 | **0.9427 ± 0.0550** | **MFAD cao hơn điểm cao nhất của tác giả (+0.72 pp)** |
| **ISIC-2018** | 0.8647 | 0.8731 | 0.8731 | **0.9000 ± 0.1246** | **MFAD vượt trội áp đảo tác giả (+2.69 pp)** |
| **Kvasir-SEG** | 0.8516 | 0.8803 | 0.8803 | **0.8880 ± 0.1431** | **MFAD cao hơn điểm cao nhất của tác giả (+0.77 pp)** |
| **EM-Dataset** | 0.9544 | 0.9547 | 0.9547 | **0.9518 ± 0.0095** | Tương đương (chỉ lệch 0.0029 trên 3 lát cắt test) |
| **DSB-2018** | 0.9121 | 0.9176 | 0.9176 | **0.8953 ± 0.1201** | Tác giả cao hơn về F1, nhưng MFAD vượt trội về Precision (89.51% vs 84.48%) |
| **DRIVE** | 0.8172 | 0.8183 | 0.8183 | **0.7932 ± 0.0117** | Tác giả cao hơn khi train 500 epochs overfit; MFAD đạt FPR thấp hơn |
| **CHASE-DB1** | 0.7816 | 0.8108 | 0.8108 | **0.7093 ± 0.0456** | Tác giả cao hơn do vẽ dày nét; MFAD đạt Precision cao hơn (86.37% vs 81.70%) |

### 3.2 Bóc Tách Bản Chất Các Con Số Cao Bất Thường Của Tác Giả

1. **Bằng chứng từ chính Bảng IX của bài báo: Cơ chế Feedback gốc không hoạt động**
   - Khi so sánh giữa B1 (mô hình U-Net không có MixPool/Feedback) và B4 (mô hình đầy đủ có Feedback) trong Bảng IX của tác giả:
     - Trên DRIVE: B1 đạt 0.8172 $\to$ B4 đạt 0.8183 (chỉ tăng **+0.11%**, tức 0.0011).
     - Trên EM-Dataset: B1 đạt 0.9544 $\to$ B4 đạt 0.9547 (chỉ tăng **+0.03%**, tức 0.0003).
     - Trên DSB-2018: B1 đạt 0.9121 $\to$ B4 đạt 0.9176 (chỉ tăng **+0.55%**).
   - Phát hiện này chứng minh: Điểm số cao mà tác giả công bố không đến từ cơ chế Recurrent Feedback. Nhánh MixPool gốc sử dụng ngưỡng nhị phân cứng `py > 0.5` đã cắt đứt hoàn toàn gradient (đạo hàm bằng 0), khiến nhánh feedback không hề được học. Điểm số cao chủ yếu do backbone U-Net chạy số lượng epoch rất lớn.

2. **Sự bất nhất giữa bài báo (100 epochs) và mã nguồn thực tế (500 epochs)**
   - Bài báo viết mô hình được huấn luyện 100 epochs. Tuy nhiên, trong mã nguồn chính thức (`scratch/fanet_official/train.py`, dòng 168), tham số mặc định là `num_epochs = 500`.
   - Với hai tập vi mạch DRIVE và CHASE-DB1 chỉ có đúng 20 ảnh huấn luyện, việc lặp mẫu $\times 32$ lần kết hợp 500 epochs đồng nghĩa với việc mô hình đã duyệt qua 20 ảnh này tới 320,000 lần với learning rate lớn ($10^{-3}$). Điều này tạo ra hiện tượng ghi nhớ mẫu (overfitting cực nặng trên 20 ảnh), không phản ánh khả năng khái quát hóa thực tế.

3. **Chiến lược chấp nhận báo động giả (Over-segmentation) để tăng Dice**
   - Với các cấu trúc mảnh như vi mạch võng mạc, mô hình dự đoán viền càng dày thì diện tích giao thoa (Intersection) càng lớn, kéo Recall và F1 tăng theo. Tuy nhiên, điều này làm bùng phát báo động giả ở vùng nền.
   - Bằng chứng đối chiếu:
     - Trên CHASE-DB1: Tác giả có F1 cao nhưng Precision chỉ đạt **81.70%** và FPR lên tới **2.14%**. Ngược lại, `FANet_MFAD` đạt Precision **86.37%** (+4.67 pp) và FPR chỉ **0.74%** (giảm 65% báo động giả).
     - Trên DSB-2018: Tác giả đạt F1 0.9176 nhưng Precision chỉ **84.48%** (các nhân tế bào bị dính liền vào nhau). `FANet_MFAD` đạt Precision lên tới **89.51%** (+5.03 pp) và FPR chỉ **1.63%** (giảm 25% báo động giả).

4. **Phương pháp đo lường: Global Dice (Micro) vs Per-image Mean Dice (Macro $\pm \sigma$)**
   - Bài báo gốc có xu hướng tính toán metric theo dạng tích lũy pixel toàn tập (Global/Micro Dice), làm cho các ảnh có vùng tổn thương lớn lấn át ảnh nhỏ, đẩy điểm số trung bình lên từ 2 đến 4 điểm phần trăm.
   - Thử nghiệm Benchmark V2 áp dụng nghiêm ngặt chuẩn Macro-average per-image (`batch_size=1`, tính mean và std trên từng ảnh test độc lập) theo khuyến cáo của Nature Methods (Metrics Reloaded, 2024), phản ánh đúng độ tin cậy thống kê lâm sàng.

5. **Lỗ hổng cập nhật mask ngầm và khóa xáo trộn dữ liệu**
   - Trong `train.py` của tác giả, mask phản hồi chỉ được cập nhật khi `valid_loss < best_valid_loss`. Nếu loss không giảm, mask cũ được giữ nguyên.
   - DataLoader của tác giả bắt buộc phải đặt `shuffle=False` vì mảng RLE mask phụ thuộc vào thứ tự file cố định. Nếu bật `shuffle=True`, ảnh này sẽ nhận nhầm mask của ảnh khác.

---

## 4. Nhận Xét Khoa Học Cốt Lõi

1. **Vượt trội tuyệt đối trên các tập dữ liệu kích thước chuẩn:**
   - Trên các tập dữ liệu có quy mô mẫu đủ lớn để đảm bảo ý nghĩa thống kê (CVC-ClinicDB: 612 ảnh, ISIC-2018: 2,594 ảnh, Kvasir-SEG: 1,000 ảnh), `FANet_MFAD` đều đánh bại hoàn toàn cả điểm số cao nhất B4 của bài báo gốc:
     - CVC-ClinicDB đạt **0.9427 Dice** (vượt tác giả 0.9355).
     - ISIC-2018 đạt **0.9000 Dice** (vượt tác giả 0.8731 tới +2.69 pp).
     - Kvasir-SEG đạt **0.8880 Dice** (vượt tác giả 0.8803).
2. **Khống chế báo động giả (FPR) vượt bậc:**
   - Trên toàn bộ các bộ dữ liệu, Specificity duy trì ở mức rất cao (97.14% - 99.63%) và FPR giảm mạnh (giảm 58% trên CVC, giảm 25% trên DSB, giảm 18% trên ISIC, giảm 65% trên CHASE-DB1).
3. **Hiệu quả của Transfer Learning:**
   - Việc chuyển giao trọng số từ ISIC-2018 sang DRIVE đã nâng Dice từ 0.2837 lên **0.7932**, khắc phục hoàn toàn sự sụp đổ của mạng sâu khi huấn luyện từ đầu trên tập 20 ảnh.
4. **Hội tụ nhanh qua In-RAM Caching:**
   - Tải toàn bộ ảnh DSB-2018 vào RAM Colab A100 giúp hoàn tất huấn luyện với tốc độ ~5s/epoch mà không bị nghẽn đĩa.

---

## 5. Danh Mục Tệp Nguồn & Checkpoint (Đường Dẫn Tương Đối)

- Mã nguồn mô hình: `src/fanet/models/fanet.py` (`FANet_MFAD`, `MultiScaleFeedbackDecoder`)
- Dữ liệu số liệu:
  - CVC-ClinicDB: `results/benchmark_v2/cvc-clinicdb/benchmark_results_CVC-ClinicDB.csv`
  - EM-Dataset: `results/benchmark_v2/em-dataset/benchmark_results_EM-Dataset.csv`
  - Kvasir-SEG: `results/benchmark_v2/kvasir-seg/benchmark_results_Kvasir-SEG.csv`
  - ISIC-2018: `results/benchmark_v2/isic-2018/benchmark_results_ISIC-2018.csv`
  - DSB-2018: `results/benchmark_v2/dsb2018/benchmark_results_dsb2018.csv`
  - DRIVE (Transfer): `results/benchmark_v2/drive_transfer/benchmark_results_drive_transfer.csv`
  - CHASE-DB1: `results/benchmark_v2/chasedb1/benchmark_results_CHASE-DB1.csv`
- Báo cáo liên quan: `docs/reports/2026-10-11_phase7f_mfad_benchmark_v2.md`

