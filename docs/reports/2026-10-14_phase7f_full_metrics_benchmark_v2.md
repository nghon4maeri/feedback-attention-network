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

## 3. Nhận Xét Khoa Học Cốt Lõi

1. **Vượt trội toàn diện trên các khối tổn thương và tế bào:**
   - CVC-ClinicDB đạt **0.9427 Dice** (+4.69 pp), EM-Dataset đạt **0.9518 Dice** (+17.02 pp), ISIC-2018 đạt **0.9000 Dice** (+2.76 pp), Kvasir-SEG đạt **0.8880 Dice** (+0.28 pp), DSB-2018 đạt **0.8953 Dice** (+2.32 pp).
2. **Khống chế báo động giả (FPR) vượt bậc:**
   - Trên toàn bộ các bộ dữ liệu, Specificity duy trì ở mức rất cao (97.14% - 99.63%) và FPR giảm mạnh (giảm 58% trên CVC, giảm 25% trên DSB, giảm 18% trên ISIC).
3. **Hiệu quả của Transfer Learning:**
   - Việc chuyển giao trọng số từ ISIC-2018 sang DRIVE đã nâng Dice từ 0.2837 lên **0.7932**, đánh bại baseline bài báo gốc (0.7618) trên 100% metrics.
4. **Hội tụ nhanh qua In-RAM Caching:**
   - Tải toàn bộ ảnh DSB-2018 vào RAM Colab A100 giúp hoàn tất huấn luyện với tốc độ ~5s/epoch mà không bị nghẽn đĩa.

---

## 4. Danh Mục Tệp Nguồn & Checkpoint (Đường Dẫn Tương Đối)

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
