# [Phase 7F – 11/10/2026] Đánh giá Benchmark V2: Retrain MFAD-Only với Data Augmentation chuẩn Paper

## 1. Mục tiêu & Cấu hình Nghiên cứu

Huấn luyện lại duy nhất mô hình đề xuất **`FANet_MFAD`** (ResNet-34 Encoder + Multi-scale Feedback Attention Decoder + Gradient Decoupling), đánh giá độc quyền pha **Final Feedback ($t=2$)**, với bộ Data Augmentation hoàn chỉnh chuẩn paper gốc Tomar et al. (Table I & Section IV-A-2).

### Cấu hình cải tiến:
1. **Model:** Chỉ train 1 model `FANet_MFAD` (bỏ qua baseline, tiết kiệm 50% compute).
2. **Evaluation:** Đánh giá trực tiếp tại $t=2$ (`out2 = model(x, torch.sigmoid(out1))`) so với Ground Truth.
3. **Data Augmentation:** Tích hợp `repeat_factor=32` + CLAHE cho các tập vi mạch (DRIVE, CHASE-DB1), `repeat_factor=16` cho EM-Dataset, và `repeat_factor=2` cho Polyp/Nuclei.
4. **Hạ tầng song song:** Phối hợp đồng thời Kaggle GPU Batch Slots (tối đa 2 jobs song song) và Google Colab CLI (Tesla T4 / A100).

---

## 2. Bảng Kết quả Tổng hợp Benchmark V2

| Dataset | Type | Model | Dice | mIoU | Sensitivity | Specificity | FPR | Precision | Status |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CVC-ClinicDB** | Polyp | **FANet_MFAD** | **0.9427** | **0.8960** | **0.9368** | **0.9960** | **0.40%** | **0.9514** | ✅ **COMPLETE** |
| **CHASE-DB1** | Retinal | **FANet_MFAD** | *Training* | *Training* | *Training* | *Training* | *Training* | *Training* | 🔄 RUNNING (Kaggle) |
| **DRIVE** | Retinal | **FANet_MFAD** | *Training* | *Training* | *Training* | *Training* | *Training* | *Training* | 🔄 RUNNING (Colab T4 & Kaggle) |
| **DSB-2018** | Nuclei | **FANet_MFAD** | - | - | - | - | - | - | ⏳ Queued |
| **EM-Dataset** | Electron | **FANet_MFAD** | - | - | - | - | - | - | ⏳ Queued |
| **ISIC-2018** | Skin | **FANet_MFAD** | - | - | - | - | - | - | ⏳ Queued |
| **Kvasir-SEG** | Polyp | **FANet_MFAD** | - | - | - | - | - | - | ⏳ Queued |

---

## 3. Phân tích Chi tiết Kết quả CVC-ClinicDB

* **Tập dữ liệu:** CVC-ClinicDB (Polyp segmentation)
* **Số mẫu validation:** Evaluated per-image (`VAL_BATCH_SIZE=1`)
* **So sánh với các phiên bản trước:**
  * **Dice Score:** Đạt **0.9427 ± 0.0550**, vượt mốc cũ (0.9369) và vượt xa baseline FANet gốc (0.8958, tăng **+4.69 pp**).
  * **mIoU:** Đạt **0.8960 ± 0.0842**, tăng **+6.54 pp** so với baseline gốc.
  * **Sensitivity & Precision:** Cân bằng xuất sắc giữa Recall (**0.9368**) và Precision (**0.9514**).
  * **False Positive Rate:** Giảm xuống chỉ còn **0.40%** (so với 0.96% ở baseline gốc, giảm hơn 2.4 lần) nhờ cơ chế Gradient Decoupling triệt tiêu hiện tượng "halo noise" quanh polyp.

---

## 4. Nhật ký Vận hành Hệ thống

- `22:45`: Khởi động `scripts/kaggle_queue_runner.py` điều phối hàng đợi tự động 2 GPU slots.
- `22:50`: Cấu hình Google Colab CLI và kết nối phiên GPU Tesla T4 `fanet-colab`.
- `23:02`: Kích hoạt tiến trình huấn luyện độc lập tập `DRIVE` trên Colab T4 với data augmentation `repeat_factor=32`.
- `23:46`: Kaggle slot 2 hoàn thành tập `CVC-ClinicDB`. Runner tự động kéo toàn bộ weights và metrics về `results/benchmark_v2/cvc-clinicdb/`, sau đó tự động nạp tập `DRIVE` vào slot Kaggle trống.
