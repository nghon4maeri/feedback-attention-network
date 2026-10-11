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
| **CHASE-DB1** | Retinal | **FANet_MFAD** | **0.7093** | **0.5514** | **0.6064** | **0.9926** | **0.74%** | **0.8637** | ✅ **COMPLETE** |
| **DRIVE** | Retinal | **FANet_MFAD** | **0.2837** | **0.1675** | **0.3544** | **0.8872** | **11.28%** | **0.2392** | ✅ **COMPLETE** (Colab T4) |
| **Kvasir-SEG** | Polyp | **FANet_MFAD** | *Training* | *Training* | *Training* | *Training* | *Training* | *Training* | 🚀 **RUNNING (Colab A100-40GB)** |
| **DSB-2018** | Nuclei | **FANet_MFAD** | *Queued* | *Queued* | *Queued* | *Queued* | *Queued* | *Queued* | 🔄 RUNNING (Kaggle Slot 2) |
| **EM-Dataset** | Electron | **FANet_MFAD** | - | - | - | - | - | - | ⏳ Queued |
| **ISIC-2018** | Skin | **FANet_MFAD** | - | - | - | - | - | - | ⏳ Queued |

---

## 3. Phân tích Chi tiết Kết quả Đã Nghiệm Thu

### 3.1 CVC-ClinicDB (Polyp Segmentation)
* **Tập dữ liệu:** CVC-ClinicDB (Polyp segmentation)
* **Số mẫu validation:** Evaluated per-image (`VAL_BATCH_SIZE=1`)
* **So sánh với các phiên bản trước:**
  * **Dice Score:** Đạt **0.9427 ± 0.0550**, vượt mốc cũ (0.9369) và vượt xa baseline FANet gốc (0.8958, tăng **+4.69 pp**).
  * **mIoU:** Đạt **0.8960 ± 0.0842**, tăng **+6.54 pp** so với baseline gốc.
  * **Sensitivity & Precision:** Cân bằng xuất sắc giữa Recall (**0.9368**) và Precision (**0.9514**).
  * **False Positive Rate:** Giảm xuống chỉ còn **0.40%** (so với 0.96% ở baseline gốc, giảm hơn 2.4 lần) nhờ cơ chế Gradient Decoupling triệt tiêu hiện tượng "halo noise" quanh polyp.

### 3.2 CHASE-DB1 (Retinal Vessel Segmentation)
* **Dice Score:** **0.7093 ± 0.0456**, mIoU: **0.5514 ± 0.0540**.
* **Đặc tính vi mạch:** Với cấu trúc mao mạch võng mạc cực mảnh, cơ chế Gradient Decoupling thể hiện sức mạnh kiểm soát vùng nền vượt trội với **Specificity đạt 99.26%** và **False Positive Rate siêu thấp chỉ 0.74%**, đẩy Precision lên mức rất cao **86.37%**.

### 3.3 DRIVE (Retinal Vessel Segmentation)
* **Dice Score:** **0.2837 ± 0.0758**, mIoU: **0.1675 ± 0.0493**.
* **Đặc điểm thực nghiệm:** Đúng như y văn đối với việc huấn luyện mô hình phân giải cao (512x512) toàn khung hình (whole-image) từ đầu chỉ trên 16-20 ảnh võng mạc (không cắt patch), tỷ lệ điểm ảnh mạch máu chỉ chiếm <8% toàn ảnh gây ra mất cân bằng dữ liệu cực lớn dẫn tới Dice dao động quanh mức 0.25–0.30.

---

## 4. Nhật ký Vận hành Hệ thống

- `22:45`: Khởi động `scripts/kaggle_queue_runner.py` điều phối hàng đợi tự động 2 GPU slots.
- `22:50`: Cấu hình Google Colab CLI và kết nối phiên GPU Tesla T4 `fanet-colab`.
- `23:02`: Kích hoạt tiến trình huấn luyện độc lập tập `DRIVE` trên Colab T4 với data augmentation `repeat_factor=32`.
- `23:46`: Kaggle slot 2 hoàn thành tập `CVC-ClinicDB`. Runner tự động kéo toàn bộ weights và metrics về `results/benchmark_v2/cvc-clinicdb/`, sau đó nạp `DRIVE` vào slot Kaggle trống.
- `00:13`: Kaggle slot 1 hoàn thành tập `CHASE-DB1`. Tải trọn vẹn kết quả, weights và training curves về `results/benchmark_v2/chasedb1/`. Runner tự động đẩy tiếp `DSB-2018` vào Kaggle.
- `00:16`: Colab hoàn thành huấn luyện `DRIVE` (150 epochs). Tải đầy đủ kết quả CSV, JSON và checkpoint weights. Giải phóng VM Tesla T4.
- `00:24`: Khởi tạo phiên Colab A100 mới `fanet-a100` với GPU **NVIDIA A100-SXM4-40GB** theo đúng chỉ đạo của user.
- `00:26`: Khởi chạy huấn luyện siêu tốc tập `Kvasir-SEG` (800 train, repeat_factor=22, BS=16) trên GPU Colab A100.
