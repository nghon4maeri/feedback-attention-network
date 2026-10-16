# Prompt: Retrain ISIC-2018 với Partition BGDNet & Đánh giá trên Official Test Set (16/10/2026)

## 1. Yêu cầu của User & Bối cảnh
- **Mục tiêu:** Huấn luyện lại mô hình GDF-Net (FANet-MFAD) trên bộ dữ liệu da liễu ISIC-2018 theo chuẩn phân chia dữ liệu học thuật mới nhất từ paper BGDNet (*AIMS Bioengineering* 2026, DOI: 10.3934/bioeng.2026019) và đánh giá trên tập **official test set** (1,000 ảnh).
- **Ràng buộc quan trọng:** Tuyệt đối không sử dụng tập validation chính thức (100 ảnh) làm tập validation ("ko nên dùng val trên official nhé").

## 2. Phân tích Literature & Data Partitioning (BGDNet Protocol)
- **Paper tham chiếu:** Zhang et al., "Boundary-guided dual-branch network for skin lesion segmentation", *AIMS Bioengineering*, 2026, 13(1): 315-334.
- **Quy tắc phân chia dữ liệu:**
  1. **Training Pool (2,594 ảnh):** Được chia theo tỷ lệ deterministic thành **2,334 ảnh huấn luyện (Train)** và **260 ảnh validation nội bộ (Val)** dùng để checkpoint model selection (chọn checkpoint có Dice cao nhất).
  2. **Official Validation Set (100 ảnh):** Bị loại trừ, không dùng làm tập thẩm định (đáp ứng đúng yêu cầu của người dùng và văn hiến).
  3. **Official Test Set (1,000 ảnh):** Dùng để đánh giá năng lực khái quát hóa cuối cùng của mô hình sau khi chọn checkpoint tối ưu từ 260 ảnh val.

## 3. Nguồn Dữ liệu & Môi trường Kaggle
- **Test Images (1,000 ảnh):** `ISIC2018_Task1-2_Test_Input` từ dataset `tschandl/isic2018-challenge-task1-data-segmentation`.
- **Test Ground Truth Masks (1,000 ảnh):** `ISIC2018_Task1_Test_GroundTruth` từ dataset `mfahimhossen/isic2018-test-groundtruth`.
- **Huấn luyện:** Khởi chạy kernel `namnguynnnn/fanet-benchmark-isic-2018` trên Kaggle với GPU P100/T4.

## 4. Trạng thái Thực thi
- Đã sinh notebook hoàn chỉnh và cập nhật `kernel-metadata.json`.
- Đã đẩy kernel thành công lên Kaggle (`Kernel version 12`) và đang ở trạng thái `KernelWorkerStatus.RUNNING`.
- Theo dõi tiến trình và cập nhật kết quả benchmark ngay khi hoàn tất.

## 6. Hoàn tất Kaggle & Kích hoạt Resume Huấn luyện trên Colab A100 (17/10/2026)
- **Kết quả Kaggle (`namnguynnnn/fanet-benchmark-isic-2018`):**
  - Hoàn tất 113 epochs với peak validation Dice **0.9026**.
  - Đánh giá trên **1.000 ảnh test chính thức**:
    - Dice Score: **0.8906 ± 0.1184**
    - mIoU: **0.8186 ± 0.1504**
    - Sensitivity: **0.9436 ± 0.1264**
    - Specificity: **0.9296 ± 0.1241**
    - Precision: **0.8671 ± 0.1348**
    - FPR: **0.0704 ± 0.1241**
    - HD95: **29.35 ± 44.65** | ASD: **10.25 ± 16.30** | BF@2: **0.3352 ± 0.2486**
- **Kích hoạt Resume Huấn luyện trên Colab A100 qua `colab-cli`:**
  - Script điều khiển: [`scripts/colab_run_isic2018_resume_a100.py`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/scripts/colab_run_isic2018_resume_a100.py).
  - Tiếp tục huấn luyện từ Epoch 114 đến mốc tối đa 200 epochs theo BGDNet trên GPU NVIDIA A100-SXM4-40GB.
  - Tự động đánh giá và kết xuất bảng chỉ số chính thức khi đạt 200 epochs.

## 7. Nghiệm thu Hoàn tất & Kéo Toàn bộ Output về Local (17/10/2026)
- **Tiến trình:** Hoàn tất 200/200 epochs trên NVIDIA A100-SXM4-40GB.
- **Đánh giá trên Official Test 1,000 ảnh:**
  - File kết quả: [`results/isic-2018_official_a100_final/benchmark_results_ISIC-2018_official_test_a100.csv`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/results/isic-2018_official_a100_final/benchmark_results_ISIC-2018_official_test_a100.csv)
  - Biểu đồ trực quan: [`results/isic-2018_official_a100_final/comparison_ISIC-2018.png`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/results/isic-2018_official_a100_final/comparison_ISIC-2018.png)
  - Checkpoint tốt nhất: `results/isic-2018_official_a100_final/GDFNet_ISIC2018_best.pth` (98 MB)
- **Báo cáo nghiên cứu chi tiết:** [`docs/reports/2026-10-17_phase7g_isic2018_bgdnet_official_test_benchmark.md`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/docs/reports/2026-10-17_phase7g_isic2018_bgdnet_official_test_benchmark.md)



