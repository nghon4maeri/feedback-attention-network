# Session Prompt Log: GDF-Netv2 Multi-Benchmark Research & Autonomous Execution

- **Ngày:** 19/10/2026
- **Mục tiêu:** Huấn luyện và thẩm định kiến trúc GDF-Netv2 trên 7 tập dữ liệu benchmark lâm sàng (ISIC-2018, CVC-ClinicDB, Kvasir-SEG, DRIVE, CHASE-DB1, DSB-2018, EM-Dataset) theo triết lý nghiên cứu mở không giới hạn trần giả định.
- **Phạm vi & Yêu cầu:**
  1. Tối ưu kiến trúc GDF-Netv2: ConvNeXt-V2-Tiny backbone, Spatial FDF, Linear SBA, MSF Gate, Composite BD-IoU Loss.
  2. Báo cáo đầy đủ 9 chỉ số lâm sàng (Dice, mIoU, Sensitivity, Specificity, Precision, FPR, HD95, ASD, BF2) với Mean, Std, Median, Min, Max.
  3. Quản lý song song môi trường Colab GPU (ISIC-2018) và Kaggle GPU (các tập benchmark còn lại).
  4. Tuân thủ nghiêm ngặt quy ước repo: không emoji, không đường dẫn tuyệt đối cục bộ, commit/push sau mỗi thay đổi file code.
