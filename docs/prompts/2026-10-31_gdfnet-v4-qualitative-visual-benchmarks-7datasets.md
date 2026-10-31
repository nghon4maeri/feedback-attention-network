# Prompt — 2026-10-31: Xuất Hình ảnh Trực quan Định tính (Qualitative Visual Figures) Top-Accuracy trên 7 Tập Dữ liệu Lâm sàng

## Mục tiêu (Objective)
Xuất bộ hình ảnh đối sánh trực quan chất lượng cao (publication-grade qualitative figures) cho mô hình chủ lực đề xuất **`GDF-Netv4-ConvFormerS18`** trên toàn bộ 7 tập dữ liệu y tế lâm sàng:
1. **ISIC-2018** (Dermoscopy Ung thư da, 384x384)
2. **CVC-ClinicDB** (Colonoscopy Polyp đại tràng, 384x384)
3. **Kvasir-SEG** (Nội soi tiêu hóa Polyp, 384x384)
4. **DSB-2018** (Kính hiển vi huỳnh quang Tế bào nhân, 256x256)
5. **EM-Dataset** (Kính hiển vi điện tử Màng nơ-ron thần kinh 3D, 512x512)
6. **CHASE-DB1** (Đáy mắt Vi mạch máu võng mạc, 768x768)
7. **DRIVE** (Đáy mắt Vi mạch máu võng mạc, 576x576)

## Yêu cầu và Ràng buộc (Constraints)
- **100% Ảnh Thật (No Synthetic/Mock)**: Tuyệt đối không dùng hình học nhân tạo hay fallback vẽ giả lập. Toàn bộ hình ảnh là ca bệnh thật từ test set, ground truth thật từ chuyên gia y tế, và dự đoán thật sinh ra bởi trọng số checkpoint `.pth` đã train.
- **Top Accuracy Selection**: Lọc và chọn các ca phân đoạn có độ chính xác cao nhất (Dice > 0.83 đến 0.99) để làm nổi bật tiềm năng tối đa của kiến trúc mạng GDF-Netv4 trước hội đồng và người phản biện.
- **Cấu trúc Panel 5 cột chuẩn IEEE / MICCAI**:
  - `(a) Input Image`
  - `(b) Ground Truth`
  - `(c) GDF-Netv4 (Proposed Prediction)`
  - `(d) Error Discrepancy Map` (TP: Xanh lá, FP: Đỏ, FN: Xanh dương, TN: Xám đậm)
  - `(e) Boundary Contour Overlay` (Viền Vàng: Ground Truth, Viền Cyan: Dự đoán GDF-Netv4)
- **Xuất bản phẩm**:
  - 7 Panel riêng lẻ cho từng dataset (độ phân giải cao 300 DPI PNG).
  - 1 Master Figure tổng hợp 7 modality (7 dòng x 5 cột) định dạng PNG và PDF vector 300 DPI.
- **Tuân thủ Git Workflow**: Commit và push ngay lên remote GitHub theo `AGENTS.md`.
