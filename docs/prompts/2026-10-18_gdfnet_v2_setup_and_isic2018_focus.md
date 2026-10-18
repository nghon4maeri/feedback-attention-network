# Prompt: Khởi tạo Phân hệ Nghiên cứu GDF-Netv2 & Tập trung Đột phá ISIC-2018 (18/10/2026)

## 1. Yêu cầu của User & Bối cảnh
- **Bối cảnh:**
  - Sau khi hoàn thành phân tích căn nguyên điểm số của GDF-Net (Mean 0.9002, Median 0.9430 trên 1.000 ảnh test chính thức) và tổng hợp y văn SOTA 2023–2026, người dùng đồng thuận triển khai kiến trúc thế hệ mới GDF-Netv2.
- **Yêu cầu cụ thể:**
  1. Tạo thư mục chuyên biệt `GDF-Netv2/` để đồng hành cùng người dùng nghiên cứu, thiết kế kiến trúc, xây dựng mã nguồn và chạy benchmark trên toàn bộ 7 tập dữ liệu.
  2. Đặt ưu tiên hàng đầu là đảm bảo mô hình cho kết quả vượt trội ở tập **ISIC-2018** trước khi mở rộng ra 6 tập dữ liệu còn lại.
  3. Cấu trúc thư mục module hóa hoàn chỉnh: kiến trúc mô hình (models), hàm mất mát đặc chế (losses), tiền xử lý và tăng cường dữ liệu (data), pipeline huấn luyện và đánh giá (training/scripts).
  4. Tuân thủ nghiêm ngặt quy trình Git workflow (commit & push sau mỗi lần tạo/sửa file), không sử dụng emoji, không để lộ đường dẫn tuyệt đối cục bộ.

## 2. Thiết kế Cấu trúc Module GDF-Netv2
- `GDF-Netv2/README.md`: Báo cáo kiến trúc, công thức toán học, chiến lược huấn luyện và bảng theo dõi 7 datasets.
- `GDF-Netv2/models/`:
  - `backbones.py`: Bộ mã hóa phân cấp hiện đại (ConvNeXt-V2, PVTv2-B2 với fallback kiến trúc).
  - `frequency_filter.py`: Bộ lọc tách tần số (FDF) triệt tiêu nhiễu lông và thước kẻ da liễu.
  - `selective_boundary.py`: Module Selective Boundary Aggregation (SBA) kết hợp nhánh ranh giới học được.
  - `gdfnet_v2.py`: Lớp mô hình hoàn chỉnh tích hợp Multi-Channel Semantic Feedback (MSF) qua Detached Gradient Firewall và MFAD-v2.
- `GDF-Netv2/losses/`:
  - `boundary_losses.py`: Boundary Difference over Union (BD-IoU) Loss, Focal Tversky Loss, Edge Loss.
- `GDF-Netv2/data/`:
  - `dataset.py` & `augmentations.py`: Pipeline nạp dữ liệu và tăng cường tối ưu hóa cho độ phân giải cao ($384 \times 384$).
- `GDF-Netv2/training/`:
  - `train_isic2018.py`: Script huấn luyện chuyên biệt cho ISIC-2018.
  - `evaluate.py`: Đo lường trọn vẹn 9 chỉ số vùng và biên.
