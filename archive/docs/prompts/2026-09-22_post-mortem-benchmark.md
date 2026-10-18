# Prompt (22/09/2026): Complexity Benchmark and Post-Mortem

**Mục tiêu**:
1. Hoàn thành Quantitative Data (Computational Complexity) cho Phase 7B để submit báo cáo khoa học.
2. Viết báo cáo "Post-Mortem & Baseline Reproduction Report" tóm tắt hành trình gỡ lỗi và benchmark cuối cùng.

**Phạm vi & Yêu cầu**:
- Tạo `scripts/benchmark_complexity.py` để đo Parameters, MACs (qua ptflops), và Inference FPS (dùng torch.cuda.Event).
- Khởi tạo 2 model: M11 (Baseline) và M12 (Phase 7B).
- In ra Markdown table và lưu file vào `paper_tables/complexity_benchmark.md`.
- Soạn thảo báo cáo Post-Mortem chuẩn mực, phân tích Root Cause của lỗi BGR/RGB mismatch, kết quả reproduce baseline, và kết luận độ phức tạp tính toán (không tốn overhead, chạy nhanh hơn).

**Kết quả mong đợi**:
Bản Markdown report Post-Mortem chất lượng cao, update `docs/reports/README.md`, và có sẵn bảng benchmark Complexity cho manuscript.
