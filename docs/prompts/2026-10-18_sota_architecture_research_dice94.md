# Prompt: Nghiên cứu Kiến trúc SOTA Đẩy Chỉ Số Dice > 0.94 Toàn Diện Bằng Scientific Research Agent (18/10/2026)

## 1. Yêu cầu của User & Bối cảnh
- **Bối cảnh:**
  - Mô hình GDF-Net hiện tại trên tập dữ liệu ISIC-2018 (official validation và test) đạt mức Dice ~0.8904 - 0.8906.
  - Mặc dù mức điểm này đã vượt qua Attention U-Net thực nghiệm (0.8562) và đạt hạng cao so với leaderboard cuộc thi gốc (Challenge Winner Jaccard 0.802 ~ 0.875-0.880 Dice), người dùng nhận định mức điểm này chưa thật sự thỏa mãn vì chưa vượt qua ngưỡng Dice > 0.94.
- **Yêu cầu cụ thể:**
  1. Sử dụng bộ công cụ **Scientific Research Agent Skills** đã thiết lập trong repo (`.agents/skills/`, bao gồm `literature-review`, `bgpt-paper-search`, `paper-lookup`, `citation-management`, `experimental-design`, `hypothesis-generation`, `statistical-analysis`, `scientific-writing`, v.v.).
  2. Đọc, phân tích và tổng hợp các công trình nghiên cứu mới nhất (State-of-the-Art, 2023 - 2026) trong phân đoạn ảnh y tế và tổn thương da (đặc biệt các nghiên cứu thử nghiệm trên ISIC-2018 và các benchmark đa dạng như Polyp/Kvasir-SEG, CVC-ClinicDB, BUSI, GlaS, DRIVE/CHASE_DB1).
  3. Xác định rõ cơ chế đột phá giúp các phương pháp đạt Dice rất cao (>0.94) và bóc tách các yếu tố: kiến trúc nền tảng (backbone/encoder hiện đại: Swin, VMamba/State Space Models, ConvNeXt v2, SAM/MedSAM adaptation), cơ chế biểu diễn không gian - tần số, boundary-aware refinement, cơ chế chú ý (attention/mamba), và hàm mục tiêu (loss formulations).
  4. Đề xuất một kiến trúc mới hoặc kế hoạch nâng cấp căn bản cho GDF-Net (Next-Generation GDF-Net / Ultra-Dice Architecture) nhằm mục tiêu đạt Dice > 0.94 một cách vững chắc và tổng quát hóa trên tất cả các tập dữ liệu đã thử nghiệm.
  5. Xuất tài liệu Literature Grounding (`docs/literature_grounding_dice94_sota.md`) với đầy đủ trích dẫn (Title, Authors, Venue, Year, DOI/arXiv, Mechanisms, Empirical Evidence) và Báo cáo nghiên cứu theo template chuẩn.

## 2. Ràng buộc & Quy ước Repo
- Tuân thủ nghiêm ngặt quy định không sử dụng emoji / biểu tượng cảm xúc.
- Không để lộ đường dẫn tuyệt đối cục bộ của máy tính.
- Thực hiện quy trình Git commit & push ngay sau khi tạo / sửa đổi file.
- Phân biệt minh bạch giữa kết quả 5-fold CV nội bộ (internal validation) và kết quả thẩm định trên tập độc lập ngoại kiểm (unseen multi-center test).
