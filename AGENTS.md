# AGENTS.md — Working conventions for this repository

## Git workflow (BẮT BUỘC)

- Sau MỖI lần tạo file mới hoặc sửa file code (scripts/, src/, analysis/,
  notebooks/, configs/, kaggle/), **commit và push lên GitHub ngay**.
- Dùng message tiếng Anh, theo phong cách hiện có của repo (vd:
  `feat: ...`, `fix: ...`, `docs: ...`, `analysis: ...`).
- Chỉ stage các file liên quan đến công việc đang làm; không commit file
  không liên quan. Commit rồi `git push` ngay trong cùng lệnh.
- Thứ tự: `git status` -> `git diff` (xem lại) -> `git add <files>` ->
  `git commit -m "..."` -> `git push`.

## Research workflow

- Mỗi phase nghiên cứu: viết report theo `docs/reports/_TEMPLATE.md`,
  cập nhật index `docs/reports/README.md` (timeline + experiment tracking).
- **Prompt/phiên làm việc:** lưu prompt giao cho AI assistant dưới
  `docs/prompts/` với tên có ngày (vd `docs/prompts/2026-09-09_<slug>.md`),
  nội dung tóm tắt mục tiêu + phạm vi + constraint để tái lập ngữ cảnh và
  theo dõi repo đang làm tới đâu. Commmit/push theo git workflow chung.
- Khi research (tìm paper, phân tích literature, xác định research gap):
  BẮT BUỘC dùng scientific agent research skills để hỗ trợ —
  literature-review, paper-lookup, citation-management, exa-search,
  research-lookup, bgpt-paper-search, hypothesis-generation,
  experimental-design, statistical-analysis, scientific-visualization,
  markdown-mermaid-writing, ... Tìm paper qua NHIỀU nguồn (OpenAlex,
  Semantic Scholar, arXiv, PubMed/Europe PMC), lưu citation đầy đủ
  (title/venue/năm/DOI) vào `docs/literature_grounding_*.md`, và phân tích
  research gap có đối chiếu bằng chứng thực nghiệm của project.
- Phân biệt bằng chứng: TRAIN (đáng tin) vs FROZEN EVAL (chỉ là khả năng
  khai thác feedback của model cũ) vs INVALID (bug) khi viết kết luận.