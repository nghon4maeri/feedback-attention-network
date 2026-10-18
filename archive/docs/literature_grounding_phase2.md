# Literature Grounding — Phase 2+3 (FANet dual-path feedback)

Search date: 31/08/2026. Database: OpenAlex (queries ghi trong `analysis/lit_search*.py`).

## Bảng papers liên quan

| # | Paper | Venue | Năm | DOI | Mức liên quan | Ý tưởng áp dụng cho FANet |
|---|---|---|---|---|---|---|
| 1 | PraNet: Parallel Reverse Attention Network for Polyp Segmentation | MICCAI | 2020 | 10.1007/978-3-030-59725-2_26 | CAO | **Reverse attention**: attention vùng background (1 - prediction) để "dọn" over-segmentation — tiền lệ trực tiếp cho dual-path m_bg |
| 2 | UACANet: Uncertainty Augmented Context Attention for Polyp Segmentation | ACM MM | 2021 | 10.1145/3474085.3475375 | CAO (đã đọc sâu) | Tách m_f/m_b/m_u từ saliency liên tục; background context pool thành vector; residual refinement — thiết kế m_bg tham khảo từ đây |
| 3 | FANet (paper gốc) | IEEE TNNLS | 2022 | 10.1109/tnnls.2022.3159394 | GỐC | Baseline; xác nhận lại 2 điểm yếu (fmask chết, hard threshold) |
| 4 | STAR-Caps: Capsule Networks with Straight-Through Attentive Routing | arXiv | 2019 | (arXiv) | TRUNG BÌNH | **Straight-through estimator cho attention routing** — bằng chứng STE dùng được cho attention rời rạc, hướng Phase 2 |
| 5 | An Improved Dice Loss for Pneumothorax Segmentation by Mining the Information of Negative Areas | IEEE Access | 2020 | 10.1109/access.2020.3020475 | TRUNG BÌNH | **Negative-area mining trong loss** — tương tự tinh thần khai thác background, nhưng ở loss thay vì feedback mask |
| 6 | High-Resolution Iterative Feedback Network for Camouflaged Object Detection | AAAI | 2023 | 10.1609/aaai.v37i1.25167 | TRUNG BÌNH | Iterative feedback + refinement cho đối tượng khó (camouflage) — polyp có tính chất tương tự (ít contrast) |
| 7 | F³Net: Fusion, Feedback and Focus for Salient Object Detection | AAAI | 2020 | 10.1609/aaai.v34i07.6916 | THẤP-TB | Feedback mechanism khác (multi-scale), tham khảo cách ablation feedback |
| 8 | The Lovász-Softmax Loss | CVPR | 2018 | 10.1109/cvpr.2018.00464 | THẤP-TB | Surrogate IoU loss — alternative nếu cần phạt FP qua loss thay vì feedback |
| 9 | Polyp-PVT | arXiv | 2021 | 10.48550/arxiv.2108.06932 | THẤP | SOTA polyp — mục tiêu so sánh tương lai (Will Do) |
| 10 | Online Hard Example Mining (OHEM) | CVPR | 2016 | 10.1109/cvpr.2016.89 | THẤP | Hard-negative mining — nền tảng ý tưởng "chỉ feedback vùng khó" |

## Đánh giá novelty của dual-path feedback

1. **Reverse attention (PraNet) tồn tại** — nhưng là attention nội tại single-pass, KHÔNG phải feedback xuyên epoch/iteration như FANet. Kết hợp "reverse/background attention" + "feedback loop" chưa thấy trong các paper tìm được trong search boundary này.
2. **UACANet có m_b (background context)** — nhưng dùng trong cùng một forward pass (stage-to-stage), không phải feedback loop epoch trước → epoch sau.
3. → **Claim novelty dự kiến**: "background-aware feedback loop" (kênh m_bg trong vòng phản hồi xuyên iteration) + "learnable fmask qua STE" — phạm vi mới so với các work trên (cần kiểm tra kỹ hơn trước khi claim trong paper).

## Kết luận áp dụng

- **m_bg thiết kế theo PraNet/UACANet**: m_bg = vùng model tin chắc là background, từ prediction mềm (p < 0.5 - tau) thay vì lấy 1 - m_fg cứng (tránh nhạy với biên).
- **STE cho fmask**: tham khảo STAR-Caps — gradient đi qua soft value trong backward, forward vẫn dùng hard gate để giữ độ mạnh prune.
- **Loss không đổi** (giữ DiceBCE để cô lập đóng góp của feedback mechanism; negative-area loss là hướng bổ sung Will Do).

## Provenance

- DB: OpenAlex API `api.openalex.org/works?search=...&select=title,publication_year,doi,cited_by_count`
- 2 rounds, 11 queries; access date 31/08/2026; scripts `analysis/lit_search.py`, `analysis/lit_search2.py`
- Giới hạn: chỉ OpenAlex; chưa query arXiv/Semantic Scholar; novelty claim cần search bổ sung (Semantic Scholar + Google Scholar) trước khi viết paper.
