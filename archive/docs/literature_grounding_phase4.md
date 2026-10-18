# Literature Grounding — Phase 4 (fallback losses + gating scope + novelty)

Search date: 31/08/2026. DB: OpenAlex (chính) + Semantic Scholar (1 query thành công, còn lại HTTP 429 rate-limited). Google Scholar chưa truy cập được từ env (cần retry thủ công hoặc qua khác).

## Fallback: loss phạt FP / negative-area

| # | Paper | Venue | Năm | DOI | Liên quan | Ý áp dụng cho FANet |
|---|---|---|---|---|---|---|
| 1 | An Improved Dice Loss for Pneumothorax Segmentation by Mining the Information of Negative Areas | IEEE Access | 2020 | 10.1109/access.2020.3020475 | CAO | **Negative-area Dice**: tách vùng ảnh hưởng tiêu cực (mô hình dự đoán sai fg) ra khỏi vùng ảnh hưởng tích cực, phạt riêng. Trực tiếp chống FP. Tích hợp: thêm term phạt vào DiceBCELoss hiện có |
| 2 | The Lovász-Softmax Loss: A Tractable Surrogate for the Optimization of the IoU | CVPR | 2018 | 10.1109/cvpr.2018.00464 | CAO | Lovász hinge = surrogate của Jaccard, phạt mạnh khi IoU thấp (FP+FN). Thay term Dice bằng Lovász-IoU nếu muốn tối ưu trực tiếp IoU |
| 3 | Jaccard Metric Losses: Optimizing the Jaccard Index with Soft Labels | arXiv | 2023 | 10.48550/arxiv.2302.05666 | TB | Phiên bản mềm hoá Jaccard loss — kết hợp được với soft feedback |
| 4 | Dice Semimetric Losses: Optimizing the Dice Score with Soft Labels | MICCAI | 2023 | 10.1007/978-3-031-43898-1_46 | TB | Tổng quát hoá Dice với soft labels — nền tảng nếu ta muốn Dice có trọng số theo độ chắc chắn |
| 5 | Background-Aware Pooling and Noise-Aware Loss for Weakly-Supervised Semantic Segmentation | CVPR | 2021 | 10.1109/cvpr46437.2021.00684 | TB | Background-aware pooling — minh chứng việc khai thác vùng background có giá trị trong segmentation |
| 6 | Unified Focal loss: Generalising Dice and cross entropy-based losses | CMIG | 2021 | 10.1016/j.compmedimag.2021.102026 | TB | Unified framework cho class imbalance — có thể điều chỉnh trọng số FP/FN qua alpha |

## Gating theo vùng (encoder-only vs decoder-only)

- **EMCAD** (CVPR 2024, 10.1109/cvpr52733.2024.01118): multi-scale convolutional attention decoding — ủng hộ ý tưởng "chỉ gate ở decoder" (tránh làm lệch phân phối encoder/BN).
- Không tìm thấy paper nào ablation rõ ràng "encoder-only vs decoder-only gating cho feedback mask" trong search boundary này → gating scope ablation là điểm mới, đáng làm.

## Novelty check: "background-aware feedback loop"

- OpenAlex `bg_feedback_loop`: không có work nào khớp "background-aware feedback loop" — các kết quả là review/HITL, không phải feedback mask loop.
- Semantic Scholar: bị 429, CHƯA xác nhận được. Google Scholar chưa chạy.
- **Kết luận tạm thời**: không tìm thấy tiền lệ trực tiếp trong OpenAlex; novelty claim "background-aware feedback loop" (kênh m_bg trong vòng phản hồi xuyên iteration) vẫn đứng vững trong search boundary hiện tại, NHƯNG cần xác nhận thêm trên Semantic Scholar + Google Scholar trước khi viết paper (ghi Will Do).

## Kết luận áp dụng cho Phase 4

- **Fallback chính nếu T_B1 bác bỏ lần nữa**: negative-area Dice loss (paper #1) — phạt trực tiếp FP trong loss, giữ nguyên feedback mechanism.
- **Fallback phụ**: decoder-only suppression — gate m_bg chỉ ở decoder (bảo toàn BN encoder), dựa trên hướng EMCAD.
- Loss hiện tại giữ nguyên DiceBCE cho 4 cell chính (để cô lập đóng góp của feedback mechanism).

## Provenance

- OpenAlex: `api.openalex.org/works?search=...`; S2: `api.semanticscholar.org/graph/v1/paper/search` (429 trên 5/6 queries).
- Script: `analysis/lit_search_phase4.py`. Access date 31/08/2026.
- Giới hạn: Semantic Scholar rate-limited, Google Scholar chưa truy cập → novelty còn pending.
