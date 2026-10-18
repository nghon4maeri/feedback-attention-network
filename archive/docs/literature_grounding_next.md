# Literature Grounding — Next Directions (04–05/09/2026)

Search date: 04/09/2026 (+ bổ sung 05/09: 7 paper đọc full-text, nguồn = tải
tay). Nguồn: OpenAlex (33 queries title.search/search), arXiv API (8 queries),
Europe PMC (5 queries), Semantic Scholar (bị 429 — thất bại), PubMed e-utilities
(bị NCBI chặn IP — thất bại). 41 ứng viên → chọn 32 paper cho bảng chính.
PDF: 29 paper tải tự động + 7 tải tay về `docs/` (log:
`docs/paper_download_log.json`; thiếu/đã sửa: `papers_not_accessible.md` ở root repo).

## Bảng papers liên quan (32 curated)

| # | Paper | Venue | Năm | DOI | Mức liên quan | Ý tưởng áp dụng cho FANet |
|---|---|---|---|---|---|---|
| 1 | FANet: A Feedback Attention Network for Improved Biomedical Image Segmentation | IEEE TNNLS | 2022 | 10.1109/tnnls.2022.3159394 | GỐC | Baseline; 2 điểm yếu đã xác nhận (fmask zero-grad, feedback fg-only) |
| 2 | Transformer-Enhanced Iterative Feedback Mechanism for Polyp Segmentation (FANetv2) | IEEE ICASSP | 2025 | 10.1109/icassp49660.2025.10890567 (arXiv:2409.05875) | CAO | Cùng lab FANet: iterative mask feedback + text-guided + refinement lúc test; feedback VẪN fg-only → novelty m_bg còn nhưng bị cạnh tranh trực tiếp |
| 3 | A Cortically-Inspired Predictive Coding Framework for Polyp Segmentation | Research Square preprint | 2026 | 10.21203/rs.3.rs-9145958/v1 | CAO | Top-down prediction-error feedback lúc inference; đo trực tiếp FPR/FDR + FP-trên-nếp-gấp — "dọn FP qua feedback" là bài toán mở 2026; đối thủ novelty gần nhất |
| 4 | FEGNet: A Feedback Enhancement Gate Network for Automatic Polyp Segmentation | IEEE JBHI | 2023 | 10.1109/jbhi.2023.3272168 | TB–CAO | Feedback refinement gate nội mạng, không thêm tham số; gợi ý đặt feedback ở decoder thay vì toàn bộ (→ X3) |
| 5 | Learning With Context Feedback Loop for Robust Medical Image Segmentation | IEEE TMI | 2021 | 10.1109/tmi.2021.3060497 | TB | Feedback = context features (không phải mask bg) — dùng để phân biệt claim của mình |
| 6 | F³Net: Fusion, Feedback and Focus for Salient Object Detection | AAAI | 2020 | 10.1609/aaai.v34i07.6916 | TB | Cascaded feedback decoder; mẫu ablation feedback đa tầng |
| 7 | PraNet: Parallel Reverse Attention Network for Polyp Segmentation | MICCAI | 2020 | 10.1007/978-3-030-59725-2_26 | CAO | Reverse attention (vùng 1−pred) "dọn" over-segmentation — tiền lệ trực tiếp của m_bg; single-pass |
| 8 | UACANet: Uncertainty Augmented Context Attention for Polyp Segmentation | ACM MM | 2021 | 10.1145/3474085.3475375 | CAO (đã đọc sâu) | Tách m_f/m_b/m_u; background context pool; residual refinement |
| 9 | Shallow Attention Network for Polyp Segmentation | MICCAI | 2021 | 10.1007/978-3-030-87193-2_66 | TB | Polyp nhỏ; baseline so sánh |
| 10 | Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformers | arXiv (CAAI AIR 2023) | 2021 | arXiv:2108.06932 | TB | CAC module attention vào prediction mềm — tinh thần "feedback mềm" |
| 11 | CaraNet: context axial reverse attention network for segmentation of small medical objects | J. Medical Imaging | 2023 | 10.1117/1.jmi.10.1.014005 | TB | Axial reverse attention cho object nhỏ |
| 12 | MEGANet: Multi-Scale Edge-Guided Attention Network for Weak Boundary Polyp Segmentation | WACV | 2024 | 10.1109/wacv57701.2024.00780 (arXiv:2309.03329) | TB | Edge-guidance cho polyp biên mờ — liên quan boundary (X2) |
| 13 | Colorectal Polyp Segmentation Based on Deep Learning Methods: A Systematic Review | J. Imaging | 2025 | 10.3390/jimaging11090293 | TB | Landscape 2025 + chuẩn metric/report |
| 14 | An Improved Dice Loss for Pneumothorax Segmentation by Mining the Information of Negative Areas | IEEE Access | 2020 | 10.1109/access.2020.3020475 | CAO (X2) | **Công thức verified 07/09/2026** (full text): WSDice = 1 − [2Σ(Ĝ·G)+ε]/[ΣĜ²+ΣG²+ε], Ĝ=w(2ŷ−1), G=w(2y−1), w=y(v2−v1)+v1 (v2=1−v1 → w_fg=v2, w_bg=v1). `src/fanet/losses.py` đã implement `WeightedSoftDiceLoss(v1=0.3)` dùng cho cell A; `NegativeAreaDiceBCELoss` cũ = soft surrogate (precision-penalty), KHÔNG phải công thức gốc |
| 15 | The Lovász-Softmax Loss | CVPR | 2018 | 10.1109/cvpr.2018.00464 | CAO (X2) | Surrogate IoU — IoU phạt FP mạnh hơn Dice khi vùng thừa lớn |
| 16 | Tversky Loss Function for Image Segmentation Using 3D FCN | MLMI (LNCS) | 2017 | 10.1007/978-3-319-67389-9_44 | TB–CAO (X2) | β<0.5 → weight FP > FN; một knob duy nhất cân bằng FP/FN |
| 17 | Boundary loss for highly unbalanced segmentation | Medical Image Analysis | 2020 | 10.1016/j.media.2020.101851 | TB (X2) | Distance-map loss cho vùng không cân bằng |
| 18 | How Distance Transform Maps Boost Segmentation CNNs: An Empirical Study | MIDL | 2020 | OpenReview 64cCPvmbXY (không có arXiv) | TB (X2) | Bằng chứng distance map giúp CNN học biên; PDF cần tải tay (link trong papers_not_accessible.md) |
| 19 | Active Boundary Loss for Semantic Segmentation | AAAI | 2022 | 10.1609/aaai.v36i2.20139 (arXiv:2102.02696) | TB (X2) | Boundary loss mới, có so sánh với các loss cũ |
| 20 | Metrics reloaded: recommendations for image analysis validation | Nature Methods | 2024 | 10.1038/s41592-023-02151-z | CAO (methodology) | Chuẩn metric/CI/report — áp cho toàn bộ eval pipeline |
| 21 | Estimating or Propagating Gradients Through Stochastic Neurons for Conditional Computation | arXiv | 2013 | arXiv:1308.3432 | TB (X4) | Nguồn gốc STE; điều kiện gradient qua ngưỡng hoạt động |
| 22 | Categorical Reparameterization with Gumbel-Softmax | ICLR | 2017 | arXiv:1611.01144 | TB (X4) | Alternative cho gate rời rạc nếu STE chết |
| 23 | STAR-Caps: Capsule Networks with Straight-Through Attentive Routing | NeurIPS | 2019 | OpenReview ycfKowdIEG (không có arXiv) | TB | STE cho attention routing — tiền lệ gần nhất (khác domain); PDF cần tải tay |
| 24 | Tent: Fully Test-time Adaptation by Entropy Minimization | ICLR | 2021 | arXiv:2006.10726 | TB | Quan điểm khác cho refinement lúc test (entropy thay vì mask loop) |
| 25 | Polyp segmentation with consistency training and continuous update of pseudo-label | Scientific Reports | 2022 | 10.1038/s41598-022-17843-3 | TB | Semi-supervised polyp; pseudo-label = "feedback mềm" ngoài vòng lặp |
| 26 | FEGNet (đọc full-text 05/09, nguồn: tải tay) | IEEE JBHI | 2023 | 10.1109/jbhi.2023.3272168 | CAO | Recurrent attention gate (T=3, soft) ở SKIP connections + deep supervision từng time-step + L=wIoU+wBCE+edge — mẫu tham khảo nếu quay lại feedback line (X3) |
| 27 | RefineU-Net (đọc full-text 05/09, nguồn: tải tay) | Pattern Recognition Letters | 2020 | 10.1016/j.patrec.2020.07.013 | TB–CAO | "Feedback" = top-down feature fusion nội mạng (KHÔNG mask feedback); attention tự học xen kẽ ROI/background → background info đã có qua attention nội tại |
| 28 | Conditional Boundary Loss (đọc full-text 05/09, nguồn: tải tay) | IEEE TIP | 2023 | 10.1109/tip.2023.3290519 | TB (X2) | Feature-space loss: pull boundary pixel về local class center, push khác class; nhắm BIÊN — không chữa FP xa biên 40.7px → không vào Gate 1 |
| 29 | BCNet (đọc full-text 05/09, nguồn: tải tay) | IEEE JBHI | 2022 | 10.1109/jbhi.2022.3173948 | TB–CAO (X2) | Dual output (area+boundary) + L=wBCE+wIoU cho từng nhánh — pattern "boundary branch rẻ" cho FANet |
| 30 | CTNet (đọc full-text 05/09, nguồn: tải tay) | IEEE TCyb | 2024 | 10.1109/tcyb.2024.3368154 | THẤP (X2) | Contrastive transformer, L=L_S+0.1·L_NCE; cần backbone mới — không drop-in; evidence contrastive giúp discrimination |
| 31 | CFA-Net (đọc full-text 05/09, nguồn: tải tay) | Pattern Recognition | 2023 | 10.1016/j.patcog.2023.109555 | CAO (X2) | wIoU+wBCE với weight (1+5μ) theo distance-to-boundary — boundary-weighted loss drop-in, candidate Gate 1 |
| 32 | BUNet (đọc full-text 05/09, nguồn: tải tay) | Neural Networks | 2024 | 10.1016/j.neunet.2023.11.050 | TB | BUM nhắm vùng score ~0.5 (bias neither fg nor bg) — khớp vấn đề FN sát biên 8.6px, KHÔNG khớp FP tràn nền |

## Đánh giá novelty của "background-aware feedback loop" (sau Phase 4)

1. **FANetv2 (ICASSP 2025)** — chính lab của FANet đã mở rộng: iterative mask
   feedback từ epoch trước + text-guided + refinement lúc test. Feedback vẫn
   foreground-only → kênh m_bg tường minh vẫn là điểm khác biệt, nhưng không
   gian "iterative mask feedback" đã có người chiếm.
2. **Predictive coding polyp (2026)** — dùng top-down prediction-error feedback
   lúc inference; prediction error ngầm mang thông tin background và paper này
   đo thẳng FPR/FDR + FP trên nếp gấp. Không có kênh m_bg tường minh, không có
   cross-epoch mask loop → khác biệt còn lại của FANet-dual-path là **"kênh
   background tin cậy tường minh trong vòng phản hồi mask xuyên epoch"** — phạm
   vi novelty CÒN NHƯNG MỎNG, và bằng chứng thực nghiệm (Phase 3: FP↑ p=1e-7;
   Phase 4: train dual xấu/collapse) không ủng hộ giá trị thực tiễn của cơ chế.
3. **Bổ sung 05/09 (sau khi đọc FEGNet + RefineU-Net)**: cả hai làm "feedback"
   nhưng ở mức feature/attention NỘI mạng (FEGNet: recurrent attention gate
   T=3 ở skip connections, soft gate, deep supervision từng time-step; RefineU-Net:
   top-down feature fusion). **Không paper nào làm kênh background tường minh
   trong mask feedback loop** → novelty hình thức vẫn đứng. NHƯNG:
   - RefineU-Net quan sát attention các tầng **tự học xen kẽ ROI/background** →
     background information đã tồn tại trong attention nội tại, làm giảm giá trị
     của kênh m_bg tường minh.
   - FEGNet trích dẫn thẳng FANet và cố ý đặt feedback ở **skip connection thay
     vì input** ("rather than merely connecting the input and output") → mask-feedback-
     at-input (thiết kế gốc của FANet) được họ coi là lựa chọn cần tránh.
4. PraNet reverse attention / UACANet m_b: single-pass, không phải feedback loop.
5. → **Phán quyết (giữ nguyên, thêm design-level)**: novelty "mỏng" không đổi;
   nay có thêm bằng chứng ở cấp thiết kế rằng (a) feedback nên ở feature level
   (skip), không ở input-mask level, (b) background thông tin tốt nhất học qua
   attention nội tại — cả hai đều chống lại định hướng dual-path m_bg tường minh.
6. → **Khuyến nghị claim**: đừng dựa paper vào novelty của dual-path; novelty
   khả dĩ hơn nằm ở (a) phân tích negative-result có hệ thống (FP/FN decomposition,
   BN mismatch confounder, bug-tracking), hoặc (b) hướng loss-side nếu X2 thành công.

## Audit PDF (05/09) — cảnh báo file sai nội dung

8 PDF tải tự động ngày 04/09 có nội dung SAI (arXiv ID sai trong danh sách gốc).
Đã sửa 6 (verify trang đầu), xóa 2 (chờ tải tay qua OpenReview):

| File | Trạng thái 05/09 | Bản đúng |
|---|---|---|
| BoundaryLoss_MICCAI2019.pdf | ✅ SỬA | arXiv 1812.07032 (Kervadec, boundary loss) |
| CaraNet_arXiv2021.pdf | ✅ SỬA | arXiv 2301.13366 (CaraNet) |
| F3Net_AAAI2020.pdf | ✅ SỬA | arXiv 1911.11445 |
| HRINet_AAAI2023.pdf | ✅ SỬA | arXiv 2203.11624 |
| SANet_MICCAI2021.pdf | ✅ SỬA | arXiv 2108.00882 |
| SINet_CVPR2020.pdf | ✅ SỬA | CVPR 2020 open access (cvf) |
| STARCaps_arXiv2019.pdf | ❌ ĐÃ XÓA — chờ tải tay | OpenReview ycfKowdIEG |
| DistanceTransform_MIDL2020.pdf | ❌ ĐÃ XÓA — chờ tải tay | OpenReview 64cCPvmbXY |

KHÔNG dùng 8 file trên cho bất kỳ claim nào trước ngày 05/09; từ 05/09 chỉ 6 file
đã sửa là dùng được.

## Provenance

- DB: OpenAlex (`api.openalex.org/works`), arXiv (`export.arxiv.org/api/query`),
  Europe PMC (`ebi.ac.uk/europepmc/webservices/rest`). Access date 04/09/2026.
- Thất bại: Semantic Scholar (HTTP 429, không key), PubMed e-utilities (NCBI
  chặn IP datacenter) — ghi nhận giới hạn coverage.
- Scripts: `analysis/lit_search*.py` (phase 2), script tạm trong
  `%TEMP%\opencode\lit_*.py` (phiên 04/09; không commit).
- Verify metadata: 14 DOI kiểm tra ngược qua OpenAlex (title/venue/year khớp).
