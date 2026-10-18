# [CN – 05/09/2026] — Review 7 paper mới + cập nhật plan X2/X3/X4

## Mục tiêu tuần này

Đọc full-text 7 paper vừa tải tay (FEGNet, RefineU-Net, Conditional Boundary
Loss, BCNet, CTNet, CFA-Net, BUNet), đối chiếu cơ chế feedback/loss với bằng
chứng Phase 1–4 và plan X2/X4/X5 (report 09/04), đồng thời audit lại 38 PDF cũ
trong `docs/` (phát hiện 8 file tải nhầm). Không chạy thí nghiệm, không code.

## Done

- [x] Extract text 7 PDF (pymupdf) → tóm tắt method/loss/results/limitation
- [x] Đọc chi tiết cơ chế feedback của FEGNet (RGM, T=3, FAAG/MSM) và RefineU-Net (GRM/LRM)
- [x] Audit trang đầu toàn bộ 38 PDF trong docs/ — phát hiện 8 file SAI nội dung
- [x] Sửa 6/8 PDF sai (verify trang đầu đúng title): BoundaryLoss (1812.07032),
      CaraNet (2301.13366), F³Net (1911.11445), HRINet (2203.11624),
      SANet (2108.00882), SINet (CVPR open access)
- [x] Xóa 2 PDF sai không sửa được (STARCaps, DistanceTransform — OpenReview
      chặn IP) + ghi link tải tay vào `papers_not_accessible.md`
- [x] Cập nhật `docs/literature_grounding_next.md`: 25 → 32 paper, novelty section,
      bảng audit PDF
- [x] Cập nhật khuyến nghị Gate 1 cho X2 dựa trên 7 paper

## Findings quan trọng

### F1. Tóm tắt 7 paper + lesson cho FANet

| # | Paper (venue, năm) | Phương pháp chính | Loss | Feedback? | Lesson cho FANet |
|---|---|---|---|---|---|
| 1 | FEGNet (JBHI 2023) | Res2Net-50 + Recurrent Gate Module (FAAG+MSM) ở skip connections f3/f4, unroll T=3 trong 1 forward pass; Edge Extractor cho f1/f2 | L=wIoU+wBCE + L_R (deep sup từng time-step) + L_edge | Feature/attention tự hồi quy, SOFT gate, KHÔNG mask, KHÔNG bg channel | Nếu quay lại feedback line: soft gate + deep supervision từng step + đặt ở SKIP (không ở input) |
| 2 | RefineU-Net (PRL 2020) | VGG-16 + GRM (top-down fusion các side-output, L2-norm fusion) + LRM (residual attention gate ở decoder) | Standard (dice/bce) | Feature-level "progressive global feedbacks" nội mạng; KHÔNG mask feedback | Attention các tầng TỰ HỌC xen kẽ ROI/background → bg info đã tồn tại nội tại; không cần kênh m_bg tường minh |
| 3 | Conditional Boundary Loss (TIP 2023) | Loss hoạt động trên FEATURE space: anchor=boundary pixel (distance transform) → pull về local class center (A2C) + push khác class (A2P&N), correctness-aware sampling | CBL phụ trợ + loss chính | Không | KHÔNG vào Gate 1: nhắm biên (không chữa FP xa biên 40.7px) + cần embeddings |
| 4 | BCNet (JBHI 2022) | CFIS cross-layer fusion (self-attention) + DUAL OUTPUT: polyp area + polyp boundary | L_b=wBCE+wIoU cho TỪNG nhánh (area & boundary) | Không | Pattern "boundary branch + BCE trên edge GT" rẻ, giúp biên; wIoU+wBCE thay DiceBCE |
| 5 | CTNet (TCyb 2024) | Contrastive transformer backbone + SMIM + CIM | L=L_S+0.1·L_NCE (contrastive) | Không | Cần backbone mới — không drop-in; evidence contrastive giúp discrimination |
| 6 | CFA-Net (PR 2023) | Res2Net-50 + boundary prediction network + two-stream seg network + CFF/BAM | L_seg = wIoU + wBCE với pixel-weight **(1+5μ)**, μ = distance-to-boundary map | Không | **Boundary-weighted wIoU+wBCE là loss drop-in rẻ nhất** — candidate Gate 1 |
| 7 | BUNet (Neural Networks 2024) | PVT + BEM (low-level boundary) + BUM (high-level uncertainty, khai thác vùng score ~0.5) + top-down deep supervision | (wIoU-style + deep sup) | Không | BUM nhắm đúng vùng ambiguous → khớp vấn đề FN sát biên 8.6px, KHÔNG khớp FP tràn nền 40.7px |

**Pattern xuyên suốt 7 paper:** 5/7 dùng wIoU+wBCE (weighted IoU + weighted BCE)
làm loss chính — không paper nào dùng DiceBCE thuần như FANet. Đây là bằng chứng
mạnh rằng bước loss-side đầu tiên của X2 nên là ĐỔI DiceBCE → wIoU+wBCE, rẻ và
được validate trên chính Kvasir-SEG bởi nhiều nhóm.

### F2. Vì sao FEGNet/RefineU-Net "feedback" thành công còn FANet dual-path thất bại

| Khía cạnh | FEGNet / RefineU-Net | FANet (H_A/H_B) |
|---|---|---|
| Loại feedback | Feature/attention **nội mạng** (FEGNet: gate tự hồi quy T=3; RefineU-Net: top-down fusion) | **Mask prediction xuyên epoch ở INPUT** → đổi phân phối đầu vào mỗi epoch → BN shock là hệ quả thiết kế, không phải bug |
| Gate | SOFT (sigmoid attention) → gradient luôn chảy | HARD threshold → fmask zero-grad (P_A1) |
| Giám sát gate | Deep supervision từng time-step (L_R ép output RGM match GT) | Chỉ loss cuối → fmask học vô định |
| Vị trí | Skip connections / decoder | Toàn bộ encoder+decoder, input-mask |
| Loss đi kèm | wIoU+wBCE (hard-pixel weighted) | DiceBCE thuần |
| Background | KHÔNG có kênh bg; RefineU-Net quan sát attention tự học xen kẽ ROI/background | kênh m_bg tường minh (H_B) → FP↑ frozen p=1e-7; train xấu/collapse |

**Kết luận đối chiếu (a):** FEGNet/RefineU-Net không xác nhận tính khả thi của
H_A/H_B — họ thành công chính xác vì KHÔNG làm cái FANet làm (mask-at-input,
hard gate, no supervision). Lesson chuyển cho X3: feedback ở skip/decoder với
soft gate + deep supervision là hình mẫu duy nhất có bằng chứng hoạt động.

### F3. Boundary-family losses vs X2 — phán quyết CÓ/KHÔNG

- **KHÔNG chọn CBL (TIP 2023) và Kervadec boundary loss cho Gate 1**: chúng nhắm
  cải thiện BIÊN, trong khi root cause Phase 1 là FP **xa biên 40.7px** (84.6%
  oracle gap ở vùng FP) — hai vấn đề khác nhau. CBL thêm yêu cầu feature
  embeddings (không drop-in). → giữ làm candidate dự phòng cho BƯỚC SAU (khi FN
  sát biên 8.6px cần chữa).
- **CÓ chọn boundary-weighted (CFA-Net (1+5μ))**: weight theo distance-to-boundary
  làm hard pixels nặng hơn mà không cần thêm nhánh mạng — drop-in, chi phí ≈ 0
  ngoài precompute distance map. Tinh thần giống wIoU+wBCE của FEGNet/BCNet/CFA.
- Lovász (plan cũ) tương đương wIoU về tinh thần (surrogate IoU) nhưng wIoU+wBCE
  có ưu thế: 5/7 paper polyp dùng đúng nó trên đúng dataset của mình.

### F4. Novelty verdict CẬP NHẬT (sau khi đọc FEGNet + RefineU-Net)

- Không paper nào có **kênh background tường minh trong mask feedback loop** →
  novelty hình thức giữ nguyên mức "còn nhưng mỏng" (report 09/04).
- NHƯNG thêm 2 bằng chứng chống:
  1. RefineU-Net: attention các tầng **tự học xen kẽ ROI/background** → thông tin
     background đã có sẵn qua attention nội tại — kênh m_bg tường minh thừa.
  2. FEGNet trích dẫn FANet và cố ý đặt feedback ở **skip connection thay vì
     input** ("rather than merely connecting the input and output") → mask-feedback-
     at-input (thiết kế gốc FANet) bị chính cộng đồng coi là cần tránh.
- **Phán quyết: giữ nguyên — "novelty mỏng, không đáng đặt cược"**, cộng thêm
  kết luận design-level: dòng dual-path m_bg không chỉ thiếu novelty mà còn đi
  ngược pattern thiết kế thành công của các work cùng họ feedback.

## Experiments (đề xuất — CHƯA chạy)

| ID | Giả thuyết | Config | Effort | Stopping rule |
|---|---|---|---|---|
| X2-Gate1 (CẬP NHẬT) | wIoU+wBCE (+neg-area hoặc boundary-weight) giảm FP ≥1pp mà không sụp Dice | Cell A: wIoU+wBCE + NegativeAreaDiceBCELoss (code sẵn); Cell B: boundary-weighted wIoU+wBCE theo CFA-Net (1+5μ); seed 43, 200ep, batch=2 | ~7 GPU-h (2 cells) + ~2h code | FP < T00−0.01 và Dice ≥ T00−0.02 ở cả 2 cells → chọn winner qua multi-seed (Gate 2 giữ nguyên như 09/04); nếu cả 2 fail → pivot |
| X4 | Chẩn đoán STE: grad variance + fmask viz + BN stats | CPU, dùng ckpt_T00/T10 trên disk | 0 GPU-h | Như report 09/04 — không đổi |
| X3 (nâng prior) | bg-suppression/feedback ở skip+decoder, soft gate + deep sup (theo FEGNet) | Chỉ khi X2 fail và muốn đóng feedback line | ~7 GPU-h + code | Collapse rule ep40 |

**Thay đổi so với 09/04:** Gate 1 của X2 chuyển từ (Lovász-mix, Tversky β=0.3)
sang **(wIoU+wBCE+neg-area, boundary-weighted wIoU+wBCE)**. Lý do: 5/7 paper
polyp vừa đọc dùng wIoU+wBCE trên chính Kvasir-SEG (evidence TỪ PAPER, không
phải suy diễn); boundary-weight là dạng "boundary loss" duy nhất đủ rẻ để vào
Gate 1; Tversky giữ làm cell thứ 3 nếu có GPU.

## Will Do (On going)

- [ ] Tải tay 2 PDF còn thiếu (STARCaps, DistanceTransform) qua link OpenReview trong `papers_not_accessible.md`
- [ ] Tải tay Negative-area Dice (IEEE Access) — **bắt buộc trước khi chạy X2** để verify `NegativeAreaDiceBCELoss` đúng công thức gốc
- [ ] Cài wIoU+wBCE (+ boundary-weight (1+5μ)) vào `src/fanet/losses.py` — không đổi kiến trúc
- [ ] Chạy X2 Gate 1 (2 cells, seed 43) trên Kaggle T4
- [ ] X4 (0 GPU) song song — đóng vấn đề STE
- [ ] Gate 2 multi-seed chỉ khi Gate 1 pass (như 09/04)

## Any Stuck / Open Questions

- NegativeAreaDiceBCELoss hiện implement "soft surrogate" (docstring tự nhận) —
  chưa đối chiếu được với paper gốc vì PDF chưa tải tay. ĐỪNG chạy Gate 1 trước
  khi verify, kẻo claim sai về loss.
- 2 PDF đúng không tải tự động được (OpenReview 403 từ IP máy) — cần tải tay.
- Vẫn giữ giới hạn: 1 split (40 val), 1 seed cho mọi số đã có; sensitivity n=40
  chỉ detect δ(Dice)≥0.13 (report 09/04 F4) — không đổi.
- Câu hỏi mở mới từ F2: FANet gốc (mask-at-input) vốn là thiết kế xấu về phân
  phối — nếu X2 fail, câu hỏi tiếp theo nên là "có nên bỏ feedback input hẳn
  (feed ảnh thuần) và xem baseline không-feedback mạnh tới đâu" thay vì thêm cơ chế.

## Đính kèm link chi tiết

- 7 PDF full-text: `docs/FEGNet_*.pdf`, `docs/1-s2.0-S0167865520302592-main.pdf`,
  `docs/Conditional_Boundary_Loss_*.pdf`, `docs/Boundary_Constraint_Network_*.pdf`,
  `docs/CTNet_*.pdf`, `docs/1-s2.0-S0031320323002558-main.pdf`,
  `docs/1-s2.0-S0893608023006731-main.pdf`
- Literature cập nhật (32 paper + novelty + audit PDF): `docs/literature_grounding_next.md`
- Danh sách tải tay còn thiếu: `papers_not_accessible.md` (root)
- Loss hiện tại: `src/fanet/losses.py` — report plan: `docs/reports/2026-09-04_next-directions.md`
- Template: `docs/reports/_TEMPLATE.md` — Index: `docs/reports/README.md`
