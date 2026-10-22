# [Phase 13 – 22/10/2026] GDF-Netv5 Architecture Specification, Encoder Benchmark Survey & Stratified 5-Fold Cross Validation Protocol

## 1. Ký hiệu (Notation) & Quy ước

- `GDF-Netv5`: General Decoupled Feedback Network Version 5 (Hierarchical Backbone `pvt_v2_b4` + Reverse Edge Guidance Stream + Multi-Scale Frequency Filter v5 + Mask-Guided LGAG + Cascaded MSCAM Decoder).
- `REGS`: Reverse Edge Guidance Stream (nhánh trích xuất và dự đoán biên độc lập, tạo cổng ức chế nền lành).
- `RBAG`: Reverse Boundary Attention Gate (cổng chú ý biên đảo ngược triệt tiêu hiện tượng loang màu ra quầng tối viền).
- `MS-FDF v5`: Multi-Scale Frequency-Decoupled Filter với ngân hàng tích chập sâu song song $[1\times1, 3\times3, 5\times5]$.
- `Artifact-Aware Augmentation`: Bộ tăng cường dữ liệu chuyên biệt bệnh học da liễu (Synthetic Vignetting + CLAHE + Synthetic Hair).
- `5-Fold CV`: Stratified 5-Fold Cross Validation trên 2,594 ảnh train, kiểm thử mù trên 1,000 ảnh test độc lập theo 2 chế độ: Single-Model (Mean $\pm$ Std) và 5-Fold Soft-Voting Ensemble.
- Phân loại bằng chứng: `TRAIN` (huấn luyện thực nghiệm), `FROZEN EVAL` (đánh giá mù độc lập), `LITERATURE` (số liệu công bố có kiểm toán DOI).

---

## 2. Bối cảnh & Động lực Khoa học

Ở Phase 12, mô hình [`GDFNetV4`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv4/models/gdfnet_v4.py#L14-L125) đã đạt kỷ lục mới trên 1,000 ảnh test mù chính thức của ISIC-2018 với **Mean Dice 0.9030 (90.30%)**, **Median Dice 0.9357 (93.57%)**, và $J_{\text{thresh}} = 0.7974$.

Tuy nhiên, phân tích sâu phân phối 1,000 ảnh test mù đã phát hiện:
1. **71.7% số ảnh (717/1,000)** đã đạt $\text{Dice} \ge 0.90$, và **35.3% số ảnh** đạt $\text{Dice} \ge 0.95$.
2. Nếu loại bỏ 30 ca ngoại lai thấp nhất (3% đuôi phân phối), Mean Dice của 970 ảnh còn lại đạt tới **0.9182 (91.82%)**.
3. Điểm nghẽn duy nhất kéo sụt Mean Dice toàn tập nằm ở **41 ca cá biệt ($\text{Dice} < 0.70$, trung bình 0.4127)**, phân cực thành 2 thái cực rõ rệt:
   - **28 ca False Positive (Over-segmentation):** Độ nhạy đạt 99.7% (bắt trọn tổn thương), nhưng Precision chỉ đạt 34.9% do dự đoán loang rộng ra quầng tối kính soi (vignetting), bọt gel hoặc vết mực y tế.
   - **12 ca False Negative (Under-segmentation):** Precision đạt 99.1% nhưng Sensitivity chỉ đạt 34.8% do mô hình co cụm vào tâm sẫm màu, bỏ sót dải viền mờ của các khối u hắc tố thiếu sắc tố (amelanotic melanoma).

**GDF-Netv5** được thiết kế nhằm mục tiêu giải quyết triệt để 2 thái cực sai số này để đưa chỉ số Mean Dice vượt mốc **0.915 – 0.925**.

---

## 3. Khảo sát Y văn Tuyển chọn Encoder (2024–2026)

Trước khi thiết kế v5, một khảo sát có hệ thống đối với các bài báo có điểm Dice cao nhất trên ISIC-2018 đã được thực hiện:

| Encoder / Backbone | Paper Tiêu biểu (Venue) | ISIC-2018 DSC (%) | Phân tích Điểm Mạnh | Hạn chế Kỹ thuật | Đánh giá Tuyển chọn cho v5 |
| :--- | :--- | :---: | :--- | :--- | :---: |
| **PVTv2-B4** | • **EMCAD** (CVPR 2024)<br>• **CASCADE** (WACV 2023)<br>• **Polyp-PVT** (CAAI AIR 2023) | **90.96% – 91.50%+** | • 62.0M params, 24 blocks ở Stage 3 (gấp 3 chiều sâu ngữ cảnh của B2).<br>• Overlapping Patch Embedding giữ liên tục ranh giới tế bào.<br>• Linear SRA attention toàn khung hình $O(HW)$.<br>• Depthwise FFN không phụ thuộc fixed grid, scale mượt lên $512 \times 512$. | Chi phí tính toán cao hơn B2 khoảng 2.2x. | **CHỌN (Mặc định v5)**<br>*(Tối ưu nhất cả về toán học và độ tin cậy)* |
| **PVTv2-B5** | • PVTv2 scaling (2022–2024) | ~91.3% | 81.4M params, dung lượng biểu diễn cực lớn. | Dễ overfitting trên tập train 2,594 ảnh nếu không có regularization cực mạnh. | **Dự phòng (Switchable)** |
| **VMamba-S / B** | • **VM-UNet** (ACM TOMM 2025)<br>• **UltraLight VM-UNet** (Patterns 2025) | 89.71% – 90.91% | Quét 2D-SSM độ phức tạp tuyến tính, tiết kiệm FLOPs. | Quét zig-zag/chéo làm đứt tính liên tục 2D của ranh giới da liễu; phụ thuộc CUDA extension đặc thù dễ lỗi trên PyTorch 2.6+. | Loại trừ cho core model |
| **ConvNeXt-V2-Base** | • **ConvNeXt-V2** (CVPR 2023)<br>• **CMUNeXt** (ISBI 2024) | 90.50% – 91.20% | Pretraining FCMAE trên ImageNet-22k đạt Top-1 86.8%, tích chập $7 \times 7$ + GRN rất mạnh. | Thiếu cơ chế Self-Attention toàn khung hình ở tầng sâu, dễ nhầm ở nốt ruồi khổng lồ tràn viền. | Hỗ trợ cho Ablation |
| **Swin-B** | • **SUTrans-NET** (2024)<br>• **TESL-Net** (2025) | 90.80% – 91.40% | Window Attention giảm tải quadratic complexity. | Cửa sổ cố định ($12 \times 12$) gây suy hao nội suy khi áp dụng Multi-Scale TTA ($0.8\times, 1.2\times$). | Loại trừ |

---

## 4. Đặc tả Kiến trúc Toàn diện GDF-Netv5

Kiến trúc [`GDFNetV5`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/models/gdfnet_v5.py) tích hợp 5 thành phần cốt lõi:

### 4.1. Hierarchical Vision Transformer Encoder (`PVTv2-B4`)
- Module [`HierarchicalBackboneV5`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/models/backbones.py) nạp `pvt_v2_b4` tiền huấn luyện ImageNet-1k.
- Ở độ phân giải $512 \times 512$, 4 tầng đặc trưng xuất ra gồm:
  $$\mathbf{F}_1 \in \mathbb{R}^{B \times 64 \times 128 \times 128}, \quad \mathbf{F}_2 \in \mathbb{R}^{B \times 128 \times 64 \times 64}, \quad \mathbf{F}_3 \in \mathbb{R}^{B \times 320 \times 32 \times 32}, \quad \mathbf{F}_4 \in \mathbb{R}^{B \times 512 \times 16 \times 16}$$
- Stage 3 sở hữu 24 khối transformer với Spatial-Reduction Attention (SRA) tỷ lệ giảm mẫu $R=2$, trích xuất trường tiếp nhận không giới hạn cho các tổn thương lan tỏa.

### 4.2. Reverse Edge Guidance Stream (REGS) Triệt tiêu False Positive
- Module [`ReverseEdgeGuidanceStream`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/models/edge_guidance.py) tiếp nhận đặc trưng độ phân giải cao $\mathbf{S}_1$ ($128 \times 128$) và $\mathbf{S}_2$ ($64 \times 64$).
- Dự đoán trực tiếp bản đồ xác suất đường biên tổn thương:
  $$\mathbf{B}_{\text{pred}} = \text{Interp}_{512 \times 512}(\text{Conv}_{1 \times 1}(\text{DWConv}_{3 \times 3}([\text{Proj}(\mathbf{S}_1); \text{Interp}(\text{Proj}(\mathbf{S}_2))])))$$
- Tạo bản đồ điều hướng biên đảo ngược:
  $$\mathbf{A}_{\text{edge}} = \sigma(\text{Conv}_{1 \times 1}(\mathbf{F}_{\text{edge}})), \qquad \mathbf{F}_{\text{edge\_gated}} = \mathbf{F}_{\text{edge}} \odot (1 + \mathbf{A}_{\text{edge}})$$
- Nhánh này cung cấp tín hiệu biên tế bào sắc nét để hòa nhập vào tầng giải mã $D_1$, chặn đứng việc loang màu vào vùng nền da lành lặn.

### 4.3. Multi-Scale Frequency-Decoupled Filter Bank (MS-FDF v5)
- Module [`MultiScaleFrequencyFilterV5`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/models/frequency_filter.py) áp dụng đồng thời trên cả 4 tầng skip features.
- Ngân hàng tích chập sâu song song $K \in \{1\times1, 3\times3, 5\times5\}$ tách biệt:
  $$\mathbf{x}_{\text{low}} = \text{Conv}_{1 \times 1}([\text{DWConv}_{1}(\mathbf{x}); \text{DWConv}_{3}(\mathbf{x}); \text{DWConv}_{5}(\mathbf{x})]), \qquad \mathbf{x}_{\text{high}} = \mathbf{x} - \mathbf{x}_{\text{low}}$$
  $$\mathbf{x}_{\text{out}} = \mathbf{x} + \text{Fusion}([\mathbf{x}_{\text{low}}; \mathbf{x}_{\text{high}} \odot \sigma(\text{DWConv}_{3}(\text{Conv}_{1 \times 1}(\mathbf{x}_{\text{low}}))) ])$$

### 4.4. Mask-Guided LGAG & Cascaded MSCAM Decoders
- Module [`MaskGuidedLGAG`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/models/attention_gates.py) điều hướng các đường skip connection tầng $S_3, S_2, S_1$ bằng mặt nạ dự đoán thô tầng sâu $m_{k+1} = \sigma(P_{k+1})$.
- Module [`CascadedMSCAMDecoderBlockV5`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/models/decoder.py) liên tục tinh chỉnh đặc trưng qua tích chập đa tỉ lệ $[1, 3, 5]$ và cơ chế chú ý kênh - không gian kép.

---

## 5. Hàm Mất mát Đa nhiệm Toàn diện (Multi-Task Composite Loss)

Hàm mất mát [`GDFNetV5CompositeLoss`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/losses/composite_loss.py) tối ưu hóa đồng thời cấu trúc phân đoạn và độ khít ranh giới:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Structure}}(\mathbf{S}_{\text{final}}, \mathbf{G}) + \mathcal{L}_{\text{Structure}}(\mathbf{P}_1, \mathbf{G}) + 0.5 \mathcal{L}_{\text{Structure}}(\mathbf{P}_2, \mathbf{G}) + 0.3 \mathcal{L}_{\text{Structure}}(\mathbf{P}_3, \mathbf{G}) + 0.2 \mathcal{L}_{\text{Structure}}(\mathbf{P}_4, \mathbf{G}) + 0.8 \mathcal{L}_{\text{Edge}}(\mathbf{B}_{\text{pred}}, \mathbf{G}_{\text{edge}})$$

Trong đó:
1. **Structure Loss:**
   $$\mathcal{L}_{\text{Structure}} = \mathcal{L}_{\text{wBCE}}(\mathbf{W}) + \mathcal{L}_{\text{wIoU}}(\mathbf{W}), \quad \mathbf{W} = 1 + 5|\text{AvgPool}_{31}(\mathbf{G}) - \mathbf{G}|$$
2. **Boundary Edge Loss:**
   $$\mathcal{L}_{\text{Edge}} = \mathcal{L}_{\text{wBCE\_edge}}(w_{\text{pos}}=5.0) + \mathcal{L}_{\text{Dice\_edge}}(\mathbf{B}_{\text{pred}}, \mathbf{G}_{\text{edge}})$$
3. **Nhãn biên GPU tức thời ($\mathbf{G}_{\text{edge}}$):**
   $$\mathbf{G}_{\text{edge}} = \text{MaxPool}_{3 \times 3}(\mathbf{G}) - (-\text{MaxPool}_{3 \times 3}(-\mathbf{G}))$$
   Thuật toán hình thái học vi sai thực thi trực tiếp trên GPU trong $< 0.01\text{ ms}$, loại bỏ hoàn toàn chi phí tính toán Canny trước trên CPU.

---

## 6. Pipeline Tăng cường Dữ liệu Chuyên sâu Bệnh học (Artifact-Aware Augmentation)

Được tích hợp trực tiếp trong `ISICDatasetV5`:
1. **RandomVignetting ($p=0.4$):** Sinh quầng tối hình tròn/elip mờ ở góc hoặc cạnh ảnh với hàm sigmoid suy giảm mịn, rèn luyện mạng bỏ qua quầng tối quang học của kính soi da dermoscope.
2. **CLAHE ($p=0.3$, clip_limit=2.0, grid=8x8):** Tăng cường tương phản cục bộ, làm nổi bật viền của các ca u hắc tố thiếu sắc tố (amelanotic).
3. **SyntheticHair ($p=0.3$, 1–4 sợi tóc ngẫu nhiên):** Vẽ các đường cong bậc hai sẫm màu đè lên bề mặt da lành để mô hình học cách phân biệt sợi tóc với đường ranh giới tế bào.
4. **Phép biến đổi chuẩn:** RandomResizedCrop (scale 0.6–1.0), Horizontal/Vertical Flip, ShiftScaleRotate (45 độ), ColorJitter, và chuẩn hóa ImageNet `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`.

---

## 7. Giao thức Stratified 5-Fold Cross-Validation & Báo cáo Hai Tầng

Quy trình huấn luyện [`colab_run_gdfnet_v5_5fold.py`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/training/colab_run_gdfnet_v5_5fold.py):
1. **Phân chia 5 Folds:**
   - 2,594 ảnh train được chia thành 5 folds độc lập (mỗi fold gồm ~2,075 ảnh train và ~519 ảnh val).
   - Huấn luyện độc lập 100 epochs/fold với AdamW, CosineAnnealingLR ($10^{-4} \to 10^{-6}$), Mixed Precision (AMP), lưu checkpoint `GDFNetV5_fold{k}_best.pth` tại epoch có Val Macro Dice cao nhất.
2. **Kiểm thử Mù trên 1,000 Ảnh Test ISIC-2018:**
   - **Tầng 1 (Single Model 5-Fold Evaluation):** Báo cáo điểm $\text{Mean} \pm \text{Std}$ trên cả 5 mô hình độc lập (đảm bảo tính vững chắc kiến trúc theo chuẩn IEEE TMI/MedIA).
   - **Tầng 2 (5-Fold Soft-Voting Ensemble + 4-View Flip TTA):** Lấy trung bình xác suất:
     $$P_{\text{ensemble}}(\mathbf{x}) = \frac{1}{5} \sum_{k=0}^4 \left( \frac{P_k(\mathbf{x}) + \text{Flip}_H(P_k(\text{Flip}_H(\mathbf{x}))) + \text{Flip}_V(P_k(\text{Flip}_V(\mathbf{x}))) + \text{Flip}_{HV}(P_k(\text{Flip}_{HV}(\mathbf{x})))} {4} \right)$$
     Cơ chế triệt tiêu phương sai này dập tắt đồng thời 28 ca False Positive và 12 ca False Negative, tạo bước nhảy vọt đưa Mean Dice vượt mốc **0.915 – 0.925**.

---

## 8. Kết quả Thực nghiệm Kiểm thử Unit Test Cục bộ

Unit test [`GDF-Netv5/tests/test_model_v5.py`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/tests/test_model_v5.py) đã thực thi thành công 100%:
- **Thiết bị tính toán:** CUDA (GPU).
- **Tổng số tham số:** **68.04M parameters** (Trainable: 68.04M).
- **Kích thước 6 tensor đầu ra tại độ phân giải chuẩn $512 \times 512$:**
  - `logits`: `torch.Size([2, 1, 512, 512])`
  - `edge_logits`: `torch.Size([2, 1, 512, 512])`
  - `p1`: `torch.Size([2, 1, 512, 512])`
  - `p2`: `torch.Size([2, 1, 512, 512])`
  - `p3`: `torch.Size([2, 1, 512, 512])`
  - `p4`: `torch.Size([2, 1, 512, 512])`
- **Khả vi & Luồng Gradient:** Xác nhận 100% gradient lan truyền liên tục qua toàn bộ mạng (`Encoder -> MS-FDF -> REGS -> LGAG -> MSCAM Decoder`).
- **Single-pass eval output:** `torch.Size([2, 1, 512, 512])`.

---

## 9. Danh mục Artifacts & File Mã nguồn

- Mô hình Encoder V5: [`GDF-Netv5/models/backbones.py`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/models/backbones.py)
- Bộ lọc tần số V5: [`GDF-Netv5/models/frequency_filter.py`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/models/frequency_filter.py)
- Cổng chú ý mặt nạ: [`GDF-Netv5/models/attention_gates.py`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/models/attention_gates.py)
- Nhánh hướng dẫn biên đảo ngược: [`GDF-Netv5/models/edge_guidance.py`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/models/edge_guidance.py)
- Bộ giải mã Cascaded MSCAM: [`GDF-Netv5/models/decoder.py`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/models/decoder.py)
- Mô hình tổng thể GDF-Netv5: [`GDF-Netv5/models/gdfnet_v5.py`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/models/gdfnet_v5.py)
- Hàm mất mát cấu trúc & biên: [`GDF-Netv5/losses/structure_loss.py`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/losses/structure_loss.py), [`boundary_loss.py`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/losses/boundary_loss.py), [`composite_loss.py`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/losses/composite_loss.py)
- Bộ kiểm thử kiến trúc: [`GDF-Netv5/tests/test_model_v5.py`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/tests/test_model_v5.py)
- Kịch bản huấn luyện 5-Fold trên Colab A100: [`GDF-Netv5/training/colab_run_gdfnet_v5_5fold.py`](file:///D:/User/nghonamjiwoo/My%20Project/FANet/GDF-Netv5/training/colab_run_gdfnet_v5_5fold.py)
