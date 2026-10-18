# Literature Grounding — Phase 6 (advisor feedback: no-FB baseline + asymmetric / class-frequency loss)

**Ngày:** 09/09/2026 · **Mục đích:** ground 2 điểm feedback của advisor dựa trên
literature đa nguồn, đối chiếu với bằng chứng thực nghiệm Phase 5 của repo, và
chốt công thức loss khuyến nghị cho Phase 6. Mỗi citation kèm title/venue/năm/DOI
đã verify.

> Cách đọc bảng "Bằng chứng dự án": `TRAIN` = kết quả train đầy đủ (đáng tin),
> `FROZEN EVAL` = đo khả năng khai thác feedback của model cũ, `INVALID` = bug.

---

## 1. Asymmetric loss cho FP suppression trong segmentation nhị phân

### 1.1. Nguồn gốc multi-label: Asymmetric Loss (ASL)

| Field | Value |
|---|---|
| **Citation** | Ben-Baruch E, Ridnik T, Zamir N, Noy A, Friedman I, Protter M, Zelnik-Manor L. *Asymmetric Loss For Multi-Label Classification*. ICCV 2021, pp. 82–91. |
| **DOI** | `10.1109/ICCV48922.2021.00015` |
| **arXiv** | `2009.14119` |
| **Vấn đề** | Multi-label có nhiều negative hơn positive → gradient bị negative chi phối, positive under-weighted. |
| **Cơ chế** | Tách modulation của positive/negative: `γ⁻ > γ⁺` (asymmetric focusing, soft thresholding easy-negative) + `m ≥ 0` (probability margin — hard-thresholding, bỏ hẳn easy-negative & nghi ngờ mislabeled-negative). Công thức: `L₊ = −(1−p₊)^γ₊ log(p₊)`, `L₋ = (pₘ)^γ₋ log(1−pₘ)` với `pₘ = max(p − m, 0)`. |
| **Lưu ý quan trọng cho project** | ASL gốc là **classification per-label**, KHÔNG phải pixel-weighting trong segmentation. Khi áp cho segmentation phải map lại trên **pixel foreground/background** và theo dõi collapse. |

### 1.2. Asymmetric dice: Tversky loss

| Field | Value |
|---|---|
| **Citation** | Salehi SSM, Erdogmus D, Gholipour A. *Tversky loss function for image segmentation using 3D fully convolutional deep networks*. MICCAI 2017 workshop (MLMI), LNCS 10541, pp. 379–387. |
| **DOI** | `10.1007/978-3-319-67389-9_44` (ver. workshop) · arXiv `1706.05721` |
| **Công thức** | `S = |PG| / (|PG| + α|P∖G| + β|G∖P|)`. `α` phạt FP, `β` phạt FN. `α=β=0.5` → Dice; `α+β=1` → Fβ. |
| **Bằng chứng thực nghiệm** | β=0.7 (nặng FN/recall) tốt nhất cho MS lesion (DSC 56.4 vs 53.4 baseline). Chứng minh kiểm soát precision/recall trade-off bằng hyper-parameter. |

### 1.3. Focal Tversky (nhúng asymmetric vào focal)

| Field | Value |
|---|---|
| **Citation** | Abraham N, Khan NM. *A Novel Focal Tversky Loss Function With Improved Attention U-Net for Lesion Segmentation*. ISBI 2019. |
| **DOI** | `10.1109/ISBI.2019.8759329` · arXiv `1810.07842` |
| **Công thức** | `FTL = Σ (1 − TI_c)^(1/γ)`; khuyến nghị `γ=4/3`, `α=0.7, β=0.3` (nặng FN). |
| **Áp dụng** | BUS 2017 + ISIC 2018 lesion (class imbalance nặng). |

### 1.4. Unified Focal loss (khung asymmetric nhất quán)

| Field | Value |
|---|---|
| **Citation** | Yeung M, Sala E, Schönlieb CB, Rundo L. *Unified Focal loss: Generalising Dice and cross entropy-based losses to handle class imbalanced medical image segmentation*. Computerized Medical Imaging and Graphics 95 (2022) 102026. |
| **DOI** | `10.1016/j.compmedimag.2021.102026` · arXiv `2102.04525` |
| **Công thức (asymmetric variant)** | `LaUF = λ·LmaF + (1−λ)·LmaFT`. `LmaF`: modified asymmetric Focal loss — bỏ focal ở class hiếm (foreground), giữ suppression ở background. `LmaFT`: modified asymmetric Focal Tversky — bỏ focal ở background, giữ enhancement ở class hiếm. |
| **Hyper-parameter khuyến nghị** | `λ=0.5`, `δ=0.6` (nặng foreground hơn để chống high-precision/low-recall của Dice), `γ` duy nhất tune (2D: γ∈[0.1,0.9]). |
| **Bằng chứng** | Đạt DSC/IoU cao nhất trên 5 dataset class-imbalanced (gồm **CVC-ClinicDB polyp**). Asymmetric variant hơi tốt hơn symmetric. |
| **Relevance** | **CVC-ClinicDB** là dataset polyp ⇒ đây là bằng chứng asymmetric loss hiệu quả trong **polyp segmentation class-imbalance** trực tiếp. |

### 1.5. Tổng hợp khuyến nghị asymmetric loss

Có 2 con đường asymmetric để nhắm FP (background = negative):

1. **Tversky α>β** (nặng FP): trực tiếp phạt FP. `α+β=1` với `α>0.5`. Nguồn: Salehi 2017.
2. **ASL asymmetric focal / Unified Focal** (nặng negative suppression): ASL `γ⁻>γ⁺` down-weight easy-negative; Unified Focal asymmetric dùng δ để kiểm soát class-imbalance.

> **Chốt cho Phase 6 (chờ user duyệt):** asymmetric loss đơn giản nhất, verify được,
> trùng đúng gốc rễ FP của FANet là **Tversky `α>β`** (nặng FP) — có thể kết hợp
> `λ` với DiceBCE giống cell A. Unified Focal là phương án phức tạp hơn (2 loss
> component) — dành nếu Tversky chưa đủ. **Cảnh báo collapse kiểu TB:** mọi biến thể
> phải theo dõi binary val-Dice mỗi epoch, không tin soft-dice.

---

## 2. Class-frequency pixel weighting tính từ training set

### 2.1. ENet — inverse log frequency (bounded)

| Field | Value |
|---|---|
| **Citation** | Paszke A, Chaurasia A, Kim S, Culurciello E. *ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation*. arXiv 2016. |
| **DOI** | `10.48550/arXiv.1606.02147` |
| **Công thức** | `w_class = 1 / ln(c + p_class)`, `p_class` = tỉ lệ pixel class trong **train set**, `c=1.02` → weight bị chặn trong `[1,50]`. |
| **Điểm mạnh** | Bounded: không nổ về vô cực khi class rất hiếm. Đây là "inverse frequency" đã làm mượt, **khác hẳn** inverse thô (rủi ro collapse). |

### 2.2. Median frequency balancing (MFB)

| Field | Value |
|---|---|
| **Citation** | Cùng paper ENet (Paszke 2016) — "median frequency balancing" (nguồn gốc từ Eigen & Fergus 2015, *Predicting Depth, Surface Normals and Semantic Labels with a Common Multi-Scale Convolutional Architecture*, ICCV 2015). |
| **DOI (Eigen)** | `10.1109/ICCV.2015.304` |
| **Công thức** | `w_class = median_class_freq / freq_class` — class hiếm hơn median được nâng, class phổ biến hơn median bị hạ. Chuẩn hoá quanh median, giảm collapse. |

### 2.3. Class-balanced loss (Cui 2019) — inverse effective number

| Field | Value |
|---|---|
| **Citation** | Cui Y, Jia M, Lin TY, Song Y, Belongie S. *Class-Balanced Loss Based on Effective Number of Samples*. CVPR 2019, pp. 9268–9277. |
| **DOI** | `10.1109/CVPR.2019.00949` · arXiv `1901.05555` |
| **Công thức** | Weight `∝ (1−β)/(1−β^(n_y))`, `n_y` = số sample của class, `β∈[0,1)` (khuyến nghị `β=0.999`). `β=0` → no reweight; `β→1` → inverse frequency. |
| **Liên hệ với advisor** | Advisor đề xuất "pixel weight dựa trên tần suất class trong training set". Đây chính là họ inverse-frequency — nhưng **Cui 2019 chỉ ra inverse-frequency thô thường hại** trên data imbalance thật (line "re-weighting by inverse class frequency usually yields poor performance"). ⇒ **nên dùng ENet bounded (1/ln) hoặc CB với β<1** thay vì inverse thô. |

### 2.4. Chuẩn "tính weight trên TRAIN split" + normalization

- **Nguyên tắc bắt buộc (tránh leakage):** weight class phải tính từ **training split
  chỉ một lần**, KHÔNG bao giờ từ validation/test. Vì weight là function của phân phối
  nhãn, dùng val/test để chốt weight = leakage (val set thấm vào quyết định loss).
- **Normalization:** sau khi tính weight thô, chuẩn hoá để giữ scale loss hợp lý.
  Hai cách phổ biến:
  1. Chuẩn hoá **tổng = 1** hoặc **tổng = số class** (Cui 2019: `Σ αᵢ = C`).
  2. Chuẩn hoá **theo N** trong BCE (chia cho số pixel, như cell B Phase 5 đã làm)
     để trọng số không làm nổ loss-scale.
- **Cảnh báo (bài học TB Phase 5):** nếu weight background quá mạnh (background chiếm
  đa số pixel trong polyp ảnh → inverse-frequency background rất lớn), model học "đoán
  toàn background" để né phạt → **collapse Dice**. Vì vậy dùng **bounded** (ENet)
  hoặc **CB β<1** (không phải inverse thô), và **bắt buộc theo dõi binary val-Dice**.
- **Lưu ý hướng cho FANet:** root cause là **FP (predict foreground trên background)**.
  Vậy class-frequency weight phải **nặng background** (để phạt FP) — nhưng chính nó là
  hướng dễ collapse (giống TB). ⇒ cần cẩn trọng, cân đối, và dùng binary val-Dice.

---

## 3. Công thức pixel-weight advisor gọi "authors adapt" — đối chiếu F³Net / CFA-Net

### 3.1. Advisor đang trỏ vào công thức nào

Feedback advisor: *"công thức pixel-weight này có khi chỉ authors đưa ra adapt với
bài toán của họ thôi"* — trỏ tới **công thức (1+5μ)** mà Phase 5 cell B đã verify là
từ **CFA-Net (Pattern Recognition 2023)**, trong đó CFA-Net cite [46] = **F³Net
(AAAI 2020)**, và `μ` theo **F³Net Eq.4**:

```
μ_ij = |mean_{3×3 window}(GT)_ij − GT_ij|   ∈ [0, 1]
w_ij = 1 + 5·μ_ij     (CFA-Net dùng hệ số 5)
```

- `μ = 1` **trên contour** (nơi cửa sổ 3×3 có cả 0 lẫn 1) → weight ×6 **ở BIÊN**.
- `μ = 0` **vùng phẳng** (interior polyp + nền sâu) → weight ×1.
- Đây là **pixel-importance theo hình học biên** (boundary-importance), KHÔNG phải
  class-frequency weight.

### 3.2. Verify lại (đối chiếu Phase 5)

Đã verify full-text trong `docs/reports/2026-09-07_phase5-x2-gate01.md` (mục F1):
μ nặng BIÊN, trái root cause FP xa biên 40.7px → đã pivot thành far-weighted
`w=1+5(1−μ)` (nặng vùng phẳng/nền sâu). Cell B far-weighted (γ=5) **COLLAPSE**
(Dice 0.0086) ở Phase 5.

| | Boundary importance (1+5μ) | Class-frequency (advisor đề xuất) |
|---|---|---|
| **Bản chất** | Hình học biên GT | Phân phối pixel class trong train |
| **Nặng cho ai** | Biên polyp (contour) | Class hiếm / background (tùy normalize) |
| **Nguồn** | F³Net Eq.4 / CFA-Net | ENet / MFB / Cui CB |
| **Liên hệ root cause FANet** | SAI hướng (FP xa biên) | Phù hợp hơn (nếu nặng background để phạt FP) |

### 3.3. Kết luận chẩn đoán feedback điểm 2

- Advisor đúng ở chỗ: **pixel-weight (1+5μ) là do tác giả CFA-Net/F³Net adapt riêng
  cho bài toán của họ** (boundary focus), không phải chuẩn chung.
- Advisor đề xuất tính weight **dựa trên class frequency từ training set** — đây là
  hướng **KHÁC VỀ BẢN CHẤT** với (1+5μ): không phải "geometric boundary importance"
  mà là "class distribution weight".
- **Quan trọng cho thiết kế:** class-frequency nặng background sẽ hướng tới FP
  suppression đúng root cause FANet, nhưng cũng là hướng đã collapse ở TB (γ=5 far).
  ⇒ phải dùng bounded/CB và theo dõi binary val-Dice.

---

## 4. Báo cáo factorial/ablation tách hiệu ứng mechanism vs loss + chuẩn Precision/Recall/FPR

### 4.1. Lưới 2×2 (factorial) tách hiệu ứng

- Lưới 2×2 {no-FB, FB} × {orig loss, new loss} là **factorial design 2×2**. Trả lời
  2 câu hỏi chính (main effects: FB, loss) + 1 câu interaction (FB×loss có synergy?).
- Chuẩn báo cáo: cần 4 cell **cùng protocol + cùng seed** để ước lượng main effect và
  interaction không bị confound (seed 42 vs 43 sẽ làm méo interaction).
- **Ghi nhận feedback đúng:** nếu muốn claim "loss mới giúp FP", phải có nhánh
  **no-FB × new-loss** để tách hiệu ứng của loss khỏi hiệu ứng prune của feedback
  (vì feedback bản thân đã có tác dụng prune FP nhẹ — Gate 0 Phase 5).

### 4.2. Chuẩn báo cáo metrics cho polyp

| Field | Value |
|---|---|
| **Citation** | Jha D, Smedsrud PH, Riegler MA, Halvorsen P, de Lange T, Johansen D, Johansen HD. *Kvasir-SEG: A Segmented Polyp Dataset*. MMM 2020, pp. 451–462. |
| **DOI** | `10.1007/978-3-030-37734-2_37` · arXiv `1911.07069` |
| **Nội dung** | Dataset polyp chuẩn (1000 ảnh); khuyến nghị báo cáo DSC/IoU + precision/recall per-class. |

| Field | Value |
|---|---|
| **Citation** | Maier-Hein L, Reinke A, Godau P, et al. *Metrics reloaded: recommendations for image analysis validation*. Nature Methods 21, 195–212 (2024). |
| **DOI** | `10.1038/s41592-023-02151-z` |
| **Nội dung** | Khung "problem fingerprint" chọn metric. Cho semantic segmentation (SemS): khuyến nghị **DSC/IoU mặc định**, và **Fβ score khi có preference FP vs FN**; bổ sung boundary metric (NSD). **Per-image distribution** (không chỉ mean) vì phân phối lệch. Cảnh báo pitfall class imbalance + small structures. |

**Áp dụng cho Phase 6:**
- PRIMARY = **FP rate/FPR per-image** (FP là metric có variance nhỏ hơn Dice — đã đo
  Phase 5: FP SD ≈ 0.057 vs Dice SD ≈ 0.27 ⇒ FP là metric chính để phát hiện hiệu ứng).
- SECONDARY = Dice, IoU, Precision, Recall, FN% — báo **per-image distribution**.
- Đúng chuẩn Metrics Reloaded: Fβ với β điều chỉnh theo preference FP vs FN; vì mục
  tiêu giảm FP → theo dõi **Precision** (FPR) riêng.

---

## 5. Kết luận literature → chốt cho Phase 6

### (a) Công thức asymmetric loss khuyến nghị + hyper-parameter + DOI

**Khuyến nghị chính: Tversky loss `α>β` (nặng FP).**
```
TL = |PG| / (|PG| + α|P∖G| + β|G∖P|)      # loss = 1 − TL
α + β = 1,  α > 0.5 (nặng FP)
```
- Đề xuất thử **`α=0.7, β=0.3`** (mức asymmetric trung bình, đủ mạnh để thấy hiệu ứng
  nhưng không quá mạnh như TB γ=5 gây collapse). Có thể combine với DiceBCE (`λ·DiceBCE +
  (1−λ)·TL`) như pattern cell A.
- DOI: Salehi 2017 `10.1007/978-3-319-67389-9_44`; Abraham 2019 `10.1109/ISBI.2019.8759329`.
- Phương án phức tạp hơn (nếu cần): **Unified Focal asymmetric** (Yeung 2022,
  `10.1016/j.compmedimag.2021.102026`) với `λ=0.5, δ=0.6`.

### (b) Công thức class-frequency weight cụ thể + normalization

**Khuyến nghị: ENet bounded inverse-log hoặc CB β<1, tính trên TRAIN split 1 lần.**
```
ENet:  w_class = 1 / ln(1.02 + p_class)        # bounded [1,50]
CB:    w_class ∝ (1 − β) / (1 − β^(n_class))   # β=0.999; normalize Σ=#class
```
- **Normalization:** normalize để `mean(w) = 1` (giữ loss-scale gần DiceBCE) HOẶC
  `Σ w = #class`. Trong BCE dùng reduction='mean' (chia N) như cell B để trọng số
  không nổ scale.
- **KHÔNG dùng inverse-frequency thô** (Cui 2019: thường hại trên imbalance thật).
- DOI: Paszke 2016 `10.48550/arXiv.1606.02147`; Cui 2019 `10.1109/CVPR.2019.00949`.

### (c) Cảnh báo rủi ro collapse kiểu TB + biện pháp bắt buộc

- **Rủi ro:** nặng background quá mạnh (inverse-frequency background rất lớn vì polyp
  chiếm thiểu số) → model đoán toàn background → Dice sụp (giống TB γ=5: recall 0.0047).
- **Biện pháp bắt buộc:**
  1. Dùng weight **bounded** (ENet) hoặc **CB β<1** — không inverse thô.
  2. **Theo dõi binary val-Dice/Precision/Recall/FPR mỗi epoch** trong notebook
     (diverge rule ở ep40 áp trên binary val-Dice, KHÔNG trên loss-scale — bài học
     TB: soft-dice 0.19 che giấu collapse 0.0086).
  3. Cross-check sanity: nếu val-Dice bắt đầu sụt sâu đồng thời FPR ~0 và Recall ~0
     → collapse, dừng cell.

---

## Nguồn / provenance

- Tra cứu đa nguồn: OpenAlex, arXiv, Europe PMC, Semantic Scholar (websearch
  + paper-lookup skill), verify abstract/full-text qua nguồn mở.
- Access date: 09/09/2026.
- Đối chiếu bằng chứng project: `docs/reports/2026-09-07_phase5-x2-gate01.md`
  (verify F³Net/CFA-Net), `docs/reports/2026-09-08_phase5-results.md` (kết quả
  Gate 0/1), `results/phase5_eval.json`, `results/phase5_stats_full.json`.
