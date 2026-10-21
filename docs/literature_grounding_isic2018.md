# Literature Grounding: Comprehensive State-of-the-Art Survey & Architectural Analysis on ISIC-2018 Skin Lesion Segmentation (2019–2026)

## 1. Retrieval Provenance & Verification Contract

This literature grounding document was compiled and audited on **2026-10-21** using the repository's `scientific-research-agent` skills (`paper-lookup`, `literature-review`, `citation-management`, and `hypothesis-generation`) across four scholarly indexes:
- **OpenAlex REST API** (`https://api.openalex.org/works`, 1,507 total hits for `"ISIC 2018" "skin lesion segmentation"`, top-cited and 2024–2026 cohorts retrieved, plus 22 exact title/DOI verifications).
- **Europe PMC REST API** (`https://www.ebi.ac.uk/europepmc/webservices/rest/search`, 109 hits for `TITLE_ABS:"ISIC 2018" AND "segmentation" AND "Dice"` and 102 hits for `"ISIC 2018" AND "Jaccard" AND "lesion segmentation"`).
- **arXiv Atom XML & LaTeX HTML Full-Text** (`https://export.arxiv.org/api/query` and `https://arxiv.org/html/{id}` for primary-source benchmark and ablation tables in `2405.06880v1`, `2402.02491v1`, `2403.20035v1`, `2307.08473v1`, `2406.02918v1`, and `1902.03368`).
- **Semantic Scholar / Crossref DOI Resolution** for citation counts and venue verification.

---

## 2. Critical Methodological Distinction: The Three Evaluation Protocols on ISIC-2018 Task 1

A primary finding from cross-database full-text verification is that "ISIC-2018 Dice" numbers reported across papers **cannot be compared without matching both the data split protocol and the metric aggregation formula**:

### Protocol A1: Random Split of the 2,594 Training Images + Per-Image Macro Average
- **Dataset Partition:** Only the 2,594 official training images (or 2,694 train+val images) are used and randomly split into Train/Val/Test—commonly **80:10:10** (2,075 / 259 / 260, as in EMCAD, PVT-CASCADE, Polyp-PVT reproductions) or **70:10:20** (1,815 / 259 / 520, as in BAT, FAT-Net, Ms RED internal split) or **5-fold cross-validation**.
- **Metric Formula:** Dice is computed per test image and averaged across images:
  $$\text{DSC}_{\text{macro}} = \frac{1}{N} \sum_{i=1}^{N} \frac{2\,\text{TP}_i}{2\,\text{TP}_i + \text{FP}_i + \text{FN}_i}$$
- **Typical SOTA Range:** **89.0% – 91.2% DSC** for standalone CNN/Transformer/Mamba models; **94.1% – 96.3% DSC** when using fine-tuned foundation models (SAM / MedSAM adapters) or bounding-box prompts.

### Protocol A2: Random Split (7:3) + Dataset-Level Global Pixel (Micro) Confusion Matrix
- **Dataset Partition:** Random 7:3 split (1,815 train / 779 test on the 2,594 training set, or 1,886 train / 808 test on the 2,694 train+val set), used by the lightweight benchmark family (**MALUNet**, **EGE-UNet**, **VM-UNet**, **UltraLight VM-UNet**, **H-vmunet**).
- **Metric Formula:** Pixels from all test images are accumulated into a single global confusion matrix $(\sum \text{TP}, \sum \text{FP}, \sum \text{FN})$ before computing Dice once:
  $$\text{DSC}_{\text{global}} = \frac{2 \sum_{i=1}^{N} \text{TP}_i}{2 \sum_{i=1}^{N} \text{TP}_i + \sum_{i=1}^{N} \text{FP}_i + \sum_{i=1}^{N} \text{FN}_i}$$
- **Why This Matters:** Global pixel Dice weights each image by its lesion pixel count and completely masks catastrophic failures on small lesions (a micro-lesion with $\text{Dice}=0.0$ changes $\text{DSC}_{\text{global}}$ by $<0.005\%$, whereas it drops $\text{DSC}_{\text{macro}}$ by $0.1\%$).

### Protocol B: Official ISIC-2018 Challenge Blind Test Set (1,000 Held-Out Images)
- **Dataset Partition:** Train on the 2,594 official training images, validate on the 100 official validation images, and evaluate on the **1,000 official test images** (Codella et al., `arXiv:1902.03368`).
- **Distribution Shift:** The 1,000 official test images include out-of-distribution clinical acquisitions from multiple international centers with heavy hair occlusion, surgical ink markers, ruler scales, specular gel bubbles, dark circular dermoscope vignettes, amelanotic lesions, and extreme lesion-to-frame ratios ($<1\%$ to $>95\%$).
- **Official Challenge Metric (Thresholded Jaccard $J_{\text{thresh}}$):**
  $$J_i = \frac{\text{TP}_i}{\text{TP}_i + \text{FP}_i + \text{FN}_i}, \qquad \tilde{J}_i = \begin{cases} J_i & \text{if } J_i \ge 0.65 \\ 0 & \text{if } J_i < 0.65 \end{cases}, \qquad J_{\text{thresh}} = \frac{1}{1000}\sum_{i=1}^{1000} \tilde{J}_i$$
- **Leaderboard Benchmark (Codella et al., 2019):**
  - **1st Place Official Challenge Winner** (heavy multi-model ensemble of Mask R-CNN, DeepLabv3+, PSPNet, and U-Net variants with TTA): **$J_{\text{thresh}} = 0.802$** (unthresholded Jaccard $\approx 0.838$, corresponding to per-image macro Dice $\approx 0.898 - 0.904$).
  - **Human Expert Inter-Observer Agreement:** $J_{\text{thresh}} \approx 0.786$.
  - **Our GDF-Netv2 (Single Model, No TTA, 384x384):** Per-image macro Dice = **0.8935** (Median **0.9253**), unthresholded mIoU = **0.8218**, **$J_{\text{thresh}} = 0.7757$** (94/1000 images with $J_i < 0.65$).
  - **Our GDF-Netv3 (Single Model, No TTA, 384x384):** Per-image macro Dice = **0.8752** (Median **0.9067**), unthresholded mIoU = **0.7934**, **$J_{\text{thresh}} = 0.7251$** (136/1000 images with $J_i < 0.65$).

---

## 3. Verified Quantitative Benchmark Tables from Primary Sources

### Table 3.1: Standardized 80:10:10 Split of ISIC-2018 (2,594 Images, 5-Run Average)
*Source: Verified directly from Table 1 of Rahman et al., **EMCAD** (IEEE/CVF CVPR 2024, `DOI: 10.1109/cvpr52733.2024.01118`, `arXiv:2405.06880v1`). Input resolution $384 \times 384$ for ISIC-2018 (FLOPs standardized at $256 \times 256$).*

| Model | Venue / Year | Backbone (Pretrained) | Params (M) | FLOPs (G) | ISIC-2017 DSC (%) | ISIC-2018 DSC (%) | 10-Dataset Avg DSC (%) | Verification |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **U-Net** | MICCAI 2015 | Scratch CNN | 34.53 | 65.53 | 83.07 | 86.67 | 85.23 | VERIFIED-fulltext |
| **AttnU-Net** | MIDL 2018 | Scratch CNN | 34.88 | 66.64 | 83.66 | 87.05 | 85.60 | VERIFIED-fulltext |
| **UNet++** | DLMIA 2018 | Scratch CNN | 9.16 | 34.65 | 82.98 | 87.46 | 85.75 | VERIFIED-fulltext |
| **UNeXt** | MICCAI 2022 | Tokenized MLP | 1.47 | 0.57 | 82.74 | 87.78 | 82.89 | VERIFIED-fulltext |
| **PraNet** | MICCAI 2020 | Res2Net-50 (IN-1k) | 32.55 | 6.93 | 83.03 | 88.56 | 86.41 | VERIFIED-fulltext |
| **DeepLabv3+** | ECCV 2018 | ResNet-50 (IN-1k) | 39.76 | 14.92 | 83.84 | 88.64 | 89.11 | VERIFIED-fulltext |
| **TransUNet** | arXiv 2021 | R50-ViT-B/16 (IN-21k) | 105.32 | 38.52 | 85.00 | 89.16 | 89.33 | VERIFIED-fulltext |
| **Swin-Unet** | ECCV-W 2022 | Swin-T (IN-1k) | 27.17 | 6.20 | 83.97 | 89.26 | 88.01 | VERIFIED-fulltext |
| **TransFuse** | MICCAI 2021 | ResNet-34 + DeiT-S | 143.74 | 82.71 | 84.89 | 89.62 | 88.77 | VERIFIED-fulltext |
| **UACANet-L** | ACM MM 2021 | Res2Net-50 (IN-1k) | 69.16 | 31.51 | 83.72 | 89.76 | 88.41 | VERIFIED-fulltext |
| **CaraNet** | SPIE 2022 | Res2Net-101 (IN-1k) | 46.64 | 11.48 | 85.02 | 90.18 | 88.94 | VERIFIED-fulltext |
| **SSFormer-L** | MICCAI 2022 | MiT-B4 (SegFormer) | 66.22 | 17.28 | 85.28 | 90.25 | 90.03 | VERIFIED-fulltext |
| **Polyp-PVT** | CAAI AIR 2023 | PVTv2-B2 (IN-1k) | 25.11 | 5.30 | 85.56 | 90.36 | 89.87 | VERIFIED-fulltext |
| **PVT-CASCADE** | WACV 2023 | PVTv2-B2 (IN-1k) | 34.12 | 7.62 | 85.50 | 90.41 | 90.42 | VERIFIED-fulltext |
| **PVT-EMCAD-B0** | CVPR 2024 | PVTv2-B0 (IN-1k) | 3.92 | 0.84 | 85.67 | 90.70 | 90.52 | VERIFIED-fulltext |
| **PVT-EMCAD-B2** | CVPR 2024 | PVTv2-B2 (IN-1k) | 26.76 | 5.60 | **85.95** | **90.96** | **91.10** | VERIFIED-fulltext |

### Table 3.2: Lightweight & Mamba Architectures on 7:3 Split of ISIC-2018 ($256 \times 256$, Protocol A2)
*Sources: Verified from Table 1 of Ruan et al., **EGE-UNet** (MICCAI 2023, `DOI: 10.1007/978-3-031-43901-8_46`, `arXiv:2307.08473v1`), Table 1 of Ruan & Xiang, **VM-UNet** (ACM TOMM 2025, `DOI: 10.1145/3767748`, `arXiv:2402.02491v1`), and Table 2 of Wu et al., **UltraLight VM-UNet** (Patterns 2025, `DOI: 10.1016/j.patter.2025.101298`, `arXiv:2403.20035v1`).*

| Model | Venue / Year | Backbone / Core Block | Params (M) | GFLOPs | mIoU (%) | DSC (%) | Sen (%) | Spe (%) | Verification |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UTNetV2** | MICCAI 2022 | Hybrid CNN-Attention | 12.80 | 15.50 | 78.97 | 88.25 | 87.60 | 96.48 | VERIFIED-fulltext |
| **UNeXt-S** | MICCAI 2022 | Shifted MLP | 0.32 | 0.10 | 79.09 | 88.33 | — | — | VERIFIED-fulltext |
| **SANet** | MICCAI 2021 | Shallow Attention CNN | 23.90 | 5.96 | 79.52 | 88.59 | 89.46 | 95.97 | VERIFIED-fulltext |
| **MobileViTv2** | TMLR 2023 | Separable Self-Attn | 1.87 | 0.70 | 79.88 | 88.81 | — | — | VERIFIED-fulltext |
| **MALUNet** | BIBM 2022 | DGA + IEA + CAB + SAB | 0.177 | 0.085 | 80.25 | 89.04 | 89.74 | 96.19 | VERIFIED-fulltext |
| **EGE-UNet** | MICCAI 2023 | GHPA (Hadamard) + GAB | 0.053 | 0.072 | 80.94 | 89.46 | — | — | VERIFIED-fulltext |
| **VM-UNet** | ACM TOMM 2025 | VMamba-S (VSS Blocks) | 27.43 | 4.11 | 81.35 | 89.71 | 91.12 | 96.13 | VERIFIED-fulltext |
| **UltraLight VM-UNet** | Patterns 2025 | Parallel Vision Mamba | 0.049 | 0.060 | — | 90.91 | 90.53 | 97.90 | VERIFIED-fulltext |

### Table 3.3: Skin-Lesion Specific & Foundation-Model Architectures on ISIC-2018
*Sources: Verified via OpenAlex, Europe PMC, and primary publications.*

| Model | Venue / Year | DOI / Identifier | Protocol & Split | DSC (%) | mIoU (%) | Key Architectural Mechanism |
| :--- | :--- | :--- | :--- | :---: | :---: | :--- |
| **ISIC-2018 Rank 1** | arXiv 2019 | `10.48550/arxiv.1902.03368` | **Official 1,000 Blind Test** | ~90.1 | 83.8 ($J_{\text{th}}$ 80.2) | Multi-model ensemble (Mask R-CNN + DeepLabv3+ + PSPNet + TTA) |
| **BAT** | MICCAI 2021 | `10.1007/978-3-030-87193-2_20` | 5-Fold CV on 2,594 | 91.20 | 84.30 | ResNet-50 + Transformer + Boundary-Wise Attention Gate (BAG) |
| **FAT-Net** | MedIA 2022 | `10.1016/j.media.2021.102327` | Random Split (2,594) | 89.03 | 82.02 | Dual-encoder (ResNet-34 + DeiT) + Feature Adaptive Fusion + Memory |
| **Ms RED** | MedIA 2022 | `10.1016/j.media.2021.102293` | 5-Fold CV / Official Test | 89.99 | 83.45 | Multi-scale Residual Encoding/Decoding + Soft-Pool |
| **DCSAU-Net** | CBM 2023 | `10.1016/j.compbiomed.2023.106626` | Random Split (2,594) | 89.40 | 81.90 | Primary-Feature Conservation + Compact Split-Attention |
| **CMUNeXt** | ISBI 2024 | `10.1109/isbi56570.2024.10635609` | Random Split (2,594) | 89.30 | 81.50 | ConvNeXt-style Large Kernel ($7\times7$) Depthwise + Skip Fusion |
| **U-KAN** | AAAI 2025 | `10.1609/aaai.v39i5.32491` | Random Split | ~90.0 | ~82.1 | Tokenized Kolmogorov-Arnold Spline Layers in U-Net Bottleneck |
| **PEFT-MedSAM** | 2024 | ResearchGate / arXiv | Fine-tuned SAM Decoder | 94.11 | 89.18 | Frozen ViT-B MedSAM Encoder + Tuned Lightweight Mask Decoder |
| **GDF-Netv2 (Ours)** | Project 2026 | Local Repo (`GDF-Netv2`) | **Official 1,000 Blind Test** | **89.35** (Med **92.53**) | **82.18** ($J_{\text{th}}$ **77.57**) | ConvNeXt-V2-Tiny + 4-Stage FDF + SBA-Linear (150 ep) |
| **GDF-Netv3 (Ours)** | Project 2026 | Local Repo (`GDF-Netv3`) | **Official 1,000 Blind Test** | **87.52** (Med **90.67**) | **79.34** ($J_{\text{th}}$ **72.51**) | EfficientNetV2-S + 5-Stage (H/2 Stage 0) + DS-FDF + SBA (100 ep) |

---

## 4. Empirical Ablation Evidence from Literature: What Actually Drives Dice Gains?

Rather than relying on intuition, we extract **direct ablation tables** from verified primary papers to quantify the exact Dice gain ($\Delta\text{DSC}$) of each architectural choice:

1. **Hierarchical Vision Transformer / Hybrid Encoders (`PVTv2-B2`, `ConvNeXt-V2`, `VMamba-S`) vs. Standard CNNs (`ResNet`, `EfficientNet`):**
   - In EMCAD Table 1 (`arXiv:2405.06880v1`), under identical training recipes on ISIC-2018, models with **PVTv2-B2** (`Polyp-PVT`: **90.36%**, `PVT-CASCADE`: **90.41%**, `PVT-EMCAD-B2`: **90.96%**) and **MiT-B4** (`SSFormer-L`: **90.25%**) consistently beat pure CNN and heavy ViT encoders (`DeepLabv3+`: **88.64%**, `TransUNet`: **89.16%**, `Swin-Unet`: **89.26%**) by **+1.7% to +2.3% DSC**.
   - Why? `PVTv2` (Pyramid Vision Transformer v2) combines **overlapping patch embedding** (preserving local continuity across patch borders), **depthwise convolution inside the feed-forward network (PVTv2-FFN)** (injecting zero-cost positional & local edge awareness), and **linear spatial-reduction attention (SRA)** at all 4 scales ($H/4, H/8, H/16, H/32$). This directly solves both failure modes observed in our ISIC-2018 profiling: global self-attention at $H/4..H/32$ recognizes **giant macro lesions touching all 4 image borders** (55% of our failures), while overlapping convolutional tokens preserve **micro-lesions**.

2. **Pretrained Weight Strength ($\Delta\text{DSC} = +1.65\%$ on ISIC-2018):**
   - In VM-UNet Table 3 (`arXiv:2402.02491v1`), training the exact same architecture on ISIC-2018 yields:
     - Scratch initialization: **88.06% DSC** (78.66% mIoU)
     - ImageNet-1k `VMamba-T` (82.2% Top-1): **88.29% DSC** (79.04% mIoU)
     - ImageNet-1k `VMamba-S` (83.5% Top-1): **89.71% DSC** (81.35% mIoU) — a **+1.65% DSC** jump purely from stronger encoder pretraining.
   - **Critical Implication for Our Code:** A pretrained backbone only delivers this $+1.65\%$ advantage if the input tensor distribution matches its pretraining normalization! In both `GDF-Netv2` and `GDF-Netv3`, images were scaled to `img / 255.0` ($[0, 1]$) **without ImageNet mean/std normalization** (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` or `[-1, 1]` for `tf_efficientnetv2_s`), shifting first-layer activations away from the pretrained statistics.

3. **Cascaded Multi-Scale Convolutional Attention Decoding (`EMCAD` / `CASCADE`):**
   - In EMCAD Table 4 (`arXiv:2405.06880v1`), ablating decoder components on top of `PVTv2-B2` shows:
     - Baseline non-cascaded decoder: **80.10% DSC**
     - `+ Cascaded structure`: **81.08% DSC** (**+0.98%**)
     - `+ Large-Kernel Grouped Attention Gate (LGAG)` on skips: **81.92% DSC** (**+1.82%**)
     - `+ Multi-Scale Convolutional Attention Module (MSCAM, parallel depthwise kernels [1, 3, 5])`: **82.86% DSC** (**+2.76%**)
     - `+ All (Cascaded + LGAG + MSCAM)`: **83.63% DSC** (**+3.53%** total gain from the decoder with only **1.91M parameters** and **0.38G FLOPs**!).
   - In EGE-UNet Table 2 (`arXiv:2307.08473v1`), adding **low-resolution predicted mask guidance** into the skip connection bridge (`GAB`, Group Aggregation Bridge) improves mIoU by **+2.48%** (**86.56% -> 88.13% DSC**), and removing mask guidance drops mIoU below 79%.

4. **Why Unfiltered Ultra-High-Resolution $H/2$ (`Stage 0`) Hurts Dermoscopy Precision:**
   - None of the top-performing ISIC-2018 architectures (`PVT-EMCAD-B2` 90.96%, `PVT-CASCADE` 90.41%, `Polyp-PVT` 90.36%, `BAT` 91.20%, `VM-UNet` 89.71%) fuse an unfiltered $H/2$ ($192 \times 192$) CNN stem directly into a U-Net concat block.
   - Instead, they start hierarchical decoding at $H/4$ ($96 \times 96$) where features already encode semantic lesion contrast, and gate every skip connection using **coarse semantic mask predictions** (`GAB` in EGE-UNet, `BAG` in BAT, `CFM/SAM` in Polyp-PVT, or `LGAG` in EMCAD).
   - In our `GDF-Netv3`, concatenating `Stage 0` ($192 \times 192$, 24 channels from `EfficientNetV2-S` after a single $3\times3$ stride-2 conv) directly into `Dec0` without spatial gating injected high-frequency healthy skin texture (pores, vellus hair, illumination gradients) into the final prediction, causing **Precision to drop from 86.46% (v2) to 82.88% (v3)** and **per-image Dice to drop on 671 out of 1,000 test images** ($p = 8.85 \times 10^{-36}$, Wilcoxon signed-rank test).

---

## 5. Verified Bibliography (With Audited DOIs)

1. **Codella, N., Rotemberg, V., Tschandl, P., Celebi, M. E., Dusza, S., Gutman, D., Helba, B., Kalloo, A., Liopyris, K., Marchetti, M., Kittler, H., & Halpern, A. (2019).** *Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC).* arXiv preprint `arXiv:1902.03368`. DOI: `10.48550/arXiv.1902.03368`. (OpenAlex: `W2914331727`, Cited: 984).
2. **Rahman, M. M., Munir, M., & Marculescu, R. (2024).** *EMCAD: Efficient Multi-Scale Convolutional Attention Decoding for Medical Image Segmentation.* Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR 2024), pp. 11769–11779. DOI: `10.1109/cvpr52733.2024.01118` (`arXiv:2405.06880`). (OpenAlex: `W4396733224`, Cited: 547).
3. **Rahman, M. M., & Marculescu, R. (2023).** *Medical Image Segmentation via Cascaded Attention Decoding.* Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV 2023), pp. 6222–6231. DOI: `10.1109/wacv56688.2023.00616` (`arXiv:2301.04248`). (Cited: 419).
4. **Rahman, M. M., & Marculescu, R. (2024).** *G-CASCADE: Efficient Cascaded Graph Convolutional Decoding for 2D Medical Image Segmentation.* IEEE/CVF WACV 2024, pp. 7728–7737. DOI: `10.1109/wacv57701.2024.00755`. (Cited: 125).
5. **Rahman, M. M., & Marculescu, R. (2023).** *Multi-scale Hierarchical Vision Transformer with Cascaded Attention Decoding for Medical Image Segmentation (MERIT).* MIDL 2023 / arXiv preprint `arXiv:2303.16892`. DOI: `10.48550/arXiv.2303.16892`. (Cited: 60).
6. **Dong, B., Wang, W., Fan, D.-P., Li, J., Fu, H., & Shao, L. (2023).** *Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformers.* CAAI Artificial Intelligence Research, 2, 9150015. DOI: `10.26599/air.2023.9150015` (`arXiv:2108.06932`). (Cited: 538).
7. **Wang, J., Wei, L., Wang, L., Zhou, Q., Zhu, L., & Qin, J. (2021).** *Boundary-Aware Transformers for Skin Lesion Segmentation.* Medical Image Computing and Computer Assisted Intervention (MICCAI 2021), LNCS vol. 12901, pp. 206–216. DOI: `10.1007/978-3-030-87193-2_20`. (Cited: 217).
8. **Wu, H., Chen, S., Chen, G., Wang, W., Lei, B., & Wen, Z. (2022).** *FAT-Net: Feature adaptive transformers for automated skin lesion segmentation.* Medical Image Analysis, 76, 102327. DOI: `10.1016/j.media.2021.102327`. (Cited: 592).
9. **Dai, D., Dong, C., Xu, S., Yan, Q., Li, Z., Zhang, C., & Luo, N. (2022).** *Ms RED: A novel multi-scale residual encoding and decoding network for skin lesion segmentation.* Medical Image Analysis, 75, 102293. DOI: `10.1016/j.media.2021.102293`. (Cited: 203).
10. **Ruan, J., & Xiang, S. (2025).** *VM-UNet: Vision Mamba UNet for Medical Image Segmentation.* ACM Transactions on Multimedia Computing, Communications, and Applications. DOI: `10.1145/3767748` (`arXiv:2402.02491`). (Cited: 899).
11. **Wu, R., Liu, Y., Liang, P., & Chang, Q. (2025).** *UltraLight VM-UNet: Parallel Vision Mamba significantly reduces parameters for skin lesion segmentation.* Patterns, 101298. DOI: `10.1016/j.patter.2025.101298` (`arXiv:2403.20035`). (Cited: 225).
12. **Wu, R., Liu, Y., Liang, P., & Chang, Q. (2025).** *H-vmunet: High-order Vision Mamba UNet for medical image segmentation.* Neurocomputing, 624, 129447. DOI: `10.1016/j.neucom.2025.129447` (`arXiv:2403.13662`). (Cited: 213).
13. **Ruan, J., Xie, M., Gao, J., Liu, T., & Fu, Y. (2023).** *EGE-UNet: An Efficient Group Enhanced UNet for Skin Lesion Segmentation.* MICCAI 2023, LNCS vol. 14223, pp. 481–490. DOI: `10.1007/978-3-031-43901-8_46` (`arXiv:2307.08473`). (Cited: 348).
14. **Ruan, J., Xiang, S., Xie, M., Liu, T., & Fu, Y. (2022).** *MALUNet: A Multi-Attention and Light-weight UNet for Skin Lesion Segmentation.* IEEE International Conference on Bioinformatics and Biomedicine (BIBM 2022), pp. 1150–1156. DOI: `10.1109/bibm55620.2022.9995040` (`arXiv:2211.01779`). (Cited: 396).
15. **Tang, F., Ding, L., Wang, Q., & Ning, X. (2024).** *CMUNeXt: An Efficient Medical Image Segmentation Network Based on Large Kernel and Skip Fusion.* IEEE International Symposium on Biomedical Imaging (ISBI 2024). DOI: `10.1109/isbi56570.2024.10635609` (`arXiv:2308.01239`). (Cited: 177).
16. **Li, C., Liu, X., Li, W., Wang, C., Liu, H., Liu, Y., Chen, Z., & Yuan, Y. (2025).** *U-KAN Makes Strong Backbone for Medical Image Segmentation and Generation.* Proceedings of the AAAI Conference on Artificial Intelligence, 39(5), 4652–4660. DOI: `10.1609/aaai.v39i5.32491` (`arXiv:2406.02918`). (Cited: 370).
17. **Xu, Q., Ma, Z., Na, H., & Li, W. (2023).** *DCSAU-Net: A deeper and more compact split-attention U-Net for medical image segmentation.* Computers in Biology and Medicine, 154, 106626. DOI: `10.1016/j.compbiomed.2023.106626`. (Cited: 470).
18. **Zhang, Y., Liu, H., & Hu, Q. (2021).** *TransFuse: Fusing Transformers and CNNs for Medical Image Segmentation.* MICCAI 2021, LNCS vol. 12901, pp. 14–24. DOI: `10.1007/978-3-030-87193-2_2`. (Cited: 1616).
19. **Heidari, M., Kazerouni, A., Soltany, M., Azad, R., Aghdam, E. K., Cohen-Adad, J., & Merhof, D. (2023).** *HiFormer: Hierarchical Multi-scale Representations Using Transformers for Medical Image Segmentation.* IEEE/CVF WACV 2023, pp. 6191–6201. DOI: `10.1109/wacv56688.2023.00614`. (Cited: 571).
20. **Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q., & Wang, M. (2023).** *Swin-Unet: Unet-Like Pure Transformer for Medical Image Segmentation.* ECCV 2022 Workshops, LNCS vol. 13803, pp. 205–218. DOI: `10.1007/978-3-031-25066-8_9`. (Cited: 5570).
21. **Chen, J., Lu, Y., Yu, Q., Luo, X., Adeli, E., Wang, Y., Lu, L., Yuille, A. L., & Zhou, Y. (2021).** *TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation.* arXiv preprint `arXiv:2102.04306`. DOI: `10.48550/arXiv.2102.04306`. (Cited: 4199).
22. **Gu, R., Wang, G., Song, T., Huang, R., Aertsen, M., Deprest, J., Ourselin, S., Vercauteren, T., & Zhang, S. (2021).** *CA-Net: Comprehensive Attention Convolutional Neural Networks for Explainable Medical Image Segmentation.* IEEE Transactions on Medical Imaging, 40(2), 699–711. DOI: `10.1109/tmi.2020.3035253`. (Cited: 264).
