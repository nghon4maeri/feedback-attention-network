# Prompt Session — 2026-10-27: GDF-Netv4 300-Epoch Kaggle Results Verification & Modern Lightweight CNN Backbone Survey

## 1. User Request
- "hi hãy kiểm tra kết quả đã train trên kaggle (đã chạy xong) và tôi thấy hình như model hơi lớn nhỉ vậy thì bạn hãy dùng agent scientific research skill tìm xem trên ImageNet benchmark có model nào CNN tốt không, như convNext chẳng hạn"

## 2. Objectives & Scope
1. **Verify & Analyze Completed Kaggle Run (`namnguynnnn/gdfnet-v4-300ep-heavyaug-isic2018`):**
   - Download all CSVs and `.log` into `GDF-Netv4/results/isic2018_300ep/`.
   - Analyze training progression (71 epochs within the 11.09h wall-clock guard, best EMA checkpoint saved at Epoch 5 with `Val Macro Dice = 0.9026`, `Val Global Dice = 0.9084`).
   - Analyze Official 1,000 Blind Test results across Single-View (`0.9006`), 4-View TTA (`0.9016`), and Official-Val Threshold-Calibrated 4-View TTA ($T_{\text{opt}} = 0.72$, achieving **Mean Dice `0.9101`**, **Median Dice `0.9399`**, **Global Micro Dice `0.9118`**, **$J_{\text{thresh}} = 0.8150`**, and only **69 failed cases**).
2. **Scientific Literature & Benchmark Research on Modern Lightweight / High-Efficiency CNN Backbones (ImageNet-1K & Medical Segmentation, 2022–2026):**
   - Activate scientific research skills (`literature-review`, `paper-lookup`, `citation-management`, `statistical-analysis`, `experimental-design`).
   - Query multiple scholarly databases (OpenAlex, Semantic Scholar, arXiv, Europe PMC) + `timm` ImageNet-1K benchmark tables to compare modern pure-CNN / ConvNet architectures against `PVTv2-B2` (25.4M) and `PVTv2-B4` (62.6M):
     - **ConvNeXt-V2 family** (`convnextv2_atto` 3.7M, `convnextv2_femto` 5.2M, `convnextv2_pico` 9.1M, `convnextv2_nano` 15.6M, `convnextv2_tiny` 28.6M — Woo et al., CVPR 2023)
     - **MogaNet family** (`moganet_xtiny` 3.0M, `moganet_tiny` 5.2M, `moganet_small` 25.3M — Li et al., ICLR 2024)
     - **InceptionNeXt family** (`inception_next_tiny` 28.1M, `inception_next_atto` 4.2M — Yu et al., CVPR 2024)
     - **UniRepLKNet / RepLKNet** (Ding et al., CVPR 2024 / CVPR 2022)
     - **SegNeXt / MSCAN & VAN (Visual Attention Network / LKA)** (`VAN-B0` 4.1M, `VAN-B1` 13.9M, `VAN-B2` 26.6M — Guo et al., CVM 2023; NeurIPS 2022)
     - **ConvFormer / MetaFormer** (`convformer_s18` 26.8M — Yu et al., TPAMI 2024)
     - **FastViT / MobileOne / EfficientMod** & Medical CNN adaptations (**CMUNeXt**, ISBI 2024; **ULite**, 2023; **I2U-Net**, MedIA 2024).
   - Document verified citations (Title, Venue, Year, DOI) in `docs/literature_grounding_modern_cnn_backbones.md` and write Phase 15 report in `docs/reports/2026-10-27_phase15-gdfnet-v4-300ep-verification-and-cnn-backbone-survey.md`.

## 3. Constraints
- Strictly follow `AGENTS.md`: commit and push all code, results, literature grounding, prompt logs, and reports to GitHub.
- Classify evidence (`TRAIN`, `FROZEN EVAL`, `INVALID`, `LITERATURE`).
