# Session Prompt: GDF-Netv5 Fold 0 Training (100 Epochs) on Kaggle GPU

- **Date:** 2026-10-25
- **Goal:** Deploy and launch GDF-Netv5 Fold 0 training (100 epochs) on Kaggle GPU (Tesla P100 / 2xT4) to verify model convergence and test set performance against the >0.91 Dice objective without consuming Google Colab compute units.
- **Scope & Constraints:**
  1. Target Model: `GDF-Netv5` with `pvt_v2_b4` backbone (pretrained on ImageNet-1k, 62.56M params in encoder, 27 blocks in Stage 3).
  2. Input Resolution: Native $512 \times 512$ resolution.
  3. Batch Size: 8 per step with gradient accumulation steps = 2 (effective batch size 16, optimized for 16GB VRAM on Tesla P100 with AMP fp16).
  4. Augmentation: Artifact-aware augmentation (`RandomVignetting`, `CLAHE`, `SyntheticHair`, `RandomResizedCrop`).
  5. Schedule: 100 epochs with `CosineAnnealingLR` (initial LR = 1e-4, min LR = 1e-6) and `AdamW` (weight decay = 1e-4).
  6. Evaluation: Full evaluation on the official 1,000 blind test images of ISIC-2018 using both Single-View and 4-View Flip Test-Time Augmentation (TTA).
  7. Verification Target: Macro Dice > 0.9100 on official test set.
  8. Environment: Kaggle Kernel (`namnguynnnn/gdfnet-v5-fold0-isic2018`) with direct `/kaggle/input` dataset mounting and internet enabled for HuggingFace model download.
  9. Working Conventions: Strictly NO emojis, commit and push git immediately after code creation.
