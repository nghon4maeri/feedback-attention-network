# Architecture Analysis Report
Date: 2026-09-22

## Overview
This report provides a comprehensive technical analysis of the FANet codebase, focusing on its architecture, data pipeline, training strategies, and ongoing research directions.

## 1. High-Level Overview & Tech Stack
- **Framework & Libraries**: PyTorch, NumPy, OpenCV, Albumentations, Scikit-Learn.
- **Project Structure**:
  - `configs/`: YAML configuration files.
  - `src/fanet/`: Installable package containing model definitions (`models/`), data handling (`data/`), metrics, and varied losses.
  - `scripts/`: Execution scripts (`train.py`, `evaluate.py`).
  - `analysis/`: Diagnostic tools for measuring gradient flow and network behaviors.
- **Core Purpose**: Improving biomedical image segmentation (specifically Kvasir-SEG sessile polyps) by addressing the "Feedback Trap"—a phenomenon where recurring spatial feedback leaks gradients and amplifies errors.

## 2. Core Architecture Analysis (FANet)
- **Encoder-Decoder Architecture**: The model does not use a standard pre-trained backbone (like ResNet). It utilizes custom `EncoderBlock` and `DecoderBlock` modules composed of custom 3x3 `ResidualBlock` modules.
- **Attention Mechanism**: `SELayer` (Squeeze-and-Excitation) is integrated within the residual blocks for channel-wise attention.
- **Feature Aggregation (MixPool)**: The signature module of FANet. `MixPool` merges internal features with a feedback mask (`m_fg`) from previous predictions.
  - *Legacy Issue*: The original binary gating `(fmask > 0.5)` was non-differentiable, freezing the `fmask` branch from learning and causing representation collapse.
  - *Phase 7B Upgrade*: Introduces **Detached Soft-OR** gating (`keep = 1.0 - (1.0 - fmask) * (1.0 - m_fg)`) combined with `detach_feedback=True`. This severs the problematic recurrent gradient loop while allowing `fmask` to actively learn to mitigate uncertainty.

## 3. Data Pipeline & Preprocessing
- **Dataset**: `DATASET` loader handles the Kvasir-SEG (sessile subset) dataset using OpenCV.
- **Augmentation**: Albumentations handles robust transformations: Rotate (±35°), Horizontal/Vertical Flips (30%), and CoarseDropout to prevent overfitting.
- **Preprocessing**: Images are resized to 256x256, scaled to `[0, 1]`, and transformed into `[C, H, W]` float32 tensors. Masks follow similarly but with a single channel.

## 4. Training, Optimization & Evaluation
- **Optimizer**: Adam optimizer with `ReduceLROnPlateau` scheduler (patience=5).
- **Loss Functions**: A heavily customized suite of losses (`losses.py`) is pivotal to the project.
  - *Base*: `DiceBCELoss`
  - *Variants*: `NegativeAreaDiceBCELoss`, `WeightedSoftDiceLoss`, `TverskyLoss` (with asymmetric penalization for FP suppression), and `FarWeightedIoUBCELoss`.
- **Metrics**: Standard metrics focus heavily on addressing over-segmentation. Priority metrics include Dice score, Precision, Recall, and False Positive Rate (FPR).

## 5. Current Research Direction & Hypothesis
- **The Problem**: Traditional Recurrent Feedback loops trap optimization (gradient attenuation, BatchNorm drift), nullifying advanced loss functions aimed at penalizing False Positives.
- **The Solution**: "Gradient Decoupling". By introducing Detached Soft-OR gating in the Phase 7B experiments, the model isolates feedback signals from backward passes.
- **Result**: This mechanism successfully "unleashes" the loss (e.g., Tversky), allowing the encoder to adapt and heavily suppress False Positives without representation collapse.

## Strategic Recommendations
1. **Extend Multi-Domain Validation**: Test the Detached Soft-OR mechanism on non-medical datasets (e.g., Cityscapes or remote sensing) to prove domain-agnostic universality of Gradient Decoupling.
2. **Dynamic Loss Weighting**: Implement an adaptive loss scheduler that scales the Tversky `alpha` penalty dynamically based on epoch confidence or bounding-box distance, rather than statically fixing it.
3. **Contrastive Feedback Pretraining**: Introduce a self-supervised contrastive objective for the `fmask` branch before end-to-end training. This could accelerate its convergence on complex, indistinct boundaries like flat sessile polyps.
