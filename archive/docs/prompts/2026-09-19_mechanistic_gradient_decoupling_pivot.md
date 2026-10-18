# 2026-09-19: Mechanistic Deep Learning Pivot & Gradient Decoupling

## 1. Context & Objectives
The Senior PI / Meta-Reviewer requested a strategic elevation of the manuscript and repository from an incremental "new attention module for polyp segmentation" to a foundational **Mechanistic Deep Learning** study.

### Core Scientific Insight:
The true scientific breakthrough is NOT merely the Soft-OR equation, but uncovering a pervasive structural pathology in recurrent feedback vision networks:
**Prediction history acting simultaneously as spatial guidance in the forward pass and as an unconstrained optimization path in the backward pass causes catastrophic error amplification (the Feedback Trap).**
The primary contribution is formulating and proving the **Feedback Firewall (Gradient Decoupling)** principle: forward spatial guidance must be decoupled from backward gradient propagation.

## 2. Four Strategic Tasks
1. **Task 1: Narrative Pivot & Title Overhaul (Framing)**
   - New Title: *Breaking the Feedback Trap: Understanding and Stabilizing Recurrent Feedback Learning in Medical Image Segmentation*.
   - Rewrite Abstract, Introduction, and Contributions to center on the discovery of the recurrent failure mode and the Gradient Decoupling principle.
2. **Task 2: Core Mechanism Ablation (Soft vs. Detach)**
   - Implement, execute, and document a 4-way ablation study disentangling continuous relaxation (Soft) from the stop-gradient operator (Detach):
     * Hard Gating + Gradient Flow ($M_{11}$ Baseline)
     * Soft-OR Only (Continuous relaxation + Flow)
     * Detached Hard Gating (Hard + Detach)
     * Detached Soft-OR ($M_{12}$ Ours)
3. **Task 3: Loss Neutralization Factorial Evidence ($3 \times 2$)**
   - Formalize tabular evidence evaluating Dice and FPR across:
     * Architecture: No Feedback (Feedforward), Hard Feedback (Baseline), Detached Soft-OR (Ours).
     * Loss Functions: Standard DiceBCE vs. Asymmetric Tversky.
   - Empirically demonstrate that Hard Feedback neutralizes the optimization benefit of asymmetric loss, whereas Detached Soft-OR unleashes it.
4. **Task 4: Generalization & Universality (Pipeline Expansion)**
   - Datasets: Scaffold ingestion and evaluation pipelines for retinal vessel benchmarks (DRIVE, CHASE_DB1) to demonstrate domain universality beyond colonoscopy.
   - Architecture: Implement and scaffold testing for Detached Soft-OR on a secondary recurrent architecture (Recurrent Residual U-Net / R2U-Net) to prove that the Feedback Trap and its cure are universal architectural principles.

## 3. Key Deliverables
- `docs/prompts/2026-09-19_mechanistic_gradient_decoupling_pivot.md`
- `src/fanet/models/r2unet.py` & `scripts/test_secondary_architecture_r2unet.py`
- `scripts/evaluate_cross_domain_retinal.py`
- `scripts/ablation_soft_vs_detach.py` & `results/ablation_soft_vs_detach.json`
- Updates to `overleaf_submission/` (`main.tex`, sections 01-05, `COVER_LETTER.tex`)
- Updates to `docs/reports/paper_full_manuscript.md`
- `docs/reports/2026-09-19_mechanistic-gradient-decoupling.md` & `docs/reports/README.md`
