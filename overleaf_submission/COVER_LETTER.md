# Cover Letter for Manuscript Submission

**To:**  
The Program Chairs / Area Chairs / Editor-in-Chief  
*Medical Image Computing and Computer Assisted Intervention (MICCAI) / IEEE Transactions on Medical Imaging (TMI)*

**Date:** September 16, 2026  
**Subject:** Submission of Original Research Article:  
*"Breaking the Feedback Trap: Understanding and Stabilizing Recurrent Feedback Learning in Medical Image Segmentation"*

Dear Program Chairs, Area Chairs, and Editorial Board,

We are pleased to submit our original research manuscript entitled **"Breaking the Feedback Trap: Understanding and Stabilizing Recurrent Feedback Learning in Medical Image Segmentation"** for consideration as a regular paper in your esteemed venue.

### Mechanistic Foundational Study Rather Than Parameter Bloat
In recent years, the medical image segmentation community has witnessed an influx of increasingly complex hybrid architectures and massive vision transformers competing for marginal gains on benchmark leaderboards. However, the foundational architectural mechanics of **recurrent feedback networks**---which mimic biological top-down cortical refinement by iteratively re-feeding past predictions into early encoder representations---have remained poorly understood and treated largely as empirical black boxes.

In this work, we present a rigorous **mechanistic study** diagnosing a pervasive mathematical pathology in recurrent medical vision architectures, which we term the **Feedback Trap**. Specifically, we examine why recurrent feedback networks (such as FANet) consistently suffer from rampant false-positive over-segmentation (over 84% of error mass bleeding into healthy mucosa), and why modern asymmetric loss functions (e.g., Tversky loss) are completely neutralized once recurrent loops are activated.

Through systematic gradient and representational auditing, we uncover two structural flaws in conventional hard binary gating ($\max(\mathbb{I}(\text{fmask} > 0.5), m_{\text{fg}})$):
1. **Internal Gradient Paralysis:** The indicator function yields zero gradients almost everywhere ($\frac{\partial \text{keep}}{\partial \text{fmask}} \equiv 0$), depriving the learnable attention branch of direct task supervision.
2. **Toxic Error Accumulation:** Recursive backpropagation through the unvalidated feedback mask tensor attenuates gradient magnitudes by over 79% and induces severe encoder distribution drift ($D_{\text{KL}} = 33.63$), trapping the model in an echo chamber of its own past mistakes.

### Principled Zero-Parameter Solution: Detached Soft-OR
To cure this pathology without discarding the multi-pass refinement benefits of recurrent networks, we introduce **Detached Soft-OR Gating**. Our reformulation relaxes the logical disjunction into continuous probabilistic space ($a + m - am$) while applying the stop-gradient operator ($\text{detach}(m_{\text{fg}})$) to the feedback prior. We prove analytically that this transformation:
- Restores continuous, non-zero gradient flow strictly scaled by background uncertainty: $\frac{\partial \text{keep}}{\partial \text{fmask}} = 1 - m_{\text{fg}}$, maximally supervising ambiguous boundary margins.
- Completely severs degenerate recursive error propagation ($\frac{\partial \text{keep}}{\partial m_{\text{fg}}} \equiv 0$).
- Requires **exactly zero additional learnable parameters** ($\Delta \Theta = 0$).

### Decisive Empirical Hard Numbers and Multi-Center Generalization
Our extensive empirical evaluation across multiple datasets and random initializations establishes overwhelming statistical superiority:
- **Over-Segmentation Suppression:** On Kvasir-SEG (Sessile), Detached Soft-OR slashes the False Positive Rate by **$42.8\%$** (dropping from $4.30\%$ to $2.46\%$, and from $21.39\%$ to $7.55\%$ at inference), while elevating mean Dice by **$+4.77$ pp** ($+19.0\%$ relative gain) and Precision by **$+7.50$ pp** ($+23.9\%$).
- **Variance Compression and Stability:** Across 5 random seeds ($200$ epochs each), inter-seed standard deviation plummets by **$66.5\%$** ($\sigma = 0.0869 \to 0.0291$), completely eliminating the catastrophic representation collapse observed in the baseline. Paired Wilcoxon signed-rank tests across $N = 200$ instances confirm over-segmentation suppression at **$p < 0.001$** ($W = 4,055.0$).
- **True Morphological Delineation (No Background Collapse):** High-resolution qualitative analysis confirms that false-positive reduction occurs while preserving dominant, high-fidelity True Positive regions ($\text{Dice} = 0.50 - 0.77$) that match true polyp bodies, rather than collapsing trivially to empty masks.
- **External Zero-Shot Generalization on CVC-ClinicDB ($N = 612$):** When tested zero-shot on an unseen clinical center (Hospital Clinic, Barcelona) without any retraining or adaptation, our model achieves a statistically significant Dice improvement of **$+2.67$ pp** ($+11.2\%$ relative gain, **$p = 1.61 \times 10^{-15}$**), enhances out-of-distribution Recall by **$+6.48$ pp** ($+14.3\%$), and compresses inter-seed variance by **$41.1\%$**.

### Compliance and Integrity
This manuscript represents entirely original work and is not under consideration for publication elsewhere. All code, trained checkpoints, and experimental pipelines will be made publicly available upon acceptance to ensure complete reproducibility.

Given the broad applicability of our findings to recurrent deep learning architectures in biomedical imaging, we believe this manuscript will be of high interest to the readership and community of MICCAI / IEEE TMI. We thank you and the reviewers for your time and constructive consideration of our submission.

Sincerely,

**The Authors**  
*FANet Research Team*  
Correspondence Email: `anonymous@domain.edu`
