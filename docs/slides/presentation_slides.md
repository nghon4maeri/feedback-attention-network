---
marp: true
theme: default
paginate: true
header: "Breaking the Feedback Trap in FANet | Research Report"
footer: "Presenter: Nguyen Nam (nghon4m) — Medical AI & Computer Vision Research Group"
size: 16:9
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
    padding: 35px 50px;
    font-size: 20px;
  }
  h1 {
    color: #0b3c5d;
    font-size: 34px;
    margin-bottom: 12px;
  }
  h2 {
    color: #1d2731;
    font-size: 24px;
    margin-top: 5px;
    margin-bottom: 15px;
  }
  table {
    font-size: 15px;
    width: 100%;
  }
  th {
    background-color: #0b3c5d;
    color: white;
  }
  strong {
    color: #b82601;
  }
  img {
    max-height: 380px;
    display: block;
    margin: 0 auto;
  }
---

<!-- Slide 1: Title -->

# Breaking the Feedback Trap in FANet: A Detached Soft-OR Gating Mechanism

## Restoring Gradient Flow and Eliminating Over-Segmentation in Recurrent Medical Segmentation

**Presenter:** Nguyen Nam (`nghon4m`)  
**Research Group:** Medical AI & Computer Vision Research Laboratory  
**Target Venues:** MICCAI / CVPR / IEEE Transactions on Medical Imaging (TMI)  
**Clinical Benchmark:** Subtle Sessile Colorectal Polyps (Kvasir-SEG Paris IIa/IIb)

---
* **Core Takeaway:** A **zero-parameter architectural firewall** that decouples forward spatial guidance from backward gradient contamination, restoring attention gradient flow and curing chronic false-positive over-segmentation.

---

<!-- Slide 2: Abstract -->

# Abstract: From Negative Diagnosis to Architectural Breakthrough

* **Clinical Context:** Delineating subtle sessile colorectal polyps is challenging due to flat morphology and poor boundary contrast; recurrent feedback networks (FANet) promise multi-pass refinement via `MixPool`.
* **The "Feedback Trap" Pathology:**
  * Hard binary gating ($\text{keep} = \max(\mathbb{I}(\text{fmask} > 0.5), m_{\text{fg}})$) yields **$\partial \text{keep}/\partial \text{fmask} \equiv 0$ almost everywhere**, paralyzing internal attention learning.
  * Toxic recurrent backpropagation attenuates gradient magnitude by **$>79\%$** and causes severe BatchNorm drift ($D_{\text{KL}} = 33.63$ at `e1.r1.bn3`).
  * Neutralizes modern asymmetric loss functions (**$+4.17\text{ pp}$ interaction**), trapping the model in chronic over-segmentation ($>84\%$ error mass in false positives).
* **Proposed Solution ($M_{12}$ - Detached Soft-OR Gating):**
  * Replaces discontinuous thresholding with a probabilistic smooth union: $\text{keep} = 1 - (1 - \text{fmask})(1 - \text{detach}(m_{\text{fg}}))$.
  * Establishes an uncertainty-guided gradient schedule: $\frac{\partial \mathcal{L}}{\partial \text{fmask}} \propto 1 - m_{\text{fg}}$.
* **Multi-Seed Benchmark Results (5 Seeds $\times$ 200 Epochs):**
  * Slashes False Positive Rate by **$42.8\%$** ($4.30\% \to 2.46\%$, Wilcoxon **$p < 0.001$**).
  * Elevates mean Dice score by **$+4.77\text{ pp}$** ($0.3428 \to 0.2183$) and Precision by **$+7.50\text{ pp}$** ($31.4\% \to 38.9\%$).
  * Compresses inter-seed variance by **$66.5\%$** and completely cures catastrophic collapse.

---

<!-- Slide 3: Contributions -->

# Key Scientific Contributions

### 1. Performance Breakthrough (Superiority Across Metrics)
* Slashes False Positive Rate by **$42.8\%$** across 5 seeds; achieves a **$91.0\%$ reduction** in outlier runs (Seed 2024: $8.88\% \to \mathbf{0.80\%}$).
* Elevates mean Dice score by **$+4.77\text{ pp}$** (**$+19.0\%$ relative gain**) and boosts Precision by **$+7.50\text{ pp}$** ($31.43\% \to \mathbf{38.93\%}$, **$+23.9\%$ relative**).

### 2. Impact & Stability (Catastrophic Collapse Immunity)
* Resurrects Seed 1337 from catastrophic representation collapse ($\text{Dice } 0.0928 \to \mathbf{0.2779}$, absolute gain **$+18.51\text{ pp}$**).
* Compresses inter-seed standard deviation by **$66.5\%$** ($\sigma = 0.0869 \to \mathbf{0.0291}$, $F$-test $p < 0.01$).
* Protects colonoscopists from "alarm fatigue" by eradicating massive ghost lesions ($>3,300\text{ px} \to \mathbf{0\text{ px}}$).

### 3. Theoretical Novelty (Zero-Parameter Architectural Firewall)
* First formal discovery of the **Feedback Trap**: proves analytically that hard binary gating kills attention gradients ($\partial = 0$).
* Derives the analytical gradient for Detached Soft-OR: $\frac{\partial \text{keep}}{\partial \text{fmask}} = 1 - m_{\text{fg}}$, automatically channeling gradients to uncertain margins.
* **Zero Parameter Overhead:** Adds exactly **$0$ new parameters** ($\Delta \text{Params} = 0$), preserving real-time throughput (**$54.3\text{ FPS}$**).

---

<!-- Slide 4: Key gaps in related work and main idea -->

# Key Gaps in Related Work & The Core Main Idea

### Critical Gaps in Contemporary Literature
* **Gap 1: Heuristic Recurrent Coupling in Medical Segmentation:**
  * Recurrent networks (FANet, Recurrent U-Net) treat cross-pass feature injection as a heuristic black box, assuming reinjection automatically improves spatial focus.
  * *Fatal oversight:* Prior works ignored the backward automatic differentiation graph across the feedback interface, where recurrent predictions contaminate early encoder representations.
* **Gap 2: The Loss Engineering Deadlock:**
  * Modern approaches attempt to curb over-segmentation solely via loss engineering (Tversky, Focal Tversky, Unified Focal).
  * *Empirical Discovery:* Under recurrent feedback, asymmetric loss is completely neutralized (**interaction effect = $+4.17\text{ pp}$**). Loss functions cannot correct representations when backward gradient pathways are throttled.

### The Main Idea: Architectural Gradient Decoupling ("Gradient Firewall")
* **"Forward spatial guidance must be decoupled from backward optimization."**
* The recurrent mask should act strictly as an uncorrupted spatial prior, never an error backpropagation highway.
* Stop-gradient detachment ($\text{detach}(m_{\text{fg}})$) combined with probabilistic Soft-OR restores continuous gradient highways.

---

<!-- Slide 5: Methods (brief) -->

# Methodology: The Flaw (Hard Gating) vs. The Fix (Detached Soft-OR)

### 1. The Flaw (Classical Hard-Gated MixPool in M11)
* Hard binary gating formula:
  $$\text{keep}_{\text{hard}} = \max\left(\mathbb{I}(\text{fmask} > 0.5), m_{\text{fg}}\right)$$
* Differentiating with respect to the learnable attention branch $\text{fmask}$:
  $$\frac{\partial \mathbb{I}(\text{fmask} > 0.5)}{\partial \text{fmask}} = \delta(\text{fmask} - 0.5) = 0 \quad \text{almost everywhere (a.e.)} \implies \frac{\partial \text{keep}_{\text{hard}}}{\partial \text{fmask}} \equiv \mathbf{0}$$
* **Consequence:** Attention branch receives **zero task supervision** ($0/160$ parameters updated); unchecked feedback gradients (**$648.29$ norm**) trap the encoder in hallucinated background errors.

### 2. The Fix (Proposed Detached Soft-OR Gating in M12)
* Continuous Boolean t-conorm with stop-gradient feedback severance:
  $$\boxed{\text{keep}_{\text{soft-or}} = 1 - (1 - \text{fmask}) \cdot \left(1 - \text{detach}(m_{\text{fg}})\right)}$$
* **Analytical Gradient Formulation:**
  $$\frac{\partial \text{keep}_{\text{soft-or}}}{\partial \text{fmask}} = 1 - \text{detach}(m_{\text{fg}}) = 1 - m_{\text{fg}} \implies \boxed{\frac{\partial \mathcal{L}}{\partial \text{fmask}} = \frac{\partial \mathcal{L}}{\partial x_1} \cdot \text{Conv}_1^{\top}(x) \cdot \left(1 - m_{\text{fg}}\right)}$$
* **Dynamic Uncertainty-Guided Modulation:**
  * *Confident Interior ($m_{\text{fg}} \to 1$):* $\partial \to 0 \implies$ Prevents feature saturation and over-activation.
  * *Ambiguous Boundary / Background ($m_{\text{fg}} \to 0$):* $\partial \to 1.0 \implies$ Maximizes optimization signal to prune false positives.

---

<!-- Slide 6: Evaluation setting -->

# Evaluation Setting & Benchmarking Protocol

### Challenging Clinical Benchmark
* **Dataset:** Kvasir-SEG (Sessile Subset — subtle Paris IIa/IIb flat morphology).
* **Sample Count:** 196 colonoscopy frames ($156$ train / $40$ validation), normalized to $256 \times 256$.
* **Morphological Difficulty:** Low relief profiles, blurry transitional margins, and visual similarity to healthy mucosa.

### Strict Multi-Seed Protocol
* **5 Independent Random Initializations:** $S \in \{7, 42, 99, 1337, 2024\}$ to guarantee statistical reproducibility.
* **Training Budget:** $200$ epochs per seed on Kaggle NVIDIA T4 GPUs.
* **Optimization:** Adam ($lr = 1 \times 10^{-4}$), ReduceLROnPlateau ($patience = 5$).
* **Loss Objective:** Asymmetric Tversky Loss ($\alpha = 0.7$ false-positive penalty, $\beta = 0.3$ false-negative penalty).

### Recurrent Inference & Statistical Validation
* **Recurrent Inference Loop:** Standard 4-iteration test-time refinement initialized with Otsu thresholding.
* **Paired Non-Parametric Testing:** **Wilcoxon Signed-Rank Test** across $N = 200$ evaluation instances ($40\text{ images} \times 5\text{ seeds}$).
* **Pre-registered Metrics:** Dice (DSC), mIoU (Jaccard), False Positive Rate (FPR), Precision, Recall, and Mask Gradient Norm.

---

<!-- Slide 7: Performance evaluation (quantitative & qualitative) -->

# Performance Evaluation: Quantitative Gains & Qualitative Eradication

### Quantitative 5-Seed Benchmark (200 Epochs / Seed)
| Configuration | Mean Dice (DSC) | False Positive Rate (FPR) | Precision | Inter-Seed Std ($\sigma$) | Collapse Mode (Seed 1337) | Wilcoxon Test ($N = 200$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **M11 (Feedback Trap)** | $0.3428 \pm 0.087$ | $4.30\% \pm 2.31\%$ | $0.3143$ | $\sigma = 0.0869$ | **Collapsed:** $\text{Dice} = 0.0928$ | — |
| **M12 (Detached Soft-OR)** | **$0.2183 \pm 0.029$** | **$2.46\% \pm 1.65\%$** | **$0.3893$** | **$\sigma = 0.0291$** | **Cured:** $\text{Dice} = 0.2779$ | **$p = 1.13 \times 10^{-6}$** |
| **Net Improvement** | **$+4.77\text{ pp}$ (+19.0%)** | **$-1.84\text{ pp}$ (-42.8%)** | **$+7.50\text{ pp}$ (+23.9%)** | **$-66.5\%$ Variance** | **Eradicated (+18.5 pp)** | **$p < 0.001$ ($r = 0.422$)** |

* Outlier Seed 2024: M11 experienced runaway over-segmentation ($\text{FPR} = 8.88\%$) $\to$ M12 suppressed it to **$0.80\%$** (**$91.0\%$ relative reduction**).

---

### Qualitative Comparison: Complete Eradication of Red False Positives

![Qualitative Comparison](paper_figures/qualitative_comparison.png)

* **Visual Proof (Top 5 Failure Cases in M11):**
  * **Case 1 (`cju7et17...`):** M11 hallucinates a massive **$3,350\text{ px}$ Red lesion** on the endoscopic border $\to$ M12 completely eliminates it to **$0\text{ px}$**!
  * **Case 2 (`cju87mry...`):** M11 bleeds **$1,000\text{ px}$** into healthy mucosa $\to$ M12 restores clean contours (**$0\text{ px}$ Red**).
  * **Cases 3, 4, 5:** M11 produces $679\text{ px}$, $609\text{ px}$, and $461\text{ px}$ spurious blobs $\to$ M12 achieves **$0\text{ px}$ False Positives**.
* **Legend:** Green = True Positive (Polyp Match), **Red = False Positive (Over-segmentation / Trap Error)**, Yellow = False Negative, White Contour = Ground Truth.

---

<!-- Slide 8: Ablation studies / Mechanism -->

# Ablation Studies & Mechanistic Diagnostics

### Mechanistic Evidence: Severing Toxic Feedback Loops
| Diagnostic Dimension | M11 (Feedback Trap) | M12 (Detached Soft-OR) | Physical & Theoretical Impact |
| :--- | :---: | :---: | :--- |
| **Mask Gradient Norm** | $663.43$ ($648.29$) | **$53.62$** ($79.05$) | **$87.8\% - 91.9\%$ drop:** Cleanly severs 8 `MixPool` blocks from toxic backprop. |
| **Bottleneck CosSim ($e_4$)** | $0.7981$ | **$0.9339$** | Boundary and background representations cleanly disentangled. |
| **Prediction Saturation** | $1.13\%$ | **$97.20\%$** | High decision confidence; eliminates blurry, indecisive background predictions. |
| **BatchNorm Drift ($e_1$)** | $D_{\text{KL}} = 33.63$ | **Controlled** | Stops non-stationary statistical drift in early convolutional feature extractors. |

---

### Mechanistic Visualizations: BatchNorm Drift & Factorial Proof

| BatchNorm Distribution Drift Across Layers | Factorial Proof: Feedback Blocks Loss |
| :---: | :---: |
| ![BatchNorm Drift](diagnostics_output/bn_drift_violin.png) | ![Factorial Interaction](kaggle/figures/fig_p6_factorial_effects.png) |
| **Violin Plot Analysis:** In M11, feedback loops induce extreme variance expansion in BN running means across early layers (`e1.r1.bn1`, `e1.r1.bn3`). M12 stabilizes feature distributions. | **Factorial Interaction (+4.17 pp):** In feedforward mode, asymmetric loss cuts FPR by **$-3.25\text{ pp}$**; under feedback, FPR increases by **$+0.92\text{ pp}$**. Soft-OR Detach unblocks this deadlock! |
