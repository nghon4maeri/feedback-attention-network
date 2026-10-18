# GDF-Netv2 & Medical Image Segmentation Research

This repository contains research and implementation for next-generation medical image segmentation architectures, focusing on continuous architectural innovation and open-ended performance maximization across standard medical segmentation benchmarks.

## Repository Structure

```
.
|-- GDF-Netv2/             # Active Research: GDF-Netv2 Architecture & Benchmarking
|   |-- models/            # ConvNeXt-V2 backbone, FDF, SBA, MSF
|   |-- losses/            # Focal-Tversky, Boundary Difference over Union (BD-IoU)
|   |-- data/              # 384x384 augmentation pipelines
|   |-- training/          # Multi-metric evaluators & Colab/Kaggle runners
|   `-- kaggle/            # Metadata and execution specs
|
|-- archive/               # Preserved Legacy Research (Phase 1 to Phase 7)
|   |-- src/               # Legacy FANet architecture & modules
|   |-- notebooks/         # Historical experiments & benchmarks
|   |-- docs/              # Research reports, literature grounding, and logs
|   |-- papers/            # Drafts and submission assets
|   `-- results/           # Benchmark outputs and metric summaries
|
|-- tools/                 # Research agent utilities and VI-Translate integration
|-- .agents/               # Agent configuration and research conventions
|-- AGENTS.md              # Working rules, research protocols, and git conventions
`-- opencode.json          # Scientific research skills mapping
```

## Active Direction: GDF-Netv2

GDF-Netv2 addresses boundary collapse and outlier failure modes observed in prior architectures via:
1. **Hierarchical Modern Backbone**: ConvNeXt-V2-Tiny preserving fine spatial resolution ($H/4$ at Stage 1) with $7 \times 7$ depthwise separable convolutions and Global Response Normalization (GRN).
2. **Frequency-Decoupled Filter (FDF)**: High/low-frequency separation on skip connections to suppress dermoscopic artifacts (hair, ruler marks, illumination glare).
3. **Selective Boundary Aggregation (SBA)**: Cross-attention boundary stream dynamically fusing edge representations into deep semantic stages.
4. **Multi-Channel Semantic Feedback (MSF)**: Recurrent semantic guidance with a detached gradient firewall to prevent gradient contamination.
5. **Composite Boundary Loss**: Focal-Tversky + Boundary Difference over Union (BD-IoU) Loss explicitly penalizing boundary band errors.

See [`GDF-Netv2/README.md`](GDF-Netv2/README.md) for complete details.
