# Prompt Session — 2026-10-28: GDF-Netv4-PureCNN (ConvFormer-S18) 300-Epoch Colab A100 Execution & Blind Test Verification

## 1. User Requests
- "mình đang train from scratch đúng ko vậy trước đó trên kaggle thì bằng chứng nào chỉ ra được model này cho ra mean dice > 0.91"
- "check thường xuyên cho tôi"
- "hiện tại có đang chạy ko"
- "Bắt đầu"
- "có thêm data aug ko đấy"
- "tôi sẽ treo máy để bạn chạy nhé"

## 2. Objectives & Scope
1. **Clarify Methodology & Evidence:**
   - Explained distinction: Task-level training (ISIC-2018 segmentation heads, decoders, attention gates) is 100% from scratch; Backbone feature extractor uses ImageNet-22K pretrained initialization (`convformer_s18.sail_in22k_ft_in1k_384`).
   - Detailed exact Kaggle pilot benchmark evidence on 1,000 Official Blind Test images (Mean Dice 0.9102, Median 0.9410, Precision 0.9008, ASD 8.16 px).
2. **Data Augmentation Verification:**
   - Verified active 15-method Heavy Augmentation Suite across 3 tiers (Spatial & Non-Rigid, 5 Clinical Dermoscopy Artifact Synthesis, Photometric & Noise).
3. **Autonomous Execution on Google Colab A100 (`a100-s18-run`):**
   - Active, automated keep-alive monitoring cycle every 180 seconds to prevent Colab idle connection reaper timeout.
   - 300 epochs completed in 2h25m (~27.2s/epoch, final loss 0.7639).
   - Checkpoint selected: Epoch 18 (Val Macro Dice 0.9086, ModelEMA).
4. **Official 1,000 Blind Test Benchmark:**
   - Single-View: Mean Dice **0.9017**, Median **0.9316**, Sensitivity **0.9455**, Specificity **0.9515**, ASD **8.84 px**.
   - 4-View Flip TTA: Mean Dice **0.9016**, Median **0.9316**, mIoU **0.8334**, ASD **8.86 px**.
   - Calibrated 4-View Flip TTA ($T_{\text{opt}} = 0.750$ on 100 Val):
     * **Mean Dice:** **0.9078**
     * **Median Dice:** **0.9431** (All-time project record!)
     * **Precision:** **0.9226** (+3.87 pp, >92%)
     * **Specificity:** **0.9712** (FPR 0.0278)
     * **ASD:** **8.48 px** (Median **4.55 px**)
     * **$J_{\text{thresh}}$:** **0.8040** (Exceeds Human Consensus 0.786)
     * **Failed cases ($J_i < 0.65$):** Only 89 / 1,000.
5. **Artifact Retention & Safe Shutdown:**
   - Bundled and downloaded `GDFNetV4_ConvFormerS18_300ep_best.pth` (118.3 MB), raw/summary CSVs, training log, history to local directory `GDF-Netv4/results/colab_a100_convformer_s18_300ep/`.
   - Safely executed `colab stop -s a100-s18-run`, preserving 3.97 Compute Units with zero burn rate.
   - Compiled Phase 16 scientific report and updated index in `docs/reports/README.md`.

## 3. Constraints & Conventions
- Strictly followed `AGENTS.md`: commit and push all code and reports to GitHub immediately.
- No emojis, concise reporting, clickable file links.
- Evidence level: `TRAIN` (verified 1,000 blind test official ground truth).
