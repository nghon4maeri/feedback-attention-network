# FANet: A Feedback Attention Network for Improved Biomedical Image Segmentation

Reproduction + analysis of:

> Tomar, Jha, Riegler, Johansen, Johansen, Rittscher, Halvorsen, Ali.
> *FANet: A Feedback Attention Network for Improved Biomedical Image Segmentation*,
> IEEE TNNLS 2022. [arXiv:2103.17235](https://arxiv.org/abs/2103.17235)

---

## 🌟 Spotlight: Executive Summary of Scientific Contributions & Improvements

**Paper:** Breaking the Feedback Trap: Understanding and Stabilizing Recurrent Feedback Learning in Medical Image Segmentation

### 1. Bước ngoặt Nhận thức (Mechanistic Deep Learning)

Thay vì đề xuất một "attention module" mang tính tiệm tiến (incremental) như các nghiên cứu trước, dự án này tạo ra bước ngoặt khi tiếp cận từ góc độ Mechanistic Deep Learning nhằm giải mã một lỗ hổng cấu trúc cơ bản trong mạng thị giác máy tính:

- **The Feedback Trap (Bẫy hồi quy):** Chúng tôi phát hiện ra rằng trong các mạng recurrent feedback truyền thống, lịch sử dự đoán (prediction history) phải gánh đồng thời 2 vai trò: làm tiền đề không gian (spatial guidance) cho lượt truyền tiến (forward pass), và làm đường dẫn tối ưu (optimization conduit) cho lượt truyền ngược (backward pass). Sự chồng chéo này dẫn đến hiện tượng khuếch đại sai số (error amplification) nghiêm trọng.
- **The Feedback Firewall:** Giải pháp của chúng tôi là nguyên lý Gradient Decoupling thông qua cơ chế Detached Soft-OR, giúp cách ly hoàn toàn dòng chảy gradient hồi quy mà vẫn duy trì được năng lực định hướng không gian của mô hình.

### 2. Bóc tách Cơ chế Cốt lõi (Core Mechanism Ablation)

Để chứng minh tính tất yếu toán học của giải pháp, chúng tôi đã thực hiện bóc tách cơ chế (4-way ablation) để làm rõ vai trò của từng thành phần:

- **Chỉ dùng Soft-OR (Continuous Relaxation):** Gây ra hiện tượng rò rỉ gradient thảm khốc (catastrophic gradient leakage), làm tăng gấp đôi chuẩn norm của sai số truyền ngược.
- **Chỉ dùng Detach (Stop-gradient):** Chặn được rò rỉ, nhưng lại "đóng băng" hoàn toàn khả năng cập nhật attention do hàm chỉ thị (indicator function) có đạo hàm bằng 0.
- **Detached Soft-OR (M12 - Đề xuất):** Là điểm giao thoa tối ưu hiệp đồng (synergistic optimal). Sự kết hợp này bảo toàn dòng cập nhật đạo hàm trơn cho attention (được dẫn hướng bởi độ bất định - uncertainty), đồng thời dựng lên một "bức tường lửa" chặn đứng hoàn toàn gradient rò rỉ qua các chu kỳ thời gian.

### 3. Phân tích Yếu tố Kép (Factorial Analysis: Loss Unleashed)

Qua thiết kế thực nghiệm 3x2 Factorial Design, chúng tôi đã giải mã được tương tác ẩn giữa Kiến trúc mạng (Architecture) và Hàm mất mát (Loss function):

- **Sự vô hiệu hóa (Neutralization):** Mạng Hard Feedback truyền thống tạo ra luồng gradient rò rỉ quá lớn, hoàn toàn lấn át và vô hiệu hóa áp lực trừng phạt dương tính giả của hàm Asymmetric Tversky Loss.
- **Giải phóng sức mạnh (Loss Unleashed):** Bằng cách áp dụng Gradient Decoupling, mô hình của chúng tôi đã giải phóng hoàn toàn hàm mất mát. Bộ mã hóa (encoder) bị ép phải hấp thụ trực tiếp hình phạt của hàm Tversky, dẫn đến tỷ lệ Dương tính giả (False Positive Rate - FPR) giảm mạnh mang tính đột phá (giảm tới hơn 40% tương đối).

### 4. Tính Bền vững & Khả năng Tổng quát hóa (Robustness & Generalization)

Mô hình đã trải qua các bài kiểm tra áp lực (stress-testing) khắc nghiệt nhất để khẳng định độ tin cậy chuẩn lâm sàng:

- **Độ ổn định đa hạt giống (Multi-seed Stability):** Đạt được sự sụt giảm đáng kể về phương sai (variance reduction), chứng minh mô hình hội tụ mượt mà và không phụ thuộc vào khởi tạo ngẫu nhiên.
- **Khả năng Tổng quát hóa Zero-shot:** Thể hiện sức mạnh vượt trội khi kiểm thử chéo trung tâm (cross-center) trên tập CVC-ClinicDB (N = 612), đạt mức ý nghĩa thống kê cực kỳ thuyết phục ($p = 1.61 \times 10^{-4}$).
- **Ưu thế tuyệt đối trên ca bệnh khó (Adversarial Flat Lesions):** Đánh bại hoàn toàn các mô hình SOTA feedforward tiêu chuẩn (U-Net, DeepLabV3+) khi đối mặt với tập dữ liệu Sessile (polyp dẹt, bờ mờ, ranh giới hòa lẫn niêm mạc), khẳng định giá trị thực tiễn cao của cơ chế định hướng phản hồi.

*Bản tóm tắt này vạch rõ một lộ trình lý luận chặt chẽ: từ việc phát hiện lỗi hệ thống -> đề xuất cơ chế giải quyết -> chứng minh toán học -> đến hiệu năng vượt trội.*

---

## Repository structure

```
FANet/
├── configs/                 # experiment configs (yaml)
│   └── kvasir_sessile.yaml
├── src/fanet/               # installable package
│   ├── models/              #   blocks (SELayer, ResidualBlock, MixPool) + FANet
│   ├── data/                #   DATASET, load_data, RLE feedback decoding
│   ├── losses.py            #   DiceLoss, DiceBCELoss
│   ├── metrics.py           #   Dice/IoU/recall/... evaluation metrics
│   ├── config.py            #   yaml config loader
│   └── utils.py             #   seeding, RLE, Otsu init-mask, logging
├── scripts/
│   ├── train.py             # FANet training with cross-epoch feedback masks
│   ├── evaluate.py          # test-time iterative refinement + metrics
│   └── report.py            # params/dataset report + refinement demo
├── analysis/
│   └── grad_flow.py         # gradient-flow check for MixPool's fmask branch
├── notebooks/
│   └── fanet_kaggle.py      # self-contained Kaggle notebook version
├── docs/                    # papers + FANet_Complete_Guide.md + reports/
├── papers/                  # research PDFs (original/ + translated/)
├── tools/
│   ├── pdf_translate.py     # PDF→Vietnamese translation wrapper (VI-Translate)
│   ├── vitranslate/         # VI-Translate submodule (runtime engine)
│   └── tests/               # integration smoke tests
├── data/                    # datasets (git-ignored, download separately)
├── assets/                  # architecture / qualitative figures
├── checkpoints/             # model weights (git-ignored)
├── logs/                    # training & analysis logs
└── results/                 # evaluation outputs (git-ignored)
```

## Setup

```bash
pip install -e .            # installs the fanet package (src/fanet)
```

Requirements: torch, numpy, opencv-python, albumentations, scikit-learn, tqdm, pyyaml.

## Data

Place datasets under `data/`:

```
data/
├── Kvasir-SEG/                # full Kvasir-SEG (1000 images)
└── sessile-main-Kvasir-SEG/   # sessile subset (196 images) with train.txt/val.txt
```

Update `dataset.root` in `configs/kvasir_sessile.yaml` if needed.

## Usage

Train (feedback masks from previous epochs, checkpoint saved on val-loss improvement):

```bash
python scripts/train.py --config configs/kvasir_sessile.yaml
```

Evaluate with iterative test-time refinement:

```bash
python scripts/evaluate.py --config configs/kvasir_sessile.yaml \
    --checkpoint checkpoints/checkpoint.pth --num-iter 10 --save-masks
```

Gradient-flow analysis (checks whether the hard binary gate in MixPool cuts
gradient to the learned fmask branch):

```bash
python analysis/grad_flow.py --config configs/kvasir_sessile.yaml
```

## PDF translation (VI-Translate)

Translate research PDFs into Vietnamese while preserving layout, formulas and
figures. Inputs live under `papers/original/`, output under `papers/translated/`.

```bash
git submodule update --init --recursive     # one-time: fetch VI-Translate
python tools/pdf_translate.py --setup        # one-time: create isolated runtime
python tools/pdf_translate.py papers/original/attention.pdf
# -> papers/translated/attention-vi.pdf
```

Google engine is the default (free, no API key, needs network). See
[`docs/pdf_translate_guide.md`](docs/pdf_translate_guide.md) for options,
engines, OCR, troubleshooting, and the OpenCode skill.

## Key findings so far

- MixPool's learned mask branch `fmask` receives **zero gradient**: the
  `(fmask > 0.5)` binarization is non-differentiable, so its conv weights
  never update (verified in `logs/analysis_grad_log.txt`). Only its
  BatchNorm running stats drift via the forward pass.
- The feedback mask itself is a hard binary mask (prediction thresholded at
  0.5 + RLE round-trip), discarding confidence/uncertainty information of
  the previous prediction.

## Citation

```bibtex
@article{tomar2022fanet,
  title={Fanet: A feedback attention network for improved biomedical image segmentation},
  author={Tomar, Nikhil Kumar and Jha, Debesh and Riegler, Michael A and Johansen, H{\aa}vard D and Johansen, Dag and Rittscher, Jens and Halvorsen, P{\aa}l and Ali, Sharib},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022}
}
```
