# Prompt Session — 2026-10-26: GDF-Netv4 300-Epoch Heavy-Augmentation Dice Optimization

## 1. User Request
- "bây giờ lấy con model có dice cao nhất xong rồi aug data thêm nhiều case hơn và train cho tôi 300 epoch để tối ưu dice"

## 2. Objective & Scope
- Take the highest-Dice model in the repository: **`GDF-Netv4`** (`PVTv2-B2` + `MS-FDF [1,3,5]` + `Mask-Guided LGAG` + `Cascaded MSCAM Decoder`, 30.68M params, $384 \times 384$, current SOTA on 1,000 ISIC-2018 Blind Test images with **Mean Dice `0.9001` Single-View / `0.9030` 4-View TTA**).
- Warm-start from `GDFNetV4_ISIC2018_best.pth` (originally saved at Epoch 8 with `Val Macro Dice = 0.9019` on the 100 Official ISIC-2018 Validation images) and enable `drop_path_rate=0.2` in `pvt_v2_b2` + `ModelEMA` (`decay=0.9995`) to prevent Transformer overfitting across long 300-epoch schedules.
- Expand the training data augmentation pipeline to cover 14 clinical dermoscopic augmentation cases:
  1. `RandomResizedCrop(384, 384, scale=(0.5, 1.0))`
  2. `HorizontalFlip(p=0.5)` & `VerticalFlip(p=0.5)` & `RandomRotate90(p=0.5)`
  3. Full-orientation `ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=180, p=0.7)`
  4. Non-rigid skin distortion: `OneOf([OpticalDistortion, GridDistortion, ElasticTransform], p=0.35)`
  5. Color & illumination: `ColorJitter`, `HueSaturationValue`, `RandomGamma`
  6. Low-contrast lesion enhancement: `CLAHE(clip_limit=2.5, p=0.3)`
  7. Sensor blur/noise: `OneOf([GaussianBlur, GaussNoise, Sharpen], p=0.25)`
  8. Occlusion completion: `CoarseDropout(p=0.25)`
  9. Calibrated `RandomVignettingTransform(p=0.25)` (dark lens corners)
  10. Multi-color `SyntheticHairTransform(p=0.30, max_hairs=5)` (dark & light hairs)
  11. `SyntheticRulerMarkTransform(p=0.15)` (dermoscope millimeter scale ticks)
  12. `SpecularGlareTransform(p=0.15)` (immersion gel reflection spots)
- Upgrade the training objective to directly maximize per-image Macro Dice: `StructureLoss (wBCE + wIoU) + 0.5 * SoftMacroDiceLoss` across all deep supervision scales.
- Train for **300 epochs** using `CosineAnnealingWarmRestarts(T_0=50, T_mult=2, eta_min=1e-6)` on 100% of the Official Training Set (`2,594` images), selecting the best checkpoint on the **100 Official ISIC-2018 Validation images**, followed by Official 1,000 Blind Test Evaluation (Single-View, 4-View TTA, and Val-Calibrated Threshold 4-View TTA).

## 3. Constraints
- Adhere strictly to `AGENTS.md` (commit and push after every code/doc update; distinguish `TRAIN` vs `FROZEN EVAL` vs `INVALID`).
- Include an 11-hour wall-clock safety guard (`39,600s`) in the Kaggle kernel so that if 300 epochs approaches Kaggle's 12-hour session timeout, the training loop exits cleanly and executes the full 1,000 Blind Test evaluation without losing outputs.
