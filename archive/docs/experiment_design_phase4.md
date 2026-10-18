# Thiết kế thí nghiệm Phase 4 — ablation train 4 models (31/08/2026)

## Các cell

| Cell | gate | dual_path | Mô tả | Checkpoint file |
|---|---|---|---|---|
| T00 | binary | False | Baseline FANet (train lại, chuẩn so sánh) | ckpt_t00_binary_single.pth |
| T10 | ste | False | fmask-STE (test T_A) | ckpt_t10_ste_single.pth |
| T01 | binary | True | dual-path (test T_B1/T_B2/T_BN) | ckpt_t01_binary_dual.pth |
| T11 | ste | True | ste+dual (test T_AB) | ckpt_t11_ste_dual.pth |

Fallback (nếu T_B1 bác bỏ): T_F1 negative-area Dice (binary-1kênh), T_F2 decoder-only suppression — chạy sau, không chạy cùng đợt.

## Điều kiện khớp CHÍNH XÁC giữa 4 cell

| Điều kiện | Giá trị | Ghi chú |
|---|---|---|
| Dataset | data/sessile-main-Kvasir-SEG | 156 train / 40 val, cùng split (train.txt/val.txt có sẵn) |
| Image size | 256x256 | |
| Batch size | **2** | Khớp local (trước đây Kaggle dùng 4 — sửa) |
| Augmentation | Rotate ±35 p=0.3, HFlip p=0.3, VFlip p=0.3, **CoarseDropout p=0.3 num_holes=10 hole_h=32 hole_w=32** | Sửa args theo API albumentations mới (max_holes→num_holes etc.) |
| Seed | 42 | seeding() |
| Optimizer | Adam lr=1e-4 | |
| Scheduler | ReduceLROnPlateau patience=5 | |
| Epochs | 200 | 1 run sanity đầu tiên (loss giảm + val metrics), rồi multi-seed ≥3 cho cell thắng |
| Loss | DiceBCELoss (0.5 BCE + 0.5 Dice) | Giữ nguyên cho 4 cell chính |
| Init mask | Otsu (init_mask) | |
| Feedback loop | mask epoch trước (RLE), cập nhật khi val loss cải thiện | Như train.py |

## Metric

**Chính**: Dice + IoU tại iter cuối (4 iters refinement) trên 40 ảnh val + mean qua iters.
**Phụ**: Precision, Recall, FP rate, FN rate, FP/FN distance-to-boundary, oracle gap, help/hurt count.
**Kiểm tra BN**: activation distribution (mean/std) của conv1 output tại từng MixPool trước/sau khi đổi gate; BN running stats của e1.p1.fmask.1 và conv1.1 trên checkpoint trained vs frozen.

## Phân tích thống kê

- Paired per-image (n=40): Wilcoxon signed-rank (2-tailed), Cohen's d (paired), bootstrap 95% CI (2000 resamples).
- Bonferroni: 6 so sánh chính (T10/T01/T11 vs T00 + T11 vs T10 + T11 vs T01) → α = 0.05/6 ≈ 0.0083.
- Báo cáo cả magnitude (đã khai báo ngưỡng +0.005) lẫn significance.

## Quy trình chạy (Kaggle T4)

1. Push data (đã có trên Kaggle: kvasir-sessile dataset) + upload code src/fanet (đảm bảo cùng version).
2. Chạy notebook train 4 cell (1 run mỗi cell, seed 42) → save checkpoint per cell.
3. Sanity check: loss giảm + val Dice trên val sau khi train xong mỗi cell.
4. Đánh giá: nạp từng checkpoint trained, chạy abl2x2_eval-style trên CÙNG 40 ảnh val (4 iters) + full test 10 iters.
5. Thống kê + figures + report.
6. Nếu T_B1 bác bỏ → chạy fallback F1/F2.

## Confounders kiểm soát

- Khớp split/seed/batch/aug giữa 4 cell (bảng trên).
- Cùng đợt chạy, cùng version code, cùng data copy.
- Không dùng augmentation khác nhau giữa cell.
- BN momentum mặc định (0.1) — giữ nguyên.

## Giới hạn

- 1 seed trong phase này → pilot; multi-seed là điều kiện claim paper.
- Kết quả gắn với dataset sessile (196 ảnh) — đa dataset là Will Do.
- Thời gian: 200ep x ~12s/ep (T4) ≈ 40 phút/cell; 4 cell ≈ 2.5-3h + eval.
