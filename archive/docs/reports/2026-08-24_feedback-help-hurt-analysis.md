# Báo cáo tuần - Thứ 2, 24/08/2026

## Mục tiêu tuần này

Hoàn thành phân tích feedback help/hurt: xác định khi nào hard binary feedback giúp và khi nào làm prediction tệ hơn, kiểm tra mức độ tương quan giữa uncertainty (confidence trước threshold) và các lỗi đó. Kết hợp với kết quả phân tích gradient flow đã có, chốt lại các điểm yếu thực sự của cơ chế feedback trong FANet làm cơ sở cho việc đề xuất cơ chế mới.

## Done

- Chạy phân tích help/hurt trên 40 ảnh val với 4 feedback variants (binary / soft / none / oracle), mỗi variant 4 iterations. Script: analysis/feedback_analysis.py, log: logs/feedback_analysis_log.txt
- Phân tích help/hurt theo từng cặp (image, iteration): binary so với none, kèm đặc trưng của mask feedback ở iteration trước (dice, error rate, mean uncertainty)
- Phân tích error rate theo 10 bins của confidence trước threshold, tách riêng FP và FN
- Diễn giải lại kết quả gradient flow: xác nhận fmask chưa từng được học bằng bằng chứng BN affine diff bằng đúng 0 (BN khởi tạo deterministic 1/0), tránh nhầm lẫn với nhiễu khởi tạo ở conv weights
- Tổng hợp hai điểm yếu đã được xác nhận bằng dữ liệu và phác thảo hướng cơ chế feedback mới

## Findings quan trọng

### 1. Hard threshold cắt gradient hoàn toàn - fmask không bao giờ học

Nhánh fmask của MixPool dùng phép so sánh (self.fmask(x) > 0.5) nên đạo hàm bằng 0 tại mọi điểm. Phân tích gradient flow ghi nhận sau 10 training steps: 0/160 tham số fmask có gradient, trong khi conv1/conv2 đủ 160/160. Weight fmask không thay đổi sau optimizer step. So với model khởi tạo, BN affine (gamma/beta khởi tạo đúng 1/0) của fmask lệch đúng 0.000 - bằng chứng quyết định rằng nhánh này chưa từng được update trong toàn bộ quá trình train 53 epochs. Nói cách khác, "feature-based attention" mà paper quảng cáo thực chất là một bộ lọc random-init, toàn bộ sức mạnh của MixPool đến từ feedback mask.

### 2. Hard threshold vứt confidence - vùng không chắc chắn bị khuếch đại lỗi

Error rate của prediction đã binarize giảm monotonic theo confidence: 48.7% ở bin confidence thấp hơn 0.1 (tương đương đoán mò) xuống 2.7% ở bin confidence cao hơn 0.35. Threshold 0.5 đối xử pixel có xác suất sai 48.7% ngang hàng với pixel sai 2.7%, đưa cả hai vào feedback mask với trọng số bằng nhau. FP tập trung chủ yếu ở vùng confidence thấp (35.3% ở bin thấp nhất so với 0.0% ở bin cao nhất), FN phân bố đều hơn.

### 3. Feedback giúp khi mask trước tốt, hại khi mask trước kém

Trong 120 cặp (image, iteration), binary feedback giúp 65, hại 46, trung lập 9, mean delta +0.031 so với không feedback. Tương quan Pearson giữa delta và chất lượng mask trước là +0.355; giữa delta và mean uncertainty của prediction trước là -0.297. Chia theo trung vị dice của mask trước (0.287): nhóm mask tốt có delta +0.050, nhóm mask kém chỉ +0.011. Feedback không đáng tin đúng khi cần nó nhất.

### 4. So sánh variants qua iterative refinement

| Iteration | Binary | Soft | None | Oracle |
|---|---|---|---|---|
| 1 | 0.315 | 0.315 | 0.315 | 0.315 |
| 2 | 0.327 | 0.311 | 0.301 | 0.370 |
| 3 | 0.333 | 0.311 | 0.301 | 0.370 |
| 4 | 0.334 | 0.311 | 0.301 | 0.370 |

Binary feedback cải thiện qua từng iteration và vượt soft (0.334 so với 0.311); soft giữ được confidence nhưng không đủ mạnh để tinh chỉnh. Khoảng cách tới oracle (0.370) cho thấy còn dư địa 0.036 - đây là upper bound tham chiếu cho cơ chế mới.

## Experiments

| Giả thuyết | Config | Kết quả | Log |
|---|---|---|---|
| Gate binary cắt gradient, fmask không học | 10 steps từ checkpoint 53 epochs | Xác nhận: 0 grad fmask / 160 grad conv; BN affine diff = 0.000; weight không đổi sau optimizer step | logs/analysis_grad_log.txt |
| Test-time refinement có cải thiện | 2 iters, 40 ảnh val | Jaccard 0.2166 sang 0.2251, F1 0.3147 sang 0.3268 | results/test_results.csv |
| Khi nào binary feedback giúp/hại; uncertainty correlate với lỗi? | 4 variants x 4 iters x 40 ảnh | 65 giúp / 46 hại; corr(prev_dice)=+0.355, corr(unc)=-0.297; err 48.7% (conf < 0.1) vs 2.7% (conf > 0.35) | logs/feedback_analysis_log.txt |

## Will Do (On going)

- Chạy lại evaluate.py đủ 10 iterations để có baseline chuẩn trước khi so sánh
- Confidence-gated feedback (trọng tâm): thay hard mask bằng feedback có trọng số theo confidence, giảm hoặc loại bỏ vùng có confidence thấp (|p - 0.5| nhỏ)
- Hồi sinh fmask (bổ trợ): straight-through estimator hoặc soft gating để gradient chảy được vào nhánh fmask
- So sánh các variants mới trên cùng 40 ảnh val: mean Dice qua iterations, tỷ lệ help/hurt, khoảng cách tới oracle bound
- Chưa mở rộng sang background context và deep supervision - giữ trọng tâm hai điểm yếu đã xác nhận

## Any Stuck / Open Questions

- Lưu ý diễn giải gradient flow: diff 0.04-0.35 ở conv weights fmask là nhiễu khởi tạo khi so với model fresh, không phải bằng chứng học. BN affine diff = 0.000 mới là bằng chứng quyết định.
- Oracle bound 0.370 so với binary 0.334: cần tách phần dư địa thuộc về chất lượng mask feedback và phần thuộc về capacity của model - dự kiến làm rõ khi thử nghiệm cơ chế mới.

## Đính kèm link chi tiết

- Script feedback analysis: analysis/feedback_analysis.py
- Script gradient flow: analysis/grad_flow.py
- Log feedback analysis: logs/feedback_analysis_log.txt
- Log gradient flow: logs/analysis_grad_log.txt
- Data feedback analysis: results/feedback_pairs.csv, results/feedback_uncertainty_bins.csv, results/feedback_summary.json
- Baseline eval: results/test_results.csv
- Checkpoint: checkpoints/checkpoint.pth
