# Prompt Session — 2026-10-27: Colab A100 GDF-Netv4 Execution & Kaggle 4 Pure-CNN Pilot Screening

## 1. User Requests
- "Chế độ Đánh giá trên 1.000 ảnh... với bảng này thì kết quả của tôi là cái nào và (mặc định) khác gì calibrated trên val và tôi nên báo cáo sao với gvhd"
- "giờ thảo luận về các ứng viên backbone dùng thuần cnn đi"
- "viết ra 4 file lấy 4 backbone cnn rồi quăng lên kaggle chạy cho tôi và trước tiên chạy vài epoch thôi để tính xem có nên chạy tiếp ko, bạn thấy thế nào"
- "bạn dùng colab cli để tự khởi động session train GDFNet_v4_300ep_HeavyAug_Colab.ipynb đi dùng gpu A100 của tôi và train trong 3 tiếng trở lại thôi lâu quá thì phải thêm cơ chế resume nhé"
- "Bản v4 Transformer 300 epochs bản này ko cần chạy trên kaggle đâu vì đã có colab rồi mà hãy chạy các bản dùng backbone là cnn thuần ấy để xem kết quả ra sao"
- "tại sao GDFNet-v4 300Ep HeavyAug ISIC2018 vẫn còn session trên kaggle vậy tôi kêu stop đi rồi mà và chạy các model dùng ứng viên backbone cnn ?"

## 2. Objectives & Scope
1. **Google Colab A100 Autonomous Execution & Verification:**
   - Execute GDF-Netv4 on NVIDIA A100-SXM4-40GB under a strict 3-hour wall-clock limit (<10,200s).
   - Reassemble chunked checkpoint on remote VM, implement resume mechanism, train 244 epochs at ~28.8s/epoch.
   - Benchmark best EMA checkpoint on 1,000 Official Blind Test images.
   - Safely stop Colab session and verify zero credit burn (preserved 22.01 CU).
2. **Kaggle Pure-CNN Pilot Screening (4 Models, 25 Epochs from Scratch):**
   - Candidate 1: `ConvNeXt-V2-Nano` (22.13M params backbone, 31.82M total) -> Mean Dice 0.9031.
   - Candidate 2: `ConvNeXt-V2-Pico` (9.1M params backbone, 13.75M total, -55% params) -> Fastest 72.1s/ep, Median Dice 0.9434, Median HD95 12.08 px.
   - Candidate 3: `ConvFormer-S18` (26.8M params backbone, 29.50M total, exact [64, 128, 320, 512] channels) -> **Mean Dice 0.9102**, Median Dice 0.9410, Precision 0.9008, ASD 8.16 px (Winner!).
   - Candidate 4: `InceptionNeXt-Tiny` (23.97M params backbone, 33.42M total) -> Mean Dice 0.8991, Precision 0.9066, Specificity 0.9635.
3. **Cross-Model Synthesis & Guidance for Supervisor:**
   - Synthesize comprehensive leaderboard comparing Transformer vs 4 Pure-CNN backbones.
   - Provide concrete reporting advice and recommendations for the supervisor.

## 3. Constraints & Conventions
- Strictly follow `AGENTS.md`: commit and push all results and reports to GitHub.
- Keep Colab runtime strictly within 3 hours.
- Evidence levels: `TRAIN` (verified blind test benchmarks) vs `FROZEN EVAL` vs `INVALID`.
