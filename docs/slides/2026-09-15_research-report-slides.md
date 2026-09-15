# Slide Deck: Breaking the Feedback Trap in FANet: A Detached Soft-OR Gating Mechanism

**Format:** Slide Deck Text Content & Visual Guidelines  
**Audience:** Academic Conference (MICCAI / CVPR / IEEE TMI) & Advisory Board  
**Target Duration:** 15–20 minutes  

---

### Slide 1: Title & Info

- **Tiêu đề Slide:** **Breaking the Feedback Trap in FANet: A Detached Soft-OR Gating Mechanism**
- **Nội dung chính:**
  - **Tác giả & Đơn vị:** FANet Research Team — Medical AI & Computer Vision Lab
  - **Bối cảnh nghiên cứu:** Phân đoạn tổn thương dạng phẳng (Sessile Colorectal Polyps) trên hình ảnh nội soi đại tràng
  - **Vấn đề trọng tâm:** Giải mã và phá vỡ hội chứng "Feedback Trap" gây quá mức phân đoạn (Over-segmentation) trong mạng hồi quy
  - **Thông điệp cốt lõi:** Cơ chế cổng mềm tách gradient (Detached Soft-OR) — Một giải pháp kiến trúc Zero-Parameter phục hồi dòng gradient và ổn định hóa hội tụ
- **Bảng biểu / Hình ảnh gợi ý:**
  - Hình minh họa tối giản: Icon nội soi đại tràng -> Sơ đồ khối FANet lặp (Recurrent Loop) với biểu tượng ổ khóa bị phá vỡ trên đường feedback gradient.

---

### Slide 2: Abstract (Executive Summary)

- **Tiêu đề Slide:** **Abstract: Tóm Tắt Nghiên Cứu**
- **Nội dung chính:**
  - **Bối cảnh:** Mạng hồi quy (FANet) hứa hẹn tinh chỉnh lặp phân đoạn polyp thông qua module `MixPool` tái tiêm mask từ epoch trước.
  - **Vấn đề (Feedback Trap):** Cổng nhị phân cứng (`Hard Binary Gating`) triệt tiêu gradient nhánh attention ($\partial \text{keep}/\partial \text{fmask} = 0$) và bóp nghẹt 79% gradient, gây over-segmentation nghiêm trọng và vô hiệu hóa hàm Asymmetric Loss.
  - **Giải pháp (M12):** Đề xuất **Detached Soft-OR Gating**: kết hợp hợp logic mờ liên tục với toán tử ngắt gradient `detach(m_fg)`, điều biến dòng học theo độ bất định ($1 - m_{\text{fg}}$).
  - **Kết quả vượt trội:** Giảm **42.8% FPR** ($4.30\% \to 2.46\%$, $p < 0.001$), tăng **+4.77 pp Dice** ($25.17\% \to 29.95\%$), tăng **+7.5 pp Precision**, giảm **66.5% phương sai**, cứu mạng hoàn toàn khỏi sụp đổ biểu diễn.
- **Bảng biểu / Hình ảnh gợi ý:**
  - Sơ đồ tóm tắt 4 pha: [Problem: Hard Gate Gradient Deadlock] -> [Theory: Derivative 1 - m_fg] -> [Ablation: 5 Seeds Benchmarking] -> [Impact: Clean Surgical Margin].

---

### Slide 3: Key Contributions

- **Tiêu đề Slide:** **Đóng Góp Khoa Học Cốt Lõi (Key Contributions)**
- **Nội dung chính:**
  - **Đột phá hiệu năng (Performance):** Cắt giảm **42.8% tỷ lệ vẽ thừa** (FPR giảm từ 4.30% xuống 2.46%), nâng cao Precision từ 31.4% lên 38.9% (+23.9% relative).
  - **Đóng góp lý thuyết (Novelty):** Lần đầu tiên phát hiện và chứng minh giải tích cơ chế tự hủy gradient của Hard Gating; thiết lập công thức Soft-OR với đạo hàm điều biến tự nhiên $\frac{\partial \text{keep}}{\partial \text{fmask}} = 1 - m_{\text{fg}}$.
  - **Thiết kế không tham số (Zero-Parameter Fix):** Không gia tăng chi phí tham số tính toán ($\Delta \text{Params} = 0$, $\Delta \text{FLOPs} < 0.1\%$), tương thích tuyệt đối với pre-trained weights.
  - **Độ bền vững & Giải phóng Loss (Impact):** Giảm **66.5% phương sai giữa các seeds**, triệt tiêu nguy cơ sụp đổ biểu diễn (Seed 1337) và kích hoạt trọn vẹn hiệu lực của Asymmetric Tversky Loss.
- **Bảng biểu / Hình ảnh gợi ý:**
  - Infographic 3 trụ cột: (1) Metric Gains (+4.77pp Dice, -42.8% FPR) | (2) Analytical Proof ($\partial = 1 - m$) | (3) Zero Parameter Overhead (0 Params added).

---

### Slide 4: Key Gaps in Related Work & Main Idea

- **Tiêu đề Slide:** **Khoảng Trống Nghiên Cứu & Ý Tưởng Cốt Lõi**
- **Nội dung chính:**
  - **Khoảng trống 1 (Heuristic Feedback Assumption):** Các mạng hồi quy y tế (FANet, Recurrent U-Net) ngây thơ giả định rằng cứ đưa mask cũ vào là mạng sẽ tự tinh chỉnh, bỏ qua hoàn toàn động lực học gradient của giao diện feedback.
  - **Khoảng trống 2 (Loss Engineering Deadlock):** Cộng đồng nỗ lực thiết kế các hàm loss bất đối xứng phức tạp (Tversky, Focal Tversky) để phạt FP, nhưng bị Feedback Trap vô hiệu hóa triệt để (Interaction $+4.17\text{ pp}$).
  - **Ý tưởng cốt lõi (Gradient Firewall):** *Dẫn đường không gian ở lượt forward phải được tách biệt tuyệt đối khỏi đồ thị lan truyền ngược backward.*
  - **Cơ chế Soft-OR:** Chuyển đổi giao diện quyết định từ nhảy bậc gián đoạn sang xác suất liên tục để duy trì gradient thông suốt.
- **Bảng biểu / Hình ảnh gợi ý:**
  - Sơ đồ so sánh hai trường phái: Lối mòn cũ (Loss Engineering bị chặn bởi Feedback Trap) vs Hướng tiếp cận của chúng ta (Architectural Firewall giải phóng tiềm năng của Asymmetric Loss).

---

### Slide 5: Methodology — The Flaw vs. The Fix

- **Tiêu đề Slide:** **Phương Pháp: Phân Tích Khuyết Tật & Cơ Chế Đề Xuất**
- **Nội dung chính:**
  - **Khuyết tật gốc (The Flaw - M11):** 
    $$\text{keep}_{\text{hard}} = \max(\mathbb{I}(\text{fmask} > 0.5), m_{\text{fg}}) \implies \frac{\partial \text{keep}}{\partial \text{fmask}} = \mathbf{0} \quad \text{(hầu khắp nơi)}$$
    $\to$ Nhánh học attention bị tê liệt; gradient độc hại tràn qua $m_{\text{fg}}$ ép mạng học theo ảo giác của chính nó.
  - **Bản vá đề xuất (The Fix - M12):**
    $$\boxed{\text{keep}_{\text{soft-or}} = 1 - (1 - \text{fmask}) \cdot (1 - \text{detach}(m_{\text{fg}}))}$$
  - **Đạo hàm điều biến bất định (Uncertainty-Modulated Gradient):**
    $$\frac{\partial \mathcal{L}}{\partial \text{fmask}} \propto 1 - m_{\text{fg}}$$
    - *Vùng tổn thương tự tin ($m_{\text{fg}} \to 1$):* $\partial \to 0$ $\to$ Chống bão hòa đặc trưng.
    - *Vùng biên bất định & nền bị nhầm ($m_{\text{fg}} \to 0$):* $\partial \to 1$ $\to$ Tập trung tối đa gradient để gọt sạch biên.
- **Bảng biểu / Hình ảnh gợi ý:**
  - Sơ đồ luồng khối module `MixPool` cải tiến: Luồng $x$ đi qua `fmask(x)` gặp toán tử Soft-OR với `detach(m_fg)` trước khi nhân phần tử với conv1.

---

### Slide 6: Experimental & Evaluation Setting

- **Tiêu đề Slide:** **Thiết Lập Thực Nghiệm & Giao Thức Đánh Giá**
- **Nội dung chính:**
  - **Tập dữ liệu thách thức:** Kvasir-SEG (Sessile subset — polyp phẳng Paris IIa/IIb khó phát hiện), 196 ảnh nội soi độ phân giải chuẩn hóa $256 \times 256$ (156 train / 40 validation).
  - **Giao thức huấn luyện chặt chẽ:** Huấn luyện $200$ epochs trên Kaggle GPU T4, tối ưu hóa Adam ($lr = 1 \times 10^{-4}$), hàm mất mát Asymmetric Tversky ($\alpha = 0.7, \beta = 0.3$).
  - **Khử nhiễu ngẫu nhiên (5 Seeds):** Đánh giá trên 5 hạt giống ngẫu nhiên độc lập $S \in \{7, 42, 99, 1337, 2024\}$ để loại bỏ hoàn toàn yếu tố may rủi.
  - **Quy chuẩn suy luận hồi quy & Thống kê:** Suy luận lặp 4 chu kỳ (4-iter recurrent refinement với Otsu initialization); kiểm định phi tham số **Wilcoxon Signed-Rank Test** trên 200 điểm đo ($40\text{ ảnh} \times 5\text{ seeds}$).
- **Bảng biểu / Hình ảnh gợi ý:**
  - Bảng tổng hợp cấu hình phần cứng/siêu tham số + Minh họa vòng lặp recurrent inference 4 bước từ mask Otsu thô đến phân đoạn cuối cùng.

---

### Slide 7: Quantitative Evaluation — Multi-Seed Benchmark

- **Tiêu đề Slide:** **Đánh Giá Định Lượng: Benchmark Đa Hạt Giống (5 Seeds)**
- **Nội dung chính:**
  - **Triệt tiêu Over-Segmentation:** FPR giảm từ **$4.30\%$ xuống $2.46\%$** (giảm **$42.8\%$** diện tích vẽ thừa). Tại Seed 2024, FPR giảm sốc từ $8.88\%$ xuống $0.80\%$.
  - **Gia tăng độ chính xác phân đoạn:** Precision trung bình tăng từ **$31.43\%$ lên $38.93\%$** ($+7.50\text{ pp}$, tương đương tăng tương đối $+23.9\%$).
  - **Nâng cao chất lượng chồng lấp:** Dice Score trung bình tăng từ **$0.2517$ lên $0.2995$** ($+4.77\text{ pp}$, tăng $+19.0\%$).
  - **Ổn định hóa & Miễn nhiễm sụp đổ:** Độ lệch chuẩn giữa các seed giảm **$66.5\%$** ($\sigma = 0.0869 \to 0.0291$); cứu thoát Seed 1337 từ mức sụp đổ thảm họa ($0.0928$) lên mức tối ưu ($0.2779$).
  - **Ý nghĩa thống kê vững chắc:** Kiểm định Wilcoxon trên 200 cặp quan sát đạt **$W = 4,055.0, p < 0.001$** (exact $p = 1.13 \times 10^{-6}$, rank-biserial $r = 0.422$).
- **Bảng biểu / Hình ảnh gợi ý:**
  - Bảng số liệu chi tiết 5 hạt giống:

| Cấu hình | Dice (Mean ± Std) | FPR (%) | Precision | Hiện tượng sụp đổ (Seed 1337) | Wilcoxon p-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **M11 (Feedback Trap)** | $0.2517 \pm 0.087$ | $4.30\% \pm 2.31\%$ | $0.3143$ | **Sụp đổ: Dice = 0.0928** | — |
| **M12 (Detached Soft-OR)** | **$0.2995 \pm 0.029$** | **$2.46\% \pm 1.65\%$** | **$0.3893$** | **Khôi phục: Dice = 0.2779** | **$p = 1.13 \times 10^{-6}$** |
| **Mức độ cải thiện** | **$+4.77\text{ pp}$ (+19.0%)** | **$-1.84\text{ pp}$ (-42.8%)** | **$+7.50\text{ pp}$ (+23.9%)** | **Xóa bỏ sụp đổ ($\sigma$ giảm 66.5%)** | **Cực kỳ thuyết phục** |

---

### Slide 8: Diagnostic Studies & Internal Mechanism

- **Tiêu đề Slide:** **Nghiên Cứu Chẩn Đoán & Cơ Chế Gradient Nội Mạng**
- **Nội dung chính:**
  - **Cắt đứt Gradient độc hại:** Mask Gradient Norm giảm từ **$663.43$** (M11) xuống **$53.62$** (M12) — Toàn bộ 8 module `MixPool` được ngắt dòng lan truyền ngược, chỉ còn gradient dư tại skip-head.
  - **Tách biệt biểu diễn ranh giới:** Bottleneck Cosine Similarity giữa biên tổn thương và nền xa tăng từ **$0.7981$ lên $0.9339$**, ngăn chặn hiện tượng đồng nhất hóa đặc trưng niêm mạc.
  - **Bão hòa dự đoán rõ nét:** Tỷ lệ dự đoán dứt khoát (Saturation) đạt **$97.20\%$** với entropy thấp ($0.0335$), chấm dứt tình trạng dự đoán mờ mịt, lưỡng lự ở vùng nền.
  - **Khắc phục trôi dạt BatchNorm:** Ngăn chặn hoàn toàn hiện tượng lệch phân phối thống kê cực đoan ($D_{\text{KL}} = 33.63$ tại `e1.r1.bn3` của M11), tái lập sự ổn định cho encoder.
- **Bảng biểu / Hình ảnh gợi ý:**
  - Biểu đồ Bar Chart 4 trục chẩn đoán: Mask Grad Norm (663 vs 54), Bottleneck CosSim (0.80 vs 0.93), Prediction Saturation (1.1% vs 97.2%), và BN Drift ($D_{KL}$ giảm mạnh).

---

### Slide 9: Qualitative Results — Eradication of Ghost Lesions

- **Tiêu đề Slide:** **Kết Quả Định Tính: Triệt Tiêu Các Báo Động Giả (Ghost Lesions)**
- **Nội dung chính:**
  - **Quan sát trực quan sắc nét:** So sánh 4 khung hình: Ảnh nội soi gốc, Ground Truth, Dự đoán M11 (Hard Gate) và Dự đoán M12 (Detached Soft-OR).
  - **Mã hóa vùng sai số:** Vùng Xanh lá (True Positive), Vùng Vàng (False Negative), Vùng Đỏ (False Positive - Over-segmentation).
  - **Xóa sổ tổn thương ma (Ghost Lesions):** Ở các ca lâm sàng có nếp gấp niêm mạc phức tạp hoặc phản quang ướt, M11 vẽ thừa diện tích khổng lồ ($>3,000\text{ pixels}$ vùng Đỏ).
  - **Ranh giới M12 sắc nét & An toàn:** M12 quét sạch hoàn toàn các đốm đỏ ($0\text{ px}$ False Positives), ôm sát ranh giới tổn thương thực tế (White contour).
- **Bảng biểu / Hình ảnh gợi ý:**
  - Hình ảnh cắt từ `paper_figures/qualitative_comparison.pdf`: Hiển thị 2 ca điển hình (Case 1 & Case 2), đối chiếu rõ rệt cột M11 ngập tràn màu Đỏ vs cột M12 sạch bóng màu Đỏ.

---

### Slide 10: Conclusion & Future Outlook

- **Tiêu đề Slide:** **Kết Luận & Định Hướng Phát Triển**
- **Nội dung chính:**
  - **Kết luận khoa học:** "Feedback Trap" là nguyên nhân gốc rễ làm tê liệt gradient và gây over-segmentation trong mạng hồi quy. Detached Soft-OR là giải pháp kiến trúc tinh gọn, giải quyết triệt để vấn đề mà không tốn thêm tham số.
  - **Ý nghĩa lâm sàng thực tiễn:** Giảm mạnh tỷ lệ báo động giả (FPR giảm 42.8% - 64.7%), loại bỏ mệt mỏi nhận thức (*alarm fatigue*) cho bác sĩ và ngăn ngừa rủi ro thủng ruột do cắt nhầm mô lành.
  - **Nguyên lý kiến trúc tổng quát:** Đóng góp bài học kinh điển cho các mạng hồi quy tương lai: *Phân ly dẫn đường xuôi (forward guidance) khỏi lan truyền ngược (backward optimization).*
  - **Định hướng tương lai:** Mở rộng cơ chế Detached Soft-OR sang dữ liệu video nội soi thời gian thực (temporal feedback) và các mô thức 3D như Cardiac MRI / CT segmentation.
- **Bảng biểu / Hình ảnh gợi ý:**
  - Sơ đồ Takeaway chốt lại: [Detached Soft-OR] = [Zero-Param] + [Stable Convergence] + [Safe Clinical Boundaries]. QR code trỏ tới GitHub repository của dự án.
