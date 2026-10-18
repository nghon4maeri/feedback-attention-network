# [T3 – 20/08/2026]

## Mục tiêu tuần này

Đọc sâu paper UACANet để hiểu cách dùng prediction của stage trước làm guidance, đối chiếu với MixPool của FANet, từ đó xác định FANet đang mất những thông tin hữu ích nào — tiền đề cho việc đề xuất cơ chế feedback mới.

## Done

- [x] **Đọc sâu UACANet** (ACM MM 2021) — phân tích module UACA chi tiết (mục dưới)
- [x] **So sánh với MixPool của FANet** — lập bảng 6 gap mất thông tin
- [x] **Refactor repo** thành cấu trúc research chuẩn: `src/fanet/` package, `scripts/`, `analysis/`, `configs/`, `docs/reports/`; untrack dataset khỏi git
- [x] **Gradient flow qua fmask**: xác nhận gate binary cắt gradient → nhánh fmask không học — `logs/analysis_grad_log.txt`
- [x] **Baseline eval**: chạy test-time refinement trên checkpoint 53 epochs — `results/test_results.csv`

## Findings quan trọng

### 1. UACANet dùng prediction stage trước như thế nào

Từ saliency map `m` **liên tục (không threshold)**, tách 3 area map:

```
m_f = max(m − 0.5, 0)    # foreground
m_b = max(0.5 − m, 0)    # background
m_u = 0.5 − |m − 0.5|    # uncertain — đỉnh tại m ≈ 0.5
```

**Vì sao không chỉ coi prediction là fg/bg đơn giản:**

1. **Biên nằm ở vùng m ≈ 0.5** — reverse attention [5] chỉ cải thiện nhẹ; vùng saliency mơ hồ chính là biên polyp → `m_u` là *edge guidance miễn phí*, không cần annotation biên
2. **Soft map giữ độ tin cậy** — pixel 0.9 ≠ 0.51; phép `max` giúp 3 vùng không chồng lấn thông tin
3. **Background không bị vứt** — pool thành context vector `v_b = Σ m_b·x` (kiểu OCR), vẫn tham gia vào mọi pixel
4. **Mỗi pixel tự chọn context** qua softmax similarity: `t_i = δ(s_f·ω(v_f) + s_b·ω(v_b) + s_u·ω(v_u))`
5. **Học residual** — output UACA cộng với saliency trước → stage sau tập trung vào vùng khó

**Bằng chứng ablation:** bỏ `m_u` (CANet-L vs UACANet-L): CVC-ClinicDB 91.2 → 92.6; **ETIS 67.8 → 76.6** (gain lớn nhất paper).

### 2. MixPool của FANet đang mất gì

```python
fmask = (self.fmask(x) > 0.5).float()               # binarize cứng
x1 = x * torch.logical_or(fmask, m > 0).float()     # gate binary OR
x2 = self.conv2(x)                                  # 50% feature gốc
x = torch.cat([x1, x2], 1)
```

| # | Thông tin UACANet giữ | MixPool mất | Hệ quả |
|---|---|---|---|
| Uncertainty | `m_u` (m≈0.5 = biên) | Threshold 0.5 xoá đúng vùng này | Mất edge guidance |
| Confidence | Confidence liên tục | Binary {0,1}, OR → mọi pixel giữ lại trọng số bằng nhau | Không phân biệt vùng tự tin / không |
| Background | Background context `v_b` | `x·gate` zero hoá background | Mất ngữ cảnh niêm mạc |
| Gradient | Gradient qua attention | `> 0.5` không differentiable → fmask không học | Nhánh "tự học" vô tác dụng |
| Residual | Residual refinement giữa stage | Feedback là mask epoch trước, không học phần dư | — |
| Blending | Blending theo nội dung | Gate nhân kênh cứng (CBAM-style) | Ít linh hoạt ở ranh giới |

### 3. Kết quả gradient flow qua fmask

- **fmask: 0 param có gradient** sau 10 training steps; conv1/conv2: 160 params có grad — `logs/analysis_grad_log.txt`
- Weight fmask **không đổi** sau optimizer.step; BN running stats vẫn trôi (do forward pass)
- Checkpoint 53 epochs vs model init: BN affine (gamma/beta init 1/0) diff **chính xác = 0** → xác nhận fmask chưa từng được update

### 4. Kết quả baseline refinement

Checkpoint 53 epochs (train chưa xong, val loss 0.505): iter 1→2: Jaccard 0.2166 → 0.2251, F1 0.3147 → 0.3268. Feedback có giúp nhưng nhẹ — cần phân tích sâu hơn (help/hurt + uncertainty).

## Experiments

| Giả thuyết | Config | Kết quả | Link log |
|-----------|--------|---------|----------|
| Gate binary cắt gradient → fmask không học | 10 steps, checkpoint 53ep | Xác nhận: 0 grad fmask / 160 grad conv | `logs/analysis_grad_log.txt` |
| Test-time refinement có cải thiện không | 2 iters, 40 ảnh val | Jaccard 0.2166→0.2251 (giúp nhẹ) | `results/test_results.csv` |

## Will Do (On going)

- [ ] **Phân tích feedback help/hurt**: với hard binary feedback, xem khi nào feedback giúp / khi nào làm prediction tệ hơn; uncertainty/confidence trước threshold có correlate với các lỗi đó không
- [ ] Từ kết quả phân tích → đề xuất cơ chế feedback mới nhắm đúng điểm yếu đã xác nhận bằng dữ liệu
- [ ] Tạm thời **không mở thêm** background context / deep supervision cho fmask — tránh loãng hướng

## Any Stuck / Open Questions

- Không. Lưu ý: việc fmask không học đã xác nhận chắc chắn bằng cả grad norm lẫn weight diff — đây là bug thật sự của implementation, không chỉ là hạn chế thiết kế.

## Đính kèm link chi tiết

- Paper UACANet: `docs/3474085.3475375.pdf` | Paper FANet: `docs/2103.17235v3.pdf`
- MixPool code: `src/fanet/models/blocks.py` (forward: `analysis/grad_flow.py`)
- Log gradient flow: `logs/analysis_grad_log.txt` | Baseline eval: `results/test_results.csv`
- Script analysis: `analysis/grad_flow.py` | Script eval: `scripts/evaluate.py`
