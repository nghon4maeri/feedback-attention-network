# FANet: A Feedback Attention Network for Improved Biomedical Image Segmentation

---

# Mục lục

1. [Bối cảnh & Động lực](#1-bối-cảnh--động-lực)
2. [Kiến trúc FANet](#2-kiến-trúc-fanet)
3. [MixPool — Feedback Attention Mechanism](#3-mixpool--feedback-attention-mechanism)
4. [Loss Functions](#4-loss-functions)
5. [Training Pipeline](#5-training-pipeline)
6. [Inference & Test-time Refinement](#6-inference--test-time-refinement)
7. [Metrics Evaluation](#7-metrics-evaluation)
8. [Từ điển thuật ngữ A-Z](#8-từ-điển-thuật-ngữ-a-z)

---

# 1. Bối cảnh & Động lực

## 1.1 Biomedical Image Segmentation

**Phân đoạn ảnh y sinh** là bài toán gán nhãn cho từng pixel trong ảnh y khoa, chia thành 2 lớp:

- **Foreground** (tiền cảnh, $y=1$): vùng cần quan tâm — polyp trong nội soi, mạch máu võng mạc, nhân tế bào, tổn thương da, v.v.
- **Background** (hậu cảnh, $y=0$): phần còn lại của ảnh.

Đầu ra là một **binary mask** $\mathbf{M} \in \{0, 1\}^{H \times W}$ — một ảnh nhị phân cùng kích thước với input, trong đó mỗi pixel $=1$ là foreground, $=0$ là background.

Đây là bài toán **pixel-wise classification** (phân loại từng pixel), khác với image classification (phân loại cả ảnh) hay object detection (phát hiện hộp bao).

---

## 1.2 Tại sao cần Deep Learning?

Các phương pháp cổ điển (thresholding, edge detection, watershed, active contours) dựa trên **hand-crafted features** (đặc trưng do con người thiết kế thủ công):

- Ngưỡng hóa Otsu, Canny edge, Gabor filter...
- Ưu điểm: nhanh, không cần dữ liệu huấn luyện.
- Nhược điểm: không tổng quát hóa; thất bại khi ảnh có nhiễu, ánh sáng không đồng đều, hoặc đối tượng có hình dạng phức tạp.

**Convolutional Neural Network (CNN)** — Mạng nơ-ron tích chập — tự động học đặc trưng phân cấp từ dữ liệu:

- Tầng thấp: học cạnh (edges), góc (corners), texture.
- Tầng giữa: học hình dạng (shapes), bộ phận (parts).
- Tầng cao: học ngữ nghĩa toàn cục (objects).

Kiến trúc điển hình cho segmentation là **U-Net** (Ronneberger et al., 2015): encoder-decoder hình chữ U, với skip connections giữa encoder và decoder cùng cấp để giữ thông tin không gian chi tiết bị mất khi downsampling.

---

## 1.3 Vấn đề: Lãng phí thông tin xuyên Epoch

Trong quá trình huấn luyện deep learning:

- **Epoch**: một lần duyệt qua **toàn bộ** tập huấn luyện. Dataset 1000 ảnh → 1 epoch = mô hình thấy cả 1000 ảnh đúng 1 lần.
- **Forward pass**: input → model → output (dự đoán).
- **Backward pass (backpropagation)**: tính gradient của loss theo tham số, dùng chain rule từ output ngược về input.
- **Gradient Descent**: cập nhật tham số $\theta$ theo hướng ngược gradient:

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}$$

trong đó $\eta$ là **learning rate** (tốc độ học).

**Vấn đề**: mask dự đoán ở epoch $t$ ($P_t$) bị **vứt bỏ hoàn toàn** khi chuyển sang epoch $t+1$. Mỗi epoch bắt đầu lại từ con số 0 — không có ký ức về những gì đã học trước đó ở dạng mask.

**Ý tưởng FANet**: mask $P_{t-1}$ chứa thông tin quý giá về việc mô hình "nghĩ" vùng nào là foreground. Hãy dùng nó như một **hard attention map** để hướng sự chú ý của mô hình ở epoch $t$, prune (tỉa) các feature không liên quan.

---

## 1.4 Attention là gì?

**Attention (cơ chế chú ý)** là kỹ thuật cho phép mô hình "tập trung" vào phần quan trọng của input. Phân biệt:

| Loại Attention | Cơ chế | Ví dụ |
|:---|:---|:---|
| **Soft Attention** | Trọng số liên tục $\alpha_i \in [0, 1]$, nhân element-wise. Mọi pixel đều được giữ lại, nhưng với cường độ khác nhau. | SE Attention, Self-Attention trong Transformer |
| **Hard Attention** | Mặt nạ nhị phân $\alpha_i \in \{0, 1\}$, pixel hoặc được giữ nguyên hoặc bị xóa sạch. | Feedback mask trong FANet |

FANet dùng **hard attention** để prune mạnh — chỉ giữ feature ở vùng foreground, xóa hoàn toàn feature ở vùng background.

**Feedback** là vòng lặp phản hồi: output của epoch $t-1$ trở thành input của epoch $t$, tạo thành một chu trình cải tiến liên tục.

---

# 2. Kiến trúc FANet

## 2.1 Tổng quan

FANet là mô hình **encoder-decoder** 4 tầng kiểu U-Net, nhận input là **cặp** `[RGB image, previous mask]`:

```
Input: [image (3×H×W), prev_mask (1×H×W)]

Encoder:
  Enc1: [3→32]  → MixPool → MaxPool(2×2)  → (32,  H/2,  W/2)
  Enc2: [32→64] → MixPool → MaxPool(2×2)  → (64,  H/4,  W/4)
  Enc3: [64→128] → MixPool → MaxPool(2×2) → (128, H/8,  W/8)
  Enc4: [128→256] → MixPool → MaxPool(2×2) → (256, H/16, W/16) ← Bottleneck

Decoder:
  Dec4: ConvTranspose2d → concat(skip Enc4) → 2×ResBlock → MixPool → (128, H/8,  W/8)
  Dec3: ConvTranspose2d → concat(skip Enc3) → 2×ResBlock → MixPool → (64,  H/4,  W/4)
  Dec2: ConvTranspose2d → concat(skip Enc2) → 2×ResBlock → MixPool → (32,  H/2,  W/2)
  Dec1: ConvTranspose2d → concat(skip Enc1) → 2×ResBlock → MixPool → (16,  H,    W)

Output: concat(Dec1_16ch, prev_mask_1ch) → Conv1×1 → (1×H×W) → Sigmoid
```

---

## 2.2 Thuật ngữ kiến trúc

### Encoder (Mã hóa)

Bộ phận **giảm kích thước không gian, tăng số channels** để trích xuất đặc trưng ngữ nghĩa cao cấp. Mỗi tầng encoder = một hoặc nhiều lớp tích chập + pooling.

### Decoder (Giải mã)

Bộ phận **tăng kích thước không gian** trở lại kích thước ban đầu để tạo mask, dùng skip connections từ encoder để khôi phục chi tiết không gian.

### Bottleneck (Cổ chai)

Điểm thắt ở **đáy chữ U**, nơi feature map có kích thước không gian **nhỏ nhất** nhưng số channels **lớn nhất**. Trong FANet: `H/16 × W/16 × 256`. Đây là biểu diễn ngữ nghĩa cô đọng nhất của toàn bộ ảnh — mỗi pixel trong bottleneck "nhìn thấy" một vùng rất rộng của ảnh gốc (**receptive field lớn**).

### MaxPool2d(2×2)

Giảm một nửa kích thước không gian bằng cách lấy giá trị lớn nhất trong mỗi cửa sổ 2×2. Không có tham số học. Tác dụng:
- Giảm tính toán.
- Tăng **receptive field** — vùng ảnh gốc mà mỗi neuron "cảm nhận".
- Tạo **translation invariance** nhẹ — mô hình ít nhạy với dịch chuyển nhỏ.

### ConvTranspose2d (Transposed Convolution)

Phép toán "ngược" của convolution trong không gian, dùng để **upsample** (tăng kích thước). Khác với nội suy (interpolation), ConvTranspose2d có tham số học được — mô hình tự học cách upsample tối ưu.

### Skip Connection (Kết nối tắt)

Feature map từ encoder được **nối trực tiếp** (concatenate theo channels) vào decoder cùng cấp. Tác dụng: khi downsampling, thông tin vị trí chi tiết bị mất; skip connection khôi phục lại, giúp decoder vừa có ngữ nghĩa cao cấp (từ decoder trên) vừa có chi tiết không gian (từ encoder).

### Receptive Field

Vùng ảnh gốc mà một neuron trong feature map "nhìn thấy". Neuron ở tầng càng sâu → receptive field càng lớn. VD: tầng 1 nhìn thấy vùng 3×3, tầng 4 (sau 4 lần pooling) có thể nhìn thấy vùng >100×100 pixel của ảnh gốc.

---

## 2.3 ResidualBlock

Mỗi ResidualBlock gồm hai nhánh:

```
Input (C channels)
  |
  ├── Main branch: Conv3x3(C→C) → BN → ReLU → Conv3x3(C→C) → BN
  |
  ├── Skip branch: Conv1x1(C→C) → BN → SELayer
  |
  └── element-wise sum (+) → ReLU → Output
```

### Residual Connection (Kết nối dư)

Đầu ra $=$ input $+$ phần dư do nhánh chính học:

$$\text{Output} = \text{ReLU}\big(\mathcal{F}(\mathbf{x}) + \mathcal{H}(\mathbf{x})\big)$$

trong đó $\mathcal{F}$ là nhánh chính (2 lớp Conv3x3) và $\mathcal{H}$ là nhánh skip (Conv1x1 + SE).

**Tại sao cần?** Với mạng rất sâu, gradient qua nhiều tầng bị suy giảm → **vanishing gradient**. Residual connection cho phép gradient truyền thẳng qua phép cộng (đạo hàm của $y=x+f(x)$ theo $x$ là $1+f'(x)$ → luôn có thành phần $1$), giúp train được mạng hàng trăm tầng.

### Batch Normalization (BN)

Với mỗi batch, chuẩn hóa activation về mean $=0$, std $=1$:

$$\hat{x}_i = \frac{x_i - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

- $\mu_{\text{batch}}, \sigma^2_{\text{batch}}$: mean và variance của batch hiện tại.
- $\gamma, \beta$: tham số học được (scale và shift).
- $\epsilon$: hằng số nhỏ tránh chia cho 0.

Tác dụng: ổn định phân phối activation giữa các tầng (giảm **internal covariate shift**), cho phép learning rate cao hơn, có tác dụng regularization nhẹ.

### ReLU (Rectified Linear Unit)

Hàm kích hoạt phi tuyến đơn giản:

$$\text{ReLU}(x) = \max(0, x)$$

So với sigmoid/tanh: không bão hòa ở vùng dương (gradient không biến mất khi $x \gg 0$), tính toán nhanh.

---

## 2.4 SELayer — Squeeze-and-Excitation

**Channel Attention** — mô hình tự học kênh nào quan trọng:

```
Input: X ∈ ℝ^{C×H×W}

Step 1 — Squeeze (Global Average Pooling):
  z_c = (1/(H×W)) · Σ_i Σ_j X_{c,i,j}     →  z ∈ ℝ^C

Step 2 — Excitation (MLP 2 tầng):
  s = σ( W_2 · δ( W_1 · z ) )
  W_1 ∈ ℝ^{(C/r)×C},   W_2 ∈ ℝ^{C×(C/r)}
  r = reduction ratio = 8

Step 3 — Scale:
  Y_c = s_c · X_c     (broadcast theo không gian)
```

Trong đó:
- $\sigma$ = sigmoid: $\sigma(x) = \frac{1}{1+e^{-x}}$, ép về $[0,1]$
- $\delta$ = ReLU

**Giải thích**:

1. **Squeeze (GAP)**: nén thông tin không gian của mỗi channel thành 1 số — "cường độ" trung bình của channel đó.
2. **Excitation (MLP)**: học mối quan hệ phi tuyến giữa các channels qua bottleneck $C \to C/8 \to C$, tạo trọng số attention cho từng channel.
3. **Scale**: channel quan trọng → $s_c \approx 1$ (giữ nguyên); channel không quan trọng → $s_c \approx 0$ (triệt tiêu).

---

# 3. MixPool — Feedback Attention Mechanism

Đây là **đóng góp cốt lõi** của FANet. MixPool kết hợp feature học được với mask từ epoch trước để prune feature map.

```
Input:
  x ∈ ℝ^{C×H×W}          — feature map hiện tại
  m ∈ {0,1}^{H'×W'}      — mask từ epoch trước (RLE-decoded)

─────────────────────────────────────────────────────

Step 1 — Feature-based attention mask (fmask):
  x → Conv3x3(C→32) → BN → ReLU → Conv3x3(32→1) → Sigmoid
  fmask = (output > 0.5) ∈ {0,1}^{H×W}

Step 2 — Spatial alignment:
  Nếu H'≠H hoặc W'≠W:
    m = MaxPool2d(m) để match (H, W)

Step 3 — Hard gating:
  attention = fmask ∨ m          (phép OR logic)
  x_gated = x ⊙ attention       (broadcast theo channels)

Step 4 — Dual convolution paths:
  conv1 = Conv3x3(C → C/2)(x_gated)    ← 50% attended
  conv2 = Conv3x3(C → C/2)(x)          ← 50% nguyên bản

Step 5 — Concatenate:
  output = concat(conv1, conv2) ∈ ℝ^{C×H×W}
```

### Phân tích từng bước

| Thành phần | Ký hiệu | Ý nghĩa |
|:---|:---|:---|
| **fmask** | $A_{\text{feat}}$ | Attention mask do **mô hình tự học** từ feature hiện tại — vùng nào của feature map là "đáng quan tâm" dựa trên dữ liệu |
| **m (prev_mask)** | $A_{\text{hist}}$ | Attention mask từ **lịch sử** — vùng nào epoch trước đã dự đoán là foreground |
| **$A_{\text{feat}} \lor A_{\text{hist}}$** | OR logic | Kết hợp cả 2 nguồn: giữ feature nếu một trong hai cho là quan trọng → **an toàn**, không bỏ sót |
| **conv1** | $\mathcal{C}_1(x \odot A)$ | Feature đã được prune — chỉ giữ vùng foreground |
| **conv2** | $\mathcal{C}_2(x)$ | Feature nguyên bản — **safety net**: nếu attention sai, conv2 vẫn giữ toàn bộ thông tin |

### Tại sao 50/50?

Nếu attention quá mạnh (100% prune), một lỗi nhỏ trong mask có thể xóa sạch foreground thật → mô hình "mù" ở vùng đó. Giữ 50% nguyên bản là **cơ chế an toàn** để mô hình không bao giờ mất hoàn toàn thông tin.

### Tiến hóa qua các epoch

| Epoch | Mask chất lượng | Tác dụng MixPool |
|:---|:---|:---|
| 0 | Otsu (thô, nhiễu) | Prune vùng rõ ràng không phải foreground (viền đen nội soi…), tập trung vào vùng trung tâm |
| 1-50 | Cải thiện dần | Prune mạnh hơn → mô hình chỉ tập trung tinh chỉnh vùng biên, vùng khó |
| 50+ | Gần hoàn hảo | Prune rất ít thay đổi → hội tụ |

→ Giống **curriculum learning** (học từ dễ đến khó).

---

## 2.5 Conv1x1 — Pointwise Convolution

Tích chập với kernel $1 \times 1$, không trộn thông tin không gian (không nhìn pixel lân cận):

$$Y_{c_{\text{out}}, i, j} = \sum_{c_{\text{in}}=1}^{C_{\text{in}}} w_{c_{\text{out}}, c_{\text{in}}} \cdot X_{c_{\text{in}}, i, j}$$

Ứng dụng trong FANet:
- **Skip branch của ResidualBlock**: thay đổi số channels nếu cần để cộng với main branch.
- **Output head**: `Conv1x1(17→1)` — tổ hợp tuyến tính của 17 kênh thành 1 kênh output.
- **SELayer bottleneck**: giảm/tăng channels.

---

# 4. Loss Functions

FANet sử dụng **DiceBCELoss**: trung bình có trọng số của Binary Cross Entropy và Dice Loss:

$$\mathcal{L}_{\text{total}} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$

---

## 4.1 Dice Loss

Xuất phát từ **Dice Similarity Coefficient (DSC)** — còn gọi là F1-score:

$$\text{Dice}(P, G) = \frac{2|P \cap G| + \epsilon}{|P| + |G| + \epsilon}$$

Trong đó:
- $P$: tập pixel predicted foreground (sau sigmoid, chưa threshold)
- $G$: tập pixel ground truth foreground
- $|P \cap G| = \sum_i P_i \cdot G_i$ (soft intersection — dùng giá trị liên tục của sigmoid, differentiable)
- $|P| = \sum_i P_i$, $|G| = \sum_i G_i$
- $\epsilon$: hằng số nhỏ ($10^{-5}$) tránh chia cho 0 (smooth)

$$\mathcal{L}_{\text{Dice}} = 1 - \text{Dice}(P, G)$$

**Ưu điểm của Dice Loss**:

- **Bất biến với class imbalance**: không bị ảnh hưởng bởi tỉ lệ foreground/background. Trong ảnh y sinh, foreground thường chỉ chiếm 1-10% diện tích → BCE sẽ bị background dominate. Dice loss tính trực tiếp trên vùng overlap.
- Trực tiếp tối ưu hóa metric quan tâm (Dice/IoU).

**Nhược điểm**: Gradient có thể dao động mạnh khi overlap rất nhỏ (mẫu số nhỏ → đạo hàm lớn).

---

## 4.2 Binary Cross Entropy (BCE)

$$\mathcal{L}_{\text{BCE}} = -\frac{1}{N}\sum_{i=1}^{N}\Big[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \Big]$$

- $N = H \times W$: tổng số pixel
- $y_i \in \{0, 1\}$: ground truth của pixel $i$
- $\hat{y}_i \in [0, 1]$: xác suất dự đoán (sau sigmoid) của pixel $i$

**Phân tích**:
- Nếu $y_i = 1$ (foreground thật): loss $= -\log(\hat{y}_i)$ → phạt nặng khi $\hat{y}_i$ thấp (dự đoán sai thành background)
- Nếu $y_i = 0$ (background thật): loss $= -\log(1-\hat{y}_i)$ → phạt nặng khi $\hat{y}_i$ cao (dự đoán sai thành foreground)

**Ưu điểm**: Gradient mượt, ổn định, dễ train.

**Nhược điểm**: Mỗi pixel được đối xử như nhau → nếu foreground nhỏ, gradient từ background lấn át.

---

## 4.3 Tổ hợp

$$\mathcal{L} = 0.5 \cdot \mathcal{L}_{\text{BCE}} + 0.5 \cdot \mathcal{L}_{\text{Dice}}$$

Kết hợp ưu điểm:
- **BCE**: gradient ổn định, hội tụ mượt.
- **Dice**: tối ưu trực tiếp overlap, chống class imbalance.

Trọng số $0.5$ mỗi thành phần — đóng góp cân bằng.

---

# 5. Training Pipeline

## 5.1 Otsu Thresholding — Mask khởi tạo

Epoch đầu tiên cần một mask để làm feedback. Vì chưa có mask nào từ epoch trước, FANet dùng **Otsu's method**:

**Otsu's method** là thuật toán **tự động chọn ngưỡng** để tách foreground/background:

1. Tính histogram của ảnh xám (256 bins)
2. Duyệt tất cả ngưỡng $T \in [0, 255]$
3. Với mỗi $T$, tính **between-class variance** $\sigma^2_B(T)$:

$$\sigma^2_B(T) = w_0(T) \cdot w_1(T) \cdot [\mu_0(T) - \mu_1(T)]^2$$

Trong đó:
- $w_0, w_1$: tỉ lệ pixel thuộc lớp 0 (≤T) và lớp 1 (>T)
- $\mu_0, \mu_1$: giá trị xám trung bình của mỗi lớp

4. Chọn $T^* = \arg\max_T \sigma^2_B(T)$ → tách foreground/background tối ưu.

**Trong code** (`utils.py:init_mask`):
```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)  # làm mờ giảm nhiễu
_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTS)
```

> **Gaussian Blur**: làm mờ ảnh bằng kernel Gaussian để giảm nhiễu trước khi thresholding. Kernel 5×5 là kích thước phổ biến.

---

## 5.2 RLE — Run-Length Encoding

**Vấn đề**: Mask float32 kích thước 256×256 = $256 \times 256 \times 4 = 256$ KB. Lưu 500 epochs × số ảnh → quá nhiều RAM.

**Giải pháp**: RLE nén mask thành string:

Ví dụ mask `[0,0,1,1,1,0,0,1,1]` → RLE: `[3, 3, 8, 2]`
- Bắt đầu tại vị trí 3, chạy 3 pixel
- Bắt đầu tại vị trí 8, chạy 2 pixel

Trong code (`utils.py:rle_encode`/`rle_decode`): dùng **Fortran column-major order** của numpy (duyệt theo cột trước, hàng sau) để nén — đây là convention từ dataset gốc.

---

## 5.3 Data Augmentation (Tăng cường dữ liệu)

Biến đổi ngẫu nhiên ảnh trong quá trình training để tạo dữ liệu "ảo", giúp mô hình tổng quát hóa tốt hơn (chống overfitting):

| Phép biến đổi | Tham số | Cơ chế | Tác dụng |
|:---|:---|:---|:---|
| **Random Rotation** | ±35° | Quay ảnh ngẫu nhiên | Bất biến với góc xoay — polyp có thể ở mọi hướng |
| **Horizontal Flip** | p=0.5 | Lật ngang | Bất biến gương |
| **Vertical Flip** | p=0.5 | Lật dọc | Bất biến gương |
| **CoarseDropout** | max_holes=8 | Xóa ngẫu nhiên vùng hình chữ nhật | Regularization — buộc mô hình không phụ thuộc vào một vùng cục bộ |

> **Augmentation** được thực hiện bởi thư viện **Albumentations** — một thư viện tăng cường ảnh nhanh, chuyên dùng trong computer vision.

---

## 5.4 Optimizer & Scheduler

### Adam Optimizer

Cập nhật tham số dùng moment bậc 1 và bậc 2:

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \quad \text{(moment bậc 1 — trung bình động của gradient)}$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \quad \text{(moment bậc 2 — trung bình động của bình phương gradient)}$$
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t} \quad \text{(bias correction)}$$
$$\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

FANet dùng $\eta = 10^{-4}$, $\beta_1=0.9$, $\beta_2=0.999$.

### ReduceLROnPlateau

Tự động giảm learning rate khi validation loss ngừng cải thiện (plateau — vùng bằng phẳng):

```
Nếu val_loss không giảm trong patience=10 epochs:
    lr ← lr × factor (vd: ×0.1)
```

Giống như giảm tốc độ khi gần tới đích — bước nhỏ hơn để không vượt quá điểm tối ưu.

---

## 5.5 Quy trình training đầy đủ

```
Bước 0: Load Kvasir-SEG dataset (train.txt / val.txt splits)
Bước 1: Khởi tạo previous_masks = Otsu(images)  → RLE encode
Bước 2: for epoch in 1..500:
          for batch in train_loader:
            prev_mask_batch = RLE_decode(prev_masks[idx])
            pred = FANet( [image, prev_mask_batch] )
            loss = DiceBCELoss(pred, ground_truth)
            loss.backward()
            optimizer.step()
            RLE_encode(pred) → lưu cho epoch sau

          val_loss = evaluate(val_loader)
          scheduler.step(val_loss)
          if val_loss < best_loss:
            save checkpoint → update prev_masks từ best predictions
```

**Lưu ý quan trọng**: mask cho epoch sau được lấy từ **best checkpoint** hiện tại, không phải từ từng batch — điều này ổn định hóa feedback loop.

---

# 6. Inference & Test-time Refinement

## 6.1 Quy trình test

Test-time refinement không cần train lại — chỉ chạy inference nhiều lần:

```
Input: test_image, ground_truth

Bước 0: prev_mask = Otsu(test_image)  →  RLE encode

Bước 1: for iteration in 0..9:
          mask = RLE_decode(prev_mask)
          pred = FANet( [test_image, mask] )
          metrics[iteration] = calculate_metrics(pred, ground_truth)
          prev_mask = RLE_encode(pred > 0.5)
```

Chất lượng mask cải thiện qua từng iteration, thường hội tụ ở iteration 3-4.

---

## 6.2 Tại sao refinement hiệu quả?

- Mask từ iteration trước chứa **thông tin bổ sung** mà model không tự suy ra được từ ảnh gốc
- MixPool dùng mask đó để prune vùng background → model tập trung refine vùng foreground
- Không cần train thêm → "miễn phí" về mặt compute
- Có thể coi như một dạng **test-time optimization**

---

# 7. Metrics Evaluation

Tất cả metrics tính trên binary prediction (ngưỡng 0.5). Định nghĩa các đại lượng cơ bản:

| Ký hiệu | Ý nghĩa |
|:---|:---|
| **TP** (True Positive) | Pixel foreground được dự đoán đúng là foreground |
| **TN** (True Negative) | Pixel background được dự đoán đúng là background |
| **FP** (False Positive) | Pixel background bị dự đoán nhầm thành foreground (báo động giả) |
| **FN** (False Negative) | Pixel foreground bị dự đoán nhầm thành background (bỏ sót) |

---

## 7.1 Jaccard Index (IoU — Intersection over Union)

$$\text{Jaccard} = \frac{TP}{TP + FP + FN} = \frac{|P \cap G|}{|P \cup G|}$$

Đo độ chồng lấn giữa vùng dự đoán và ground truth. Giá trị [0, 1], 1 = hoàn hảo. Nhạy cảm hơn Dice — phạt nặng hơn khi sai lệch nhỏ.

> Mối liên hệ với Dice: $\text{Jaccard} = \frac{\text{Dice}}{2 - \text{Dice}}$

---

## 7.2 Dice Score (F1-score)

$$\text{Dice} = \frac{2 \cdot TP}{2 \cdot TP + FP + FN} = \frac{2|P \cap G|}{|P| + |G|}$$

Trung bình điều hòa (harmonic mean) của Precision và Recall.

---

## 7.3 Recall (Sensitivity / Độ nhạy)

$$\text{Recall} = \frac{TP}{TP + FN}$$

Tỉ lệ foreground thật được phát hiện. Cao → ít bỏ sót. Trong y tế: bỏ sót bệnh (FN cao) nguy hiểm hơn báo nhầm (FP cao) → recall rất quan trọng.

---

## 7.4 Precision (Độ chính xác)

$$\text{Precision} = \frac{TP}{TP + FP}$$

Tỉ lệ pixel dự đoán foreground thật sự đúng. Cao → ít báo nhầm.

---

## 7.5 Specificity (Độ đặc hiệu)

$$\text{Specificity} = \frac{TN}{TN + FP}$$

Tỉ lệ background được phân loại đúng. Đối ngẫu của Recall cho background.

---

## 7.6 Accuracy (Độ chính xác tổng thể)

$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

Tỉ lệ pixel đúng trên toàn ảnh. **Cảnh báo**: nếu foreground chỉ chiếm 1% ảnh, mô hình dự đoán toàn bộ là background → accuracy = 99% nhưng hoàn toàn vô dụng! Không nên dùng accuracy một mình trong segmentation.

---

## 7.7 F2-score

$$F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}$$

Với $\beta = 2$, Recall được coi trọng gấp $2^2 = 4$ lần Precision. Trong medical imaging, bỏ sót bệnh (FN) nguy hiểm hơn báo nhầm (FP) → dùng F2 thay vì F1.

---

# 8. Từ điển thuật ngữ A-Z

## A

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **Activation Function** | Hàm phi tuyến áp dụng lên output của mỗi neuron. Không có nó, toàn bộ mạng chỉ là tổ hợp tuyến tính → không học được quan hệ phức tạp. VD: ReLU, Sigmoid, Tanh |
| **Adam** | **Ada**ptive **M**oment Estimation — optimizer dùng moment bậc 1 và bậc 2 của gradient để tự điều chỉnh learning rate cho từng tham số |
| **Attention** | Cơ chế cho phép mô hình "tập trung" vào phần quan trọng của input bằng cách gán trọng số khác nhau cho các phần khác nhau |
| **Augmentation** | Biến đổi ngẫu nhiên dữ liệu huấn luyện (xoay, lật, thêm nhiễu…) để tăng độ đa dạng và chống overfitting |

## B

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **Backpropagation** | Thuật toán lan truyền ngược: dùng chain rule để tính gradient của loss theo từng tham số, từ tầng cuối ngược về tầng đầu |
| **Background** | Phần ảnh không phải đối tượng cần phân đoạn (hậu cảnh) |
| **Batch** | Một nhóm mẫu được xử lý đồng thời trong một lần cập nhật trọng số |
| **Batch Normalization (BN)** | Chuẩn hóa activation trong mỗi batch về mean=0, std=1, giúp ổn định và tăng tốc training |
| **Binary Mask** | Ảnh nhị phân cùng kích thước với input, mỗi pixel = 0 hoặc 1, biểu diễn foreground/background |
| **Bottleneck** | Điểm thắt của mạng nơ-ron — nơi feature map có kích thước nhỏ nhất nhưng nhiều channels nhất, chứa biểu diễn cô đọng nhất |

## C

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **Channel** | Chiều thứ ba của feature map (sau height, width). Mỗi channel mã hóa một loại đặc trưng. Input RGB có 3 channels |
| **Channel Attention** | Cơ chế attention áp dụng trên chiều channels: học trọng số cho từng channel. VD: SE Attention |
| **Class Imbalance** | Tình trạng các lớp phân bố không đồng đều. Segmentation: foreground thường chỉ 1-10% ảnh |
| **CNN** | Convolutional Neural Network — mạng nơ-ron dùng phép tích chập thay vì fully connected, chuyên cho dữ liệu dạng lưới (ảnh) |
| **Conv1x1** | Pointwise convolution — kernel 1×1, chỉ trộn giữa các channels, không trộn không gian |
| **Conv3x3** | Tích chập kernel 3×3 — kích thước nhỏ nhất bắt được pattern không gian (cần pixel trung tâm + 8 lân cận) |
| **ConvTranspose2d** | Transposed convolution — phép toán ngược trong không gian của convolution, dùng để upsample, có tham số học được |
| **Cross-Entropy** | Hàm đo khoảng cách giữa 2 phân phối xác suất. BCE = binary cross-entropy cho 2 lớp |
| **Curriculum Learning** | Chiến lược huấn luyện: bắt đầu từ mẫu dễ, tăng dần độ khó. FANet tự nhiên làm được điều này qua feedback loop |

## D

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **Decoder** | Bộ phận tăng kích thước không gian, khôi phục mask từ biểu diễn nén của encoder |
| **Deep Learning** | Nhánh của machine learning dùng mạng nơ-ron nhiều tầng để tự động học biểu diễn phân cấp |
| **Dice Coefficient (F1)** | $\frac{2\|P \cap G\|}{\|P\|+\|G\|}$ — metric đo độ chồng lấn, trung bình điều hòa của Precision và Recall |
| **Dice Loss** | $1 - \text{Dice}(P, G)$ — loss function dựa trên Dice coefficient, bất biến với class imbalance |
| **Differentiable** | Tính chất "khả vi" — hàm có đạo hàm tại mọi điểm. Cần thiết để backpropagation hoạt động. Sigmoid output dùng trong Dice loss là differentiable |

## E

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **Encoder** | Bộ phận giảm kích thước không gian, tăng channels để trích xuất đặc trưng ngữ nghĩa cao cấp |
| **Encoder-Decoder** | Kiến trúc gồm 2 phần: encoder nén input, decoder giải nén ra output. U-Net là một encoder-decoder |
| **Epoch** | Một lần duyệt qua **toàn bộ** tập huấn luyện |

## F

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **F2-score** | F-beta với $\beta=2$ — Recall được coi trọng gấp 4 lần Precision ($\beta^2=4$) |
| **Feature Map** | Đầu ra của một tầng CNN, dạng (C×H×W). Tầng càng sâu, feature map càng nhỏ về không gian, càng nhiều channels |
| **Feedback** | Vòng lặp phản hồi: output của hệ thống được đưa ngược lại làm input cho lần chạy sau |
| **FLOPs** | Floating Point Operations — số phép tính dấu phẩy động, đo độ phức tạp tính toán của model |
| **Foreground** | Vùng ảnh chứa đối tượng cần phân đoạn (tiền cảnh) |
| **Forward Pass** | Quá trình dữ liệu đi từ input qua các tầng đến output |
| **FPS** | Frames Per Second — số ảnh xử lý được mỗi giây, đo tốc độ inference |

## G

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **Generalization** | Khả năng mô hình hoạt động tốt trên dữ liệu chưa từng thấy (test set) |
| **Global Average Pooling (GAP)** | Tính trung bình toàn bộ không gian của mỗi channel → vector C chiều. Dùng trong SE Attention |
| **Gradient** | Vector đạo hàm riêng của loss theo từng tham số. Chỉ hướng loss tăng nhanh nhất → đi ngược lại để giảm loss |
| **Gradient Descent** | Thuật toán cập nhật tham số theo hướng ngược gradient: $\theta \leftarrow \theta - \eta \nabla \mathcal{L}$ |
| **Ground Truth** | Nhãn đúng do con người gán — "đáp án chuẩn" để so sánh |

## H

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **Hand-crafted Features** | Đặc trưng do con người thiết kế thủ công (SIFT, HOG, Gabor…), trái ngược với learned features của deep learning |
| **Hard Attention** | Mặt nạ attention nhị phân {0,1} — hoặc giữ nguyên feature (×1), hoặc xóa hoàn toàn (×0) |
| **Hyperparameter** | Tham số do con người chọn trước khi train (learning rate, batch size, số epochs…), không được học tự động |

## I

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **Inference** | Quá trình dùng mô hình đã train để dự đoán trên dữ liệu mới. Chỉ có forward pass, không backward pass |
| **Internal Covariate Shift** | Hiện tượng phân phối activation thay đổi liên tục trong quá trình training — BN giải quyết vấn đề này |
| **IoU (Jaccard)** | $\frac{\|P \cap G\|}{\|P \cup G\|}$ — metric đo độ chồng lấn, nhạy hơn Dice |
| **Iteration** | Một lần cập nhật trọng số (= xử lý 1 batch) |

## K

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **Kernel (Filter)** | Ma trận trọng số nhỏ (3×3, 5×5…) dùng trong phép tích chập, trượt trên toàn bộ ảnh |

## L

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **Learning Rate ($\eta$)** | Độ lớn của mỗi bước cập nhật trong gradient descent. Quá lớn → vượt quá tối ưu; quá nhỏ → học chậm |
| **Loss Function** | Hàm đo độ sai khác giữa dự đoán và ground truth. Mô hình học bằng cách tối thiểu hóa hàm này |

## M

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **MaxPool** | Giảm kích thước bằng cách lấy max trong mỗi cửa sổ (thường 2×2). Không có tham số |
| **MixPool** | Khối attention chính của FANet — gating feature map bằng fmask ∨ prev_mask |

## O

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **Optimizer** | Thuật toán cập nhật tham số để giảm loss (SGD, Adam, RMSprop…) |
| **Otsu's Method** | Thuật toán tự động chọn ngưỡng bằng cách tối đa hóa between-class variance |
| **Overfitting** | Mô hình học quá khớp dữ liệu train → performance kém trên dữ liệu mới |

## P

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **Pixel-wise Classification** | Bài toán phân loại từng pixel độc lập |
| **Precision** | $\frac{TP}{TP+FP}$ — tỉ lệ dự đoán foreground đúng / tổng dự đoán foreground |
| **Pruning** | Loại bỏ (tỉa) các phần không cần thiết của feature map |

## R

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **Recall (Sensitivity)** | $\frac{TP}{TP+FN}$ — tỉ lệ foreground thật được phát hiện |
| **Receptive Field** | Vùng ảnh gốc mà một neuron "nhìn thấy". Tầng càng sâu, receptive field càng lớn |
| **ReduceLROnPlateau** | Scheduler tự động giảm learning rate khi validation loss ngừng cải thiện |
| **Regularization** | Kỹ thuật chống overfitting: dropout, weight decay, augmentation, BN… |
| **ReLU** | $\text{ReLU}(x) = \max(0, x)$ — hàm kích hoạt phi tuyến phổ biến nhất |
| **Residual Connection** | $y = \mathcal{F}(x) + x$ — kết nối tắt giúp gradient truyền thẳng, cho phép train mạng rất sâu |
| **RLE (Run-Length Encoding)** | Nén mask bằng cách lưu vị trí bắt đầu + độ dài của các đoạn foreground liên tiếp |

## S

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **Scheduler** | Cơ chế điều chỉnh learning rate trong quá trình training |
| **SE (Squeeze-and-Excitation)** | Channel attention: GAP → MLP → Sigmoid → scale channels |
| **Segmentation** | Bài toán phân chia ảnh thành các vùng có ý nghĩa (phân đoạn) |
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ — hàm kích hoạt ép output về $[0, 1]$, dùng ở tầng cuối cho binary classification |
| **Skip Connection** | Kết nối nối trực tiếp feature map từ encoder sang decoder cùng cấp |
| **Soft Attention** | Trọng số attention liên tục $\alpha \in [0,1]$, nhân element-wise |
| **Specificity** | $\frac{TN}{TN+FP}$ — tỉ lệ background được phân loại đúng |

## T

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **Test-time Refinement** | Cải thiện prediction bằng cách chạy model nhiều lần, dùng output lần trước làm input lần sau — không cần train lại |
| **Thresholding** | Chuyển ảnh xám → nhị phân: pixel > T → 1, pixel ≤ T → 0 |
| **Transposed Convolution** | Phép toán upsample có tham số học được |

## U

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **U-Net** | Kiến trúc encoder-decoder hình chữ U, có skip connections, kinh điển cho biomedical segmentation |
| **Upsampling** | Tăng kích thước không gian của feature map |

## V

| Thuật ngữ | Định nghĩa |
|:---|:---|
| **Validation Set** | Tập dữ liệu dùng để đánh giá mô hình **trong quá trình training** — không dùng để train, chỉ để theo dõi overfitting và điều chỉnh hyperparameter |
| **Vanishing Gradient** | Gradient trở nên cực nhỏ ở các tầng đầu khi train mạng sâu → các tầng đầu hầu như không học được. Residual connection + BN + ReLU giúp giảm thiểu |
| **Vanishing Gradient** | Khi train mạng quá sâu, gradient truyền ngược bị nhân với nhiều trọng số nhỏ → suy giảm về 0 ở các tầng đầu → không học được |

---

# Tài liệu tham khảo

- **Bài báo gốc**: Nikhil Kumar Tomar et al., *"FANet: A Feedback Attention Network for Improved Biomedical Image Segmentation"*, IEEE Transactions on Neural Networks and Learning Systems, 2022. arXiv: [2103.17235](https://arxiv.org/abs/2103.17235)
- **Source code**: [github.com/nikhilroxtomar/FANet](https://github.com/nikhilroxtomar/FANet)
- **U-Net**: Ronneberger et al., *"U-Net: Convolutional Networks for Biomedical Image Segmentation"*, MICCAI 2015
- **SENet**: Hu et al., *"Squeeze-and-Excitation Networks"*, CVPR 2018
- **ResNet**: He et al., *"Deep Residual Learning for Image Recognition"*, CVPR 2016
