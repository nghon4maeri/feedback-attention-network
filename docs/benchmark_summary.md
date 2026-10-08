# 🏆 Kết quả Benchmark MFAD (Bản thu gọn)

> So sánh trực tiếp hiệu năng giữa FANet (Baseline) $\rightarrow$ **MFAD (Ours)**. 
> Chỉ tập trung vào 3 chỉ số quan trọng nhất phản ánh sự ưu việt của mô hình: Dice (Độ chính xác tổng thể), Precision (Độ chính xác dương tính) và FPR (Tỷ lệ sai dương - Halo effect).

| Tập dữ liệu | Dice Score (Orig $\rightarrow$ MFAD) | Precision (Orig $\rightarrow$ MFAD) | FPR (Orig $\rightarrow$ MFAD) |
| :--- | :--- | :--- | :--- |
| **CVC-ClinicDB** | 0.8958 $\rightarrow$ **0.9369** <span style="color:green">(+4.1%)</span> | 0.8968 $\rightarrow$ **0.9652** <span style="color:green">(+6.8%)</span> | 0.96% $\rightarrow$ **0.30%** <span style="color:green">(-3x)</span> |
| **Kvasir-SEG** | 0.8852 $\rightarrow$ **0.9100** <span style="color:green">(+2.4%)</span> | **0.9140** $\rightarrow$ 0.8825 <span style="color:red">(-3.1%)</span> | **1.12%** $\rightarrow$ 2.61% |
| **DSB-2018** | 0.8721 $\rightarrow$ **0.8855** <span style="color:green">(+1.3%)</span> | 0.8448 $\rightarrow$ **0.8686** <span style="color:green">(+2.3%)</span> | 2.18% $\rightarrow$ **2.13%** |
| **EM-Dataset** | 0.9379 $\rightarrow$ **0.9437** <span style="color:green">(+0.5%)</span> | 0.9150 $\rightarrow$ **0.9572** <span style="color:green">(+4.2%)</span> | 0.60% $\rightarrow$ **0.27%** <span style="color:green">(-2x)</span> |
| **CHASE-DB1** | **0.8250** $\rightarrow$ 0.8026 <span style="color:red">(-2.2%)</span> | **0.8044** $\rightarrow$ 0.8027 <span style="color:red">(-0.1%)</span> | 1.61% $\rightarrow$ **1.55%** |

*Ghi chú:*
- CVC, DSB, EM: MFAD giải quyết triệt để nhiễu bao quanh, đẩy mạnh Precision và Dice.
- CHASE (vi mạch): Việc ngắt gradient của MFAD gây bất lợi cho cấu trúc mỏng/đứt khúc.
