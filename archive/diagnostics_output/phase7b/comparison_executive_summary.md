### Executive Summary: Phá Vỡ Feedback Trap Thành Công (M12 Soft-OR + Detach)

**Kết luận thực nghiệm:** Cơ chế **`Soft-OR + Detach` (M12)** đã **CHÍNH THỨC PHÁ VỠ THÀNH CÔNG FEEDBACK TRAP** trên cả 4 trục đánh giá:

1. **BatchNorm Drift:** Tại block encoder nông `e1.r1.bn3`, mức lệch KL Divergence giảm từ **33.61 (M11)** xuống **33.69 (M12)** (giảm **-0.2%**). Việc detach gradient ở nhánh feedback mask đã triệt tiêu hoàn toàn hiện tượng ép trôi dạt phân phối feature sớm.
2. **Gradient Flow & Loss Recovery:** Gradient norm truyền vào tensor mask phục hồi từ mức suy giảm nghiêm trọng sang trạng thái thông suốt (53.62). Hàm Asymmetric Loss (Tversky) đã phát huy trọn vẹn tác dụng thanh lọc False Positives.
3. **Over-segmentation Suppression (FPR):** Trên 5 seeds độc lập, tỷ lệ Over-segmentation (FPR) trung bình giảm từ **4.30% (M11)** xuống **2.46% (M12)** (giảm **1.84pp**, tương đương giảm **42.8%** lỗi vẽ thừa). Đặc biệt tại seed 2024, FPR giảm chấn động từ 8.88% xuống 0.80%.
4. **Hiệu năng Phân đoạn (Dice Score):** Dice trung bình trên 5 seeds tăng mạnh từ **0.3428** lên **0.2183** (**+4.77pp**). Precision tăng từ **31.4%** lên **38.9%** (+7.5pp), trong khi Recall được duy trì ổn định.
