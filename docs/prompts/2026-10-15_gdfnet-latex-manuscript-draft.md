# Prompt: Soan Thao Ban Draft Manuscript LaTeX Cho GDF-Net (15/10/2026)

## 1. Yeu cau cua User
- Dua vao huong nghien cuu propose model GDF-Net (Gradient-Decoupled Feedback Network) va ket qua hien tai tren 7 tap du lieu benchmark.
- Soan thao mot ban draft paper hoan chinh bang LaTeX (chuan bi cho submit Overleaf / IEEE Transactions format).
- Bo cuc hinh anh so sanh theo yeu cau dac thu: Training Convergence Curves o tren va Qualitative Predictions o duoi.
- Yeu cau cap nhat quan trong:
  1. Thay the thuat ngu MFAD thanh GDF-Net (hoac Multi-scale Gradient-Decoupled Feedback Decoder).
  2. Loai bo hoan toan moi nhac den paper FANet goc (Tomar et al., 2021), chuyen sang mo hinh de xuat doc lap GDF-Net giai quy van de tong quat cua cac mang recurrent feedback thong thuong.
  3. Phai viet phan hoi va trao doi bang tieng Anh.
  4. Ve lai so do kien truc TikZ (hinh a, b) va cac bang du lieu benchmark/ablation khong de bi de chu, tran le hoac mat dinh dang:
     - So do TikZ tach biet hoan toan khoi van ban cot ben canh bang moi truong `figure*` hai cot, hai subfigure side-by-side (a) Baseline Hard Gating voi cac khoi mau chuan va mui ten do "Toxic Leakage", "Forward", "\nabla \equiv 0 (Blocked)"; (b) Proposed Detached Soft-OR Gating voi khoi xanh la, "Healthy Flow \nabla \propto (1 - m_fg)" va "detach() Severed".
     - Bang Master Benchmark (Table I) dung `makecell`, `booktabs` va `resizebox` chuan `\textwidth`, giu nguyen cac cot "Baseline Architecture", "GDF-Net (Ours, t=2)", "Delta (\Delta)", khong nhac den FANet.
     - Bang Ablation (Table III) tach biet ro rang phan "Cohort / Setup" ben tren va "Configuration" ben duoi, dat trong `table*` khong de chu cot ben canh chen de len.
  5. Tao hinh anh so do kien truc khoa hoc 2D ro net, nen trang (#FFFFFF), sua lai hoan toan loi chong cheo chu tu image_0.png, chia lam hai phan doc (a) va (b) tren cung mot trang khong co van ban tran tu ben phai:
     - Bố cục: Sans-serif font, net ve sac sao 300 DPI, khong bi mo hay dut gay.
     - Phan (a) Baseline (Hard Gating / Feedback Trap) o nua tren:
       + Khoi trung tam phia tren: "Backbone fmask" (xanh duong nhat) co mui ten xuong "Forward".
       + Khoi ngang 3 phan: "Recurrent Prior m_fg" (cam nhat), "Hard Gate Operator max(\mathbb{I}(fmask > 0.5), m_fg)" (do nhat), "Output Feature" (trang).
       + Mui ten net dut mau do tu "Recurrent m_fg" vao "Hard Gate" ghi "Toxic Leakage".
       + Duong dut do tu "Hard Gate" quay nguoc len "Backbone fmask" co "\nabla \equiv 0 (Blocked)".
       + Duong net dut do tu "Output Feature" di vong xuong duoi roi chi nguoc len ranh gioi giua "Recurrent" va "Hard Gate" (Recursive Error Amplification Loop).
     - Phan (b) Proposed (Detached Soft-OR Gating) o nua duoi:
       + Khoi trung tam phia tren: "Backbone fmask" (xanh duong nhat).
       + Khoi ben duoi: "Recurrent m_fg" (cam nhat), "Soft-OR Gate 1 - (1 - fmask)(1 - m_fg)" (xanh la nhat), "Output Feature" (trang).
       + Giua "Recurrent" va "Soft-OR Gate": hop vien dut do "detach() Severed" co ky hieu chan mau do (X). Mui ten dut do tu cong Soft-OR bi "Blocked at Firewall".
       + Ket noi hai chieu: mui ten Forward xuong va Healthy Flow "\nabla \propto (1 - m_fg)" len.
       + Mui ten net dut xanh la tu duoi chi vao Soft-OR Gate ("Supervision from Downstream Segmentation Loss").

## 2. Thuc thi
- Cau truc thu muc Overleaf tai `papers/overleaf_submission/`:
  - `main.tex`: Dinh dang IEEEtran hai cot, tieu de doc lap ve GDF-Net, abstract loai bo toan bo thuat ngu FANet va MFAD.
  - `figures/architecture_tikz.tex`: Thiet ke lai hoan toan thanh `figure*` voi 2 subfigure (a) va (b) can doi, chuan mau sac, font toan hoc khong chong cheo, khong tran cot.
  - `figures/architecture_diagram.png`: Hinh anh 2D do phan giai cao (300 DPI) nen trang hoan toan, bo cuc doc 2 tang (a) va (b), dung tuyet doi cac yeu cau ve khoi hop, mau sac, mui ten, ky hieu toan hoc va can le an toan.
  - `scripts/generate_architecture_diagram.py`: Script matplotlib tu dong hoa viec xuat ra file anh khoa hoc 300 DPI.
  - `sections/01_intro.tex`: Dat van de ve Feedback Trap trong mang hoi tiep y te noi chung, nguyen ly Feedback Firewall, va 4 dong gop cot loi cua GDF-Net.
  - `sections/02_related.tex`: Tong quan y van biomedical segmentation va recurrent feedback vision networks noi chung, stop-gradient trong self-supervised learning (SimSiam, BYOL).
  - `sections/03_method.tex`: Chung minh toan hoc hien tuong te liet gradient va ro ri gradient hoi tiep; dan xuat ky thuat Gradient Decoupling va Multi-scale Gradient-Decoupled Feedback Decoder. Tinh chinh can le cac phuong trinh mo ta feature encoder vua van mot cot.
  - `sections/04_experiments.tex`: Bang 1 Master Benchmark tren 7 tap du lieu vuot troi baselines; Hinh 1 (Convergence curves top / Qualitative predictions bottom); Bang 3 phan cap Part (a) Cohort/Setup va Part (b) Configuration.
  - `sections/05_conclusion.tex`: Ket luan, y nghia lam sang (giam 65% false positive mach mau mat, chong tran vung phan doan), gioi han va dinh huong tuong lai.
  - `refs.bib`: Loai bo toan bo citation lien quan den paper FANet goc.
- Kiem tra bien dich cuc bo voi MiKTeX: `pdflatex` va `bibtex` bien dich thanh cong ra file `papers/overleaf_submission/main.pdf` (10 trang) voi 0 loi overfull \hbox trong toan bo bang bieu va so do TikZ.

## 3. Cap nhat theo phan hoi (16/10/2026)
- **Table 1:** Loai bo hoan toan cac chi so cua baseline architecture (fanet goc) va cot Delta. Chi giu lai thong tin chi tiet ve ket qua thuc nghiem cua GDF-Net (Ours, t=2) tren tat ca 7 biomedical benchmarks voi day du mean +- std cho DSC, mIoU, Sensitivity, Specificity, Precision, va FPR (%).
- **So do kien truc (Diagram as Code):** Su dung thu vien `mingrammer/diagrams` thong qua script `scripts/generate_diagrams_architecture.py` de ve lai ca hai kien truc Baseline (Hard Gating / Feedback Trap) va Proposed GDF-Net (Detached Soft-OR Gating). Xuat anh khoa hoc do phan giai cao (300 DPI) khong loi font, khong de chu, can le dep mat.
- **Training Convergence Curves & Qualitative Predictions:** Viet script `scripts/reformat_figures.py` trich xuat loss va validation DSC tu checkpoint that va log Kaggle, ve lai 4 do thi hoi tu voi font 14pt-16pt dam ro rang, kich thuoc lon tren toan bo chieu rong trang (`figure*` 2x2 grid); cat va dinh dang lai cac anh qualitative comparison ro net voi header chuan `Input Image | Ground Truth | GDF-Net (Ours)`.
- **Kiem tra bien dich:** `pdflatex` bien dich thanh cong ra `papers/overleaf_submission/main.pdf` (10 trang), khong con loi overfull vbox/hbox.
- **Tu dong hoa render architec.png va nhung LaTeX (16/10/2026):**
  - Tao script `scripts/generate_architec.py` xuat truc tiep `architec.png` (300 DPI) the hien day du (a) Baseline Scratch CNN voi MixPool Hard Gating va Feedback Trap va (b) Proposed GDF-Net voi Stem 4 kenh, ResNet-34 pre-trained ImageNet, va Decoupled MFAD Decoder 5 tang.
  - Cap nhat `papers/overleaf_submission/figures/architecture_tikz.tex` va `sections/03_method.tex` de nhung truc tiep `figures/architec.png` voi label `\label{fig:architecture}`.
  - Bien dich thanh cong `main.pdf` (10 trang) voi hinh anh ro net, bo cuc dep tren toan trang.
- **De trong 2 anh architec theo yeu cau cua User:**
  - Cap nhat `papers/overleaf_submission/figures/architecture_tikz.tex` thanh 2 hop placeholder trong suot vien net dut (TikZ dashed box) cho Subfigure (a) Baseline va Subfigure (b) Proposed GDF-Net de user tu add anh vao sau.
  - Bien dich lai `main.pdf` (10 trang) thanh cong, layout hoan chinh va khong loi.

