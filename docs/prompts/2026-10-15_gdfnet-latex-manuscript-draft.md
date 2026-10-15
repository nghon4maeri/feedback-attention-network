# Prompt: Soan Thao Ban Draft Manuscript LaTeX Cho GDF-Net (15/10/2026)

## 1. Yeu cau cua User
- Dua vao huong nghien cuu propose model GDF-Net (Gradient Decoupling Feedback / FANet-MFAD) va ket qua hien tai tren 7 tap du lieu benchmark.
- Soan thao mot ban draft paper hoan chinh bang LaTeX (chuan bi cho submit Overleaf / IEEE Transactions format).
- Tham khao toan bo cac tai lieu, bao cao va bang chung thuc nghiem trong repo.
- Bo cuc hinh anh so sanh theo yeu cau dac thu: Training Convergence Curves o tren va Qualitative Predictions o duoi.

## 2. Thuc thi
- Cau truc thu muc Overleaf tai `papers/overleaf_submission/`:
  - `main.tex`: Dinh dang IEEEtran hai cot, cau hinh day du goi lenh toan hoc, TikZ, bang bieu va tham chieu.
  - `sections/01_intro.tex`: Dat van de ve Feedback Trap trong mang hoi tiep y te, nguyen ly Feedback Firewall, va 4 dong gop cot loi.
  - `sections/02_related.tex`: Tong quan y van tu U-Net, Attention U-Net, FANet, PraNet den ky thuat stop-gradient trong self-supervised learning (SimSiam, BYOL).
  - `sections/03_method.tex`: Chung minh toan hoc hien tuong te liet gradient va ro ri gradient hoi tiep; dan xuat ky thuat Gradient Decoupling va khoi giai ma hoi tiep da ty le MFAD.
  - `sections/04_experiments.tex`: Bang 1 Master Benchmark tren 7 tap du lieu vuot troi baseline; Hinh 1 kien truc so sanh hoi tu o tren va du doan dinh tinh o duoi; Bang 2 doi chieu phap y y van (Tomar et al. Bang IX); Bang 3 phan tich cat bo (ablation studies).
  - `sections/05_conclusion.tex`: Ket luan, y nghia lam sang (giam 65% false positive mach mau vong mac, chong tran vung phan doan), gioi han va dinh huong tuong lai.
  - `refs.bib`: Bo sung day du cac trich dan hoc thuat tieu chuan.
  - `figures/`: Tap hop day du bieu do hoi tu va hinh anh du doan phan doan dinh tinh cho CVC-ClinicDB, EM-Dataset, ISIC-2018 va CHASE-DB1.
- Kiem tra bien dich cuc bo voi MiKTeX: `pdflatex` va `bibtex` bien dich thanh cong ra file `papers/overleaf_submission/main.pdf` (10 trang).
