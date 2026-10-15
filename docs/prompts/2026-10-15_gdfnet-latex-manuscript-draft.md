# Prompt: Soan Thao Ban Draft Manuscript LaTeX Cho GDF-Net (15/10/2026)

## 1. Yeu cau cua User
- Dua vao huong nghien cuu propose model GDF-Net (Gradient-Decoupled Feedback Network) va ket qua hien tai tren 7 tap du lieu benchmark.
- Soan thao mot ban draft paper hoan chinh bang LaTeX (chuan bi cho submit Overleaf / IEEE Transactions format).
- Bo cuc hinh anh so sanh theo yeu cau dac thu: Training Convergence Curves o tren va Qualitative Predictions o duoi.
- Yeu cau cap nhat quan trong:
  1. Thay the thuat ngu MFAD thanh GDF-Net (hoac Multi-scale Gradient-Decoupled Feedback Decoder).
  2. Loai bo hoan toan moi nhac den paper FANet goc (Tomar et al., 2021), chuyen sang mo hinh de xuat doc lap GDF-Net giai quy van de tong quat cua cac mang recurrent feedback thong thuong.
  3. Phai viet phan hoi va trao doi bang tieng Anh.

## 2. Thuc thi
- Cau truc thu muc Overleaf tai `papers/overleaf_submission/`:
  - `main.tex`: Dinh dang IEEEtran hai cot, tieu de doc lap ve GDF-Net, abstract loai bo toan bo thuat ngu FANet va MFAD.
  - `sections/01_intro.tex`: Dat van de ve Feedback Trap trong mang hoi tiep y te noi chung, nguyen ly Feedback Firewall, va 4 dong gop cot loi cua GDF-Net.
  - `sections/02_related.tex`: Tong quan y van biomedical segmentation va recurrent feedback vision networks noi chung, stop-gradient trong self-supervised learning (SimSiam, BYOL).
  - `sections/03_method.tex`: Chung minh toan hoc hien tuong te liet gradient va ro ri gradient hoi tiep; dan xuat ky thuat Gradient Decoupling va Multi-scale Gradient-Decoupled Feedback Decoder.
  - `sections/04_experiments.tex`: Bang 1 Master Benchmark tren 7 tap du lieu vuot troi baselines; Hinh 1 (Convergence curves top / Qualitative predictions bottom); Bang 2 phan tich dong luc hoc hoi tiep qua cac pass ($t=1, 2, 4$); Bang 3 phan tich cat bo (ablation studies).
  - `sections/05_conclusion.tex`: Ket luan, y nghia lam sang (giam 65% false positive mach mau mat, chong tran vung phan doan), gioi han va dinh huong tuong lai.
  - `refs.bib`: Loai bo toan bo citation lien quan den paper FANet goc.
- Kiem tra bien dich cuc bo voi MiKTeX: `pdflatex` va `bibtex` bien dich thanh cong ra file `papers/overleaf_submission/main.pdf` (10 trang).
