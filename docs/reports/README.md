# Research Reports — GDF-Netv2 Multi-Benchmark Project

Index tổng hợp các báo cáo nghiên cứu và kết quả thực nghiệm của dự án GDF-Netv2 (General Decoupled Feedback Network v2).

## Timeline

| # | File | Ngày | Nội dung chính |
|---|------|------|----------------|
| 31 | [2026-10-19_phase8-gdfnet-v2-benchmarks.md](2026-10-19_phase8-gdfnet-v2-benchmarks.md) | 19/10/2026 | **Phase 8: Nghiệm thu Toàn diện Thực nghiệm Độc lập GDF-Netv2 trên 7 Tập Dữ liệu Lâm sàng (9 Metrics)**: Triển khai 150 epochs trên Kaggle GPU; CVC-ClinicDB đạt Mean Dice **0.9437** (Median **0.9631**); Kvasir-SEG đạt Mean Dice **0.9181** (Median **0.9641**); EM-Dataset đạt kỷ lục Mean Dice **0.9660** (Median **0.9675**); DSB-2018 đạt Mean Dice **0.8912** (Precision 90.63%); CHASE-DB1 đạt Mean Dice **0.7902** (+8.09 pp, Sensitivity 81.50%); ISIC-2018 đạt Mean Dice **0.8935** (Median **0.9253**, Sensitivity **95.01%** trên 1.000 blind test images); DRIVE thiết lập baseline từ scratch **0.2356** sẵn sàng transfer learning. |

## Experiment Tracking

| Giả thuyết / Mục đích | Trạng thái | Kết quả chính | Artifacts |
|----------------------|------------|---------------|-----------|
| GDF-Netv2 vượt baseline trên CVC-ClinicDB (N=61 test) | Hoàn thành | Dice Mean **0.9437**, Median **0.9631**; mIoU Mean **0.8987**, Median **0.9289**; FPR **0.0049**; ASD **1.72 px** | `GDF-Netv2/results/cvc_clinicdb/benchmark_summary_CVC.csv`, `GDFNetV2_CVC_best.pth` |
| GDF-Netv2 vượt baseline trên Kvasir-SEG (N=100 test) | Hoàn thành | Dice Mean **0.9181**, Median **0.9641**; mIoU Mean **0.8701**, Median **0.9307**; Sensitivity **0.9520**; ASD **2.05 px** | `GDF-Netv2/results/kvasir_seg/benchmark_summary_Kvasir.csv`, `GDFNetV2_Kvasir_best.pth` |
| GDF-Netv2 trên lát cắt EM-Dataset (512x512, N=6 test) | Hoàn thành | Dice Mean **0.9660**, Median **0.9675**, Max **0.9780**; mIoU Mean **0.9344**; BF2 **0.9803**; ASD **0.61 px** | `GDF-Netv2/results/em_dataset/benchmark_summary_EMDataset.csv`, `GDFNetV2_EMDataset_best.pth` |
| GDF-Netv2 trên tổn thương da ISIC-2018 (384x384, N=1,000 blind test) | Hoàn thành | Dice Mean **0.8935**, Median **0.9253** (+2.11 pp); Sensitivity **0.9501** (+6.10 pp); mIoU **0.8218** | `GDF-Netv2/results/isic2018/benchmark_summary_ISIC2018.csv`, `GDFNetV2_ISIC2018_best.pth` |
| GDF-Netv2 trên nhân tế bào DSB-2018 (256x256, N=67 test) | Hoàn thành | Dice Mean **0.8912**, Median **0.9149**; Precision **0.9063** (+6.15 pp); FPR **0.0139**; ASD **0.72 px** | `GDF-Netv2/results/dsb2018/benchmark_summary_DSB2018.csv`, `GDFNetV2_DSB2018_best.pth` |
| GDF-Netv2 trên vi mạch võng mạc CHASE-DB1 (512x512, N=4 test) | Hoàn thành | Dice Mean **0.7902**, Median **0.7843** (+8.09 pp); Sensitivity **0.8150** (+20.86 pp); BF2 **0.9097** | `GDF-Netv2/results/chasedb1/benchmark_summary_CHASE-DB1.csv`, `GDFNetV2_CHASE-DB1_best.pth` |
| GDF-Netv2 trên vi mạch võng mạc DRIVE (512x512, N=6 test) | Hoàn thành (scratch) | Dice Mean **0.2356**, Median **0.2471** (baseline huấn luyện từ scratch với 14 ảnh; sẵn sàng transfer learning) | `GDF-Netv2/results/drive/benchmark_summary_DRIVE.csv`, `GDFNetV2_DRIVE_best.pth` |
