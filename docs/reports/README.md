# Research Reports — GDF-Netv2 Multi-Benchmark Project

Index tổng hợp các báo cáo nghiên cứu và kết quả thực nghiệm của dự án GDF-Netv2 (General Decoupled Feedback Network v2).

## Timeline

| # | File | Ngày | Nội dung chính |
|---|------|------|----------------|
| 31 | [2026-10-19_phase8-gdfnet-v2-benchmarks.md](2026-10-19_phase8-gdfnet-v2-benchmarks.md) | 19/10/2026 | **Phase 8: Nghiệm thu Thực nghiệm Độc lập GDF-Netv2 trên CVC-ClinicDB & Kvasir-SEG (9 Metrics)**: Triển khai 150 epochs trên Kaggle GPU; CVC-ClinicDB phá kỷ lục với Mean Dice **0.9437**, Median Dice **0.9631**, FPR cực thấp 0.49%; Kvasir-SEG đạt Mean Dice **0.9181**, Median Dice **0.9641**; giải quyết triệt để lỗi OOM/cuFFT qua Linear SBA + Spatial FDF; khởi động tiếp DRIVE & CHASE-DB1. |

## Experiment Tracking

| Giả thuyết / Mục đích | Trạng thái | Kết quả chính | Artifacts |
|----------------------|------------|---------------|-----------|
| GDF-Netv2 vượt baseline trên CVC-ClinicDB (N=61 test) | ✅ Done | Dice Mean **0.9437**, Median **0.9631**; mIoU Mean **0.8987**, Median **0.9289**; FPR **0.0049**; ASD **1.72 px** | `GDF-Netv2/results/cvc_clinicdb/benchmark_summary_CVC.csv`, `GDFNetV2_CVC_best.pth` |
| GDF-Netv2 vượt baseline trên Kvasir-SEG (N=100 test) | ✅ Done | Dice Mean **0.9181**, Median **0.9641**; mIoU Mean **0.8701**, Median **0.9307**; Sensitivity **0.9520**; ASD **2.05 px** | `GDF-Netv2/results/kvasir_seg/benchmark_summary_Kvasir.csv`, `GDFNetV2_Kvasir_best.pth` |
| GDF-Netv2 trên vi mạch võng mạc DRIVE (512x512, 16x Aug) | ⏳ Đang chạy | Kernel `namnguynnnn/gdfnet-v2-drive` đang train trên Kaggle GPU | `GDF-Netv2/kaggle/drive/` |
| GDF-Netv2 trên vi mạch võng mạc CHASE-DB1 (512x512, 16x Aug) | ⏳ Đang chạy | Kernel `namnguynnnn/gdfnet-v2-chasedb1` đang train trên Kaggle GPU | `GDF-Netv2/kaggle/chasedb1/` |
| GDF-Netv2 trên tổn thương da ISIC-2018 (384x384, N=1,000 blind test) | ⏳ Đang chạy | Session `gdfnet-v2-isic` đang train trên Colab T4 GPU | `GDF-Netv2/training/colab_run_gdfnet_v2_isic2018.py` |
| GDF-Netv2 trên nhân tế bào DSB-2018 (256x256) | 📋 Chờ GPU | Đã sẵn sàng script và metadata tại `GDF-Netv2/kaggle/dsb2018/` | `GDF-Netv2/kaggle/dsb2018/` |
| GDF-Netv2 trên lát cắt EM-Dataset (512x512, 16x Aug) | 📋 Chờ GPU | Đã sẵn sàng script và metadata tại `GDF-Netv2/kaggle/em_dataset/` | `GDF-Netv2/kaggle/em_dataset/` |
