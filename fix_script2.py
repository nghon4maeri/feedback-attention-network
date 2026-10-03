with open('scripts/push_benchmark_kernels.py', 'r', encoding='utf-8') as f:
    text = f.read()
text = text.replace('"fanet_benchmark_', '"benchmark_v2/fanet_vs_mfad_')
with open('scripts/push_benchmark_kernels.py', 'w', encoding='utf-8', newline='
') as f:
    f.write(text)
