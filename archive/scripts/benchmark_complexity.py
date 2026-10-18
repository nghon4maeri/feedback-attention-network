import os
import sys
import time
import torch
import torch.nn as nn
from pathlib import Path
from ptflops import get_model_complexity_info

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from fanet.models import FANet

class ModelWrapper(nn.Module):
    """Wrapper to handle FANet's list input format [img, mask] for ptflops."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        b, c, h, w = x.size()
        m = torch.zeros((b, 1, h, w), device=x.device)
        return self.model([x, m])

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def benchmark_fps(model, device, size=(256, 256), warmup=50, iters=300):
    model.eval()
    dummy_img = torch.randn(1, 3, size[0], size[1], device=device)
    dummy_mask = torch.randn(1, 1, size[0], size[1], device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model([dummy_img, dummy_mask])
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        for _ in range(iters):
            _ = model([dummy_img, dummy_mask])
    end_event.record()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        elapsed_time_ms = start_event.elapsed_time(end_event)
    else:
        # Fallback for CPU
        start_time = time.time()
        with torch.no_grad():
            for _ in range(iters):
                _ = model([dummy_img, dummy_mask])
        elapsed_time_ms = (time.time() - start_time) * 1000.0

    avg_latency_ms = elapsed_time_ms / iters
    fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0
    return fps

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Benchmarking on device: {device}")
    
    models_to_test = {
        "M11 (Baseline)": FANet(gating_mode="hard", detach_feedback=False),
        "M12 (Phase 7B)": FANet(gating_mode="soft_or", detach_feedback=True)
    }
    
    results = {}
    
    for name, model in models_to_test.items():
        print(f"--- Benchmarking {name} ---")
        model = model.to(device)
        
        params_m = count_parameters(model) / 1e6
        
        # FLOPs using ptflops
        wrapper = ModelWrapper(model)
        macs, _ = get_model_complexity_info(wrapper, (3, 256, 256), as_strings=False, print_per_layer_stat=False, verbose=False)
        flops_g = (macs * 2) / 1e9 # Usually 1 MAC = 2 FLOPs, we'll report MACs in G
        macs_g = macs / 1e9
        
        # FPS
        fps = benchmark_fps(model, device)
        
        results[name] = {
            "Params (M)": f"{params_m:.2f}",
            "MACs (G)": f"{macs_g:.2f}",
            "FPS": f"{fps:.1f}"
        }
        
        print(f"  Params: {params_m:.2f} M")
        print(f"  MACs:   {macs_g:.2f} G")
        print(f"  FPS:    {fps:.1f}\n")

    # Generate Markdown Table
    md_table = "### Computational Complexity Benchmark\n\n"
    md_table += "| Model | Parameters (M) | MACs (G) | Inference FPS |\n"
    md_table += "|---|---|---|---|\n"
    
    for name, res in results.items():
        md_table += f"| **{name}** | {res['Params (M)']} | {res['MACs (G)']} | {res['FPS']} |\n"
        
    print(md_table)
    
    out_dir = REPO_ROOT / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "complexity_benchmark.md"
    
    with open(out_file, "w") as f:
        f.write(md_table)
        
    print(f"Saved benchmark table to {out_file}")

if __name__ == "__main__":
    main()
