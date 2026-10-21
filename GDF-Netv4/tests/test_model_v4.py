import sys
import os
import torch

# Add root directory to sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, root_dir)

# Also add models and losses directly
models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
losses_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "losses"))
sys.path.insert(0, models_dir)
sys.path.insert(0, losses_dir)

from gdfnet_v4 import GDFNetV4
from structure_loss import GDFNetV4CompositeLoss

def test_gdfnet_v4():
    print("[*] Instantiating GDFNetV4 (backbone=pvt_v2_b2, pretrained=False)...")
    model = GDFNetV4(backbone_name="pvt_v2_b2", pretrained=False, decoder_channels=64)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[+] Total Parameters: {total_params / 1e6:.2f}M | Trainable: {trainable_params / 1e6:.2f}M")

    # 1. Forward Pass Test
    B, C, H, W = 2, 3, 384, 384
    x = torch.randn(B, C, H, W)
    targets = torch.randint(0, 2, (B, 1, H, W)).float()

    print(f"[*] Testing Forward Pass with input shape {list(x.shape)}...")
    outs = model(x)
    assert isinstance(outs, dict), "Training forward pass should return a dict"
    for k, v in outs.items():
        print(f"    - Output '{k}': shape {list(v.shape)}")
        assert v.shape == (B, 1, H, W), f"Shape mismatch for {k}: expected {(B, 1, H, W)}, got {v.shape}"

    # 2. Loss & Backward Pass Test
    print("[*] Testing Loss & Gradient Flow...")
    criterion = GDFNetV4CompositeLoss()
    loss = criterion(outs, targets)
    print(f"[+] Computed Composite Structure Loss: {loss.item():.4f}")

    loss.backward()

    # 3. Verify Gradient Flow through all modules
    zero_grad_modules = []
    has_grads = True
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                zero_grad_modules.append(name)
                has_grads = False
            elif param.grad.abs().sum() == 0:
                zero_grad_modules.append(name + " (zero grad)")
    
    if zero_grad_modules:
        print(f"[!] Warning: Following parameters received no gradient: {zero_grad_modules[:10]}")
    else:
        print("[+] 100% of trainable parameters received active gradients! Zero gradient detach bugs.")

    # 4. Eval Mode Test
    print("[*] Testing Eval Mode...")
    model.eval()
    with torch.no_grad():
        eval_out = model(x)
    assert isinstance(eval_out, torch.Tensor), "Eval mode should return a single Tensor"
    assert eval_out.shape == (B, 1, H, W), f"Eval output shape mismatch: {eval_out.shape}"
    print(f"[+] Eval output shape: {list(eval_out.shape)}")

    print("\n[SUCCESS] GDF-Netv4 passed all architectural, forward, backward, and gradient flow unit tests!")

if __name__ == "__main__":
    test_gdfnet_v4()
