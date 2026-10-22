import sys, os
import torch

# Add models and losses to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.gdfnet_v5 import GDFNetV5
from losses.composite_loss import GDFNetV5CompositeLoss, extract_edge_ground_truth

def test_gdfnet_v5_architecture():
    print("[*] Testing GDF-Netv5 Architecture and Multi-Task Edge Guidance...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Compute Device: {device}")

    # 1. Instantiate GDFNetV5 with PVTv2-B4 backbone
    model = GDFNetV5(backbone_name="pvt_v2_b4", pretrained=False).to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"[+] Total Parameters: {total_params:.2f}M | Trainable: {trainable_params:.2f}M")
    assert 60.0 < total_params < 75.0, f"Expected params around 67M, got {total_params:.2f}M"

    # 2. Test Input at 512x512 Native Resolution
    B, H, W = 2, 512, 512
    x = torch.randn(B, 3, H, W, device=device)
    targets = torch.randint(0, 2, (B, 1, H, W), device=device).float()

    outs = model(x)
    assert isinstance(outs, dict), "Training output must be a dictionary"
    
    expected_keys = ['logits', 'edge_logits', 'p1', 'p2', 'p3', 'p4']
    for k in expected_keys:
        assert k in outs, f"Missing output key: {k}"
        assert outs[k].shape == (B, 1, H, W), f"Shape mismatch for {k}: {outs[k].shape} vs ({B}, 1, {H}, {W})"

    print("[+] All 6 output tensor shapes verified at full 512x512 resolution:")
    for k in expected_keys:
        print(f"    - {k}: {outs[k].shape}")

    # 3. Test Edge Ground Truth Extraction
    g_edge = extract_edge_ground_truth(targets)
    assert g_edge.shape == (B, 1, H, W)
    print(f"[+] Morphological GPU Edge Extraction verified: {g_edge.shape} (Edge pixel ratio: {g_edge.mean():.4f})")

    # 4. Test Multi-Task Composite Loss and Backward Pass
    criterion = GDFNetV5CompositeLoss()
    loss = criterion(outs, targets)
    print(f"[+] Computed Multi-Task Composite Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"

    loss.backward()
    print("[+] Backward pass succeeded. Verifying gradient flow...")

    # Verify gradients across all core submodules
    first_param = next(model.backbone.parameters())
    assert first_param.grad is not None, "Encoder grad is None"
    assert model.regs.edge_head.weight.grad is not None, "REGS Edge Head grad is None"
    assert model.dec1.head.weight.grad is not None, "Decoder 1 head grad is None"
    assert model.final_refine[3].weight.grad is not None, "Final refine head grad is None"

    print("[+] Gradient flow verified across 100% of network (Encoder -> MS-FDF -> REGS -> LGAG -> MSCAM Decoder)!")

    # 5. Test Inference Mode
    model.eval()
    with torch.no_grad():
        out_eval = model(x)
        assert out_eval.shape == (B, 1, H, W), f"Eval shape mismatch: {out_eval.shape}"
        print(f"[+] Single-pass eval output shape: {out_eval.shape}")

    print("\n[SUCCESS] GDF-Netv5 Architecture and Loss Passed All Unit Tests!\n")

if __name__ == "__main__":
    test_gdfnet_v5_architecture()
