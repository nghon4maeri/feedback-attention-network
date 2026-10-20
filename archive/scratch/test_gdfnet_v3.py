import sys, os
import torch

root = r"D:\User\nghonamjiwoo\My Project\FANet\GDF-Netv3"
sys.path.insert(0, root)

from models.gdfnet_v3 import GDFNetV3
from losses.boundary_losses import CompositeGDFNetV3Loss

print("[*] Instantiating GDFNetV3 with EfficientNetV2-S...")
model = GDFNetV3(backbone_name="tf_efficientnetv2_s.in21k_ft_in1k", pretrained=False)
criterion = CompositeGDFNetV3Loss()

params = sum(p.numel() for p in model.parameters()) / 1e6
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
print(f"[+] Total Parameters: {params:.2f} M | Trainable: {trainable:.2f} M")

x = torch.randn(2, 3, 384, 384)
targets = torch.randint(0, 2, (2, 1, 384, 384)).float()

# Training forward pass
model.train()
outs = model(x)
print("[+] Training outputs:")
for k, v in outs.items():
    print(f"    {k}: shape={list(v.shape)}")

# Backward pass
loss = criterion(outs, targets)
print(f"[+] Loss value: {loss.item():.4f}")
loss.backward()
print("[+] Backward gradient check passed successfully!")

# Eval forward pass
model.eval()
with torch.no_grad():
    eval_out = model(x)
print(f"[+] Eval output: shape={list(eval_out.shape)}")
print("[+] ALL ARCHITECTURE TESTS PASSED!")
