import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleFrequencyFilterV5(nn.Module):
    """
    Multi-Scale Frequency-Decoupled Filter (MS-FDF v5).
    
    Decouples multi-scale spatial representations:
    1. Multi-scale Low-Frequency Context (x_low): Extracted via parallel depthwise
       kernels [1x1, 3x3, 5x5] to capture homogeneous skin tone and broad illumination gradients.
    2. Multi-scale High-Frequency Residual (x_high = x - x_low): Captures sharp lesion edges,
       pigment networks, and fine boundary transitions.
    3. Gated High-Frequency Enhancement: Uses semantic low-frequency context to suppress
       spurious high-frequency noise (e.g. hair, dermoscope grid marks) while boosting true perimeter edges.
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        mid_channels = max(in_channels // reduction, 16)

        # Parallel multi-scale depthwise convolutions for low-pass extraction
        self.dw_conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
        self.dw_conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.dw_conv5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False)
        
        self.pw_low = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        # High-frequency attention gate: extracts semantic guidance from x_low
        self.hf_gate = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

        # Fusion projection
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1. Multi-scale low-pass feature
        low1 = self.dw_conv1(x)
        low3 = self.dw_conv3(x)
        low5 = self.dw_conv5(x)
        x_low = self.pw_low(torch.cat([low1, low3, low5], dim=1))

        # 2. High-pass residual
        x_high = x - x_low

        # 3. Gated high-frequency enhancement
        gate = self.hf_gate(x_low)
        x_high_filtered = x_high * gate

        # 4. Fusion + Residual
        out = self.fusion(torch.cat([x_low, x_high_filtered], dim=1))
        return out + x
