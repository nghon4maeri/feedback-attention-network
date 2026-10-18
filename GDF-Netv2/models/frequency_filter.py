import torch
import torch.nn as nn
import torch.nn.functional as F

class FrequencyDecoupledFilter(nn.Module):
    """
    Frequency-Decoupled Filter (FDF) for Dermoscopic & Medical Feature Maps.
    
    Skin lesion and endoscopy images often suffer from high-frequency non-semantic artifacts
    (hair, air bubbles, ruler markings, specular glare) which confuse standard convolutions.
    
    FDF separates the feature map into:
    1. Low-Frequency Stream (Semantic Morphology): extracted via depthwise adaptive low-pass pooling.
    2. High-Frequency Stream (Boundary & Noise): residual difference X - LowPass(X).
    
    An adaptive gating mechanism selectively suppresses high-frequency noise while preserving
    sharp lesion boundaries before passing features to the decoder skip connections.
    """
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.in_channels = in_channels
        mid_channels = max(in_channels // reduction, 16)

        # Low-pass filter branch (smooth tissue structure)
        self.low_pass = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels)
        )

        # High-frequency gating (attenuates spurious hair/glare while preserving authentic borders)
        self.hf_gate = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1, groups=mid_channels),
            nn.Sigmoid()
        )

        # Fusion projection
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 1. Extract low-frequency semantic base
        x_low = self.low_pass(x)

        # 2. Extract high-frequency components (lesion boundaries + artifacts)
        x_high = x - x_low

        # 3. Modulate high-frequency map with adaptive spatial-channel gate
        gate = self.hf_gate(x_low)
        x_high_filtered = x_high * gate

        # 4. Recombine filtered boundary details with semantic morphology
        out = self.fusion(torch.cat([x_low, x_high_filtered], dim=1))
        return out + x  # Residual connection
