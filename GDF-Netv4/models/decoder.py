import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleConvolutionalAttentionModule(nn.Module):
    """
    Multi-Scale Convolutional Attention Module (MSCAM, adapted from EMCAD CVPR 2024).
    
    Uses parallel depthwise convolutions with kernels [1x1, 3x3, 5x5] to aggregate
    multi-scale contextual semantics, followed by channel and spatial attention.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Parallel multi-scale depthwise convolutions
        self.dw1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, groups=in_channels, bias=False)
        self.dw3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False)
        self.dw5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2, groups=in_channels, bias=False)
        self.bn_dw = nn.BatchNorm2d(in_channels * 3)

        # Pointwise projection
        self.pw = nn.Sequential(
            nn.Conv2d(in_channels * 3, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Channel Attention
        mid_ch = max(out_channels // 4, 16)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, mid_ch, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # Spatial Attention
        self.sa = nn.Sequential(
            nn.Conv2d(out_channels, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

        # 3x3 refinement
        self.refine = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        d1 = self.dw1(x)
        d3 = self.dw3(x)
        d5 = self.dw5(x)
        dw_cat = F.relu(self.bn_dw(torch.cat([d1, d3, d5], dim=1)), inplace=True)
        feat = self.pw(dw_cat)
        ca_weight = self.ca(feat)
        feat = feat * ca_weight
        sa_weight = self.sa(feat)
        feat = feat * sa_weight
        return self.refine(feat)


class CascadedMSCAMDecoderBlock(nn.Module):
    """
    Cascaded MSCAM Decoder Block with Deep Supervision.
    
    1. Upsamples previous decoder feature by 2x.
    2. Concatenates with mask-gated skip features.
    3. Refines with MSCAM (multi-scale kernels [1, 3, 5]).
    4. Predicts auxiliary segmentation logits P_k.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.mscam = MultiScaleConvolutionalAttentionModule(in_channels + skip_channels, out_channels)
        self.head = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x, gated_skip):
        x_up = self.upsample(x)
        if x_up.shape[2:] != gated_skip.shape[2:]:
            x_up = F.interpolate(x_up, size=gated_skip.shape[2:], mode='bilinear', align_corners=False)
        fused = torch.cat([x_up, gated_skip], dim=1)
        out_feat = self.mscam(fused)
        pred_logits = self.head(out_feat)
        return out_feat, pred_logits
