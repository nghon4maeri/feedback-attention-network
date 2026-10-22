import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskGuidedLGAG(nn.Module):
    """
    Mask-Guided Large-Kernel Grouped Attention Gate (Mask-Guided LGAG).
    
    Combines:
    1. Large-Kernel Grouped Attention (EMCAD, CVPR 2024): Depthwise convolutions with dilation 1 & 2
       expand the receptive field to capture long-range contextual dependencies at minimal parameters.
    2. Deep Mask Guidance (EGE-UNet, MICCAI 2023): Coarse mask probability predictions from deeper
       stages act as an explicit spatial prior, suppressing healthy skin regions on skip features.
    """
    def __init__(self, skip_channels, gating_channels, out_channels=None):
        super().__init__()
        out_channels = out_channels or skip_channels
        self.skip_channels = skip_channels
        self.gating_channels = gating_channels

        # Project gating features to match skip channels
        self.g_proj = nn.Sequential(
            nn.Conv2d(gating_channels, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels)
        )

        # Depthwise grouped large-kernel convolutions (dilation 1 and dilation 2)
        mid_ch = max(skip_channels // 2, 16)
        self.conv1x1 = nn.Conv2d(skip_channels + 1, mid_ch, kernel_size=1, bias=False)  # +1 for mask prior
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.relu = nn.ReLU(inplace=True)

        self.dw_dil1 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, dilation=1, groups=mid_ch, bias=False)
        self.dw_dil2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=2, dilation=2, groups=mid_ch, bias=False)
        self.bn_dw = nn.BatchNorm2d(mid_ch)

        self.attn_proj = nn.Sequential(
            nn.Conv2d(mid_ch, skip_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.Sigmoid()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, skip, gating, mask_prob=None):
        B, C, H, W = skip.shape

        # Upsample gating feature to match skip resolution
        g_up = F.interpolate(gating, size=(H, W), mode='bilinear', align_corners=False)
        g_proj = self.g_proj(g_up)

        # Upsample mask probability prior
        if mask_prob is not None:
            m_up = F.interpolate(mask_prob, size=(H, W), mode='bilinear', align_corners=False)
        else:
            m_up = torch.zeros((B, 1, H, W), device=skip.device, dtype=skip.dtype)

        # Fuse skip, gating, and mask prior
        feat = skip + g_proj
        feat_with_mask = torch.cat([feat, m_up], dim=1)

        # Compute large-kernel attention coefficients
        x = self.relu(self.bn1(self.conv1x1(feat_with_mask)))
        x = self.relu(self.bn_dw(self.dw_dil1(x) + self.dw_dil2(x)))
        attn = self.attn_proj(x)

        # Apply attention gating: healthy skin background is suppressed
        gated_skip = skip * attn
        return self.out_conv(gated_skip)
