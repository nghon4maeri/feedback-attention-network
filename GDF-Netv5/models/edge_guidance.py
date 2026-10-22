import torch
import torch.nn as nn
import torch.nn.functional as F

class ReverseEdgeGuidanceStream(nn.Module):
    """
    Reverse Edge Guidance Stream (REGS) for GDF-Netv5.
    
    Designed specifically to eliminate the 28 False Positive over-segmentation failure cases
    identified in ISIC-2018 profiling (where lesions spilled into vignetted healthy skin).
    
    Key Mechanisms:
    1. Multi-scale High-Resolution Edge Extraction: Fuses Stage 1 (H/4) and Stage 2 (H/8)
       skip features to capture crisp gradient boundaries.
    2. Explicit Boundary Head: Predicts full-resolution boundary logits B_pred supervised
       by Canny/Laplacian ground truth edge map G_edge.
    3. Reverse Boundary Modulation: Modulates decoder features by suppressing spurious
       activations in non-boundary healthy skin regions, forcing sharp perimeter cuts.
    """
    def __init__(self, c1, c2, mid_channels=64):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.mid_channels = mid_channels

        # Project Stage 2 (H/8) to match mid_channels
        self.proj_c2 = nn.Sequential(
            nn.Conv2d(c2, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Project Stage 1 (H/4) to match mid_channels
        self.proj_c1 = nn.Sequential(
            nn.Conv2d(c1, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Edge feature fusion and refinement
        self.edge_conv = nn.Sequential(
            nn.Conv2d(mid_channels * 2, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Boundary prediction head
        self.edge_head = nn.Conv2d(mid_channels, 1, kernel_size=1)

        # Reverse Boundary Attention Generator
        # Projects edge features to modulate decoder feature maps
        self.edge_gate_proj = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.Sigmoid()
        )

    def forward(self, f1, f2, target_size):
        """
        Args:
            f1: Stage 1 feature map (H/4 x W/4)
            f2: Stage 2 feature map (H/8 x W/8)
            target_size: (H, W) tuple of full input resolution
        Returns:
            edge_logits: Full resolution boundary prediction (B x 1 x H x W)
            edge_feat: Modulated edge features at H/4 resolution for decoder integration
        """
        H4, W4 = f1.shape[2], f1.shape[3]

        # 1. Align Stage 2 to H/4
        f2_up = F.interpolate(self.proj_c2(f2), size=(H4, W4), mode='bilinear', align_corners=False)
        f1_proj = self.proj_c1(f1)

        # 2. Fuse multi-scale boundary features
        edge_feat = self.edge_conv(torch.cat([f1_proj, f2_up], dim=1))

        # 3. Predict boundary logits and upsample to full resolution
        edge_logits_low = self.edge_head(edge_feat)
        edge_logits = F.interpolate(edge_logits_low, size=target_size, mode='bilinear', align_corners=False)

        # 4. Compute boundary attention modulation map
        edge_attn = self.edge_gate_proj(edge_feat)
        edge_feat_gated = edge_feat * (1.0 + edge_attn)

        return edge_logits, edge_feat_gated
