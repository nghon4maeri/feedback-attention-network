import torch
import torch.nn as nn
import torch.nn.functional as F

class DifferentiableEdgeStream(nn.Module):
    """
    Differentiable Edge Extraction Stream.
    Uses horizontal and vertical Sobel kernel filters initialized with fixed gradient weights
    and refined through learnable depthwise convs to extract high-precision boundary cues.
    """
    def __init__(self, out_channels=32):
        super().__init__()
        # Sobel filters for horizontal and vertical edge detection
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]).unsqueeze(0).unsqueeze(0)

        # Register non-trainable initial kernels (repeated for 3 input channels)
        self.register_buffer('sobel_x', sobel_x.repeat(1, 3, 1, 1))
        self.register_buffer('sobel_y', sobel_y.repeat(1, 3, 1, 1))

        # Learnable refinement layers
        self.edge_proj = nn.Sequential(
            nn.Conv2d(2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.edge_head = nn.Conv2d(out_channels, 1, kernel_size=1)

    def forward(self, x):
        # x: input image [B, 3, H, W]
        gx = F.conv2d(x, self.sobel_x, padding=1)
        gy = F.conv2d(x, self.sobel_y, padding=1)
        grad_mag = torch.cat([gx, gy], dim=1)  # [B, 2, H, W]

        edge_feat = self.edge_proj(grad_mag)   # [B, out_channels, H, W]
        edge_pred = self.edge_head(edge_feat)  # [B, 1, H, W] (auxiliary edge supervision)
        return edge_feat, edge_pred


class SelectiveBoundaryAggregation(nn.Module):
    """
    Selective Boundary Aggregation (SBA) Module (inspired by DuAT, Tang et al., PRCV 2023).
    
    Fuses high-level semantic features from the decoder with low-level structural edge features
    from the edge stream through cross-attentive gating:
    - Attenuates false interior edge activations.
    - Sharpens outer lesion perimeters where semantic confidence is ambiguous.
    """
    def __init__(self, semantic_channels, edge_channels=32, out_channels=64):
        super().__init__()
        self.semantic_conv = nn.Sequential(
            nn.Conv2d(semantic_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.edge_conv = nn.Sequential(
            nn.Conv2d(edge_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        # Cross-attention gating
        self.cross_gate = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, 1, kernel_size=1)
        )

    def forward(self, semantic_feat, edge_feat):
        # semantic_feat: [B, C_sem, H, W]
        # edge_feat: [B, C_edge, H, W]
        if semantic_feat.shape[2:] != edge_feat.shape[2:]:
            semantic_feat = F.interpolate(semantic_feat, size=edge_feat.shape[2:], mode='bilinear', align_corners=False)

        s_proj = self.semantic_conv(semantic_feat)
        e_proj = self.edge_conv(edge_feat)

        # Compute cross-gating weight
        gate = self.cross_gate(torch.cat([s_proj, e_proj], dim=1))
        
        # Gated fusion: edge enhances uncertain semantic boundaries
        e_gated = e_proj * gate
        fused = torch.cat([s_proj, e_gated], dim=1)
        out_mask = self.out_conv(fused)
        return out_mask
