import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import HierarchicalBackboneV3
from .frequency_filter import FrequencyDecoupledFilter
from .selective_boundary import DifferentiableEdgeStream, SelectiveBoundaryAggregationLinear

class MultiScaleDecoderBlock(nn.Module):
    """
    Decoder block with bilinear upsampling, depthwise-pointwise convolutions, and channel attention.
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, max(out_channels // 4, 8), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(out_channels // 4, 8), out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        out = torch.cat([x, skip], dim=1)
        out = self.conv(out)
        attn = self.ca(out)
        return out * attn


class GDFNetV3(nn.Module):
    """
    GDF-Netv3: General Decoupled Feedback Network Version 3
    
    Core Architectural Innovations:
    1. 5-Stage Spatial-Preserving Encoder (EfficientNet-V2-S / EfficientNet-B3):
       Features an ultra-high-resolution Stage 0 at H/2 x W/2 (preserves 2-5% micro lesions
       that were completely crushed by ResNet's 4x stem compression).
    2. 5-Stage Multi-Scale Frequency-Decoupled Filter (FDF):
       Filters dermoscopic hair, bubbles, and ruler artifacts at all spatial resolutions,
       especially the ultra-fine H/2 x W/2 stream.
    3. Multi-Channel Semantic Feedback (MSF-v3):
       Detached gradient firewall recycling rich contextual representations into early layers.
    4. 5-Stage High-Resolution Boundary Decoder:
       Seamlessly ascends from H/32 -> H/16 -> H/8 -> H/4 -> H/2 -> H.
    5. Linear Selective Boundary Aggregation (SBA-Linear):
       O(HW) memory-efficient edge fusion sharpening fine lesion perimeters.
    """
    def __init__(
        self,
        backbone_name="tf_efficientnetv2_s.in21k_ft_in1k",
        pretrained=True,
        in_chans=3,
        feedback_iterations=2,
        feedback_channels=32,
        decoder_channels=48
    ):
        super().__init__()
        self.feedback_iterations = feedback_iterations
        self.feedback_channels = feedback_channels

        # 1. 5-Stage Spatial-Preserving Backbone
        self.backbone = HierarchicalBackboneV3(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans
        )
        c0, c1, c2, c3, c4 = self.backbone.get_channels()

        # 2. 5 Multi-Scale Frequency-Decoupled Filters for Skip Connections
        self.fdf0 = FrequencyDecoupledFilter(c0)  # H/2 (192x192) - Filters fine hair artifacts
        self.fdf1 = FrequencyDecoupledFilter(c1)  # H/4 (96x96)
        self.fdf2 = FrequencyDecoupledFilter(c2)  # H/8 (48x48)
        self.fdf3 = FrequencyDecoupledFilter(c3)  # H/16 (24x24)
        self.fdf4 = FrequencyDecoupledFilter(c4)  # H/32 (12x12)

        # 3. Multi-Channel Semantic Feedback (MSF-v3)
        # Detached Gradient Firewall recycling representations back to Stage 1
        self.feedback_adapter = nn.Sequential(
            nn.Conv2d(decoder_channels * 2, feedback_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feedback_channels),
            nn.ReLU(inplace=True)
        )
        self.feedback_gate = nn.Sequential(
            nn.Conv2d(c1 + feedback_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Sigmoid()
        )

        # 4. 5-Stage Multi-Scale Decoder
        # Bottleneck projection
        self.dec4_conv = nn.Sequential(
            nn.Conv2d(c4, decoder_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels * 4),
            nn.ReLU(inplace=True)
        )
        # Stage 4: H/32 -> H/16 (fused with c3)
        self.dec3 = MultiScaleDecoderBlock(decoder_channels * 4, c3, decoder_channels * 3)
        # Stage 3: H/16 -> H/8 (fused with c2)
        self.dec2 = MultiScaleDecoderBlock(decoder_channels * 3, c2, decoder_channels * 2)
        # Stage 2: H/8 -> H/4 (fused with c1)
        self.dec1 = MultiScaleDecoderBlock(decoder_channels * 2, c1, decoder_channels * 2)
        # Stage 1: H/4 -> H/2 (fused with c0) <-- ULTRA-HIGH-RES PRESERVATION
        self.dec0 = MultiScaleDecoderBlock(decoder_channels * 2, c0, decoder_channels)

        # 5. Differentiable Edge Guidance & Linear SBA at full resolution H
        self.edge_stream = DifferentiableEdgeStream(out_channels=32)
        self.sba = SelectiveBoundaryAggregationLinear(
            semantic_channels=decoder_channels,
            edge_channels=32,
            out_channels=decoder_channels
        )

        # Final full-resolution upsample (H/2 -> H)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Heads
        self.head_final = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.head_aux1  = nn.Conv2d(decoder_channels, 1, kernel_size=1)
        self.head_aux2  = nn.Conv2d(decoder_channels * 2, 1, kernel_size=1)

    def forward_single_pass(self, x, feedback_token=None):
        B, C, H, W = x.shape

        # 1. Edge extraction
        edge_feat, edge_pred = self.edge_stream(x)

        # 2. 5-Stage Hierarchical Feature Extraction
        f0, f1, f2, f3, f4 = self.backbone(x)

        # Apply Frequency-Decoupled Filters
        s0 = self.fdf0(f0)
        s1 = self.fdf1(f1)
        s2 = self.fdf2(f2)
        s3 = self.fdf3(f3)
        s4 = self.fdf4(f4)

        # Apply Semantic Feedback if available (Detached Gradient Firewall)
        if feedback_token is not None:
            fb = F.interpolate(feedback_token, size=s1.shape[2:], mode='bilinear', align_corners=False)
            gate = self.feedback_gate(torch.cat([s1, fb], dim=1))
            s1 = s1 * gate + s1

        # 3. Decoder Walkthrough
        d4 = self.dec4_conv(s4)
        d3 = self.dec3(d4, s3)
        d2 = self.dec2(d3, s2)
        d1 = self.dec1(d2, s1)
        d0 = self.dec0(d1, s0)  # [B, decoder_channels, H/2, W/2]

        # 4. Upsample to full resolution H and apply SBA with edge stream
        d_full = self.final_upsample(d0)
        refined_feat = self.sba(d_full, edge_feat)
        out_final = self.head_final(refined_feat)

        # Extract auxiliary predictions
        aux1 = F.interpolate(self.head_aux1(d0), size=(H, W), mode='bilinear', align_corners=False)
        aux2 = F.interpolate(self.head_aux2(d1), size=(H, W), mode='bilinear', align_corners=False)

        # Next feedback token (detached)
        next_feedback = self.feedback_adapter(d1).detach()

        return out_final, aux1, aux2, edge_pred, next_feedback

    def forward(self, x):
        feedback = None
        for it in range(self.feedback_iterations):
            out_final, aux1, aux2, edge_pred, feedback = self.forward_single_pass(x, feedback)

        if self.training:
            return {
                'logits': out_final,
                'aux1': aux1,
                'aux2': aux2,
                'edge_pred': edge_pred
            }
        else:
            return out_final
