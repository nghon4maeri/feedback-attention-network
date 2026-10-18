import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import HierarchicalBackbone
from .frequency_filter import FrequencyDecoupledFilter
from .selective_boundary import DifferentiableEdgeStream, SelectiveBoundaryAggregation

class MultiScaleDecoderBlock(nn.Module):
    """
    MFAD-v2 Multi-Scale Decoder Block with Channel Attention.
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
        # Channel-wise attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1),
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


class GDFNetV2(nn.Module):
    """
    GDF-Netv2 (Ultra-GDFNet): Gradient-Decoupled Feedback Network v2
    
    Architectural Breakthroughs:
    1. Modern Hierarchical Backbone: ConvNeXt-V2 / PVTv2-B2 with inverted bottleneck 7x7 depthwise convs.
    2. Frequency-Decoupled Filter (FDF): filters out dermoscopic hair and ruler noise on skip connections.
    3. Multi-Channel Semantic Feedback (MSF): Detached Gradient Firewall recycling 32-channel feature tokens.
    4. Differentiable Edge Guidance & Selective Boundary Aggregation (SBA): sharpens true lesion contours.
    5. MFAD-v2 Decoder: Hierarchical multi-scale fusion with channel attention.
    """
    def __init__(
        self,
        backbone_name="convnextv2_tiny",
        pretrained=True,
        in_chans=3,
        feedback_iterations=2,
        feedback_channels=32,
        decoder_channels=64
    ):
        super().__init__()
        self.feedback_iterations = feedback_iterations
        self.feedback_channels = feedback_channels

        # 1. Hierarchical Backbone
        self.backbone = HierarchicalBackbone(
            backbone_name=backbone_name,
            pretrained=pretrained,
            in_chans=in_chans
        )
        c1, c2, c3, c4 = self.backbone.get_channels()

        # 2. Frequency-Decoupled Filters for Skip Connections
        self.fdf1 = FrequencyDecoupledFilter(c1)
        self.fdf2 = FrequencyDecoupledFilter(c2)
        self.fdf3 = FrequencyDecoupledFilter(c3)
        self.fdf4 = FrequencyDecoupledFilter(c4)

        # 3. Multi-Channel Semantic Feedback Adapter (MSF)
        # Detached Gradient Firewall operates at Stage 1 scale (H/4, W/4)
        self.feedback_adapter = nn.Sequential(
            nn.Conv2d(decoder_channels, feedback_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(feedback_channels),
            nn.ReLU(inplace=True)
        )
        self.feedback_gate = nn.Sequential(
            nn.Conv2d(c1 + feedback_channels, c1, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.Sigmoid()
        )

        # 4. Decoder Path (MFAD-v2)
        self.dec4_conv = nn.Sequential(
            nn.Conv2d(c4, decoder_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.dec3 = MultiScaleDecoderBlock(decoder_channels * 4, c3, decoder_channels * 2)
        self.dec2 = MultiScaleDecoderBlock(decoder_channels * 2, c2, decoder_channels)
        self.dec1 = MultiScaleDecoderBlock(decoder_channels, c1, decoder_channels)

        # 5. Differentiable Edge Guidance & SBA
        self.edge_stream = DifferentiableEdgeStream(out_channels=32)
        self.sba = SelectiveBoundaryAggregation(
            semantic_channels=decoder_channels,
            edge_channels=32,
            out_channels=64
        )

        # Auxiliary Head (Iteration 1 initial guess)
        self.aux_head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(decoder_channels // 2, 1, kernel_size=1)
        )

    def _decode_features(self, s1, s2, s3, s4):
        """Pass multi-stage features through MFAD-v2 decoder."""
        f4 = self.fdf4(s4)
        f3 = self.fdf3(s3)
        f2 = self.fdf2(s2)
        f1 = self.fdf1(s1)

        d4 = self.dec4_conv(f4)
        d3 = self.dec3(d4, f3)
        d2 = self.dec2(d3, f2)
        d1 = self.dec1(d2, f1)
        return d1

    def forward(self, x):
        B, _, H, W = x.shape

        # Step 1: Extract edge features from raw image
        edge_feat, edge_pred = self.edge_stream(x)

        # Step 2: First Forward Pass (t=1, Feedforward)
        stages = self.backbone(x)
        s1, s2, s3, s4 = stages

        d1_t1 = self._decode_features(s1, s2, s3, s4)
        pred_t1 = self.aux_head(d1_t1)
        pred_t1_up = F.interpolate(pred_t1, size=(H, W), mode='bilinear', align_corners=False)

        if self.feedback_iterations <= 1:
            # Single-pass evaluation
            final_pred = self.sba(d1_t1, edge_feat)
            if self.training:
                return {
                    'final_pred': final_pred,
                    'pred_t1': pred_t1_up,
                    'edge_pred': edge_pred
                }
            return final_pred

        # Step 3: Recurrent Pass (t=2, Multi-Channel Semantic Feedback)
        # Apply Detached Gradient Firewall to prevent feedback collapse / saturation
        fb_feat = self.feedback_adapter(d1_t1).detach()
        if fb_feat.shape[2:] != s1.shape[2:]:
            fb_feat = F.interpolate(fb_feat, size=s1.shape[2:], mode='bilinear', align_corners=False)

        # Soft-OR gating: modulate stage 1 features with recurrent semantic prior
        gate = self.feedback_gate(torch.cat([s1, fb_feat], dim=1))
        s1_refined = s1 + (s1 * gate)

        # Re-decode with refined semantic features
        d1_t2 = self._decode_features(s1_refined, s2, s3, s4)

        # Step 4: Selective Boundary Aggregation (SBA)
        final_pred = self.sba(d1_t2, edge_feat)

        if self.training:
            return {
                'final_pred': final_pred,
                'pred_t1': pred_t1_up,
                'edge_pred': edge_pred
            }
        return final_pred
