import torch
import torch.nn as nn
import timm

class HierarchicalBackboneV4(nn.Module):
    """
    Hierarchical 4-Stage Encoder for GDF-Netv4.
    
    Supports:
    - 'pvt_v2_b2' (default): Pyramid Vision Transformer v2 with overlapping patch embeddings
      and linear spatial-reduction self-attention. Proven SOTA for medical segmentation (EMCAD, Polyp-PVT).
    - 'convnextv2_tiny': Pure convolutional alternative with 7x7 depthwise convolutions and GRN.
    - 'resnet50': Classic ResNet baseline.
    
    Returns 4 stages of feature maps:
    - Stage 1: H/4 x W/4
    - Stage 2: H/8 x W/8
    - Stage 3: H/16 x W/16
    - Stage 4: H/32 x W/32
    """
    def __init__(self, backbone_name="pvt_v2_b2", pretrained=True, in_chans=3):
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=in_chans,
            features_only=True,
            out_indices=(0, 1, 2, 3)
        )
        self.channels = self.backbone.feature_info.channels()

    def get_channels(self):
        return tuple(self.channels)

    def forward(self, x):
        feats = self.backbone(x)
        return feats[0], feats[1], feats[2], feats[3]
