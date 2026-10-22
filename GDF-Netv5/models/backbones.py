import torch
import torch.nn as nn
import timm

class HierarchicalBackboneV5(nn.Module):
    """
    Hierarchical 4-Stage Encoder for GDF-Netv5.
    
    Supports:
    - 'pvt_v2_b4' (default): Pyramid Vision Transformer v2 with 62.0M parameters,
      24 transformer blocks in Stage 3 (H/16), overlapping patch embeddings, and
      linear spatial-reduction self-attention. Delivers 3x the contextual depth of B2
      for faint amelanotic lesion contrast and fine boundary detail.
    - 'pvt_v2_b5': 81.4M parameter extreme-capacity variant (40 blocks in Stage 3).
    - 'pvt_v2_b2': 24.8M parameter efficient baseline (v4 default).
    - 'convnextv2_base': 87.7M parameter modern convolutional alternative with FCMAE pretraining.
    
    Returns 4 hierarchical stages of feature maps:
    - Stage 1: H/4 x W/4   (128x128 for 512x512 input)
    - Stage 2: H/8 x W/8   (64x64 for 512x512 input)
    - Stage 3: H/16 x W/16 (32x32 for 512x512 input)
    - Stage 4: H/32 x W/32 (16x16 for 512x512 input)
    """
    def __init__(self, backbone_name="pvt_v2_b4", pretrained=True, in_chans=3):
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
