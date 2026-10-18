import torch
import torch.nn as nn
import timm

class HierarchicalBackbone(nn.Module):
    """
    Unified Hierarchical Feature Extractor for GDF-Netv2.
    Supports modern vision backbones:
    - convnextv2_tiny / convnext_tiny
    - pvt_v2_b2
    - resnet34 (legacy baseline compatibility)
    Extracts multi-stage pyramid feature maps at 4 scales:
    Stage 1: H/4,  W/4
    Stage 2: H/8,  W/8
    Stage 3: H/16, W/16
    Stage 4: H/32, W/32
    """
    def __init__(self, backbone_name="convnextv2_tiny", pretrained=True, in_chans=3):
        super().__init__()
        self.backbone_name = backbone_name

        if "convnext" in backbone_name:
            self.encoder = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                in_chans=in_chans,
                features_only=True,
                out_indices=(0, 1, 2, 3)
            )
            # Channel dimensions for convnext_tiny / convnextv2_tiny: [96, 192, 384, 768]
            self.feature_channels = self.encoder.feature_info.channels()
        elif "pvt" in backbone_name:
            self.encoder = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                in_chans=in_chans,
                features_only=True,
                out_indices=(0, 1, 2, 3)
            )
            # Channel dimensions for pvt_v2_b2: [64, 128, 320, 512]
            self.feature_channels = self.encoder.feature_info.channels()
        elif "resnet" in backbone_name:
            self.encoder = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                in_chans=in_chans,
                features_only=True,
                out_indices=(1, 2, 3, 4)
            )
            # Channel dimensions for resnet34: [64, 128, 256, 512]
            self.feature_channels = self.encoder.feature_info.channels()
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

    def forward(self, x):
        features = self.encoder(x)
        # Returns [stage1, stage2, stage3, stage4]
        return features

    def get_channels(self):
        return self.feature_channels
