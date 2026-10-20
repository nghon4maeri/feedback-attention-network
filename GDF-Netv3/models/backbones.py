import torch
import torch.nn as nn
import timm

class HierarchicalBackboneV3(nn.Module):
    """
    Unified Hierarchical Feature Extractor for GDF-Netv3.
    Specialized in preserving fine spatial details via 5 multi-scale stages:
    Stage 0: H/2,  W/2  (Ultra-high spatial detail, preserves 2-5% small lesions)
    Stage 1: H/4,  W/4  (Sub-lesion structures)
    Stage 2: H/8,  W/8  (Contextual semantics)
    Stage 3: H/16, W/16 (Global morphology)
    Stage 4: H/32, W/32 (High-level abstract semantics)

    Supported backbones:
    - tf_efficientnetv2_s.in21k_ft_in1k (default, compound-scaled depthwise SE)
    - efficientnet_b3.ra2_in1k (classic efficientnet B3 with SE attention)
    - convnextv2_tiny (4-stage fallback with dummy stage 0 interpolation)
    """
    def __init__(self, backbone_name="tf_efficientnetv2_s.in21k_ft_in1k", pretrained=True, in_chans=3):
        super().__init__()
        self.backbone_name = backbone_name

        if "efficientnet" in backbone_name:
            self.encoder = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                in_chans=in_chans,
                features_only=True,
                out_indices=(0, 1, 2, 3, 4)
            )
            self.feature_channels = self.encoder.feature_info.channels()
            self.has_stage0 = True
        elif "convnext" in backbone_name:
            self.encoder = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                in_chans=in_chans,
                features_only=True,
                out_indices=(0, 1, 2, 3)
            )
            raw_channels = self.encoder.feature_info.channels()
            # Synthesize stage 0 channel for compatibility
            self.feature_channels = [raw_channels[0] // 2] + raw_channels
            self.has_stage0 = False
            self.synth_stage0 = nn.Sequential(
                nn.ConvTranspose2d(raw_channels[0], raw_channels[0] // 2, kernel_size=2, stride=2),
                nn.BatchNorm2d(raw_channels[0] // 2),
                nn.ReLU(inplace=True)
            )
        else:
            raise ValueError(f"Unsupported backbone for GDF-Netv3: {backbone_name}")

    def forward(self, x):
        features = self.encoder(x)
        if self.has_stage0:
            # Returns 5 stages: [stage0 (H/2), stage1 (H/4), stage2 (H/8), stage3 (H/16), stage4 (H/32)]
            return features
        else:
            f0 = self.synth_stage0(features[0])
            return [f0] + features

    def get_channels(self):
        return self.feature_channels
