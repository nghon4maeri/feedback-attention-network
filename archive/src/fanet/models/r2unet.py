"""
Recurrent Residual U-Net (R2U-Net) with Configurable Recurrent Feedback.

This module provides a secondary recurrent architecture to evaluate the universality
of the Feedback Trap failure mode and verify that Detached Soft-OR (Feedback Firewall)
generalizes beyond the FANet architecture.

Reference:
    Alom et al., "Recurrent Residual Convolutional Neural Network based on
    U-Net (R2U-Net) for Medical Image Segmentation", arXiv:1802.06955 (2018).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RecurrentConv(nn.Module):
    """Recurrent Convolutional unit with t recurrent steps."""
    def __init__(self, out_ch, t=2):
        super(RecurrentConv, self).__init__()
        self.t = t
        self.conv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            else:
                x1 = self.conv(x + x1)
        return x1


class RRCNNBlock(nn.Module):
    """Recurrent Residual Convolutional Block (RRCL)."""
    def __init__(self, in_ch, out_ch, t=2):
        super(RRCNNBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        self.rcnn1 = RecurrentConv(out_ch, t=t)
        self.rcnn2 = RecurrentConv(out_ch, t=t)

    def forward(self, x):
        x_res = self.conv1x1(x)
        x1 = self.rcnn1(x_res)
        x2 = self.rcnn2(x1)
        return x_res + x2


class RecurrentGatingBlock(nn.Module):
    """
    Universal Recurrent Feedback Gating Module.
    Combines feature tensor x with recurrent feedback mask m.
    
    Supports:
      - gating_mode: "hard" (indicator function) or "soft_or" (continuous relaxation)
      - detach_feedback: bool (Feedback Firewall / Gradient Decoupling)
    """
    def __init__(self, in_ch, gating_mode="soft_or", detach_feedback=True):
        super(RecurrentGatingBlock, self).__init__()
        self.gating_mode = gating_mode
        self.detach_feedback = detach_feedback

        self.att = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, m):
        fmask = self.att(x)

        # Downsample recurrent feedback mask m to current spatial resolution
        if m.shape[-2:] != x.shape[-2:]:
            m = F.interpolate(m, size=x.shape[-2:], mode="nearest")

        # Feedback Firewall: sever backward error propagation
        if self.detach_feedback:
            m = m.detach()

        if self.gating_mode == "hard":
            keep = torch.maximum((fmask > 0.5).float(), m)
        else:  # soft_or
            keep = 1.0 - (1.0 - fmask) * (1.0 - m)

        return x * keep


class R2UNet(nn.Module):
    """
    R2U-Net with recurrent feedback injection across encoder levels.
    Allows toggling between:
      1. Conventional coupled recurrent feedback (Feedback Trap)
      2. Decoupled Detached Soft-OR feedback (Feedback Firewall)
    """
    def __init__(self, in_ch=3, out_ch=1, t=2, gating_mode="soft_or", detach_feedback=True):
        super(R2UNet, self).__init__()
        self.t = t
        self.gating_mode = gating_mode
        self.detach_feedback = detach_feedback

        # Encoder
        self.rrcl1 = RRCNNBlock(in_ch, 32, t=t)
        self.gate1 = RecurrentGatingBlock(32, gating_mode=gating_mode, detach_feedback=detach_feedback)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.rrcl2 = RRCNNBlock(32, 64, t=t)
        self.gate2 = RecurrentGatingBlock(64, gating_mode=gating_mode, detach_feedback=detach_feedback)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.rrcl3 = RRCNNBlock(64, 128, t=t)
        self.gate3 = RecurrentGatingBlock(128, gating_mode=gating_mode, detach_feedback=detach_feedback)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.rrcl4 = RRCNNBlock(128, 256, t=t)
        self.gate4 = RecurrentGatingBlock(256, gating_mode=gating_mode, detach_feedback=detach_feedback)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck = RRCNNBlock(256, 512, t=t)

        # Decoder
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_rrcl4 = RRCNNBlock(512, 256, t=t)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_rrcl3 = RRCNNBlock(256, 128, t=t)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_rrcl2 = RRCNNBlock(128, 64, t=t)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec_rrcl1 = RRCNNBlock(64, 32, t=t)

        # Final prediction head
        self.head = nn.Conv2d(32, out_ch, kernel_size=1)

    def forward(self, inputs):
        """
        inputs: tuple or list (image, recurrent_mask)
          - image: [B, 3, H, W]
          - recurrent_mask: [B, 1, H, W]
        """
        x, m = inputs[0], inputs[1]

        # Encoder with recurrent feedback gating
        e1 = self.rrcl1(x)
        e1_g = self.gate1(e1, m)
        p1 = self.pool1(e1_g)

        e2 = self.rrcl2(p1)
        e2_g = self.gate2(e2, m)
        p2 = self.pool2(e2_g)

        e3 = self.rrcl3(p2)
        e3_g = self.gate3(e3, m)
        p3 = self.pool3(e3_g)

        e4 = self.rrcl4(p3)
        e4_g = self.gate4(e4, m)
        p4 = self.pool4(e4_g)

        b = self.bottleneck(p4)

        # Decoder
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4_g], dim=1)
        d4 = self.dec_rrcl4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3_g], dim=1)
        d3 = self.dec_rrcl3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2_g], dim=1)
        d2 = self.dec_rrcl2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1_g], dim=1)
        d1 = self.dec_rrcl1(d1)

        out = self.head(d1)
        return out
