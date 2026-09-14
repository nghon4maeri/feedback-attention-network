import torch
import torch.nn as nn
from .blocks import ResidualBlock, MixPool


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, gate="binary", dual_path=False,
                 gating_mode=None, detach_feedback=False, name=None):
        super(EncoderBlock, self).__init__()

        self.name = name
        self.r1 = ResidualBlock(in_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)
        self.p1 = MixPool(out_c, out_c, gate=gate, dual_path=dual_path,
                          gating_mode=gating_mode, detach_feedback=detach_feedback)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs, masks):
        x = self.r1(inputs)
        x = self.r2(x)
        p = self.p1(x, masks)
        o = self.pool(p)
        return o, x


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c, gate="binary", dual_path=False,
                 gating_mode=None, detach_feedback=False, name=None):
        super(DecoderBlock, self).__init__()

        self.upsample = nn.ConvTranspose2d(in_c, in_c, kernel_size=4, stride=2, padding=1)
        self.r1 = ResidualBlock(in_c+in_c, out_c)
        self.r2 = ResidualBlock(out_c, out_c)
        self.p1 = MixPool(out_c, out_c, gate=gate, dual_path=dual_path,
                          gating_mode=gating_mode, detach_feedback=detach_feedback)

    def forward(self, inputs, skip, masks):
        x = self.upsample(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.r1(x)
        x = self.r2(x)
        p = self.p1(x, masks)
        return p


class FANet(nn.Module):
    """FANet with configurable MixPool gating.

    Backward-compatible: FANet() reproduces original behaviour exactly.

    New Phase-7B parameters:
        gating_mode: "hard" | "hard_ste" | "soft_max" | "soft_or"
        detach_feedback: bool — severs BN-drift / gradient-attenuation trap
    """
    def __init__(self, gate="binary", dual_path=False,
                 gating_mode=None, detach_feedback=False):
        super(FANet, self).__init__()

        self.gate = gate
        self.dual_path = dual_path
        self.gating_mode = gating_mode
        self.detach_feedback = detach_feedback

        # Shared kwargs forwarded to every Encoder/DecoderBlock
        mp_kwargs = dict(gate=gate, dual_path=dual_path,
                         gating_mode=gating_mode, detach_feedback=detach_feedback)

        self.e1 = EncoderBlock(3, 32, **mp_kwargs)
        self.e2 = EncoderBlock(32, 64, **mp_kwargs)
        self.e3 = EncoderBlock(64, 128, **mp_kwargs)
        self.e4 = EncoderBlock(128, 256, **mp_kwargs)

        self.d1 = DecoderBlock(256, 128, **mp_kwargs)
        self.d2 = DecoderBlock(128, 64, **mp_kwargs)
        self.d3 = DecoderBlock(64, 32, **mp_kwargs)
        self.d4 = DecoderBlock(32, 16, **mp_kwargs)

        # Output head: concat d4 with m_fg (1 channel)
        self.output = nn.Conv2d(16+1, 1, kernel_size=1, padding=0)

    def forward(self, x):
        inputs, masks = x[0], x[1]

        p1, s1 = self.e1(inputs, masks)
        p2, s2 = self.e2(p1, masks)
        p3, s3 = self.e3(p2, masks)
        p4, s4 = self.e4(p3, masks)

        d1 = self.d1(p4, s4, masks)
        d2 = self.d2(d1, s3, masks)
        d3 = self.d3(d2, s2, masks)
        d4 = self.d4(d3, s1, masks)

        m_fg = masks[:, 0:1]
        d5 = torch.cat([d4, m_fg], axis=1)
        output = self.output(d5)

        return output


if __name__ == "__main__":
    x = torch.randn((2, 3, 256, 256))
    m = torch.randn((2, 1, 256, 256))

    # Original behaviour
    model = FANet()
    y = model([x, m])
    print("original (binary, no detach):", y.shape)

    # Phase-7B: Detached Soft-OR
    model7b = FANet(gating_mode="soft_or", detach_feedback=True)
    y7b = model7b([x, m])
    print("Phase-7B (soft_or + detach):  ", y7b.shape)
