import torch
import torch.nn as nn

""" Squeeze and Excitation block """
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

""" 3x3->3x3 Residual block """
class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_c)

        self.se = SELayer(out_c, out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)

        x2 = self.conv2(x1)
        x2 = self.bn2(x2)

        x3 = self.conv3(x)
        x3 = self.bn3(x3)
        x3 = self.se(x3)

        x4 = x2 + x3
        x4 = self.relu(x4)

        return x4

""" Mixpool block: Merging the image features and the mask """
class MixPool(nn.Module):
    """MixPool with configurable gating.

    BACKWARD-COMPATIBLE: existing checkpoints load without any change.
    Default args reproduce original FANet behaviour exactly.

    ── Legacy gate= parameter (kept for compat) ──────────────────────────
    gate:
        "binary" - hard threshold (original FANet, zero gradient to fmask)
        "ste"    - hard forward + straight-through gradient to fmask
        "soft"   - soft gating max(fmask, m_fg), full gradient

    ── Phase-7B: new parameters ──────────────────────────────────────────
    gating_mode (takes priority over gate= if set explicitly):
        "hard"      - identical to gate="binary"   [DEFAULT, compat]
        "hard_ste"  - identical to gate="ste"
        "soft_max"  - identical to gate="soft"
        "soft_or"   - Smooth Probabilistic OR: 1-(1-fmask)(1-m_fg)
                      Fully differentiable; saturates gracefully near 0/1.
        "learned_residual" - Phase 7C: Solves Monotonicity Trap. Uses a learned 
                             1x1 Conv gate allowing network to suppress prior False Positives.

    detach_feedback:
        False  - original: m_fg participates in backward graph  [DEFAULT]
        True   - m_fg.detach() before gating; severs the Feedback Trap:
                   * eliminates BN drift (e1.r1.bn3 KL=33.63 -> 0)
                   * restores gradient norm to nhánh fmask
                   * fmask MUST learn to suppress FP via Tversky loss

    dual_path:
        False - mask m is [B,1,H,W] foreground only               [DEFAULT]
        True  - mask m is [B,2,H,W] = [m_fg, m_bg]
    """
    _GATE_ALIAS = {"binary": "hard", "ste": "hard_ste", "soft": "soft_max"}
    _VALID_MODES = ("hard", "hard_ste", "soft_max", "soft_or", "learned_residual")

    def __init__(self, in_c, out_c, gate="binary", dual_path=False,
                 gating_mode=None, detach_feedback=False):
        super(MixPool, self).__init__()

        # ── Resolve gating_mode (new) vs gate (legacy) ──────────────────
        if gating_mode is not None:
            assert gating_mode in self._VALID_MODES, (
                f"Unknown gating_mode '{gating_mode}'. Valid: {self._VALID_MODES}"
            )
            self.gating_mode = gating_mode
        else:
            assert gate in self._GATE_ALIAS, (
                f"Unknown gate '{gate}'. Valid: {list(self._GATE_ALIAS)}"
            )
            self.gating_mode = self._GATE_ALIAS[gate]

        self.gate = gate              # preserve legacy attribute
        self.dual_path = dual_path
        self.detach_feedback = detach_feedback

        self.fmask = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, 1, kernel_size=1, padding=0),
            nn.Sigmoid()
        )

        # Phase 7C: Learned Residual Gate (1x1 Conv)
        # We instantiate this regardless of gating mode to keep state_dict compatible
        self.learned_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, out_c//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c//2),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c, out_c//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_c//2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, m):
        fmask = self.fmask(x)   # [B,1,h,w] in (0,1), always differentiable

        # ── Downsample mask to feature map resolution ────────────────────
        stride_h = m.shape[2] // x.shape[2]
        stride_w = m.shape[3] // x.shape[3]
        m = nn.MaxPool2d((stride_h, stride_w))(m)
        m_fg = m[:, 0:1]
        m_bg = (m[:, 1:2] if (self.dual_path and m.shape[1] > 1)
                else torch.zeros_like(m_fg))

        # ── Phase-7B: detach feedback from gradient graph ────────────────
        # Severs the Feedback Trap:
        #   m_fg carries spatial hint at inference, but ZERO gradient
        #   flows back into the recurrent loop. fmask now owns all FP
        #   gradient signal → Tversky loss can reshape encoder weights.
        if self.detach_feedback:
            m_fg = m_fg.detach()
            m_bg = m_bg.detach()

        # ── Gating ──────────────────────────────────────────────────────
        if self.gating_mode == "hard":
            # Original FANet — hard OR, gradient killed by (>0.5)
            fmask_g = (fmask > 0.5).float()
            keep = torch.maximum(fmask_g, m_fg)

        elif self.gating_mode == "hard_ste":
            # STE: hard forward, straight-through backward
            fmask_g = (fmask > 0.5).float() + fmask - fmask.detach()
            keep = torch.maximum(fmask_g, m_fg)

        elif self.gating_mode == "soft_max":
            # Soft max (old "soft" gate) — continuous element-wise max
            keep = torch.maximum(fmask, m_fg)

        elif self.gating_mode == "soft_or":
            # Smooth Probabilistic OR:  keep = 1 - (1-p)(1-q)
            # Properties:
            #   • keep=0 only when BOTH p=0 AND q=0
            #   • keep=1 when either p=1 or q=1
            #   • Fully differentiable everywhere
            #   • With detach_feedback: d(keep)/d(fmask) = (1-m_fg)
            #     → gradient scales inversely with prior mask confidence,
            #     so the network is pushed hardest where mask is uncertain.
            keep = 1.0 - (1.0 - fmask) * (1.0 - m_fg)

        elif self.gating_mode == "learned_residual":
            # Phase 7C: Learned Residual Decoupling (Solves Monotonicity Trap)
            # Instead of a monotonic mathematical OR gate, we let a 1x1 conv learn
            # a dynamic spatial attention map that can both augment AND suppress (prune).
            # We strictly preserve the gradient decoupling by passing m_fg (which is detached)
            combined = torch.cat([fmask, m_fg], dim=1)
            dynamic_gate = self.learned_gate(combined)
            
            # The gate modulates the prior mask, and we add the backbone's feature mask
            # This allows the network to zero-out the prior (prune) when dynamic_gate -> 0
            keep = (dynamic_gate * m_fg) + fmask
            
            # Clamp to [0,1] to maintain valid probability range
            keep = torch.clamp(keep, 0.0, 1.0)

        # ── Feature Map Probing Hook ──────────────────────────────────────
        if getattr(self, 'probe_heatmaps', False):
            self.saved_fmask = fmask.detach().cpu()
            self.saved_m_fg = m_fg.detach().cpu()
            self.saved_keep = keep.detach().cpu()

        # ── Background suppression (dual_path only) ──────────────────────
        if self.dual_path:
            keep = keep * (1.0 - m_bg)


        # ── Split-Transform-Merge ────────────────────────────────────────
        x1 = x * keep
        x1 = self.conv1(x1)
        x2 = self.conv2(x)
        x = torch.cat([x1, x2], axis=1)
        return x
