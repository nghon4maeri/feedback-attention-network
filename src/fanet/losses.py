import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

        return 1 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = (0.5 * BCE) + (0.5 * dice_loss)

        return Dice_BCE


class NegativeAreaDiceBCELoss(nn.Module):
    """DiceBCE + FP-precision penalty (soft surrogate). NOT the original IEEE Access loss.

    NOTE (verified 05/09/2026 against the IEEE Access 2020 paper abstract):
    the original "Improved Dice Loss ... Negative Areas" is a **weighted soft
    Dice loss (WSDice)** that maps labels/preds to [-1,1] via (2y-1)/(2yhat-1)
    and adds the background with a small label-derived weight w_ij. The present
    class is a hand-rolled soft surrogate built from the same *intent*
    (penalize predicted-foreground on GT-background) and is NOT the paper's
    formula. Prefer `WeightedSoftDiceLoss` for a faithful implementation.

    Surrogate: fp_frac = mean over predicted-fg mass that falls on GT-bg.
        fp_frac = sum(p * (1-targets)) / (sum(p) + smooth)
        loss = DiceBCE + fp_weight * fp_frac
    """
    def __init__(self, fp_weight=0.5, smooth=1.0):
        super(NegativeAreaDiceBCELoss, self).__init__()
        self.fp_weight = fp_weight
        self.smooth = smooth

    def forward(self, inputs, targets, smooth=None):
        if smooth is None:
            smooth = self.smooth
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)

        # standard Dice
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        dice_bce = 0.5 * BCE + 0.5 * dice_loss

        # negative-area (FP) penalty: foreground predicted on GT-background
        bg_targets = (1.0 - targets)
        # soft overlap of predicted fg with GT bg, normalized by GT bg
        fp_overlap = (inputs * bg_targets).sum()
        fp_norm = bg_targets.sum() + smooth
        fp_penalty = 1.0 - (fp_overlap + smooth) / (inputs.sum() + smooth + 1e-8)
        # alternative bounded form: how much of predicted fg is on bg
        fp_frac = (inputs * bg_targets).sum() / (inputs.sum() + smooth + 1e-8)

        return dice_bce + self.fp_weight * fp_frac


class FocalDiceBCELoss(nn.Module):
    """(Reserved) Not used in Phase 4 primary cells."""
    pass


class SoftIoULoss(nn.Module):
    """Soft IoU (Jaccard) loss, differentiable surrogate for 1 - IoU.

    Plain (no pixel weighting). Matches the "global restriction" term used in
    PraNet / CFA-Net / F³Net when their pixel-importance weight is dropped.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        inter = (inputs * targets).sum()
        union = (inputs + targets - inputs * targets).sum()
        return 1.0 - (inter + self.smooth) / (union + self.smooth)


class IoUBCELoss(nn.Module):
    """Plain wIoU + wBCE (no pixel-importance weight): SoftIoULoss + BCE.

    Used as the loss backbone of Phase 5 cell A (boundary-agnostic, class-
    balanced by construction of IoU). Equal weighting like F³Net L = LwIoU + LwBCE.
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
        self.iou = SoftIoULoss(smooth=smooth)

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        iou_l = 1.0 - ( (inputs*targets).sum() + self.smooth ) / \
                      ( (inputs + targets - inputs*targets).sum() + self.smooth )
        bce = F.binary_cross_entropy(inputs, targets, reduction='mean')
        return iou_l + bce


class WeightedSoftDiceLoss(nn.Module):
    """Faithful Weighted Soft Dice loss (WSDice), Wang et al. IEEE Access 2020.

    Original formula (Eq. 4 of the paper), verified via the published text:

        L = 1 - [ 2*sum(Ghat*G) + eps ] / [ sum(Ghat^2) + sum(G^2) + eps ]
        Ghat = w_ij * (2*yhat_ij - 1)          # map pred to [-1, 1]
        G    = w_ij * (2*y_ij   - 1)           # map label to [-1, 1]
        w_ij = y_ij*(v2 - v1) + v1,  v2 = 1 - v1   # w_fg = v2, w_bg = v1

    Background gets the small weight v1 so negative-area pixels enter the Dice
    numerator/denominator without dominating. v2 = 1 - v1 keeps the two-class
    weights complementary. Label smoothing: because predictions are mapped to
    [-1, 1] with a per-pixel label-derived weight, confident positives are
    implicitly softened.

    NOTE: v1 is a free hyper-parameter; the paper does not fix its value in the
    accessible text. Default v1=0.3 (w_fg=0.7, w_bg=0.3) is a heuristic choice.
    """
    def __init__(self, v1=0.3, smooth=1.0):
        super().__init__()
        self.v1 = v1
        self.v2 = 1.0 - v1
        self.smooth = smooth

    def forward(self, inputs, targets):
        yhat = torch.sigmoid(inputs)
        y = targets
        w = y * (self.v2 - self.v1) + self.v1            # [B,1,H,W]
        Ghat = w * (2.0 * yhat - 1.0)
        G = w * (2.0 * y - 1.0)
        num = 2.0 * (Ghat * G).sum()
        den = (Ghat * Ghat).sum() + (G * G).sum()
        return 1.0 - (num + self.smooth) / (den + self.smooth)


class Phase5CellALoss(nn.Module):
    """Phase 5 Gate 1 cell A: plain wIoU+wBCE + WSDice.

    L = (SoftIoU + BCE) + lambda_w * WSDice(v1)

    Targets FP from two complementary angles: IoU contracts over-segmented
    unions; WSDice adds the small-weighted background channel so predicted-FP
    mass is penalized inside the Dice family (faithful to IEEE Access 2020).
    """
    def __init__(self, v1=0.3, lambda_w=1.0, smooth=1.0):
        super().__init__()
        self.lambda_w = lambda_w
        self.iou_bce = IoUBCELoss(smooth=smooth)
        self.wsdice = WeightedSoftDiceLoss(v1=v1, smooth=smooth)

    def forward(self, inputs, targets):
        return self.iou_bce(inputs, targets) + self.lambda_w * self.wsdice(inputs, targets)


class TverskyLoss(nn.Module):
    """Asymmetric Tversky loss (FP-heavy) for false-positive suppression.

    Faithful to Salehi et al. (MICCAI-MLMI 2017), Tversky index:
        TI = |PG| / (|PG| + alpha*|P\\G| + beta*|G\\P|),   loss = 1 - TI
    alpha penalizes false positives, beta penalizes false negatives.
    alpha=beta=0.5 recovers Dice; alpha+beta=1 recovers F_beta.

    Phase 6 cell TC/TD (advisor feedback #2: asymmetric loss for FP reduction):
    alpha=0.7, beta=0.3 -> FP (over-prediction) penalized ~2.3x harder than FN.
    Intended to be combined with DiceBCE as lambda*DiceBCE + (1-lambda)*Tversky
    (Phase6AsymmetricBCELoss), mirroring the Cell-A / Unified-Focal pattern.
    """
    def __init__(self, alpha=0.7, beta=0.3, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, inputs, targets):
        p = torch.sigmoid(inputs).view(-1)
        g = targets.view(-1)
        tp = (p * g).sum()
        fp = (p * (1 - g)).sum()
        fn = ((1 - p) * g).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class Phase6AsymmetricBCELoss(nn.Module):
    """Phase 6 cell: lambda*DiceBCE + (1-lambda)*Tversky(alpha=0.7, beta=0.3).

    Combines the balanced DiceBCE backbone (keeps Dice level stable) with an
    FP-heavy Tversky term to suppress over-prediction without the collapse risk
    of far-weighted cell B (Phase 5 TB). lambda=0.5 default (Unified-Focal-style).
    """
    def __init__(self, alpha=0.7, beta=0.3, lam=0.5, smooth=1.0):
        super().__init__()
        self.lam = lam
        self.dice_bce = DiceBCELoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    def forward(self, inputs, targets):
        return self.lam * self.dice_bce(inputs, targets) + (1.0 - self.lam) * self.tversky(inputs, targets)


class FarWeightedIoUBCELoss(nn.Module):
    """Cell B: wIoU+wBCE with FAR-from-boundary pixel importance weight.

    Flipped F³Net/CFA-Net weighting. F³Net defines pixel importance
        mu_ij = |mean_{N(i,j)}(GT) - GT_ij| in [0, 1]
    mu is ~1 ON the GT contour and ~0 in flat regions (deep interior/background).
    CFA-Net uses (1+5*mu) => weights BOUNDARY pixels ~6x, far ~1x.

    This cell is the *far* variant:
        w_ij = 1 + gamma * (1 - mu_ij)
    so flat/background pixels far from the polyp boundary get ~6x weight while
    the contour gets ~1x. Rationale (X2): FANet's over-segmentation is FP lying
    FAR from the GT boundary (Phase 1: FP distance-to-boundary ~40.7 px), so the
    loss should emphasize suppressing FP mass in the deep background, not on the
    boundary. mu is computed on-the-fly from the target batch (equivalent to
    precomputing from GT since targets are deterministic).

    Note: this is NOT what CFA-Net reports; it is a Phase-5 diagnostic variant
    chosen to match the measured root cause.
    """
    def __init__(self, gamma=5.0, smooth=1.0):
        super().__init__()
        self.gamma = gamma
        self.smooth = smooth

    def _mu(self, targets):
        """mu = |local-mean(GT) - GT|, local window 3x3 (F³Net neighbourhood)."""
        local = F.avg_pool2d(targets, kernel_size=3, stride=1, padding=1)
        return (local - targets).abs()

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        mu = self._mu(targets)                                   # [B,1,H,W]
        w = 1.0 + self.gamma * (1.0 - mu)                        # far = high weight
        wp = w * inputs
        # weighted IoU (absolute pixel weighting; far mass in union is costly)
        inter = (wp * targets).sum()
        union = (w * (inputs + targets - inputs * targets)).sum()
        w_iou = 1.0 - (inter + self.smooth) / (union + self.smooth)
        # weighted BCE: reduction='mean' divides by N (NOT by sum(w)) so the
        # far/background pixels keep their full gamma-fold absolute weight.
        # (normalising by sum(w) would cancel the weighting because flat/bg
        # pixels are the majority of the image)
        w_bce = F.binary_cross_entropy(inputs, targets, weight=w, reduction='mean')
        return w_iou + w_bce


class AdaptiveTverskyLoss(nn.Module):
    """
    Curriculum Tversky Loss.
    Dynamically scales the Tversky alpha (FP penalty) and beta (FN penalty) based on the training epoch.
    Starts balanced to establish boundaries, then transitions to an FP-heavy penalty to suppress over-segmentation.
    """
    def __init__(self, start_alpha=0.5, start_beta=0.5, end_alpha=0.8, end_beta=0.2, 
                 total_epochs=500, transition_start_pct=0.3, smooth=1e-6):
        super().__init__()
        self.start_alpha = start_alpha
        self.start_beta = start_beta
        self.end_alpha = end_alpha
        self.end_beta = end_beta
        self.total_epochs = total_epochs
        self.transition_start_epoch = int(total_epochs * transition_start_pct)
        self.smooth = smooth
        self.current_epoch = 0

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def forward(self, inputs, targets):
        if self.current_epoch < self.transition_start_epoch:
            alpha = self.start_alpha
            beta = self.start_beta
        else:
            progress = min(1.0, (self.current_epoch - self.transition_start_epoch) / max(1, self.total_epochs - self.transition_start_epoch))
            alpha = self.start_alpha + (self.end_alpha - self.start_alpha) * progress
            beta = self.start_beta + (self.end_beta - self.start_beta) * progress

        p = torch.sigmoid(inputs).view(-1)
        g = targets.view(-1)
        tp = (p * g).sum()
        fp = (p * (1 - g)).sum()
        fn = ((1 - p) * g).sum()
        tversky = (tp + self.smooth) / (tp + alpha * fp + beta * fn + self.smooth)
        return 1.0 - tversky


class HybridRecallLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.dice_bce = DiceBCELoss()
        self.tversky = TverskyLoss(alpha=alpha, beta=beta, smooth=smooth)

    def forward(self, inputs, targets):
        return 0.5 * self.dice_bce(inputs, targets) + 0.5 * self.tversky(inputs, targets)
