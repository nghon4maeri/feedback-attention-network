import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss (Abraham & Khan, ISBI 2019).
    Focuses gradient updates on hard-to-segment false negatives (lesion boundaries & micro lesions).
    """
    def __init__(self, alpha=0.6, beta=0.4, gamma=1.33, smooth=1e-5):
        super().__init__()
        self.alpha = alpha  # penalty on False Positives
        self.beta = beta    # penalty on False Negatives (higher beta = focus on lesion recall)
        self.gamma = gamma  # focal modulation factor
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        tp = (probs_flat * targets_flat).sum()
        fp = (probs_flat * (1.0 - targets_flat)).sum()
        fn = ((1.0 - probs_flat) * targets_flat).sum()

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        focal_tversky = torch.pow(1.0 - tversky, 1.0 / self.gamma)
        return focal_tversky


class BoundaryDifferenceIoULoss(nn.Module):
    """
    Boundary Difference over Union (BD-IoU) Loss (MICCAI 2023).
    Concentrates optimization exclusively along the boundary discrepancy band.
    """
    def __init__(self, kernel_size=5, smooth=1e-5):
        super().__init__()
        self.kernel_size = kernel_size
        self.smooth = smooth
        self.dilation = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def _extract_boundary_band(self, targets):
        dilated = self.dilation(targets)
        eroded = 1.0 - self.dilation(1.0 - targets)
        boundary_band = dilated - eroded
        return boundary_band

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        boundary_band = self._extract_boundary_band(targets).detach()

        p_boundary = probs * boundary_band
        g_boundary = targets * boundary_band

        intersection = (p_boundary * g_boundary).sum(dim=(2, 3))
        union = (p_boundary + g_boundary - (p_boundary * g_boundary)).sum(dim=(2, 3))

        bd_iou = (intersection + self.smooth) / (union + self.smooth)
        return (1.0 - bd_iou).mean()


class CompositeGDFNetV3Loss(nn.Module):
    """
    Composite Loss for GDF-Netv3 with 5-stage Deep Supervision and Edge Guidance:
    L_total = L_BCE(final, gt) + 1.0 * L_FT(final, gt) + 0.5 * L_BDIoU(final, gt)
            + 0.3 * L_FT(aux1, gt) + 0.2 * L_FT(aux2, gt) + 0.3 * L_BCE(edge_pred, edge_gt)
    """
    def __init__(self, ft_alpha=0.6, ft_beta=0.4, ft_gamma=1.33, bd_kernel=5):
        super().__init__()
        self.focal_tversky = FocalTverskyLoss(alpha=ft_alpha, beta=ft_beta, gamma=ft_gamma)
        self.bd_iou = BoundaryDifferenceIoULoss(kernel_size=bd_kernel)
        self.bce = nn.BCEWithLogitsLoss()
        self.dilation = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def _generate_edge_gt(self, targets):
        dilated = self.dilation(targets)
        eroded = 1.0 - self.dilation(1.0 - targets)
        edge_gt = dilated - eroded
        return edge_gt

    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            final_pred = outputs['logits']
            aux1       = outputs['aux1']
            aux2       = outputs['aux2']
            edge_pred  = outputs['edge_pred']

            loss_bce_final = self.bce(final_pred, targets)
            loss_ft_final  = self.focal_tversky(final_pred, targets)
            loss_bd_final  = self.bd_iou(final_pred, targets)

            loss_aux1 = self.focal_tversky(aux1, targets)
            loss_aux2 = self.focal_tversky(aux2, targets)

            edge_gt = self._generate_edge_gt(targets)
            loss_edge = self.bce(edge_pred, edge_gt)

            total_loss = loss_bce_final + loss_ft_final + 0.5 * loss_bd_final + \
                         0.3 * loss_aux1 + 0.2 * loss_aux2 + 0.3 * loss_edge
            return total_loss
        else:
            loss_bce = self.bce(outputs, targets)
            loss_ft  = self.focal_tversky(outputs, targets)
            loss_bd  = self.bd_iou(outputs, targets)
            return loss_bce + loss_ft + 0.5 * loss_bd
