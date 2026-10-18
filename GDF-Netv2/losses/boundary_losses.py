import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    r"""
    Focal Tversky Loss (Abraham & Khan, ISBI 2019).
    Focuses gradient updates on hard-to-segment false negatives (lesion boundaries & small lesions).
    Formula: TI = (|P ∩ G|) / (|P ∩ G| + alpha * |P \ G| + beta * |G \ P|)
             FTL = (1 - TI)^(1 / gamma)
    """
    def __init__(self, alpha=0.7, beta=0.3, gamma=1.33, smooth=1e-5):
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
    Boundary Difference over Union (BD-IoU) Loss (Sun, Luo, Li, MICCAI 2023).
    Concentrates optimization exclusively along the boundary discrepancy zone:
    Omega_B = Dilate(G, d) - Erode(G, d)
    """
    def __init__(self, kernel_size=5, smooth=1e-5):
        super().__init__()
        self.kernel_size = kernel_size
        self.smooth = smooth
        # MaxPool2d acts as morphological dilation on binary mask
        self.dilation = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def _extract_boundary_band(self, targets):
        dilated = self.dilation(targets)
        # Invert, dilate, invert to achieve erosion
        eroded = 1.0 - self.dilation(1.0 - targets)
        boundary_band = dilated - eroded
        return boundary_band

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        boundary_band = self._extract_boundary_band(targets).detach()

        # Weight probabilities and targets by the boundary band
        p_boundary = probs * boundary_band
        g_boundary = targets * boundary_band

        intersection = (p_boundary * g_boundary).sum(dim=(2, 3))
        union = (p_boundary + g_boundary - (p_boundary * g_boundary)).sum(dim=(2, 3))

        bd_iou = (intersection + self.smooth) / (union + self.smooth)
        return (1.0 - bd_iou).mean()


class CompositeGDFNetV2Loss(nn.Module):
    """
    Composite Loss for GDF-Netv2:
    L_total = L_FT(pred_t2, gt) + 0.5 * L_BDIoU(pred_t2, gt) + 0.4 * L_FT(pred_t1, gt) + 0.3 * L_Edge(edge_pred, edge_gt)
    """
    def __init__(self, ft_alpha=0.7, ft_beta=0.3, ft_gamma=1.33, bd_kernel=5):
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
        final_pred = outputs['final_pred']
        pred_t1    = outputs['pred_t1']
        edge_pred  = outputs['edge_pred']

        # 1. Primary Loss at t=2
        loss_ft_final = self.focal_tversky(final_pred, targets)
        loss_bd_final = self.bd_iou(final_pred, targets)

        # 2. Deep Supervision at t=1
        loss_ft_t1    = self.focal_tversky(pred_t1, targets)

        # 3. Auxiliary Edge Supervision
        edge_gt = self._generate_edge_gt(targets)
        loss_edge = self.bce(edge_pred, edge_gt)

        total_loss = loss_ft_final + 0.5 * loss_bd_final + 0.4 * loss_ft_t1 + 0.3 * loss_edge
        return total_loss, {
            'loss_total': total_loss.item(),
            'loss_ft_final': loss_ft_final.item(),
            'loss_bd_final': loss_bd_final.item(),
            'loss_ft_t1': loss_ft_t1.item(),
            'loss_edge': loss_edge.item()
        }
