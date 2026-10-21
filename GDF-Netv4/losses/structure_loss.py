import torch
import torch.nn as nn
import torch.nn.functional as F

class StructureLoss(nn.Module):
    """
    Structure Loss (Weighted BCE + Weighted IoU).
    
    Standard SOTA loss in medical image segmentation (PraNet, Polyp-PVT, EMCAD).
    Adaptively increases pixel weights near ambiguous lesion boundaries (up to 6x)
    based on local gradient variance, penalizing boundary errors while preventing
    over-segmentation into healthy skin.
    """
    def __init__(self, kernel_size=31, weight_factor=5.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.weight_factor = weight_factor
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, pred_logits, targets):
        """
        pred_logits: [B, 1, H, W] (raw logits)
        targets: [B, 1, H, W] (binary {0, 1})
        """
        # 1. Compute adaptive boundary weight map
        with torch.no_grad():
            local_mean = self.avg_pool(targets)
            boundary_diff = torch.abs(targets - local_mean)
            weight = 1.0 + self.weight_factor * boundary_diff

        # 2. Weighted BCE Loss
        bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='none')
        wbce = (weight * bce).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + 1e-8)

        # 3. Weighted IoU Loss
        probs = torch.sigmoid(pred_logits)
        intersection = (weight * probs * targets).sum(dim=(2, 3))
        union = (weight * (probs + targets - probs * targets)).sum(dim=(2, 3))
        wiou = 1.0 - (intersection + 1.0) / (union + 1.0)

        return (wbce + wiou).mean()


class GDFNetV4CompositeLoss(nn.Module):
    """
    Composite Deep Supervision Loss for GDF-Netv4.
    
    Computes StructureLoss on the final aggregated prediction and all 4 cascaded stages:
    L_total = L(P_final) + L(P_1) + 0.5 * L(P_2) + 0.3 * L(P_3) + 0.2 * L(P_4)
    """
    def __init__(self):
        super().__init__()
        self.structure_loss = StructureLoss()

    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            p_final = outputs['logits']
            p1 = outputs['p1']
            p2 = outputs['p2']
            p3 = outputs['p3']
            p4 = outputs['p4']

            l_final = self.structure_loss(p_final, targets)
            l_p1 = self.structure_loss(p1, targets)
            l_p2 = self.structure_loss(p2, targets)
            l_p3 = self.structure_loss(p3, targets)
            l_p4 = self.structure_loss(p4, targets)

            return l_final + l_p1 + 0.5 * l_p2 + 0.3 * l_p3 + 0.2 * l_p4
        else:
            return self.structure_loss(outputs, targets)
