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
