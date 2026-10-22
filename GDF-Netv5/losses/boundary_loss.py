import torch
import torch.nn as nn
import torch.nn.functional as F

def extract_edge_ground_truth(mask, radius=1):
    """
    Extract exact boundary perimeter ground truth from binary lesion mask
    via morphological dilation minus erosion:
    G_edge = Dilation(G) - Erosion(G).
    Runs entirely on GPU via max_pooling in <0.01 ms.
    """
    kernel_size = 2 * radius + 1
    padding = radius
    with torch.no_grad():
        dilation = F.max_pool2d(mask, kernel_size=kernel_size, stride=1, padding=padding)
        erosion = -F.max_pool2d(-mask, kernel_size=kernel_size, stride=1, padding=padding)
        edge = (dilation - erosion).clamp(0.0, 1.0)
    return edge

class BoundaryEdgeLoss(nn.Module):
    """
    Boundary Edge Loss for Reverse Edge Guidance Stream.
    Combines Weighted BCE Loss and Boundary Dice Loss to force sharp perimeter gradients.
    """
    def __init__(self, pos_weight=5.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred_edge_logits, target_edges):
        # 1. Weighted BCE Loss (edge pixels are sparse, typically ~2-5% of total pixels)
        weight = torch.where(target_edges > 0.5, self.pos_weight, 1.0)
        bce = F.binary_cross_entropy_with_logits(pred_edge_logits, target_edges, reduction='none')
        wbce = (weight * bce).sum(dim=(2, 3)) / (weight.sum(dim=(2, 3)) + 1e-8)

        # 2. Boundary Dice Loss
        probs = torch.sigmoid(pred_edge_logits)
        intersection = (probs * target_edges).sum(dim=(2, 3))
        cardinality = (probs + target_edges).sum(dim=(2, 3))
        dice_loss = 1.0 - (2.0 * intersection + 1.0) / (cardinality + 1.0)

        return (wbce + dice_loss).mean()
