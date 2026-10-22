import torch
import torch.nn as nn

try:
    from .structure_loss import StructureLoss
    from .boundary_loss import BoundaryEdgeLoss, extract_edge_ground_truth
except (ImportError, ValueError):
    from structure_loss import StructureLoss
    from boundary_loss import BoundaryEdgeLoss, extract_edge_ground_truth

class GDFNetV5CompositeLoss(nn.Module):
    """
    Composite Multi-Task Loss for GDF-Netv5.
    
    Total Loss Formulation:
    L_total = L_struct(logits, G) + L_struct(p1, G) + 0.5 * L_struct(p2, G)
              + 0.3 * L_struct(p3, G) + 0.2 * L_struct(p4, G)
              + 0.8 * L_edge(edge_logits, G_edge)
    """
    def __init__(self, edge_weight=0.8):
        super().__init__()
        self.structure_loss = StructureLoss()
        self.edge_loss = BoundaryEdgeLoss()
        self.edge_weight = edge_weight

    def forward(self, outputs, targets):
        if isinstance(outputs, dict):
            p_final = outputs['logits']
            p1 = outputs['p1']
            p2 = outputs['p2']
            p3 = outputs['p3']
            p4 = outputs['p4']
            edge_logits = outputs.get('edge_logits')

            l_final = self.structure_loss(p_final, targets)
            l_p1 = self.structure_loss(p1, targets)
            l_p2 = self.structure_loss(p2, targets)
            l_p3 = self.structure_loss(p3, targets)
            l_p4 = self.structure_loss(p4, targets)

            seg_loss = l_final + l_p1 + 0.5 * l_p2 + 0.3 * l_p3 + 0.2 * l_p4

            if edge_logits is not None:
                g_edge = extract_edge_ground_truth(targets)
                l_edge = self.edge_loss(edge_logits, g_edge)
                return seg_loss + self.edge_weight * l_edge
            return seg_loss
        else:
            return self.structure_loss(outputs, targets)
