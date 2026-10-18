import numpy as np
import pandas as pd
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt, binary_erosion

def compute_surface_distances(mask_gt, mask_pred):
    """Compute HD95 and ASD between two binary masks."""
    if mask_gt.sum() == 0 and mask_pred.sum() == 0:
        return 0.0, 0.0
    if mask_gt.sum() == 0 or mask_pred.sum() == 0:
        diag = np.sqrt(mask_gt.shape[0]**2 + mask_gt.shape[1]**2)
        return float(diag), float(diag / 2.0)

    # Extract contours via morphological difference
    border_gt = mask_gt ^ binary_erosion(mask_gt, iterations=1)
    border_pred = mask_pred ^ binary_erosion(mask_pred, iterations=1)

    pts_gt = np.argwhere(border_gt)
    pts_pred = np.argwhere(border_pred)

    if len(pts_gt) == 0 or len(pts_pred) == 0:
        return 0.0, 0.0

    # Directed Hausdorff Distances
    d_gt_to_pred = directed_hausdorff(pts_gt, pts_pred)[0]
    d_pred_to_gt = directed_hausdorff(pts_pred, pts_gt)[0]
    hd = max(d_gt_to_pred, d_pred_to_gt)

    # Average Surface Distance via Distance Transforms
    dt_gt = distance_transform_edt(~border_gt)
    dt_pred = distance_transform_edt(~border_pred)
    asd = (dt_gt[border_pred].mean() + dt_pred[border_gt].mean()) / 2.0

    # Approximate 95th percentile
    hd95 = hd * 0.95
    return float(hd95), float(asd)


def compute_boundary_f1(mask_gt, mask_pred, theta=2):
    """Compute Boundary F1 score at tolerance threshold theta (pixels)."""
    if mask_gt.sum() == 0 and mask_pred.sum() == 0:
        return 1.0
    if mask_gt.sum() == 0 or mask_pred.sum() == 0:
        return 0.0

    border_gt = mask_gt ^ binary_erosion(mask_gt, iterations=1)
    border_pred = mask_pred ^ binary_erosion(mask_pred, iterations=1)

    if border_gt.sum() == 0 or border_pred.sum() == 0:
        return 0.0

    dt_gt = distance_transform_edt(~border_gt)
    dt_pred = distance_transform_edt(~border_pred)

    precision = (dt_gt[border_pred] <= theta).sum() / border_pred.sum()
    recall = (dt_pred[border_gt] <= theta).sum() / border_gt.sum()

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def evaluate_binary_metrics(preds, targets, threshold=0.5):
    """
    Evaluates 9 region and boundary metrics across a set of binary masks:
    Returns per-image dictionary and aggregated statistical summary (Mean, Std, Median, Min, Max).
    """
    results = []

    for i in range(len(preds)):
        p = (preds[i] > threshold).astype(bool)
        t = (targets[i] > 0.5).astype(bool)

        tp = np.logical_and(p, t).sum()
        fp = np.logical_and(p, ~t).sum()
        fn = np.logical_and(~p, t).sum()
        tn = np.logical_and(~p, ~t).sum()

        dice = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-8)
        miou = tp / (tp + fp + fn + 1e-8)
        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        fpr  = fp / (fp + tn + 1e-8)

        hd95, asd = compute_surface_distances(t, p)
        bf2 = compute_boundary_f1(t, p, theta=2)

        results.append({
            'dice': float(dice),
            'miou': float(miou),
            'sensitivity': float(sens),
            'specificity': float(spec),
            'precision': float(prec),
            'fpr': float(fpr),
            'hd95': float(hd95),
            'asd': float(asd),
            'bf2': float(bf2)
        })

    df = pd.DataFrame(results)
    summary = []
    for metric in df.columns:
        summary.append({
            'Metric': metric,
            'Mean': df[metric].mean(),
            'Std': df[metric].std(),
            'Median': df[metric].median(),
            'Min': df[metric].min(),
            'Max': df[metric].max()
        })

    return df, pd.DataFrame(summary)
