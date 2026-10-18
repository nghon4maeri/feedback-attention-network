"""Segmentation metrics used by FANet evaluation."""
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def precision_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)


def recall_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)


def F2_score(y_true, y_pred, beta=2):
    p = precision_score(y_true, y_pred)
    r = recall_score(y_true, y_pred)
    return (1 + beta**2.) * (p * r) / float(beta**2 * p + r + 1e-15)


def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


def calculate_metrics(y_true, y_pred):
    """y_true, y_pred: torch tensors (already sigmoid-applied for y_pred)."""
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()

    y_pred = (y_pred > 0.5).reshape(-1).astype(np.uint8)
    y_true = (y_true > 0.5).reshape(-1).astype(np.uint8)

    score_jaccard = jac_score(y_true, y_pred)
    score_f1 = dice_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_fbeta = F2_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    confusion = confusion_matrix(y_true, y_pred)
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        score_specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    else:
        score_specificity = 0.0

    return [score_jaccard, score_f1, score_recall, score_precision,
            score_specificity, score_acc, score_fbeta]
