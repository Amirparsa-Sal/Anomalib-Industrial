"""Metric computation for pixel-level anomaly detection evaluation.

Computes Average Precision (AP), Area Under Precision-Recall Curve (AUPRC),
and Area Under ROC Curve (AUROC) at the pixel level.
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    auc,
)


def compute_pixel_ap(pred_map: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute pixel-level Average Precision for a single image.

    Args:
        pred_map: Predicted anomaly map, shape (H, W), values in [0, 1].
        gt_mask: Ground truth binary mask, shape (H, W), values in {0, 1}.

    Returns:
        Average Precision score. Returns 0.0 if gt_mask is all zeros,
        1.0 if gt_mask is all ones and pred is perfect.
    """
    pred_flat = pred_map.flatten()
    gt_flat = gt_mask.flatten().astype(np.int32)

    if gt_flat.sum() == 0:
        return 1.0 if pred_flat.max() == 0 else float(1.0 - pred_flat.max())

    return float(average_precision_score(gt_flat, pred_flat))


def compute_pixel_metrics(
    pred_maps: list[np.ndarray],
    gt_masks: list[np.ndarray],
) -> dict[str, float]:
    """Compute aggregate pixel-level metrics across multiple images.

    Computes metrics by pooling all pixel predictions and ground truths,
    giving a single global metric value.

    Args:
        pred_maps: List of predicted anomaly maps, each shape (H, W).
        gt_masks: List of ground truth binary masks, each shape (H, W).

    Returns:
        Dictionary with 'AP', 'AUPRC', and 'AUROC' keys.
    """
    all_preds = np.concatenate([m.flatten() for m in pred_maps])
    all_gts = np.concatenate([m.flatten().astype(np.int32) for m in gt_masks])

    results = {}

    # Average Precision
    if all_gts.sum() > 0:
        results["AP"] = float(average_precision_score(all_gts, all_preds))
    else:
        results["AP"] = 1.0

    # AUPRC (same as AP for binary classification but computed via curve)
    if all_gts.sum() > 0:
        precision, recall, _ = precision_recall_curve(all_gts, all_preds)
        results["AUPRC"] = float(auc(recall, precision))
    else:
        results["AUPRC"] = 1.0

    # AUROC
    if all_gts.sum() > 0 and all_gts.sum() < len(all_gts):
        results["AUROC"] = float(roc_auc_score(all_gts, all_preds))
    else:
        results["AUROC"] = 1.0

    return results


def compute_per_image_ap(pred_map: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute pixel-level AP for a single image.

    For normal images (all-zero mask), returns a score based on how
    inactive the prediction is (lower predictions = better).

    Args:
        pred_map: Predicted anomaly map, shape (H, W), values in [0, 1].
        gt_mask: Ground truth binary mask, shape (H, W).

    Returns:
        AP score for this image.
    """
    return compute_pixel_ap(pred_map, gt_mask)


def compute_anomaly_activity(pred_map: np.ndarray) -> float:
    """Compute how 'active' a predicted segmentation map is.

    Used to rank normal image predictions: higher activity means worse
    prediction (more false positive area).

    Args:
        pred_map: Predicted anomaly map, shape (H, W), values in [0, 1].

    Returns:
        Activity score (mean of predicted anomaly values).
    """
    return float(pred_map.mean())
