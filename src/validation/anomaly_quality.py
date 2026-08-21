"""
Industrial Anomaly evaluation metrics: Image AUROC, Pixel AUROC, and Per-Region Overlap (AU-PRO).
"""

from typing import List, Union
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score


def compute_image_auroc(y_true: Union[List[int], np.ndarray], y_score: Union[List[float], np.ndarray]) -> float:
    """
    Computes image-level Area Under the ROC Curve for binary classification (normal vs anomalous).

    Args:
        y_true: Binary ground-truth labels (0 = normal, 1 = anomalous).
        y_score: Continuous anomaly prediction scores.

    Returns:
        Scalar AUROC in [0.0, 1.0].
    """
    y_true_arr = np.asarray(y_true, dtype=np.int32)
    y_score_arr = np.asarray(y_score, dtype=np.float32)

    # Handle edge case where only one class is present in test set
    if len(np.unique(y_true_arr)) < 2:
        return 1.0 if np.all(y_true_arr == 0) else 0.5

    return float(round(roc_auc_score(y_true_arr, y_score_arr), 4))


def compute_pixel_auroc(
    ground_truth_masks: Union[List[np.ndarray], np.ndarray],
    anomaly_maps: Union[List[np.ndarray], np.ndarray],
) -> float:
    """
    Computes pixel-level Area Under the ROC Curve for localized defect segmentation.

    Args:
        ground_truth_masks: Binary ground-truth masks [N, H, W] or list of [H, W] arrays.
        anomaly_maps: Continuous pixel anomaly intensity maps.

    Returns:
        Scalar pixel AUROC in [0.0, 1.0].
    """
    flat_masks = np.concatenate([m.flatten() for m in ground_truth_masks]).astype(np.int32)
    flat_maps = np.concatenate([a.flatten() for a in anomaly_maps]).astype(np.float32)

    # Threshold masks to binary 0 / 1
    flat_masks = (flat_masks > 0.5).astype(np.int32)

    if len(np.unique(flat_masks)) < 2:
        return 1.0

    return float(round(roc_auc_score(flat_masks, flat_maps), 4))


def compute_aupro(
    ground_truth_masks: List[np.ndarray],
    anomaly_maps: List[np.ndarray],
    max_fpr: float = 0.3,
    num_thresholds: int = 200,
) -> float:
    """
    Computes Area Under the Per-Region Overlap (AU-PRO) curve up to maximum False Positive Rate.

    Standard metric for industrial defect inspection on MVTec-AD benchmark.

    Args:
        ground_truth_masks: List of binary ground-truth defect masks [H, W].
        anomaly_maps: List of continuous anomaly maps [H, W].
        max_fpr: Maximum False Positive Rate limit for area integration (default: 0.30).
        num_thresholds: Number of threshold evaluation points.

    Returns:
        Normalized AU-PRO score in [0.0, 1.0].
    """
    # Collect connected components (defect regions)
    regions = []
    total_normal_pixels = 0

    for mask, a_map in zip(ground_truth_masks, anomaly_maps):
        bin_mask = (mask > 0.5).astype(np.uint8)
        total_normal_pixels += int(np.sum(bin_mask == 0))

        labeled_mask, num_features = label(bin_mask)
        for region_id in range(1, num_features + 1):
            region_mask = (labeled_mask == region_id)
            regions.append(a_map[region_mask])

    if not regions or total_normal_pixels == 0:
        return 1.0

    # Determine threshold range using percentiles
    all_scores = np.concatenate([a.flatten() for a in anomaly_maps])
    thresholds = np.unique(np.percentile(all_scores, np.linspace(0, 100, num_thresholds)))[::-1]

    fpr_list = [0.0]
    pro_list = [0.0]

    for t in thresholds:
        fp_pixels = sum(
            np.sum((a_map >= t) & (mask <= 0.5))
            for mask, a_map in zip(ground_truth_masks, anomaly_maps)
        )
        fpr = float(fp_pixels / total_normal_pixels)

        region_overlaps = [float(np.mean(region_scores >= t)) for region_scores in regions]
        pro = float(np.mean(region_overlaps))

        fpr_list.append(fpr)
        pro_list.append(pro)

    fpr_arr = np.array(fpr_list)
    pro_arr = np.array(pro_list)

    # Sort by ascending FPR
    sort_idx = np.argsort(fpr_arr)
    fpr_arr = fpr_arr[sort_idx]
    pro_arr = pro_arr[sort_idx]

    # Deduplicate FPR values
    unique_fpr, unique_indices = np.unique(fpr_arr, return_index=True)
    unique_pro = pro_arr[unique_indices]

    # Interpolate PRO across uniform FPR grid from 0 to max_fpr
    fpr_grid = np.linspace(0.0, max_fpr, 500)
    pro_interp = np.interp(fpr_grid, unique_fpr, unique_pro)

    aupro = float(np.trapezoid(pro_interp, fpr_grid) / max_fpr)
    return float(round(np.clip(aupro, 0.0, 1.0), 4))
