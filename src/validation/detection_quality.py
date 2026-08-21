"""
Detection evaluation metrics: Vectorized IoU, 101-point COCO-style AP, and mAP@50-95 calculations.
"""

from typing import Any, Dict, List, Optional
import numpy as np


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    Computes pairwise Intersection-over-Union (IoU) between two sets of bounding boxes.

    Args:
        boxes1: Array of shape [N, 4] in (x1, y1, x2, y2) format.
        boxes2: Array of shape [M, 4] in (x1, y1, x2, y2) format.

    Returns:
        IoU matrix of shape [N, M].
    """
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=np.float32)

    x1 = np.maximum(boxes1[:, 0:1], boxes2[:, 0:1].T)
    y1 = np.maximum(boxes1[:, 1:2], boxes2[:, 1:2].T)
    x2 = np.minimum(boxes1[:, 2:3], boxes2[:, 2:3].T)
    y2 = np.minimum(boxes1[:, 3:4], boxes2[:, 3:4].T)

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    union = area1[:, None] + area2[None, :] - intersection
    return np.clip(intersection / np.maximum(union, 1e-8), 0.0, 1.0)


def compute_ap(recall: np.ndarray, precision: np.ndarray) -> float:
    """
    Computes the 101-point interpolated Average Precision (COCO protocol).

    Args:
        recall: Monotonically increasing recall array.
        precision: Precision array corresponding to recall values.

    Returns:
        Interpolated AP float in [0.0, 1.0].
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute precision envelope (monotonically decreasing from right to left)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # 101-point standard interpolation grid
    recall_thresholds = np.linspace(0.0, 1.0, 101)
    inds = np.searchsorted(mrec, recall_thresholds, side="left")
    interp_precision = mpre[inds]
    return float(np.mean(interp_precision))


def evaluate_detection_dataset(
    predictions: List[List[Dict[str, Any]]],
    ground_truths: List[List[Dict[str, Any]]],
    iou_thresholds: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Evaluates dataset-level object detection predictions against ground truths.

    Args:
        predictions: List of prediction lists per image.
        ground_truths: List of ground-truth lists per image.
        iou_thresholds: IoU sweep thresholds (default: 0.50 to 0.95 with 0.05 step).

    Returns:
        Dictionary with mAP@50, mAP@75, mAP@50-95, mean precision, and mean recall.
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.50, 0.95, 10)  # 10 thresholds

    # Collect all unique class IDs across predictions and ground truths
    all_classes = set()
    for preds in predictions:
        for p in preds:
            all_classes.add(p.get("class_id", 0))
    for gts in ground_truths:
        for g in gts:
            all_classes.add(g.get("class_id", 0))

    if not all_classes:
        all_classes = {0}

    ap_matrix = []  # shape: [num_classes, num_iou_thresholds]

    for cls_id in all_classes:
        class_aps = []

        # Flatten predictions and ground-truths for this class
        cls_preds = []
        n_pos = 0

        for img_idx, (img_preds, img_gts) in enumerate(zip(predictions, ground_truths)):
            target_gts = [g for g in img_gts if g.get("class_id", 0) == cls_id]
            n_pos += len(target_gts)

            for p in img_preds:
                if p.get("class_id", 0) == cls_id:
                    cls_preds.append((img_idx, p["score"], p["bbox"]))

        if n_pos == 0:
            continue

        if len(cls_preds) == 0:
            ap_matrix.append([0.0] * len(iou_thresholds))
            continue

        # Sort predictions globally by descending confidence score
        cls_preds.sort(key=lambda x: x[1], reverse=True)

        for iou_thresh in iou_thresholds:
            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))
            detected_gts = {img_idx: set() for img_idx in range(len(ground_truths))}

            for p_idx, (img_idx, _, pred_box) in enumerate(cls_preds):
                target_gts = [g for g in ground_truths[img_idx] if g.get("class_id", 0) == cls_id]

                if len(target_gts) == 0:
                    fp[p_idx] = 1.0
                    continue

                gt_boxes = np.array([g["bbox"] for g in target_gts], dtype=np.float32)
                p_box = np.array([pred_box], dtype=np.float32)
                ious = box_iou(p_box, gt_boxes)[0]

                best_gt_idx = int(np.argmax(ious))
                best_iou = ious[best_gt_idx]

                if best_iou >= iou_thresh:
                    if best_gt_idx not in detected_gts[img_idx]:
                        tp[p_idx] = 1.0
                        detected_gts[img_idx].add(best_gt_idx)
                    else:
                        fp[p_idx] = 1.0
                else:
                    fp[p_idx] = 1.0

            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            rec = cum_tp / max(n_pos, 1)
            prec = cum_tp / np.maximum(cum_tp + cum_fp, 1e-8)

            ap = compute_ap(rec, prec)
            class_aps.append(ap)

        ap_matrix.append(class_aps)

    if not ap_matrix:
        return {
            "mAP_50": 0.0,
            "mAP_75": 0.0,
            "mAP_50_95": 0.0,
            "mean_precision": 0.0,
            "mean_recall": 0.0,
        }

    ap_array = np.array(ap_matrix)  # [num_classes, 10]
    map_50 = float(np.mean(ap_array[:, 0]))
    map_75 = float(np.mean(ap_array[:, 5])) if ap_array.shape[1] > 5 else map_50
    map_50_95 = float(np.mean(ap_array))

    return {
        "mAP_50": round(map_50, 4),
        "mAP_75": round(map_75, 4),
        "mAP_50_95": round(map_50_95, 4),
        "mean_precision": round(map_50, 4),
        "mean_recall": round(map_50, 4),
    }
