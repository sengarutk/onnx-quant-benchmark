"""
Quantization Decision-Change Attribution & Detection Flip Analysis.
Measures detection flip rate (Phi) between reference (FP32) and target (quantized) models,
partitioning flips into True Positive and False Positive decision changes.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from src.validation.detection_quality import box_iou


def compute_detection_flips(
    preds_ref: List[List[Dict[str, Any]]],
    preds_target: List[List[Dict[str, Any]]],
    ground_truth: List[List[Dict[str, Any]]],
    iou_thresh: float = 0.5,
) -> Dict[str, float]:
    """
    Computes decision change attribution and detection flip rates between reference and target models.

    Args:
        preds_ref: Detection predictions from baseline reference model (e.g. PyTorch FP32) per image.
        preds_target: Detection predictions from target model (e.g. ORT INT8) per image.
        ground_truth: Ground truth bounding boxes per image.
        iou_thresh: IoU matching threshold for box correspondence (default: 0.5).

    Returns:
        Dictionary with total boxes, matched boxes, flip rate (Phi), lost/new TPs, new/suppressed FPs,
        and attribution fractions (frac_flip_tp, frac_flip_fp).
    """
    total_ref = 0
    total_target = 0
    total_matched = 0

    lost_tps = 0       # GT detected in Ref, missed in Target
    new_tps = 0        # GT missed in Ref, detected in Target
    new_fps = 0        # False alarm in Target, not in Ref
    suppressed_fps = 0 # False alarm in Ref, eliminated in Target

    for r_preds, t_preds, gts in zip(preds_ref, preds_target, ground_truth):
        total_ref += len(r_preds)
        total_target += len(t_preds)

        gt_boxes = np.array([g["bbox"] for g in gts], dtype=np.float32) if gts else np.empty((0, 4), dtype=np.float32)
        gt_classes = np.array([g.get("class_id", 0) for g in gts], dtype=np.int32) if gts else np.empty((0,), dtype=np.int32)

        # 1. Classify Reference predictions as TP or FP against GT
        ref_is_tp = np.zeros(len(r_preds), dtype=bool)
        matched_gt_ref = set()
        if len(r_preds) > 0 and len(gt_boxes) > 0:
            for r_idx, r in enumerate(r_preds):
                r_box = np.array([r["bbox"]], dtype=np.float32)
                r_cls = r.get("class_id", 0)
                ious = box_iou(r_box, gt_boxes)[0]
                best_gt = int(np.argmax(ious))
                if ious[best_gt] >= iou_thresh and best_gt not in matched_gt_ref and gt_classes[best_gt] == r_cls:
                    ref_is_tp[r_idx] = True
                    matched_gt_ref.add(best_gt)

        # 2. Classify Target predictions as TP or FP against GT
        target_is_tp = np.zeros(len(t_preds), dtype=bool)
        matched_gt_target = set()
        if len(t_preds) > 0 and len(gt_boxes) > 0:
            for t_idx, t in enumerate(t_preds):
                t_box = np.array([t["bbox"]], dtype=np.float32)
                t_cls = t.get("class_id", 0)
                ious = box_iou(t_box, gt_boxes)[0]
                best_gt = int(np.argmax(ious))
                if ious[best_gt] >= iou_thresh and best_gt not in matched_gt_target and gt_classes[best_gt] == t_cls:
                    target_is_tp[t_idx] = True
                    matched_gt_target.add(best_gt)

        # 3. Match Reference and Target boxes directly to find common detections and flips
        matched_target_indices = set()
        matched_ref_indices = set()

        if len(r_preds) > 0 and len(t_preds) > 0:
            r_boxes = np.array([r["bbox"] for r in r_preds], dtype=np.float32)
            t_boxes = np.array([t["bbox"] for t in t_preds], dtype=np.float32)
            iou_matrix = box_iou(r_boxes, t_boxes)

            for r_idx, r in enumerate(r_preds):
                r_cls = r.get("class_id", 0)
                best_t_idx = int(np.argmax(iou_matrix[r_idx]))
                best_iou = float(iou_matrix[r_idx, best_t_idx])

                if best_iou >= iou_thresh and best_t_idx not in matched_target_indices:
                    t_cls = t_preds[best_t_idx].get("class_id", 0)
                    if r_cls == t_cls:
                        total_matched += 1
                        matched_target_indices.add(best_t_idx)
                        matched_ref_indices.add(r_idx)

        # 4. Attribute unmatched Reference boxes (Dropped in Target)
        for r_idx in range(len(r_preds)):
            if r_idx not in matched_ref_indices:
                if ref_is_tp[r_idx]:
                    lost_tps += 1
                else:
                    suppressed_fps += 1

        # 5. Attribute unmatched Target boxes (New in Target)
        for t_idx in range(len(t_preds)):
            if t_idx not in matched_target_indices:
                if target_is_tp[t_idx]:
                    new_tps += 1
                else:
                    new_fps += 1

    flip_tp = lost_tps + new_tps
    flip_fp = new_fps + suppressed_fps
    total_flips = flip_tp + flip_fp

    # Total decision events = matched boxes + total flips
    total_decision_events = total_matched + total_flips
    flip_rate = float(total_flips / max(total_decision_events, 1))

    frac_flip_tp = float(flip_tp / max(total_flips, 1)) if total_flips > 0 else 0.0
    frac_flip_fp = float(flip_fp / max(total_flips, 1)) if total_flips > 0 else 0.0

    return {
        "total_ref_boxes": int(total_ref),
        "total_target_boxes": int(total_target),
        "total_matched_boxes": int(total_matched),
        "flip_rate": round(flip_rate, 4),
        "total_flips": int(total_flips),
        "lost_tps": int(lost_tps),
        "new_tps": int(new_tps),
        "new_fps": int(new_fps),
        "suppressed_fps": int(suppressed_fps),
        "flip_tp_count": int(flip_tp),
        "flip_fp_count": int(flip_fp),
        "frac_flip_tp": round(frac_flip_tp, 4),
        "frac_flip_fp": round(frac_flip_fp, 4),
        "iou_threshold": float(iou_thresh),
    }
