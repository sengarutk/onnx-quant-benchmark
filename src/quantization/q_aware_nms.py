"""
Quantization-Aware Post-Processing (Q-Aware NMS).

Q-Aware NMS re-selects the confidence and IoU suppression thresholds (tau_conf, tau_iou) separately for every
(precision, runtime) pair by maximizing detection F1 on a labeled calibration split that is disjoint from the
test split. Raw model outputs are cached once, so the full grid search only re-runs post-processing.
"""

import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import torch

from src.common.logging import setup_logger
from src.models.yolo_adapter import YOLOAdapter
from src.validation.detection_quality import box_iou

logger = setup_logger("q_aware_nms")

DEFAULT_CONF = 0.25
DEFAULT_IOU = 0.45


def detection_counts(
    predictions: List[List[Dict[str, Any]]],
    ground_truths: List[List[Dict[str, Any]]],
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """
    Greedy, class-aware, score-ordered matching of predictions to ground truth (each GT matched at most once).

    Returns:
        (true positives, false positives, false negatives) summed over the dataset.
    """
    tp = fp = fn = 0
    for preds, gts in zip(predictions, ground_truths):
        if not gts:
            fp += len(preds)
            continue
        gt_boxes = np.array([g["bbox"] for g in gts], dtype=np.float32)
        gt_cls = np.array([g.get("class_id", 0) for g in gts], dtype=np.int64)
        matched = np.zeros(len(gts), dtype=bool)
        for p in sorted(preds, key=lambda d: d.get("score", 0.0), reverse=True):
            ious = box_iou(np.array([p["bbox"]], dtype=np.float32), gt_boxes)[0]
            ious[(gt_cls != p.get("class_id", 0)) | matched] = -1.0
            j = int(np.argmax(ious))
            if ious[j] >= iou_threshold:
                matched[j] = True
                tp += 1
            else:
                fp += 1
        fn += int((~matched).sum())
    return tp, fp, fn


def compute_f1_score(
    predictions: List[List[Dict[str, Any]]],
    ground_truths: List[List[Dict[str, Any]]],
    iou_threshold: float = 0.5,
) -> float:
    """Dataset-level micro-averaged F1 (TP/FP/FN pooled over all images and classes)."""
    tp, fp, fn = detection_counts(predictions, ground_truths, iou_threshold)
    if tp == 0:
        return 0.0
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    return float(2.0 * prec * rec / (prec + rec))


def _postprocess_all(
    cache: Sequence[Dict[str, Any]], adapter: YOLOAdapter, conf: float, iou: float
) -> List[List[Dict[str, Any]]]:
    return [adapter.postprocess(c["raw"], c["orig_shape"], c["ratio"], c["pad"], conf, iou) for c in cache]


def calibrate_q_aware_nms(
    calib_cache: Sequence[Dict[str, Any]],
    adapter: Optional[YOLOAdapter] = None,
    precision: str = "int8",
    runtime: str = "ort_cpu",
    latency_budget_ms: Optional[float] = None,
    conf_grid: Optional[Sequence[float]] = None,
    iou_grid: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """
    Grid-searches (tau_conf, tau_iou) maximizing calibration-split F1.

    Args:
        calib_cache: Cached raw outputs from ``src.validation.task_eval.run_detector`` on the CALIBRATION split
            (each item: raw, orig_shape, ratio, pad, gt). Must contain ground-truth boxes.
        adapter: YOLOAdapter used for decoding/NMS.
        latency_budget_ms: Optional ceiling on mean per-image post-processing time; slower grid points are skipped.
        conf_grid / iou_grid: Candidate thresholds (default 0.10..0.95 and 0.05..0.70 in steps of 0.05, 252 points).

    Returns:
        Dictionary with the default and calibrated operating points and their calibration-split F1.
    """
    if not calib_cache:
        raise ValueError("No valid calibration samples available for Q-Aware NMS calibration.")
    gts = [c["gt"] for c in calib_cache]
    if sum(len(g) for g in gts) == 0:
        raise ValueError("Q-Aware NMS calibration requires a labeled calibration split (no ground-truth boxes found).")

    adapter = adapter or YOLOAdapter()
    conf_grid = np.round(np.arange(0.10, 0.95 + 1e-9, 0.05), 2) if conf_grid is None else np.asarray(conf_grid)
    iou_grid = np.round(np.arange(0.05, 0.70 + 1e-9, 0.05), 2) if iou_grid is None else np.asarray(iou_grid)

    baseline_f1 = compute_f1_score(_postprocess_all(calib_cache, adapter, DEFAULT_CONF, DEFAULT_IOU), gts)
    best = (baseline_f1, DEFAULT_CONF, DEFAULT_IOU)
    evaluated = skipped = 0
    t_start = time.perf_counter()

    for conf in conf_grid:
        for iou in iou_grid:
            t0 = time.perf_counter()
            preds = _postprocess_all(calib_cache, adapter, float(conf), float(iou))
            post_ms = (time.perf_counter() - t0) / len(calib_cache) * 1000.0
            if latency_budget_ms is not None and post_ms > latency_budget_ms:
                skipped += 1
                continue
            evaluated += 1
            f1 = compute_f1_score(preds, gts)
            # Strict improvement required; ties keep the earlier (lower-threshold) candidate.
            if f1 > best[0] + 1e-12:
                best = (f1, float(conf), float(iou))

    result = {
        "precision": precision,
        "runtime": runtime,
        "baseline_conf": DEFAULT_CONF,
        "baseline_iou": DEFAULT_IOU,
        "baseline_f1": round(float(baseline_f1), 4),
        "optimal_conf": round(best[1], 2),
        "optimal_iou": round(best[2], 2),
        "optimal_f1": round(float(best[0]), 4),
        "f1_delta": round(float(best[0] - baseline_f1), 4),
        "grid_evaluated": int(evaluated),
        "grid_skipped_latency": int(skipped),
        "calib_samples": int(len(calib_cache)),
        "search_seconds": round(time.perf_counter() - t_start, 3),
        "optimum_on_grid_boundary": bool(best[1] in (float(conf_grid[0]), float(conf_grid[-1]))
                                         or best[2] in (float(iou_grid[0]), float(iou_grid[-1]))),
    }
    logger.info(
        f"Q-Aware NMS [{precision}/{runtime}]: calib F1 {result['baseline_f1']:.4f} @ (0.25, 0.45) -> "
        f"{result['optimal_f1']:.4f} @ ({result['optimal_conf']:.2f}, {result['optimal_iou']:.2f})"
    )
    return result


def apply_q_aware_nms(
    predictions: Union[List[Any], np.ndarray, torch.Tensor],
    policy_dict: Dict[str, Any],
    original_shapes: Optional[List[Tuple[int, int]]] = None,
    ratios: Optional[List[Tuple[float, float]]] = None,
    pads: Optional[List[Tuple[float, float]]] = None,
    yolo_adapter: Optional[YOLOAdapter] = None,
) -> List[Any]:
    """
    Applies a calibrated policy.

    Raw tensors (single or list) are fully re-post-processed with (optimal_conf, optimal_iou). Already-decoded
    detection lists can only be re-filtered by confidence (NMS is not re-run on them).
    """
    opt_conf = float(policy_dict.get("optimal_conf", policy_dict.get("conf_threshold", DEFAULT_CONF)))
    opt_iou = float(policy_dict.get("optimal_iou", policy_dict.get("iou_threshold", DEFAULT_IOU)))
    adapter = yolo_adapter or YOLOAdapter(conf_threshold=opt_conf, iou_threshold=opt_iou)

    def _meta(i: int) -> Tuple[Tuple[int, int], Tuple[float, float], Tuple[float, float]]:
        shape = original_shapes[i] if original_shapes and i < len(original_shapes) else (640, 640)
        ratio = ratios[i] if ratios and i < len(ratios) else (1.0, 1.0)
        pad = pads[i] if pads and i < len(pads) else (0.0, 0.0)
        return shape, ratio, pad

    if isinstance(predictions, (torch.Tensor, np.ndarray)):
        return adapter.postprocess(predictions, *_meta(0), conf_threshold=opt_conf, iou_threshold=opt_iou)

    if isinstance(predictions, list) and predictions and isinstance(predictions[0], (torch.Tensor, np.ndarray)):
        return [
            adapter.postprocess(raw, *_meta(i), conf_threshold=opt_conf, iou_threshold=opt_iou)
            for i, raw in enumerate(predictions)
        ]

    if isinstance(predictions, list):
        if predictions and isinstance(predictions[0], list):
            return [[p for p in img if p.get("score", 0.0) >= opt_conf] for img in predictions]
        return [p for p in predictions if p.get("score", 0.0) >= opt_conf]

    return predictions
