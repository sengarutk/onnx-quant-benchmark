"""
Quantization-Aware Post-Processing (Q-Aware NMS) Subsystem.
Calibrates optimal confidence and IoU suppression thresholds per precision/runtime on disjoint calibration splits.
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging import setup_logger
from src.models.preprocess import preprocess_detection_image
from src.models.yolo_adapter import YOLOAdapter
from src.validation.detection_quality import box_iou, evaluate_detection_dataset

logger = setup_logger("q_aware_nms")


def compute_f1_score(
    predictions: List[List[Dict[str, Any]]],
    ground_truths: List[List[Dict[str, Any]]],
    iou_threshold: float = 0.5,
) -> float:
    """
    Computes dataset-level macro F1-score between predictions and ground-truth boxes.

    Args:
        predictions: List of detection dicts per image.
        ground_truths: List of ground truth bounding box dicts per image.
        iou_threshold: IoU threshold for matching (default: 0.5).

    Returns:
        F1-score in [0.0, 1.0].
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for img_preds, img_gts in zip(predictions, ground_truths):
        if not img_gts and not img_preds:
            continue
        if not img_gts:
            total_fp += len(img_preds)
            continue
        if not img_preds:
            total_fn += len(img_gts)
            continue

        gt_boxes = np.array([g["bbox"] for g in img_gts], dtype=np.float32)
        gt_classes = np.array([g.get("class_id", 0) for g in img_gts], dtype=np.int32)
        matched_gt = set()

        sorted_preds = sorted(img_preds, key=lambda x: x.get("score", 0.0), reverse=True)

        for p in sorted_preds:
            p_box = np.array([p["bbox"]], dtype=np.float32)
            p_cls = p.get("class_id", 0)
            ious = box_iou(p_box, gt_boxes)[0]

            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])

            if best_iou >= iou_threshold and best_idx not in matched_gt and gt_classes[best_idx] == p_cls:
                total_tp += 1
                matched_gt.add(best_idx)
            else:
                total_fp += 1

        total_fn += len(img_gts) - len(matched_gt)

    prec = total_tp / max(total_tp + total_fp, 1e-8)
    rec = total_tp / max(total_tp + total_fn, 1e-8)
    if prec + rec < 1e-8:
        return 0.0
    return float(2.0 * (prec * rec) / (prec + rec))


def calibrate_q_aware_nms(
    model_adapter: Any,
    calib_loader: Union[List[Dict[str, Any]], str, Path],
    precision: str = "int8",
    runtime: str = "ort_cpu",
    latency_budget_ms: Optional[float] = None,
    conf_min: float = 0.10,
    conf_max: float = 0.60,
    conf_step: float = 0.05,
    iou_min: float = 0.30,
    iou_max: float = 0.70,
    iou_step: float = 0.05,
) -> Dict[str, Any]:
    """
    Calibrates optimal confidence and IoU suppression thresholds per precision/runtime on disjoint calibration data.

    Args:
        model_adapter: YOLOAdapter instance or inference runtime.
        calib_loader: List of calibration dicts, or Path to manifest.json/manifest.csv.
        precision: Model precision (e.g. 'fp32', 'fp16', 'int8').
        runtime: Execution runtime backend (e.g. 'pytorch', 'ort_cpu', 'ort_cuda').
        latency_budget_ms: Optional latency ceiling constraint in milliseconds.
        conf_min: Minimum confidence threshold to search.
        conf_max: Maximum confidence threshold to search.
        conf_step: Search step size for confidence.
        iou_min: Minimum IoU threshold to search.
        iou_max: Maximum IoU threshold to search.
        iou_step: Search step size for IoU.

    Returns:
        Dictionary containing optimal thresholds, baseline/calibrated F1, delta, and grid metadata.
    """
    logger.info(f"Starting Q-Aware NMS calibration [Precision: {precision}, Runtime: {runtime}]...")

    # Load calibration sample metadata
    samples = []
    if isinstance(calib_loader, (str, Path)):
        p = Path(calib_loader)
        if p.is_file() and p.suffix == ".json":
            data = json.loads(p.read_text(encoding="utf-8"))
            samples = data.get("detection_samples", data.get("samples", []))
        elif p.is_file() and p.suffix == ".csv":
            import csv
            with open(p, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("category") == "detection":
                        samples.append({"image_path": r["path"], "ground_truth_boxes": []})
    elif isinstance(calib_loader, list):
        samples = calib_loader

    if not samples:
        # Fallback to default detection sample dataset
        sample_manifest = PROJECT_ROOT / "data" / "sample_images" / "manifest.json"
        if sample_manifest.is_file():
            data = json.loads(sample_manifest.read_text(encoding="utf-8"))
            samples = data.get("detection_samples", [])

    # If samples lack ground truth boxes, extract from corresponding label file
    processed_samples = []
    for idx, s in enumerate(samples):
        img_p = PROJECT_ROOT / s["image_path"] if not Path(s["image_path"]).is_absolute() else Path(s["image_path"])
        gt_boxes = list(s.get("ground_truth_boxes", []))
        if not gt_boxes and img_p.is_file():
            lbl_p = img_p.parent.parent / "labels" / f"{img_p.stem}.txt"
            if lbl_p.is_file():
                for line in lbl_p.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:5])
                        x1 = int(round((cx - w / 2) * 640))
                        y1 = int(round((cy - h / 2) * 640))
                        x2 = int(round((cx + w / 2) * 640))
                        y2 = int(round((cy + h / 2) * 640))
                        gt_boxes.append({"bbox": [x1, y1, x2, y2], "class_id": cid})
        processed_samples.append({"image_path": str(img_p), "ground_truth_boxes": gt_boxes})

    # Cache raw model predictions to accelerate 2D grid search
    cached_inferences = []
    yolo_adapter = model_adapter if isinstance(model_adapter, YOLOAdapter) else YOLOAdapter()

    for item in processed_samples:
        img_p = Path(item["image_path"])
        if not img_p.is_file():
            continue

        inp_t, ratio, pad, orig_shape = preprocess_detection_image(img_p)

        if hasattr(model_adapter, "predict"):
            out_dict = model_adapter.predict({"images": inp_t.numpy()})
            raw_out = out_dict.get("output0", list(out_dict.values())[0])
        elif hasattr(model_adapter, "forward"):
            raw_out = model_adapter.forward(inp_t)
        else:
            raw_out = yolo_adapter.forward(inp_t)

        cached_inferences.append({
            "raw_output": raw_out,
            "orig_shape": orig_shape,
            "ratio": ratio,
            "pad": pad,
            "gt_boxes": item["ground_truth_boxes"],
        })

    if not cached_inferences:
        raise ValueError("No valid calibration samples available for Q-Aware NMS calibration.")

    # Generate 2D search grid
    conf_grid = np.arange(conf_min, conf_max + 1e-5, conf_step)
    iou_grid = np.arange(iou_min, iou_max + 1e-5, iou_step)

    # 1. Baseline Evaluation (conf=0.25, iou=0.45)
    baseline_preds = []
    gt_list = [c["gt_boxes"] for c in cached_inferences]

    for c in cached_inferences:
        preds = yolo_adapter.postprocess(
            c["raw_output"],
            c["orig_shape"],
            c["ratio"],
            c["pad"],
            conf_threshold=0.25,
            iou_threshold=0.45,
        )
        baseline_preds.append(preds)

    baseline_f1 = compute_f1_score(baseline_preds, gt_list)

    # 2. Grid Search Optimization
    best_f1 = baseline_f1
    best_conf = 0.25
    best_iou = 0.45
    grid_count = 0

    for conf_val in conf_grid:
        for iou_val in iou_grid:
            grid_count += 1
            cand_preds = []
            t0 = time.perf_counter()

            for c in cached_inferences:
                preds = yolo_adapter.postprocess(
                    c["raw_output"],
                    c["orig_shape"],
                    c["ratio"],
                    c["pad"],
                    conf_threshold=float(conf_val),
                    iou_threshold=float(iou_val),
                )
                cand_preds.append(preds)

            t1 = time.perf_counter()
            avg_postprocess_ms = ((t1 - t0) / max(len(cached_inferences), 1)) * 1000.0

            if latency_budget_ms is not None and avg_postprocess_ms > latency_budget_ms:
                continue

            f1 = compute_f1_score(cand_preds, gt_list)

            # Maximize F1; prefer slightly higher confidence on ties for lower false positives
            if f1 > best_f1 or (np.isclose(f1, best_f1) and conf_val > best_conf):
                best_f1 = f1
                best_conf = float(conf_val)
                best_iou = float(iou_val)

    delta_f1 = float(best_f1 - baseline_f1)
    logger.info(
        f"Q-Aware NMS Calibration Complete: Baseline F1={baseline_f1:.4f} (conf=0.25, iou=0.45) -> "
        f"Calibrated F1={best_f1:.4f} (conf={best_conf:.2f}, iou={best_iou:.2f}) [Delta F1: {delta_f1:+.4f}]"
    )

    return {
        "precision": precision,
        "runtime": runtime,
        "baseline_conf": 0.25,
        "baseline_iou": 0.45,
        "baseline_f1": round(float(baseline_f1), 4),
        "optimal_conf": round(float(best_conf), 2),
        "optimal_iou": round(float(best_iou), 2),
        "optimal_f1": round(float(best_f1), 4),
        "f1_delta": round(float(delta_f1), 4),
        "grid_evaluated": int(grid_count),
        "calib_samples": int(len(cached_inferences)),
    }


def apply_q_aware_nms(
    predictions: Union[List[List[Dict[str, Any]]], List[Dict[str, Any]], Any],
    policy_dict: Dict[str, Any],
    original_shapes: Optional[List[Tuple[int, int]]] = None,
    ratios: Optional[List[Tuple[float, float]]] = None,
    pads: Optional[List[Tuple[float, float]]] = None,
    yolo_adapter: Optional[YOLOAdapter] = None,
) -> List[Any]:
    """
    Applies calibrated Q-Aware NMS post-processing using calibrated thresholds.

    Args:
        predictions: Raw tensor outputs or list of candidate detections.
        policy_dict: Calibration dictionary containing 'optimal_conf' and 'optimal_iou'.
        original_shapes: Original image shapes (H, W).
        ratios: Letterbox scaling ratios.
        pads: Letterbox padding shifts.
        yolo_adapter: Optional YOLOAdapter instance for raw tensor decoding.

    Returns:
        Calibrated detections list.
    """
    opt_conf = float(policy_dict.get("optimal_conf", policy_dict.get("conf_threshold", 0.25)))
    opt_iou = float(policy_dict.get("optimal_iou", policy_dict.get("iou_threshold", 0.45)))
    adapter = yolo_adapter or YOLOAdapter(conf_threshold=opt_conf, iou_threshold=opt_iou)

    # Case 1: Raw output array or tensor
    if isinstance(predictions, (torch.Tensor, np.ndarray)) and original_shapes and ratios and pads:
        return adapter.postprocess(
            predictions,
            original_shapes[0],
            ratios[0],
            pads[0],
            conf_threshold=opt_conf,
            iou_threshold=opt_iou,
        )

    # Case 2: List of raw output tensors
    if isinstance(predictions, list) and predictions and isinstance(predictions[0], (torch.Tensor, np.ndarray)):
        results = []
        for i, raw in enumerate(predictions):
            shape = original_shapes[i] if original_shapes and i < len(original_shapes) else (640, 640)
            ratio = ratios[i] if ratios and i < len(ratios) else (1.0, 1.0)
            pad = pads[i] if pads and i < len(pads) else (0.0, 0.0)
            det = adapter.postprocess(raw, shape, ratio, pad, conf_threshold=opt_conf, iou_threshold=opt_iou)
            results.append(det)
        return results

    # Case 3: Already parsed detection dicts -> re-filter by optimal thresholds
    if isinstance(predictions, list):
        if predictions and isinstance(predictions[0], list):
            filtered = []
            for img_preds in predictions:
                kept = [p for p in img_preds if p.get("score", 0.0) >= opt_conf]
                filtered.append(kept)
            return filtered
        else:
            return [p for p in predictions if p.get("score", 0.0) >= opt_conf]

    return predictions
