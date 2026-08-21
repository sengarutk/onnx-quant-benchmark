"""
Numerical equivalence validation and tensor parity checks across inference runtimes.
"""

from typing import Any, Dict, List
import numpy as np


def compute_tensor_diff(tensor_a: np.ndarray, tensor_b: np.ndarray) -> Dict[str, float]:
    """
    Computes element-wise differences, norms, and numerical parity metrics between two tensors.

    Args:
        tensor_a: Reference baseline NumPy tensor (e.g. PyTorch FP32).
        tensor_b: Candidate test NumPy tensor (e.g. ONNX FP16 / TensorRT INT8).

    Returns:
        Dict containing max_abs_error, mean_abs_error, rmse, cosine_similarity, and anomaly counts.
    """
    a = np.asarray(tensor_a, dtype=np.float64)
    b = np.asarray(tensor_b, dtype=np.float64)

    nan_count = int(np.isnan(a).sum() + np.isnan(b).sum())
    inf_count = int(np.isinf(a).sum() + np.isinf(b).sum())

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch in tensor comparison: {a.shape} vs {b.shape}")

    diff = np.abs(a - b)
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))

    # Cosine similarity across flattened vectors
    norm_a = np.linalg.norm(a.flatten())
    norm_b = np.linalg.norm(b.flatten())
    if norm_a > 1e-8 and norm_b > 1e-8:
        cosine_sim = float(np.dot(a.flatten(), b.flatten()) / (norm_a * norm_b))
    else:
        cosine_sim = 1.0 if max_err < 1e-6 else 0.0

    return {
        "max_abs_error": float(max_err),
        "mean_abs_error": float(mean_err),
        "rmse": float(rmse),
        "cosine_similarity": float(cosine_sim),
        "nan_count": nan_count,
        "inf_count": inf_count,
    }


def check_detection_output_consistency(
    detections_a: List[Dict[str, Any]],
    detections_b: List[Dict[str, Any]],
    iou_match_threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Compares two detection sets (e.g. PyTorch reference vs ONNX/TensorRT output).

    Args:
        detections_a: Reference detection list.
        detections_b: Candidate detection list.
        iou_match_threshold: IoU threshold for matching corresponding bounding boxes.

    Returns:
        Summary dict containing count differences, matched box deviations, and class agreement rate.
    """
    count_a = len(detections_a)
    count_b = len(detections_b)

    if count_a == 0 and count_b == 0:
        return {
            "count_a": 0,
            "count_b": 0,
            "matched_boxes": 0,
            "mean_box_mae": 0.0,
            "mean_score_mae": 0.0,
            "class_match_rate": 1.0,
        }

    matched = 0
    box_diffs = []
    score_diffs = []
    class_matches = 0

    from src.validation.detection_quality import box_iou

    if count_a > 0 and count_b > 0:
        boxes_a = np.array([d["bbox"] for d in detections_a], dtype=np.float32)
        boxes_b = np.array([d["bbox"] for d in detections_b], dtype=np.float32)
        ious = box_iou(boxes_a, boxes_b)

        for i in range(count_a):
            best_j = int(np.argmax(ious[i]))
            if ious[i, best_j] >= iou_match_threshold:
                matched += 1
                box_diffs.append(np.mean(np.abs(boxes_a[i] - boxes_b[best_j])))
                score_diffs.append(abs(detections_a[i]["score"] - detections_b[best_j]["score"]))
                if detections_a[i]["class_id"] == detections_b[best_j]["class_id"]:
                    class_matches += 1

    return {
        "count_a": count_a,
        "count_b": count_b,
        "matched_boxes": matched,
        "mean_box_mae": float(np.mean(box_diffs)) if box_diffs else 0.0,
        "mean_score_mae": float(np.mean(score_diffs)) if score_diffs else 0.0,
        "class_match_rate": float(class_matches / max(matched, 1)),
    }
