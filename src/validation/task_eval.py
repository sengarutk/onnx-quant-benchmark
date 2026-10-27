"""
Runtime-agnostic task-quality evaluation shared by training, baselines, quantization gating and benchmarking.

Every evaluator takes a ``predict_fn`` mapping a float32 NCHW NumPy array to the model's output array(s), so the
same code path scores PyTorch, ONNX Runtime FP32/FP16/INT8 and TensorRT engines.
"""

from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Union
import cv2
import numpy as np

from src.models.preprocess import preprocess_detection_image, preprocess_industrial_image
from src.validation.anomaly_quality import compute_aupro, compute_image_auroc, compute_pixel_auroc
from src.validation.detection_quality import evaluate_detection_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# mAP is computed over (almost) the full score range, as in the COCO protocol; precision/recall/F1 are
# reported at the deployment operating point.
MAP_CONF_THRESHOLD = 0.001
DEPLOY_CONF_THRESHOLD = 0.25
DEPLOY_IOU_THRESHOLD = 0.45

PredictFn = Callable[[np.ndarray], Union[np.ndarray, Sequence[np.ndarray]]]


def _first(out: Union[np.ndarray, Sequence[np.ndarray], Dict[str, np.ndarray]]) -> np.ndarray:
    if isinstance(out, dict):
        return np.asarray(list(out.values())[0])
    if isinstance(out, (list, tuple)):
        return np.asarray(out[0])
    return np.asarray(out)


def run_detector(predict_fn: PredictFn, samples: List[Dict[str, Any]], dtype: Any = np.float32) -> List[Dict[str, Any]]:
    """Runs inference once per sample and caches raw outputs with letterbox metadata."""
    cache = []
    for s in samples:
        tensor, ratio, pad, orig_shape = preprocess_detection_image(PROJECT_ROOT / s["image_path"])
        raw = _first(predict_fn(tensor.numpy().astype(dtype))).astype(np.float32)
        cache.append({"raw": raw, "ratio": ratio, "pad": pad, "orig_shape": orig_shape, "gt": s["ground_truth_boxes"]})
    return cache


def score_detections(
    cache: List[Dict[str, Any]], adapter: Any, conf: float = DEPLOY_CONF_THRESHOLD, iou: float = DEPLOY_IOU_THRESHOLD
) -> List[List[Dict[str, Any]]]:
    return [adapter.postprocess(c["raw"], c["orig_shape"], c["ratio"], c["pad"], conf, iou) for c in cache]


def evaluate_detector(predict_fn: PredictFn, samples: List[Dict[str, Any]], adapter: Any, dtype: Any = np.float32) -> Dict[str, float]:
    """mAP@50 / mAP@50-95 (conf >= 0.001) and P / R / F1 at the deployment threshold (conf 0.25, IoU 0.45)."""
    from src.quantization.q_aware_nms import detection_counts

    cache = run_detector(predict_fn, samples, dtype)
    gts = [c["gt"] for c in cache]
    ap = evaluate_detection_dataset(score_detections(cache, adapter, MAP_CONF_THRESHOLD), gts)
    tp, fp, fn = detection_counts(score_detections(cache, adapter), gts)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {
        "mAP_50": ap["mAP_50"],
        "mAP_50_95": ap["mAP_50_95"],
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
    }


def evaluate_autoencoder(predict_fn: PredictFn, samples: List[Dict[str, Any]], dtype: Any = np.float32) -> Dict[str, float]:
    """
    Image AUROC (smoothed top-k% absolute-error score), pixel AUROC and AU-PRO (FPR <= 0.3) on the smoothed maps.
    Scoring hyper-parameters come from load_scoring_params() (selected on the validation split).
    """
    from src.models.industrial_model_adapter import load_scoring_params, top_k_anomaly_score

    sigma = load_scoring_params()["blur_sigma"]

    y_true, y_score, masks, maps = [], [], [], []
    for s in samples:
        x = preprocess_industrial_image(PROJECT_ROOT / s["image_path"]).numpy()
        out = predict_fn(x.astype(dtype))
        if isinstance(out, dict):
            out = list(out.values())
        if isinstance(out, (list, tuple)) and len(out) > 1:
            a_map = np.asarray(out[1], dtype=np.float32)
        else:
            recon = _first(out).astype(np.float32)
            a_map = np.mean(np.abs(x - recon), axis=1, keepdims=True)
        a_map = a_map.reshape(a_map.shape[-2:])
        y_true.append(int(s["is_anomalous"]))
        y_score.append(top_k_anomaly_score(a_map))
        if sigma > 0:
            a_map = cv2.GaussianBlur(a_map, (0, 0), sigma)
        if s["mask_path"]:
            m = cv2.imread(str(PROJECT_ROOT / s["mask_path"]), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        else:
            m = np.zeros(a_map.shape, dtype=np.float32)
        masks.append(m)
        maps.append(a_map)
    return {
        "image_auroc": compute_image_auroc(y_true, y_score),
        "pixel_auroc": compute_pixel_auroc(masks, maps),
        "aupro": compute_aupro(masks, maps),
    }
