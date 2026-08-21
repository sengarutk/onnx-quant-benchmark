"""
Quantization quality degradation validator and acceptance decision gating engine.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
import cv2
import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import QualityThresholdConfig
from src.common.logging import setup_logger
from src.models.preprocess import preprocess_detection_image, preprocess_industrial_image
from src.models.yolo_adapter import YOLOAdapter
from src.validation.anomaly_quality import compute_aupro, compute_image_auroc, compute_pixel_auroc
from src.validation.detection_quality import evaluate_detection_dataset

logger = setup_logger("validate_quantization")


def validate_quantized_model(
    model_name: str,
    precision: str,
    onnx_path: Union[str, Path],
    manifest_path: Union[str, Path],
    baseline_metrics_path: Union[str, Path],
    quality_threshold_config: Optional[QualityThresholdConfig] = None,
) -> Dict[str, Any]:
    """
    Validates task quality for a quantized ONNX model against PyTorch FP32 baseline and enforces acceptance gates.

    Args:
        model_name: 'yolo_nano' or 'industrial_autoencoder'.
        precision: 'fp16', 'int8', etc.
        onnx_path: Path to the quantized ONNX graph.
        manifest_path: Path to sample evaluation dataset manifest.json.
        baseline_metrics_path: Path to reference FP32 baseline_metrics.json.
        quality_threshold_config: Threshold constraints for quality degradation.

    Returns:
        Validation report dictionary with metrics, deltas, and acceptance status.
    """
    onnx_p = Path(onnx_path)
    if not onnx_p.is_file():
        raise FileNotFoundError(f"Quantized ONNX model not found: {onnx_path}")

    thresh_cfg = quality_threshold_config or QualityThresholdConfig()

    # Load baseline metrics
    b_path = Path(baseline_metrics_path)
    if not b_path.is_file():
        raise FileNotFoundError(f"Baseline metrics not found: {baseline_metrics_path}")
    baseline_data = json.loads(b_path.read_text(encoding="utf-8"))
    baseline_metrics = baseline_data["metrics"]

    # Load evaluation manifest
    m_path = Path(manifest_path)
    if not m_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")
    manifest = json.loads(m_path.read_text(encoding="utf-8"))

    # Initialize ONNX Runtime session
    sess_opts = ort.SessionOptions()
    session = ort.InferenceSession(str(onnx_p), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    is_fp16_input = "float16" in session.get_inputs()[0].type

    if model_name == "yolo_nano":
        adapter = YOLOAdapter()
        det_samples = manifest["detection_samples"]
        all_preds = []
        all_gts = []

        for item in det_samples:
            img_p = PROJECT_ROOT / item["image_path"]
            tensor, ratio, pad, orig_shape = preprocess_detection_image(img_p)
            np_in = tensor.numpy().astype(np.float16) if is_fp16_input else tensor.numpy()

            ort_out = session.run(None, {input_name: np_in})[0]
            if ort_out.dtype == np.float16:
                ort_out = ort_out.astype(np.float32)

            detections = adapter.postprocess(torch.from_numpy(ort_out), orig_shape, ratio, pad)

            all_preds.append(detections)
            all_gts.append(item["ground_truth_boxes"])

        eval_metrics = evaluate_detection_dataset(all_preds, all_gts)

        base_map50 = baseline_metrics["mAP_50"]
        base_map50_95 = baseline_metrics.get("mAP_50_95", base_map50)
        curr_map50 = eval_metrics["mAP_50"]
        curr_map50_95 = eval_metrics["mAP_50_95"]

        delta_map50 = float(round(base_map50 - curr_map50, 6))
        delta_map50_95 = float(round(base_map50_95 - curr_map50_95, 6))

        # Hard quality gate: max allowed drop in mAP
        passed = delta_map50_95 <= thresh_cfg.max_map_drop
        status = "PASS" if passed else "REJECTED"
        rejection_reason = None if passed else f"mAP drop ({delta_map50_95:.4f}) exceeds threshold ({thresh_cfg.max_map_drop})"

        return {
            "model_name": model_name,
            "precision": precision,
            "onnx_path": str(onnx_p),
            "status": status,
            "rejection_reason": rejection_reason,
            "metrics": eval_metrics,
            "baseline_metrics": baseline_metrics,
            "delta_map_50": delta_map50,
            "delta_map_50_95": delta_map50_95,
            "max_allowed_drop": thresh_cfg.max_map_drop,
        }

    elif model_name == "industrial_autoencoder":
        from src.models.industrial_model_adapter import IndustrialModelAdapter

        adapter = IndustrialModelAdapter()
        ind_samples = manifest["industrial_samples"]

        y_true = []
        y_scores = []
        gt_masks = []
        anomaly_maps = []

        for item in ind_samples:
            img_p = PROJECT_ROOT / item["image_path"]
            tensor = preprocess_industrial_image(img_p)
            np_in = tensor.numpy().astype(np.float16) if is_fp16_input else tensor.numpy()

            ort_outs = session.run(None, {input_name: np_in})
            recon = ort_outs[0].astype(np.float32) if ort_outs[0].dtype == np.float16 else ort_outs[0]
            a_map = ort_outs[1].astype(np.float32) if ort_outs[1].dtype == np.float16 else ort_outs[1]

            score = adapter.compute_anomaly_score(torch.from_numpy(a_map))
            is_anom = 1 if item["is_anomalous"] else 0

            y_true.append(is_anom)
            y_scores.append(score)

            if item["mask_path"]:
                mask_img = cv2.imread(str(PROJECT_ROOT / item["mask_path"]), cv2.IMREAD_GRAYSCALE)
                mask_arr = (mask_img / 255.0).astype(np.float32)
            else:
                mask_arr = np.zeros((256, 256), dtype=np.float32)

            gt_masks.append(mask_arr)
            anomaly_maps.append(a_map.squeeze())

        img_auroc = compute_image_auroc(y_true, y_scores)
        pix_auroc = compute_pixel_auroc(gt_masks, anomaly_maps)
        aupro = compute_aupro(gt_masks, anomaly_maps)

        eval_metrics = {
            "image_auroc": img_auroc,
            "pixel_auroc": pix_auroc,
            "aupro": aupro,
        }

        base_auroc = baseline_metrics["image_auroc"]
        base_aupro = baseline_metrics.get("aupro", 1.0)

        delta_auroc = float(round(base_auroc - img_auroc, 6))
        delta_aupro = float(round(base_aupro - aupro, 6))

        passed = delta_auroc <= thresh_cfg.max_auroc_drop
        status = "PASS" if passed else "REJECTED"
        rejection_reason = None if passed else f"AUROC drop ({delta_auroc:.4f}) exceeds threshold ({thresh_cfg.max_auroc_drop})"

        return {
            "model_name": model_name,
            "precision": precision,
            "onnx_path": str(onnx_p),
            "status": status,
            "rejection_reason": rejection_reason,
            "metrics": eval_metrics,
            "baseline_metrics": baseline_metrics,
            "delta_auroc": delta_auroc,
            "delta_aupro": delta_aupro,
            "max_allowed_drop": thresh_cfg.max_auroc_drop,
        }
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
