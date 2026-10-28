"""
Quality gate for converted / quantized ONNX artifacts against the PyTorch FP32 baseline.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

from src.common.config import QualityThresholdConfig, load_quality_thresholds
from src.common.logging import setup_logger
from src.validation.numerical_equivalence import _rel, evaluate_onnx_dataset

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
    Evaluates the artifact on the test split with ONNX Runtime (CPU) and applies the precision tier's gate
    (configs/<model>/<precision>.yaml unless an explicit config is passed).

    Returns:
        Report with metrics, baseline metrics, delta of the gated metric and PASS / REJECTED status.
    """
    onnx_p = Path(onnx_path)
    if not onnx_p.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
    b_path = Path(baseline_metrics_path)
    if not b_path.is_file():
        raise FileNotFoundError(f"Baseline metrics not found: {baseline_metrics_path}")

    thresholds = quality_threshold_config or load_quality_thresholds(model_name, precision)
    baseline = json.loads(b_path.read_text(encoding="utf-8"))["metrics"]
    metrics = evaluate_onnx_dataset(model_name, onnx_p, manifest_path)["metrics"]

    if model_name == "yolo_nano":
        key, max_drop = "mAP_50", thresholds.max_map_drop
    elif model_name == "industrial_autoencoder":
        key, max_drop = "image_auroc", thresholds.max_auroc_drop
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    delta = float(metrics[key] - baseline[key])
    passed = delta >= -max_drop
    return {
        "model_name": model_name,
        "precision": precision,
        "onnx_path": _rel(onnx_p),
        "status": "PASS" if passed else "REJECTED",
        "rejection_reason": None if passed else f"{key} drop {-delta:.4f} exceeds {max_drop}",
        "gated_metric": key,
        "metrics": metrics,
        "baseline_metrics": baseline,
        "delta": delta,
        "max_allowed_drop": max_drop,
    }
