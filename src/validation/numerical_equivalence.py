"""
Numerical equivalence and parity validation engine comparing PyTorch and ONNX Runtime.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.preprocess import preprocess_detection_image, preprocess_industrial_image
from src.models.yolo_adapter import YOLOAdapter
from src.validation.anomaly_quality import compute_aupro, compute_image_auroc, compute_pixel_auroc
from src.validation.detection_quality import evaluate_detection_dataset
from src.validation.output_checks import compute_tensor_diff


def compare_pytorch_vs_onnx(
    torch_model: nn.Module,
    onnx_path: Union[str, Path],
    input_tensor: torch.Tensor,
    providers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compares the raw output tensors of a PyTorch Module and ONNX Runtime InferenceSession.

    Args:
        torch_model: PyTorch model in eval mode.
        onnx_path: Path to the target ONNX file.
        input_tensor: PyTorch tensor matching model input dimensions.
        providers: ONNX Runtime execution provider list.

    Returns:
        Dictionary of parity comparison metrics per output tensor.
    """
    torch_model.eval()
    with torch.inference_mode():
        torch_out = torch_model(input_tensor)

    # Initialize ONNX Runtime session
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(onnx_path),
        sess_options=sess_opts,
        providers=providers or ["CPUExecutionProvider"],
    )

    input_name = session.get_inputs()[0].name
    np_input = input_tensor.detach().cpu().numpy()
    ort_outputs = session.run(None, {input_name: np_input})

    results = {}
    if isinstance(torch_out, torch.Tensor):
        torch_arrs = [torch_out.detach().cpu().numpy()]
    elif isinstance(torch_out, (tuple, list)):
        torch_arrs = [t.detach().cpu().numpy() for t in torch_out]
    else:
        raise TypeError(f"Unsupported torch model output type: {type(torch_out)}")

    output_names = [out.name for out in session.get_outputs()]

    all_passed = True
    for out_name, t_arr, o_arr in zip(output_names, torch_arrs, ort_outputs):
        diff_stats = compute_tensor_diff(t_arr, o_arr)
        passed = diff_stats["max_abs_error"] < 1e-4 and diff_stats["cosine_similarity"] > 0.99999
        if not passed:
            all_passed = False

        results[out_name] = {
            **diff_stats,
            "passed": bool(passed),
        }

    return {
        "model_name": Path(onnx_path).stem,
        "providers": session.get_providers(),
        "all_passed": bool(all_passed),
        "outputs": results,
    }


def evaluate_onnx_dataset(
    model_type: str,
    onnx_path: Union[str, Path],
    sample_manifest_path: Union[str, Path] = "data/sample_images/manifest.json",
    provider: str = "CPUExecutionProvider",
) -> Dict[str, Any]:
    """
    Runs full dataset evaluation using ONNX Runtime and compares metrics against PyTorch FP32 baseline.

    Args:
        model_type: One of 'yolo_nano' or 'industrial_autoencoder'.
        onnx_path: Path to the exported ONNX model.
        sample_manifest_path: Path to the sample dataset manifest.
        provider: ORT execution provider.

    Returns:
        Evaluation report dictionary with baseline parity comparison.
    """
    sess_opts = ort.SessionOptions()
    session = ort.InferenceSession(str(onnx_path), sess_options=sess_opts, providers=[provider])

    manifest_file = Path(sample_manifest_path)
    if not manifest_file.is_file():
        raise FileNotFoundError(f"Sample manifest not found: {sample_manifest_path}")

    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))

    if model_type == "yolo_nano":
        adapter = YOLOAdapter()
        det_samples = manifest["detection_samples"]
        all_preds = []
        all_gts = []

        input_name = session.get_inputs()[0].name

        for item in det_samples:
            img_path = PROJECT_ROOT / item["image_path"]
            tensor, ratio, pad, orig_shape = preprocess_detection_image(img_path)

            ort_out = session.run(None, {input_name: tensor.numpy()})[0]
            detections = adapter.postprocess(torch.from_numpy(ort_out), orig_shape, ratio, pad)

            all_preds.append(detections)
            all_gts.append(item["ground_truth_boxes"])

        metrics = evaluate_detection_dataset(all_preds, all_gts)

        # Load PyTorch baseline for comparison
        baseline_file = PROJECT_ROOT / "results" / "raw" / "pytorch_fp32" / "yolo_nano" / "baseline_metrics.json"
        baseline_metrics = json.loads(baseline_file.read_text(encoding="utf-8"))["metrics"] if baseline_file.is_file() else {}

        delta_map_50 = abs(metrics["mAP_50"] - baseline_metrics.get("mAP_50", metrics["mAP_50"]))
        parity_passed = delta_map_50 <= 0.0001

        return {
            "model_type": model_type,
            "onnx_path": str(onnx_path),
            "provider": provider,
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "delta_mAP_50": float(round(delta_map_50, 6)),
            "parity_passed": bool(parity_passed),
        }

    elif model_type == "industrial_autoencoder":
        from src.models.industrial_model_adapter import IndustrialModelAdapter

        adapter = IndustrialModelAdapter()
        ind_samples = manifest["industrial_samples"]

        input_name = session.get_inputs()[0].name
        y_true = []
        y_scores = []
        gt_masks = []
        anomaly_maps = []

        for item in ind_samples:
            img_path = PROJECT_ROOT / item["image_path"]
            tensor = preprocess_industrial_image(img_path)

            ort_outs = session.run(None, {input_name: tensor.numpy()})
            recon, a_map = ort_outs[0], ort_outs[1]

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

        image_auroc = compute_image_auroc(y_true, y_scores)
        pixel_auroc = compute_pixel_auroc(gt_masks, anomaly_maps)
        aupro = compute_aupro(gt_masks, anomaly_maps)

        metrics = {
            "image_auroc": image_auroc,
            "pixel_auroc": pixel_auroc,
            "aupro": aupro,
        }

        # Load PyTorch baseline for comparison
        baseline_file = PROJECT_ROOT / "results" / "raw" / "pytorch_fp32" / "industrial_autoencoder" / "baseline_metrics.json"
        baseline_metrics = json.loads(baseline_file.read_text(encoding="utf-8"))["metrics"] if baseline_file.is_file() else {}

        delta_auroc = abs(image_auroc - baseline_metrics.get("image_auroc", image_auroc))
        parity_passed = delta_auroc <= 0.0001

        return {
            "model_type": model_type,
            "onnx_path": str(onnx_path),
            "provider": provider,
            "metrics": metrics,
            "baseline_metrics": baseline_metrics,
            "delta_auroc": float(round(delta_auroc, 6)),
            "parity_passed": bool(parity_passed),
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
