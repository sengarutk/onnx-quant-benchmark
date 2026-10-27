"""
Numerical-equivalence and dataset-parity validation between PyTorch and ONNX Runtime.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from src.data.writers import load_manifest
from src.validation.output_checks import compute_tensor_diff
from src.validation.task_eval import evaluate_autoencoder, evaluate_detector

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# FP32 export parity gate (relative to the reference output's dynamic range)
FP32_MAX_REL_ERROR = 1e-5
FP32_MIN_COSINE = 0.999999


def compare_pytorch_vs_onnx(
    torch_model: nn.Module,
    onnx_path: Union[str, Path],
    input_tensor: torch.Tensor,
    providers: Optional[List[str]] = None,
    max_rel_error: float = FP32_MAX_REL_ERROR,
    min_cosine: float = FP32_MIN_COSINE,
) -> Dict[str, Any]:
    """Compares every output tensor of a PyTorch module and an ONNX Runtime session on the same input."""
    torch_model.eval()
    with torch.inference_mode():
        torch_out = torch_model(input_tensor)
    torch_arrs = [t.detach().cpu().numpy() for t in (torch_out if isinstance(torch_out, (tuple, list)) else [torch_out])]

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers or ["CPUExecutionProvider"])
    ort_outputs = session.run(None, {session.get_inputs()[0].name: input_tensor.detach().cpu().numpy()})

    results = {}
    for out, t_arr, o_arr in zip(session.get_outputs(), torch_arrs, ort_outputs):
        d = compute_tensor_diff(t_arr, o_arr)
        results[out.name] = {**d, "passed": bool(d["max_rel_error"] <= max_rel_error and d["cosine_similarity"] >= min_cosine)}

    return {
        "model_name": Path(onnx_path).stem,
        "providers": session.get_providers(),
        "gate": {"max_rel_error": max_rel_error, "min_cosine_similarity": min_cosine},
        "all_passed": all(r["passed"] for r in results.values()),
        "outputs": results,
    }


def evaluate_onnx_dataset(
    model_type: str,
    onnx_path: Union[str, Path],
    sample_manifest_path: Union[str, Path] = PROJECT_ROOT / "data" / "sample_images" / "manifest.json",
    provider: str = "CPUExecutionProvider",
) -> Dict[str, Any]:
    """Evaluates the task metrics of an ONNX model on the test split with ONNX Runtime."""
    session = ort.InferenceSession(str(onnx_path), providers=[provider])
    in_name = session.get_inputs()[0].name
    is_fp16 = "float16" in session.get_inputs()[0].type
    dtype = np.float16 if is_fp16 else np.float32
    manifest = load_manifest(sample_manifest_path)

    def predict(x: np.ndarray):
        return session.run(None, {in_name: x})

    if model_type == "yolo_nano":
        from src.models.yolo_adapter import YOLOAdapter
        metrics = evaluate_detector(predict, manifest["detection_samples"], YOLOAdapter(), dtype=dtype)
    elif model_type == "industrial_autoencoder":
        metrics = evaluate_autoencoder(predict, manifest["industrial_samples"], dtype=dtype)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return {"model_type": model_type, "onnx_path": str(onnx_path), "provider": provider, "metrics": metrics}
