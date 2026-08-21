"""
Canonical ONNX export routines targeting Opset 17 with metadata injection and checksum hashing.
"""

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import onnx
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.hashes import compute_file_sha256
from src.common.logging import setup_logger

logger = setup_logger("export_onnx")


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return "UNKNOWN"


def export_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    output_path: Union[str, Path],
    input_names: List[str],
    output_names: List[str],
    opset_version: int = 17,
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Exports a PyTorch Module to a standard ONNX graph with metadata properties and SHA-256 verification.

    Args:
        model: PyTorch model in evaluation mode.
        dummy_input: Representative dummy input tensor with static dimensions.
        output_path: Destination path for the .onnx file.
        input_names: Names of input tensor nodes in the graph.
        output_names: Names of output tensor nodes in the graph.
        opset_version: Target ONNX opset version (default: 17).
        dynamic_axes: Optional dynamic dimension mapping.
        metadata: Key-value metadata dictionary to inject into ONNX metadata_props.

    Returns:
        Path to the saved ONNX file.
    """
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    logger.info(f"Exporting model -> {target_path} (Opset: {opset_version})...")

    with torch.inference_mode():
        torch.onnx.export(
            model,
            dummy_input,
            str(target_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )

    # Ingest exported model to inject custom metadata properties
    onnx_model = onnx.load(str(target_path))

    meta_props = metadata or {}
    meta_props.setdefault("opset", str(opset_version))
    meta_props.setdefault("git_commit", _get_git_commit())
    meta_props.setdefault("framework", f"PyTorch {torch.__version__}")
    meta_props.setdefault("export_timestamp", datetime.now(timezone.utc).isoformat())

    for k, v in meta_props.items():
        meta = onnx_model.metadata_props.add()
        meta.key = str(k)
        meta.value = str(v)

    onnx.save(onnx_model, str(target_path))

    # Compute and persist SHA-256 checksum
    sha256_hash = compute_file_sha256(target_path)
    sha_file = target_path.with_name(target_path.name + ".sha256")
    sha_file.write_text(f"{sha256_hash}  {target_path.name}\n", encoding="utf-8")

    # Persist structured companion JSON metadata
    meta_json_path = target_path.with_name(target_path.stem + ".metadata.json")
    meta_payload = {
        "file_name": target_path.name,
        "sha256": sha256_hash,
        "input_names": input_names,
        "output_names": output_names,
        "opset_version": opset_version,
        "metadata_props": meta_props,
    }
    meta_json_path.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

    logger.info(f"Export complete -> {target_path} (SHA-256: {sha256_hash[:16]}...)")
    return target_path


def export_model_family(
    model_type: str,
    weights_path: Optional[str] = None,
    output_dir: Union[str, Path] = "models/exported",
    model_instance: Optional[nn.Module] = None,
) -> Path:
    """
    Instantiates and exports standard models (yolo_nano or industrial_autoencoder) to ONNX.

    Args:
        model_type: One of 'yolo_nano' or 'industrial_autoencoder'.
        weights_path: Optional path to checkpoint weights.
        output_dir: Directory where exported models are stored.
        model_instance: Optional pre-instantiated PyTorch model instance.

    Returns:
        Path to the exported ONNX model.
    """
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    if model_type == "yolo_nano":
        from src.models.yolo_adapter import YOLOAdapter

        if model_instance is None:
            adapter = YOLOAdapter(weights_path=weights_path)
            pytorch_model = adapter.get_pytorch_model()
        else:
            pytorch_model = model_instance

        dummy_input = torch.randn(1, 3, 640, 640, dtype=torch.float32)

        out_file = out_dir_path / "yolo_nano_fp32_opset17.onnx"
        meta = {
            "model_name": "yolo_nano",
            "task": "detection",
            "input_shape": json.dumps([1, 3, 640, 640]),
            "output_shape": json.dumps([1, 84, 8400]),
            "conf_threshold": "0.25",
            "iou_threshold": "0.45",
        }
        return export_to_onnx(
            model=pytorch_model,
            dummy_input=dummy_input,
            output_path=out_file,
            input_names=["images"],
            output_names=["output0"],
            opset_version=17,
            metadata=meta,
        )

    elif model_type == "industrial_autoencoder":
        from src.models.industrial_model_adapter import IndustrialModelAdapter

        class AutoencoderExportWrapper(nn.Module):
            def __init__(self, core_model: nn.Module):
                super().__init__()
                self.core = core_model

            def forward(self, x: torch.Tensor):
                recon = self.core(x)
                a_map = torch.mean(torch.abs(x - recon), dim=1, keepdim=True)
                return recon, a_map

        if model_instance is None:
            adapter = IndustrialModelAdapter(weights_path=weights_path)
            export_model = AutoencoderExportWrapper(adapter.get_pytorch_model())
        else:
            export_model = model_instance

        dummy_input = torch.randn(1, 3, 256, 256, dtype=torch.float32)

        out_file = out_dir_path / "industrial_autoencoder_fp32_opset17.onnx"
        meta = {
            "model_name": "industrial_autoencoder",
            "task": "reconstruction",
            "input_shape": json.dumps([1, 3, 256, 256]),
            "output_shapes": json.dumps({"reconstruction": [1, 3, 256, 256], "anomaly_map": [1, 1, 256, 256]}),
        }
        return export_to_onnx(
            model=export_model,
            dummy_input=dummy_input,
            output_path=out_file,
            input_names=["input"],
            output_names=["reconstruction", "anomaly_map"],
            opset_version=17,
            metadata=meta,
        )
    else:
        raise ValueError(f"Unknown model_type for export: {model_type}")
