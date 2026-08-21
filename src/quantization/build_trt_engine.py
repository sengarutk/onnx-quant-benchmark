"""
TensorRT Engine Compilation & Plan Serialization utility.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.hashes import compute_file_sha256
from src.common.logging import setup_logger

logger = setup_logger("build_trt_engine")

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    trt = None


def build_tensorrt_engine(
    onnx_path: Union[str, Path],
    engine_output_path: Union[str, Path],
    precision: str = "fp16",
    workspace_gb: float = 4.0,
    static_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> Path:
    """
    Compiles an ONNX model into a serialized TensorRT execution engine plan.

    Args:
        onnx_path: Path to input ONNX file.
        engine_output_path: Destination path for .engine plan file.
        precision: 'fp32', 'fp16', or 'int8'.
        workspace_gb: Workspace memory pool limit in GB.
        static_shapes: Static min/opt/max shapes dictionary.

    Returns:
        Path to compiled .engine artifact.
    """
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not available in current environment.")

    in_p = Path(onnx_path)
    out_p = Path(engine_output_path)
    if not in_p.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Building TensorRT engine: {in_p.name} -> {out_p.name} (Precision: {precision})...")

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    onnx_bytes = in_p.read_bytes()
    if not parser.parse(onnx_bytes):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"Failed to parse ONNX graph: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 1024**3))

    if precision.lower() == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision.lower() == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)

    # Static optimization profile
    if static_shapes:
        profile = builder.create_optimization_profile()
        for name, shape in static_shapes.items():
            profile.set_shape(name, min=shape, opt=shape, max=shape)
        config.add_optimization_profile(profile)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT engine compilation failed for {onnx_path}")

    out_p.write_bytes(plan)

    # Compute and persist SHA-256
    sha256_hash = compute_file_sha256(out_p)
    sha_file = out_p.with_name(out_p.name + ".sha256")
    sha_file.write_text(f"{sha256_hash}  {out_p.name}\n", encoding="utf-8")

    # Persist build manifest
    manifest_dir = out_p.parent / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_dir / f"{out_p.stem}.json"

    manifest_data = {
        "engine_name": out_p.name,
        "onnx_source": in_p.name,
        "precision": precision,
        "workspace_gb": workspace_gb,
        "sha256": sha256_hash,
        "tensorrt_version": getattr(trt, "__version__", "Unknown"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown",
    }
    manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    logger.info(f"TensorRT Engine build complete -> {out_p} (SHA-256: {sha256_hash[:16]}...)")
    return out_p
