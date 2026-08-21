"""
FP16 half-precision ONNX model conversion utility using onnxconverter_common.
"""

import sys
from pathlib import Path
from typing import Optional, Union
import onnx
from onnxconverter_common import float16

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.hashes import compute_file_sha256
from src.common.logging import setup_logger

logger = setup_logger("convert_fp16")


def convert_onnx_to_fp16(
    input_onnx_path: Union[str, Path],
    output_onnx_path: Optional[Union[str, Path]] = None,
    keep_io_types: bool = True,
) -> Path:
    """
    Converts a standard FP32 ONNX model into numerical-stable FP16 representation.

    Args:
        input_onnx_path: Path to the source FP32 ONNX model.
        output_onnx_path: Target path for the converted FP16 model (defaults to *_fp16.onnx).
        keep_io_types: Keep graph input and output tensors in float32 for seamless I/O compatibility.

    Returns:
        Path to the saved FP16 model.
    """
    in_p = Path(input_onnx_path)
    if not in_p.is_file():
        raise FileNotFoundError(f"Input ONNX model not found: {input_onnx_path}")

    if output_onnx_path:
        out_p = Path(output_onnx_path)
    else:
        out_p = in_p.parent / f"{in_p.stem.replace('_fp32_opset17', '')}_fp16.onnx"

    out_p.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Converting FP32 model -> FP16: {in_p.name} -> {out_p.name}...")

    model_fp32 = onnx.load(str(in_p))
    try:
        model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=keep_io_types)
        onnx.checker.check_model(model_fp16, full_check=True)
    except Exception as exc:
        logger.info(f"Retrying FP16 conversion with keep_io_types=False (reason: {exc})...")
        model_fp16 = float16.convert_float_to_float16(model_fp32, keep_io_types=False)
        onnx.checker.check_model(model_fp16, full_check=True)

    # Save converted model
    onnx.save(model_fp16, str(out_p))

    # Compute and persist SHA-256 digest
    sha256_hash = compute_file_sha256(out_p)
    sha_file = out_p.with_name(out_p.name + ".sha256")
    sha_file.write_text(f"{sha256_hash}  {out_p.name}\n", encoding="utf-8")

    logger.info(f"FP16 conversion complete -> {out_p} (SHA-256: {sha256_hash[:16]}...)")
    return out_p
