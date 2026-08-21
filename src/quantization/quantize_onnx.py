"""
Static Post-Training Quantization (PTQ) engine using ONNX Runtime.
"""

import sys
from pathlib import Path
from typing import List, Optional, Union
import onnx
import onnxruntime.quantization as ort_quant

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.hashes import compute_file_sha256
from src.common.logging import setup_logger
from src.quantization.calibration_reader import BenchmarkCalibrationDataReader

logger = setup_logger("quantize_onnx")


def quantize_onnx_static(
    input_onnx_path: Union[str, Path],
    output_onnx_path: Union[str, Path],
    calibration_data_reader: BenchmarkCalibrationDataReader,
    quant_format: str = "QDQ",
    calibrate_method: str = "MinMax",
    per_channel: bool = True,
    weight_type: str = "QInt8",
    activation_type: str = "QUInt8",
    op_types_to_quantize: Optional[List[str]] = None,
    nodes_to_exclude: Optional[List[str]] = None,
) -> Path:
    """
    Performs static 8-bit quantization with activation calibration over an ONNX graph.

    Args:
        input_onnx_path: Path to the validated FP32 ONNX graph.
        output_onnx_path: Destination path for the static INT8 ONNX graph.
        calibration_data_reader: Configured calibration data reader.
        quant_format: 'QDQ' or 'QOperator'.
        calibrate_method: 'MinMax', 'Entropy', or 'Percentile'.
        per_channel: Symmetrically quantize weights per-channel.
        weight_type: 'QInt8' or 'QUInt8'.
        activation_type: 'QInt8' or 'QUInt8'.
        op_types_to_quantize: Operators to quantize (default: ['Conv', 'MatMul', 'Gemm']).
        nodes_to_exclude: List of specific node names to leave in FP32.

    Returns:
        Path to the quantized ONNX model.
    """
    in_p = Path(input_onnx_path)
    out_p = Path(output_onnx_path)
    if not in_p.is_file():
        raise FileNotFoundError(f"Input ONNX file not found: {input_onnx_path}")

    out_p.parent.mkdir(parents=True, exist_ok=True)

    # 1. Inspect ONNX computation graph to exclude sensitive non-compute routing operations
    exclude_ops = {"Concat", "Split", "Reshape", "Transpose", "Sigmoid", "Slice", "Resize", "Softmax"}
    auto_excluded = []
    try:
        model_proto = onnx.load(str(in_p))
        for node in model_proto.graph.node:
            if node.op_type in exclude_ops and node.name:
                auto_excluded.append(node.name)
    except Exception as e:
        logger.warning(f"Could not inspect ONNX nodes for automatic exclusion: {e}")

    final_nodes_to_exclude = list(set((nodes_to_exclude or []) + auto_excluded))

    logger.info(
        f"Starting static INT8 PTQ -> {in_p.name} -> {out_p.name} "
        f"[Format: {quant_format}, Calibrate: {calibrate_method}, Weights: {weight_type}, Act: {activation_type}, "
        f"Excluded nodes: {len(final_nodes_to_exclude)}]..."
    )

    # Map configuration strings to ORT Quantization Enums
    format_map = {
        "QDQ": ort_quant.QuantFormat.QDQ,
        "QOperator": ort_quant.QuantFormat.QOperator,
    }
    calib_map = {
        "MinMax": ort_quant.CalibrationMethod.MinMax,
        "Entropy": ort_quant.CalibrationMethod.Entropy,
        "Percentile": ort_quant.CalibrationMethod.Percentile,
    }
    type_map = {
        "QInt8": ort_quant.QuantType.QInt8,
        "QUInt8": ort_quant.QuantType.QUInt8,
    }

    q_format = format_map.get(quant_format, ort_quant.QuantFormat.QDQ)
    c_method = calib_map.get(calibrate_method, ort_quant.CalibrationMethod.MinMax)
    w_type = type_map.get(weight_type, ort_quant.QuantType.QInt8)
    a_type = type_map.get(activation_type, ort_quant.QuantType.QInt8)
    target_ops = op_types_to_quantize or ["Conv", "MatMul", "Gemm"]

    calibration_data_reader.rewind()

    extra_opts = {
        "EnableSubgraph": True,
        "ForceQuantizeNoInputCheck": True,
        "MatMulConstBOnly": True,
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
    }

    ort_quant.quantize_static(
        model_input=str(in_p),
        model_output=str(out_p),
        calibration_data_reader=calibration_data_reader,
        quant_format=q_format,
        calibrate_method=c_method,
        per_channel=per_channel,
        weight_type=w_type,
        activation_type=a_type,
        op_types_to_quantize=target_ops,
        nodes_to_exclude=final_nodes_to_exclude if final_nodes_to_exclude else None,
        extra_options=extra_opts,
    )

    # Compute and persist SHA-256 digest
    sha256_hash = compute_file_sha256(out_p)
    sha_file = out_p.with_name(out_p.name + ".sha256")
    sha_file.write_text(f"{sha256_hash}  {out_p.name}\n", encoding="utf-8")

    logger.info(f"Static INT8 PTQ complete -> {out_p} (SHA-256: {sha256_hash[:16]}...)")
    return out_p
