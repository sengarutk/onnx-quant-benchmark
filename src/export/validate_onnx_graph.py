"""
ONNX graph integrity validation, checker verification, and shape inference propagation.
"""

from pathlib import Path
from typing import Any, Dict, Union
import onnx


def validate_onnx_graph(onnx_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Validates structural integrity, IR format, tensor data types, and node topology.

    Args:
        onnx_path: Path to the target ONNX file.

    Returns:
        Validation report dictionary.
    """
    p = Path(onnx_path)
    if not p.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    model = onnx.load(str(p))

    # 1. Full ONNX Checker Validation
    onnx.checker.check_model(model, full_check=True)

    # 2. Shape Inference Propagation
    inferred_model = onnx.shape_inference.infer_shapes(model)

    # Extract input signatures
    inputs = {}
    for inp in inferred_model.graph.input:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in inp.type.tensor_type.shape.dim]
        inputs[inp.name] = {
            "shape": shape,
            "elem_type": onnx.TensorProto.DataType.Name(inp.type.tensor_type.elem_type),
        }

    # Extract output signatures
    outputs = {}
    for out in inferred_model.graph.output:
        shape = [d.dim_value if d.dim_value > 0 else d.dim_param for d in out.type.tensor_type.shape.dim]
        outputs[out.name] = {
            "shape": shape,
            "elem_type": onnx.TensorProto.DataType.Name(out.type.tensor_type.elem_type),
        }

    # Extract metadata properties
    metadata_dict = {prop.key: prop.value for prop in model.metadata_props}

    # Extract opset version
    opset_ver = 0
    for op in model.opset_import:
        if op.domain == "" or op.domain == "ai.onnx":
            opset_ver = op.version
            break

    return {
        "valid": True,
        "ir_version": model.ir_version,
        "opset_version": opset_ver,
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "inputs": inputs,
        "outputs": outputs,
        "metadata": metadata_dict,
        "node_count": len(model.graph.node),
    }
