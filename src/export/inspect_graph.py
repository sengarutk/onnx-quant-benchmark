"""
ONNX graph structural inspection, layer distribution analysis, and memory footprint estimation.
"""

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Union
import onnx
import numpy as np


def inspect_onnx_graph(onnx_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Inspects and summarizes nodes, operators, initializers, and memory parameters.

    Args:
        onnx_path: Path to the target ONNX file.

    Returns:
        Structured inspection report dictionary.
    """
    p = Path(onnx_path)
    if not p.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    model = onnx.load(str(p))
    graph = model.graph

    # Operator frequency
    op_counts = Counter(node.op_type for node in graph.node)

    # Initializer parameter count and byte size
    total_params = 0
    total_bytes = 0
    for init in graph.initializer:
        arr = onnx.numpy_helper.to_array(init)
        total_params += arr.size
        total_bytes += arr.nbytes

    inputs_info = [
        {"name": inp.name, "dims": [d.dim_value for d in inp.type.tensor_type.shape.dim]}
        for inp in graph.input
    ]
    outputs_info = [
        {"name": out.name, "dims": [d.dim_value for d in out.type.tensor_type.shape.dim]}
        for out in graph.output
    ]

    return {
        "file_name": p.name,
        "file_size_mb": round(p.stat().st_size / (1024.0 * 1024.0), 3),
        "total_nodes": len(graph.node),
        "operator_counts": dict(op_counts.most_common()),
        "total_parameters": int(total_params),
        "parameter_size_mb": round(total_bytes / (1024.0 * 1024.0), 3),
        "inputs": inputs_info,
        "outputs": outputs_info,
    }
