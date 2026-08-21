"""
ONNX graph simplification utility with constant folding and equivalence auditing.
"""

import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import onnx
import onnxruntime as ort
import onnxsim

from src.common.hashes import compute_file_sha256
from src.common.logging import setup_logger

logger = setup_logger("simplify_onnx")


def simplify_onnx_graph(
    input_onnx_path: Union[str, Path],
    output_onnx_path: Optional[Union[str, Path]] = None,
    check_n: int = 3,
) -> Tuple[Path, Dict[str, Any]]:
    """
    Simplifies an ONNX graph using onnxsim, verifies numerical equivalence, and writes the output.

    Args:
        input_onnx_path: Path to the original ONNX model.
        output_onnx_path: Destination path for the simplified model (defaults to overwriting input).
        check_n: Number of numerical checks across random inputs.

    Returns:
        Tuple of (output_path, simplification_report_dict).
    """
    in_path = Path(input_onnx_path)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input ONNX file not found: {input_onnx_path}")

    out_path = Path(output_onnx_path) if output_onnx_path else in_path

    model = onnx.load(str(in_path))
    nodes_before = len(model.graph.node)

    logger.info(f"Simplifying ONNX graph -> {in_path.name} ({nodes_before} nodes)...")
    simplified_model, check_ok = onnxsim.simplify(model, check_n=check_n)

    if not check_ok:
        logger.warning(f"onnxsim numerical check returned False for {in_path.name}")

    nodes_after = len(simplified_model.graph.node)

    # Validate simplified model structure
    onnx.checker.check_model(simplified_model, full_check=True)

    # Save simplified model to target
    out_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(simplified_model, str(out_path))

    # Update SHA-256 checksum
    sha256_hash = compute_file_sha256(out_path)
    sha_file = out_path.with_name(out_path.name + ".sha256")
    sha_file.write_text(f"{sha256_hash}  {out_path.name}\n", encoding="utf-8")

    report = {
        "model_name": in_path.stem,
        "nodes_before": nodes_before,
        "nodes_after": nodes_after,
        "nodes_eliminated": nodes_before - nodes_after,
        "reduction_percentage": round(100.0 * (nodes_before - nodes_after) / max(nodes_before, 1), 2),
        "check_ok": bool(check_ok),
        "sha256": sha256_hash,
    }

    logger.info(
        f"Simplification complete: {nodes_before} -> {nodes_after} nodes "
        f"({report['reduction_percentage']}% reduction)"
    )
    return out_path, report
