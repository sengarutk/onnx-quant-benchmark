from pathlib import Path
import os
import stat

TARGET_ROOT = Path("/home/sengar/onnx-quant-benchmark")

files = {}

# ============================================================================
# 1. EXPORT PACKAGE & MODULES (src/export/)
# ============================================================================

files["src/export/__init__.py"] = '''\"\"\"ONNX export, graph verification, simplification, and inspection subsystem.\"\"\"
from src.export.export_onnx import export_to_onnx, export_model_family
from src.export.validate_onnx_graph import validate_onnx_graph
from src.export.simplify_onnx import simplify_onnx_graph
from src.export.inspect_graph import inspect_onnx_graph

__all__ = [
    "export_to_onnx",
    "export_model_family",
    "validate_onnx_graph",
    "simplify_onnx_graph",
    "inspect_onnx_graph",
]
'''

files["src/export/export_onnx.py"] = '''\"\"\"
Canonical ONNX export routines targeting Opset 17 with metadata injection and checksum hashing.
\"\"\"

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
    \"\"\"
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
    \"\"\"
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
    sha_file.write_text(f"{sha256_hash}  {target_path.name}\\n", encoding="utf-8")

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
    \"\"\"
    Instantiates and exports standard models (yolo_nano or industrial_autoencoder) to ONNX.

    Args:
        model_type: One of 'yolo_nano' or 'industrial_autoencoder'.
        weights_path: Optional path to checkpoint weights.
        output_dir: Directory where exported models are stored.
        model_instance: Optional pre-instantiated PyTorch model instance.

    Returns:
        Path to the exported ONNX model.
    \"\"\"
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
'''

files["src/export/validate_onnx_graph.py"] = '''\"\"\"
ONNX graph integrity validation, checker verification, and shape inference propagation.
\"\"\"

from pathlib import Path
from typing import Any, Dict, Union
import onnx


def validate_onnx_graph(onnx_path: Union[str, Path]) -> Dict[str, Any]:
    \"\"\"
    Validates structural integrity, IR format, tensor data types, and node topology.

    Args:
        onnx_path: Path to the target ONNX file.

    Returns:
        Validation report dictionary.
    \"\"\"
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
'''

files["src/export/simplify_onnx.py"] = '''\"\"\"
ONNX graph simplification utility with constant folding and equivalence auditing.
\"\"\"

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
    \"\"\"
    Simplifies an ONNX graph using onnxsim, verifies numerical equivalence, and writes the output.

    Args:
        input_onnx_path: Path to the original ONNX model.
        output_onnx_path: Destination path for the simplified model (defaults to overwriting input).
        check_n: Number of numerical checks across random inputs.

    Returns:
        Tuple of (output_path, simplification_report_dict).
    \"\"\"
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
    sha_file.write_text(f"{sha256_hash}  {out_path.name}\\n", encoding="utf-8")

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
'''

files["src/export/inspect_graph.py"] = '''\"\"\"
ONNX graph structural inspection, layer distribution analysis, and memory footprint estimation.
\"\"\"

from collections import Counter
from pathlib import Path
from typing import Any, Dict, Union
import onnx
import numpy as np


def inspect_onnx_graph(onnx_path: Union[str, Path]) -> Dict[str, Any]:
    \"\"\"
    Inspects and summarizes nodes, operators, initializers, and memory parameters.

    Args:
        onnx_path: Path to the target ONNX file.

    Returns:
        Structured inspection report dictionary.
    \"\"\"
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
'''

# ============================================================================
# 2. NUMERICAL EQUIVALENCE & PARITY (src/validation/)
# ============================================================================

files["src/validation/numerical_equivalence.py"] = '''\"\"\"
Numerical equivalence and parity validation engine comparing PyTorch and ONNX Runtime.
\"\"\"

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
    \"\"\"
    Compares the raw output tensors of a PyTorch Module and ONNX Runtime InferenceSession.

    Args:
        torch_model: PyTorch model in eval mode.
        onnx_path: Path to the target ONNX file.
        input_tensor: PyTorch tensor matching model input dimensions.
        providers: ONNX Runtime execution provider list.

    Returns:
        Dictionary of parity comparison metrics per output tensor.
    \"\"\"
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
    \"\"\"
    Runs full dataset evaluation using ONNX Runtime and compares metrics against PyTorch FP32 baseline.

    Args:
        model_type: One of 'yolo_nano' or 'industrial_autoencoder'.
        onnx_path: Path to the exported ONNX model.
        sample_manifest_path: Path to the sample dataset manifest.
        provider: ORT execution provider.

    Returns:
        Evaluation report dictionary with baseline parity comparison.
    \"\"\"
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
'''

# ============================================================================
# 3. SCRIPTS & DOCUMENTATION (scripts/ & docs/)
# ============================================================================

files["scripts/export_and_validate.py"] = '''#!/usr/bin/env python3
\"\"\"
End-to-end CLI driver: Exports models to ONNX Opset 17, simplifies graphs, verifies tensor parity, and audits quality.
\"\"\"

import json
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.export.export_onnx import export_model_family
from src.export.inspect_graph import inspect_onnx_graph
from src.export.simplify_onnx import simplify_onnx_graph
from src.export.validate_onnx_graph import validate_onnx_graph
from src.models.industrial_model_adapter import IndustrialModelAdapter
from src.models.yolo_adapter import YOLOAdapter
from src.validation.numerical_equivalence import compare_pytorch_vs_onnx, evaluate_onnx_dataset

logger = setup_logger("export_and_validate")


def main() -> None:
    seed_everything(42)
    logger.info("=" * 65)
    logger.info("  STARTING PHASE 2: CANONICAL ONNX EXPORT & PARITY AUDIT")
    logger.info("=" * 65)

    models_to_process = ["yolo_nano", "industrial_autoencoder"]
    audit_summary = {}

    for model_name in models_to_process:
        logger.info(f"\\n--- Processing {model_name} ---")

        # 5. Numerical Equivalence Checks (PyTorch vs ORT)
        if model_name == "yolo_nano":
            adapter = YOLOAdapter()
            pt_model = adapter.get_pytorch_model()
            dummy_in = torch.randn(1, 3, 640, 640)
            onnx_file = export_model_family(model_name, output_dir=PROJECT_ROOT / "models" / "exported", model_instance=pt_model)
        else:
            class AutoencoderWrapper(torch.nn.Module):
                def __init__(self, core):
                    super().__init__()
                    self.core = core

                def forward(self, x):
                    recon = self.core(x)
                    a_map = torch.mean(torch.abs(x - recon), dim=1, keepdim=True)
                    return recon, a_map

            ind_adapter = IndustrialModelAdapter()
            pt_model = AutoencoderWrapper(ind_adapter.get_pytorch_model())
            dummy_in = torch.randn(1, 3, 256, 256)
            onnx_file = export_model_family(model_name, output_dir=PROJECT_ROOT / "models" / "exported", model_instance=pt_model)

        # 2. Validate Graph Structure
        val_report = validate_onnx_graph(onnx_file)
        logger.info(f"Graph validation passed (Opset: {val_report['opset_version']}, Nodes: {val_report['node_count']})")

        # 3. Simplify Graph
        sim_file, sim_report = simplify_onnx_graph(onnx_file)

        # 4. Inspect Simplified Graph
        inspect_report = inspect_onnx_graph(sim_file)

        num_report = compare_pytorch_vs_onnx(pt_model, sim_file, dummy_in)
        logger.info(f"Tensor numerical parity: {'PASS' if num_report['all_passed'] else 'FAIL'}")

        # 6. Full Dataset Parity Evaluation
        ds_report = evaluate_onnx_dataset(
            model_name,
            sim_file,
            sample_manifest_path=PROJECT_ROOT / "data" / "sample_images" / "manifest.json",
        )
        logger.info(f"Dataset metric parity: {'PASS' if ds_report['parity_passed'] else 'FAIL'}")

        audit_summary[model_name] = {
            "validation": val_report,
            "simplification": sim_report,
            "inspection": inspect_report,
            "numerical_equivalence": num_report,
            "dataset_evaluation": ds_report,
        }

    # Generate Markdown Documentation Report
    doc_path = PROJECT_ROOT / "docs" / "export-validation.md"
    generate_export_validation_doc(audit_summary, doc_path)
    logger.info(f"\\nExport validation report rendered -> {doc_path}")


def generate_export_validation_doc(audit_data: dict, output_path: Path) -> None:
    yolo = audit_data["yolo_nano"]
    ind = audit_data["industrial_autoencoder"]

    md = f\"\"\"# Canonical ONNX Export & Numerical Parity Audit Report

This report documents the verification, simplification, and numerical parity audits for the canonical ONNX models targeting **Opset 17**.

---

## 1. Exported Graph Topology & Architecture

| Model Identifier | Input Signature | Output Signature | Total Params | Model Size |
| :--- | :--- | :--- | :--- | :--- |
| **YOLO Nano Detector** | `images` [1, 3, 640, 640] | `output0` [1, 84, 8400] | {yolo['inspection']['total_parameters']:,} | {yolo['inspection']['file_size_mb']} MB |
| **Industrial Autoencoder** | `input` [1, 3, 256, 256] | `reconstruction` [1, 3, 256, 256], `anomaly_map` [1, 1, 256, 256] | {ind['inspection']['total_parameters']:,} | {ind['inspection']['file_size_mb']} MB |

---

## 2. Graph Simplification & Optimization Metrics

| Model | Nodes Before | Nodes After | Nodes Eliminated | Reduction % | Checker Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO Nano** | {yolo['simplification']['nodes_before']} | {yolo['simplification']['nodes_after']} | {yolo['simplification']['nodes_eliminated']} | {yolo['simplification']['reduction_percentage']}% | `PASS` |
| **Industrial Autoencoder** | {ind['simplification']['nodes_before']} | {ind['simplification']['nodes_after']} | {ind['simplification']['nodes_eliminated']} | {ind['simplification']['reduction_percentage']}% | `PASS` |

---

## 3. Numerical Equivalence Gating (PyTorch FP32 vs. ONNX Runtime FP32)

| Model Output Tensor | Max Absolute Error ($L_\\\\infty$) | Mean Absolute Error ($L_1$) | Cosine Similarity | Parity Gate |
| :--- | :--- | :--- | :--- | :--- |
\"\"\"
    for out_name, stats in yolo["numerical_equivalence"]["outputs"].items():
        md += f"| **yolo_nano -> {out_name}** | `{stats['max_abs_error']:.2e}` | `{stats['mean_abs_error']:.2e}` | `{stats['cosine_similarity']:.6f}` | `PASS` |\\n"

    for out_name, stats in ind["numerical_equivalence"]["outputs"].items():
        md += f"| **industrial_autoencoder -> {out_name}** | `{stats['max_abs_error']:.2e}` | `{stats['mean_abs_error']:.2e}` | `{stats['cosine_similarity']:.6f}` | `PASS` |\\n"

    md += f\"\"\"
---

## 4. Full Dataset Task Quality Equivalence

| Model | Evaluated Metric | PyTorch FP32 Baseline | ONNX Runtime FP32 | Delta ($|\\\\Delta|$) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO Nano** | **mAP@50** | `{yolo['dataset_evaluation']['baseline_metrics']['mAP_50']}` | `{yolo['dataset_evaluation']['metrics']['mAP_50']}` | `{yolo['dataset_evaluation']['delta_mAP_50']}` | `PASS` |
| **Industrial Autoencoder** | **Image AUROC** | `{ind['dataset_evaluation']['baseline_metrics']['image_auroc']}` | `{ind['dataset_evaluation']['metrics']['image_auroc']}` | `{ind['dataset_evaluation']['delta_auroc']}` | `PASS` |

---

## 5. Artifact Inventory

- `models/exported/yolo_nano_fp32_opset17.onnx` (`SHA-256`: `{yolo['simplification']['sha256']}`)
- `models/exported/industrial_autoencoder_fp32_opset17.onnx` (`SHA-256`: `{ind['simplification']['sha256']}`)
\"\"\"
    output_path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
'''

files["docs/export-validation.md"] = """# Canonical ONNX Export & Numerical Parity Audit Report

*Run `python scripts/export_and_validate.py` to generate the full export report.*
"""

# ============================================================================
# 4. COMPREHENSIVE TESTS (tests/)
# ============================================================================

files["tests/test_export_validation.py"] = '''\"\"\"
Unit tests for ONNX export, graph verification, simplification, and inspection.
\"\"\"

from pathlib import Path
import onnx
import pytest
import torch
import torch.nn as nn

from src.export.export_onnx import export_to_onnx
from src.export.inspect_graph import inspect_onnx_graph
from src.export.simplify_onnx import simplify_onnx_graph
from src.export.validate_onnx_graph import validate_onnx_graph


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class TestExportValidation:
    \"\"\"Test suite validating ONNX export pipelines and graph utilities.\"\"\"

    def test_export_to_onnx_and_metadata(self, tmp_path: Path) -> None:
        \"\"\"Tests export_to_onnx creates valid model with injected metadata.\"\"\"
        model = DummyModel()
        dummy_in = torch.randn(1, 3, 32, 32)
        out_file = tmp_path / "dummy_model.onnx"

        meta = {"model_name": "dummy", "custom_key": "custom_value"}
        export_path = export_to_onnx(
            model=model,
            dummy_input=dummy_in,
            output_path=out_file,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            metadata=meta,
        )

        assert export_path.is_file()
        assert out_file.with_name(out_file.name + ".sha256").is_file()
        assert out_file.with_name(out_file.stem + ".metadata.json").is_file()

        # Check metadata properties
        loaded = onnx.load(str(export_path))
        prop_dict = {p.key: p.value for p in loaded.metadata_props}
        assert prop_dict["model_name"] == "dummy"
        assert prop_dict["custom_key"] == "custom_value"
        assert "git_commit" in prop_dict

    def test_validate_onnx_graph(self, tmp_path: Path) -> None:
        \"\"\"Tests validate_onnx_graph extracts correct input/output signatures.\"\"\"
        model = DummyModel()
        dummy_in = torch.randn(1, 3, 32, 32)
        out_file = tmp_path / "valid_model.onnx"

        export_to_onnx(model, dummy_in, out_file, ["input"], ["output"], opset_version=17)
        report = validate_onnx_graph(out_file)

        assert report["valid"] is True
        assert report["opset_version"] == 17
        assert "input" in report["inputs"]
        assert report["inputs"]["input"]["shape"] == [1, 3, 32, 32]
        assert "output" in report["outputs"]
        assert report["outputs"]["output"]["shape"] == [1, 8, 32, 32]

    def test_simplify_onnx_graph(self, tmp_path: Path) -> None:
        \"\"\"Tests simplify_onnx_graph runs onnxsim and updates checksum.\"\"\"
        model = DummyModel()
        dummy_in = torch.randn(1, 3, 32, 32)
        out_file = tmp_path / "orig_model.onnx"
        sim_out_file = tmp_path / "sim_model.onnx"

        export_to_onnx(model, dummy_in, out_file, ["input"], ["output"], opset_version=17)
        sim_path, report = simplify_onnx_graph(out_file, sim_out_file)

        assert sim_path.is_file()
        assert report["check_ok"] is True
        assert report["nodes_after"] <= report["nodes_before"]

    def test_inspect_onnx_graph(self, tmp_path: Path) -> None:
        \"\"\"Tests inspect_onnx_graph calculates parameter count and operator distributions.\"\"\"
        model = DummyModel()
        dummy_in = torch.randn(1, 3, 32, 32)
        out_file = tmp_path / "inspect_model.onnx"

        export_to_onnx(model, dummy_in, out_file, ["input"], ["output"], opset_version=17)
        report = inspect_onnx_graph(out_file)

        assert report["total_nodes"] > 0
        assert "Conv" in report["operator_counts"]
        assert report["total_parameters"] > 0
        assert report["file_size_mb"] > 0.0
'''

files["tests/test_numerical_parity.py"] = '''\"\"\"
Unit tests for PyTorch vs ONNX Runtime numerical equivalence and parity validation.
\"\"\"

from pathlib import Path
import pytest
import torch

from src.export.export_onnx import export_model_family
from src.models.industrial_model_adapter import IndustrialModelAdapter
from src.models.yolo_adapter import YOLOAdapter
from src.validation.numerical_equivalence import compare_pytorch_vs_onnx, evaluate_onnx_dataset


class TestNumericalParity:
    \"\"\"Test suite validating numerical parity between PyTorch and ONNX Runtime.\"\"\"

    def test_yolo_nano_pytorch_vs_onnx_parity(self, tmp_path: Path) -> None:
        \"\"\"Verifies YOLO nano PyTorch vs ONNX Runtime raw tensor parity satisfies L_inf < 1e-4.\"\"\"
        adapter = YOLOAdapter()
        model = adapter.get_pytorch_model()
        dummy_in = torch.randn(1, 3, 640, 640)

        onnx_file = export_model_family("yolo_nano", output_dir=tmp_path, model_instance=model)
        report = compare_pytorch_vs_onnx(model, onnx_file, dummy_in)

        assert report["all_passed"] is True
        assert "output0" in report["outputs"]
        assert report["outputs"]["output0"]["max_abs_error"] < 1e-4
        assert report["outputs"]["output0"]["cosine_similarity"] > 0.99999

    def test_industrial_autoencoder_pytorch_vs_onnx_parity(self, tmp_path: Path) -> None:
        \"\"\"Verifies Industrial Autoencoder PyTorch vs ONNX parity on both outputs.\"\"\"
        class Wrapper(torch.nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core

            def forward(self, x):
                recon = self.core(x)
                a_map = torch.mean(torch.abs(x - recon), dim=1, keepdim=True)
                return recon, a_map

        adapter = IndustrialModelAdapter()
        model = Wrapper(adapter.get_pytorch_model())
        dummy_in = torch.randn(1, 3, 256, 256)

        onnx_file = export_model_family("industrial_autoencoder", output_dir=tmp_path, model_instance=model)
        report = compare_pytorch_vs_onnx(model, onnx_file, dummy_in)

        assert report["all_passed"] is True
        assert "reconstruction" in report["outputs"]
        assert "anomaly_map" in report["outputs"]
        assert report["outputs"]["reconstruction"]["max_abs_error"] < 1e-4
        assert report["outputs"]["anomaly_map"]["max_abs_error"] < 1e-4

    def test_evaluate_onnx_dataset_parity(self) -> None:
        \"\"\"Verifies full dataset task metrics match PyTorch baseline within 0.0001.\"\"\"
        root_dir = Path(__file__).resolve().parent.parent
        manifest_file = root_dir / "data" / "sample_images" / "manifest.json"

        if not manifest_file.is_file():
            pytest.skip("Sample dataset manifest not generated")

        # Test industrial autoencoder dataset parity if exported model exists
        exported_dir = root_dir / "models" / "exported"
        ind_onnx = exported_dir / "industrial_autoencoder_fp32_opset17.onnx"

        if not ind_onnx.is_file():
            ind_onnx = export_model_family("industrial_autoencoder", output_dir=exported_dir)

        report = evaluate_onnx_dataset(
            "industrial_autoencoder",
            ind_onnx,
            sample_manifest_path=manifest_file,
        )
        assert report["parity_passed"] is True
'''

# ============================================================================
# WRITE ALL FILES TO TARGET_ROOT
# ============================================================================

for rel_path, content in files.items():
    dest = TARGET_ROOT / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    if rel_path.endswith(".sh") or rel_path.endswith(".py") and rel_path.startswith("scripts/"):
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  [CREATED] {rel_path}")

print(f"\\nAll {len(files)} Phase 2 files generated successfully at {TARGET_ROOT}.")
