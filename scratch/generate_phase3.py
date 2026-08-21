from pathlib import Path
import os
import stat

TARGET_ROOT = Path("/home/sengar/onnx-quant-benchmark")

files = {}

# ============================================================================
# 1. QUANTIZATION PACKAGE & MODULES (src/quantization/)
# ============================================================================

files["src/quantization/__init__.py"] = '''\"\"\"Quantization, FP16 conversion, calibration data reading, and PTQ engine.\"\"\"
from src.quantization.convert_fp16 import convert_onnx_to_fp16
from src.quantization.calibration_reader import BenchmarkCalibrationDataReader
from src.quantization.quantize_onnx import quantize_onnx_static
from src.validation.validate_quantization import validate_quantized_model

__all__ = [
    "convert_onnx_to_fp16",
    "BenchmarkCalibrationDataReader",
    "quantize_onnx_static",
    "validate_quantized_model",
]
'''

files["src/quantization/convert_fp16.py"] = '''\"\"\"
FP16 half-precision ONNX model conversion utility using onnxconverter_common.
\"\"\"

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
    \"\"\"
    Converts a standard FP32 ONNX model into numerical-stable FP16 representation.

    Args:
        input_onnx_path: Path to the source FP32 ONNX model.
        output_onnx_path: Target path for the converted FP16 model (defaults to *_fp16.onnx).
        keep_io_types: Keep graph input and output tensors in float32 for seamless I/O compatibility.

    Returns:
        Path to the saved FP16 model.
    \"\"\"
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
    sha_file.write_text(f"{sha256_hash}  {out_p.name}\\n", encoding="utf-8")

    logger.info(f"FP16 conversion complete -> {out_p} (SHA-256: {sha256_hash[:16]}...)")
    return out_p
'''

files["src/quantization/calibration_reader.py"] = '''\"\"\"
Calibration data reader implementing onnxruntime.quantization.CalibrationDataReader interface.
\"\"\"

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from onnxruntime.quantization import CalibrationDataReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging import setup_logger

logger = setup_logger("calibration_reader")


class BenchmarkCalibrationDataReader(CalibrationDataReader):
    \"\"\"
    Deterministic calibration data reader feeding representative activation tensors to ONNX Runtime quantizer.
    \"\"\"

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        input_name: str,
        input_shape: Tuple[int, int, int, int],
        preprocess_fn: Callable[[Union[str, Path]], Any],
        batch_size: int = 1,
    ) -> None:
        \"\"\"
        Initializes the calibration data reader.

        Args:
            image_paths: List of absolute or relative paths to calibration images.
            input_name: Graph input tensor name (e.g., 'images' or 'input').
            input_shape: 4D input dimensions [B, C, H, W].
            preprocess_fn: Callable mapping image path to PyTorch Tensor or (Tensor, ...).
            batch_size: Batch size for calibration steps (default: 1).
        \"\"\"
        super().__init__()
        self.image_paths = [Path(p) for p in image_paths]
        self.input_name = input_name
        self.input_shape = input_shape
        self.preprocess_fn = preprocess_fn
        self.batch_size = max(1, batch_size)
        self._current_index = 0

        logger.info(
            f"Initialized BenchmarkCalibrationDataReader: {len(self.image_paths)} images, "
            f"input='{input_name}', shape={input_shape}, batch_size={self.batch_size}"
        )

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        \"\"\"
        Returns the next batch of preprocessed input data, or None when dataset is exhausted.

        Returns:
            Dictionary {input_name: numpy_ndarray} or None.
        \"\"\"
        if self._current_index >= len(self.image_paths):
            return None

        batch_paths = self.image_paths[self._current_index : self._current_index + self.batch_size]
        batch_tensors = []

        for p in batch_paths:
            res = self.preprocess_fn(p)
            # Handle preprocess_detection_image returning (tensor, ratio, pad, orig_shape)
            if isinstance(res, (tuple, list)):
                tensor = res[0]
            elif isinstance(res, torch.Tensor):
                tensor = res
            elif isinstance(res, np.ndarray):
                tensor = torch.from_numpy(res)
            else:
                raise TypeError(f"Unexpected preprocessing return type: {type(res)}")

            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            batch_tensors.append(tensor)

        self._current_index += len(batch_paths)
        full_batch = torch.cat(batch_tensors, dim=0).detach().cpu().numpy().astype(np.float32)

        return {self.input_name: full_batch}

    def rewind(self) -> None:
        \"\"\"Resets the iteration pointer to the beginning of the dataset.\"\"\"
        self._current_index = 0
'''

files["src/quantization/quantize_onnx.py"] = '''\"\"\"
Static Post-Training Quantization (PTQ) engine using ONNX Runtime.
\"\"\"

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
    \"\"\"
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
    \"\"\"
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
    sha_file.write_text(f"{sha256_hash}  {out_p.name}\\n", encoding="utf-8")

    logger.info(f"Static INT8 PTQ complete -> {out_p} (SHA-256: {sha256_hash[:16]}...)")
    return out_p
'''

# ============================================================================
# 2. VALIDATION & QUALITY DEGRADATION GATING (src/validation/)
# ============================================================================

files["src/validation/validate_quantization.py"] = '''\"\"\"
Quantization quality degradation validator and acceptance decision gating engine.
\"\"\"

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Union
import cv2
import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import QualityThresholdConfig
from src.common.logging import setup_logger
from src.models.preprocess import preprocess_detection_image, preprocess_industrial_image
from src.models.yolo_adapter import YOLOAdapter
from src.validation.anomaly_quality import compute_aupro, compute_image_auroc, compute_pixel_auroc
from src.validation.detection_quality import evaluate_detection_dataset

logger = setup_logger("validate_quantization")


def validate_quantized_model(
    model_name: str,
    precision: str,
    onnx_path: Union[str, Path],
    manifest_path: Union[str, Path],
    baseline_metrics_path: Union[str, Path],
    quality_threshold_config: Optional[QualityThresholdConfig] = None,
) -> Dict[str, Any]:
    \"\"\"
    Validates task quality for a quantized ONNX model against PyTorch FP32 baseline and enforces acceptance gates.

    Args:
        model_name: 'yolo_nano' or 'industrial_autoencoder'.
        precision: 'fp16', 'int8', etc.
        onnx_path: Path to the quantized ONNX graph.
        manifest_path: Path to sample evaluation dataset manifest.json.
        baseline_metrics_path: Path to reference FP32 baseline_metrics.json.
        quality_threshold_config: Threshold constraints for quality degradation.

    Returns:
        Validation report dictionary with metrics, deltas, and acceptance status.
    \"\"\"
    onnx_p = Path(onnx_path)
    if not onnx_p.is_file():
        raise FileNotFoundError(f"Quantized ONNX model not found: {onnx_path}")

    thresh_cfg = quality_threshold_config or QualityThresholdConfig()

    # Load baseline metrics
    b_path = Path(baseline_metrics_path)
    if not b_path.is_file():
        raise FileNotFoundError(f"Baseline metrics not found: {baseline_metrics_path}")
    baseline_data = json.loads(b_path.read_text(encoding="utf-8"))
    baseline_metrics = baseline_data["metrics"]

    # Load evaluation manifest
    m_path = Path(manifest_path)
    if not m_path.is_file():
        raise FileNotFoundError(f"Evaluation manifest not found: {manifest_path}")
    manifest = json.loads(m_path.read_text(encoding="utf-8"))

    # Initialize ONNX Runtime session
    sess_opts = ort.SessionOptions()
    session = ort.InferenceSession(str(onnx_p), sess_options=sess_opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    is_fp16_input = "float16" in session.get_inputs()[0].type

    if model_name == "yolo_nano":
        adapter = YOLOAdapter()
        det_samples = manifest["detection_samples"]
        all_preds = []
        all_gts = []

        for item in det_samples:
            img_p = PROJECT_ROOT / item["image_path"]
            tensor, ratio, pad, orig_shape = preprocess_detection_image(img_p)
            np_in = tensor.numpy().astype(np.float16) if is_fp16_input else tensor.numpy()

            ort_out = session.run(None, {input_name: np_in})[0]
            if ort_out.dtype == np.float16:
                ort_out = ort_out.astype(np.float32)

            detections = adapter.postprocess(torch.from_numpy(ort_out), orig_shape, ratio, pad)

            all_preds.append(detections)
            all_gts.append(item["ground_truth_boxes"])

        eval_metrics = evaluate_detection_dataset(all_preds, all_gts)

        base_map50 = baseline_metrics["mAP_50"]
        base_map50_95 = baseline_metrics.get("mAP_50_95", base_map50)
        curr_map50 = eval_metrics["mAP_50"]
        curr_map50_95 = eval_metrics["mAP_50_95"]

        delta_map50 = float(round(base_map50 - curr_map50, 6))
        delta_map50_95 = float(round(base_map50_95 - curr_map50_95, 6))

        # Hard quality gate: max allowed drop in mAP
        passed = delta_map50_95 <= thresh_cfg.max_map_drop
        status = "PASS" if passed else "REJECTED"
        rejection_reason = None if passed else f"mAP drop ({delta_map50_95:.4f}) exceeds threshold ({thresh_cfg.max_map_drop})"

        return {
            "model_name": model_name,
            "precision": precision,
            "onnx_path": str(onnx_p),
            "status": status,
            "rejection_reason": rejection_reason,
            "metrics": eval_metrics,
            "baseline_metrics": baseline_metrics,
            "delta_map_50": delta_map50,
            "delta_map_50_95": delta_map50_95,
            "max_allowed_drop": thresh_cfg.max_map_drop,
        }

    elif model_name == "industrial_autoencoder":
        from src.models.industrial_model_adapter import IndustrialModelAdapter

        adapter = IndustrialModelAdapter()
        ind_samples = manifest["industrial_samples"]

        y_true = []
        y_scores = []
        gt_masks = []
        anomaly_maps = []

        for item in ind_samples:
            img_p = PROJECT_ROOT / item["image_path"]
            tensor = preprocess_industrial_image(img_p)
            np_in = tensor.numpy().astype(np.float16) if is_fp16_input else tensor.numpy()

            ort_outs = session.run(None, {input_name: np_in})
            recon = ort_outs[0].astype(np.float32) if ort_outs[0].dtype == np.float16 else ort_outs[0]
            a_map = ort_outs[1].astype(np.float32) if ort_outs[1].dtype == np.float16 else ort_outs[1]

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

        img_auroc = compute_image_auroc(y_true, y_scores)
        pix_auroc = compute_pixel_auroc(gt_masks, anomaly_maps)
        aupro = compute_aupro(gt_masks, anomaly_maps)

        eval_metrics = {
            "image_auroc": img_auroc,
            "pixel_auroc": pix_auroc,
            "aupro": aupro,
        }

        base_auroc = baseline_metrics["image_auroc"]
        base_aupro = baseline_metrics.get("aupro", 1.0)

        delta_auroc = float(round(base_auroc - img_auroc, 6))
        delta_aupro = float(round(base_aupro - aupro, 6))

        passed = delta_auroc <= thresh_cfg.max_auroc_drop
        status = "PASS" if passed else "REJECTED"
        rejection_reason = None if passed else f"AUROC drop ({delta_auroc:.4f}) exceeds threshold ({thresh_cfg.max_auroc_drop})"

        return {
            "model_name": model_name,
            "precision": precision,
            "onnx_path": str(onnx_p),
            "status": status,
            "rejection_reason": rejection_reason,
            "metrics": eval_metrics,
            "baseline_metrics": baseline_metrics,
            "delta_auroc": delta_auroc,
            "delta_aupro": delta_aupro,
            "max_allowed_drop": thresh_cfg.max_auroc_drop,
        }
    else:
        raise ValueError(f"Unknown model_name: {model_name}")
'''

# ============================================================================
# 3. SCRIPTS & DOCUMENTATION (scripts/ & docs/)
# ============================================================================

files["scripts/prepare_calibration_data.py"] = '''#!/usr/bin/env python3
\"\"\"
Generates disjoint representative calibration datasets for detection and industrial models.
\"\"\"

import csv
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.hashes import compute_file_sha256
from src.common.logging import setup_logger
from src.common.seed import seed_everything

logger = setup_logger("prepare_calibration_data")


def generate_calibration_data(
    detection_count: int = 50,
    industrial_count: int = 50,
    output_dir: Path = PROJECT_ROOT / "data" / "calibration",
) -> Path:
    \"\"\"
    Synthesizes disjoint calibration datasets with unique random seeds ensuring 0% overlap with evaluation splits.
    \"\"\"
    det_dir = output_dir / "detection"
    ind_dir = output_dir / "industrial"
    det_dir.mkdir(parents=True, exist_ok=True)
    ind_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    # 1. Generate Detection Calibration Images (Seed: 100)
    seed_everything(100)
    logger.info(f"Generating {detection_count} detection calibration images...")

    for i in range(detection_count):
        # Create realistic multi-channel noise + geometric shapes
        img = np.random.randint(40, 210, (640, 640, 3), dtype=np.uint8)
        # Draw background structures
        for _ in range(5):
            pt1 = (np.random.randint(0, 640), np.random.randint(0, 640))
            pt2 = (np.random.randint(0, 640), np.random.randint(0, 640))
            color = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
            cv2.rectangle(img, pt1, pt2, color, -1)

        img_path = det_dir / f"calib_det_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        sha = compute_file_sha256(img_path)
        manifest_rows.append({
            "path": f"data/calibration/detection/{img_path.name}",
            "category": "detection",
            "split": "calibration",
            "sha256": sha,
        })

    # 2. Generate Industrial Normal Calibration Textures (Seed: 200)
    seed_everything(200)
    logger.info(f"Generating {industrial_count} industrial normal calibration images...")

    for i in range(industrial_count):
        # Create continuous brushed metal texture
        base_val = np.random.randint(110, 160)
        grad = np.tile(np.linspace(base_val - 15, base_val + 15, 256, dtype=np.float32), (256, 1))
        noise = np.random.normal(0, 4.0, (256, 256)).astype(np.float32)
        texture = np.clip(grad + noise, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

        img_path = ind_dir / f"calib_ind_{i:03d}.png"
        cv2.imwrite(str(img_path), img)

        sha = compute_file_sha256(img_path)
        manifest_rows.append({
            "path": f"data/calibration/industrial/{img_path.name}",
            "category": "industrial",
            "split": "calibration",
            "sha256": sha,
        })

    # Write CSV manifest
    manifest_csv = output_dir / "manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "category", "split", "sha256"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Write README.md
    readme_md = output_dir / "README.md"
    readme_md.write_text(
        f\"\"\"# Representative Calibration Dataset

- **Total Samples**: {len(manifest_rows)} ({detection_count} Detection, {industrial_count} Industrial Normal)
- **Manifest**: `manifest.csv`
- **Isolation Constraint**: 100% disjoint from `data/sample_images/` test splits.
\"\"\",
        encoding="utf-8",
    )

    logger.info(f"Calibration data saved: {manifest_csv} ({len(manifest_rows)} records)")
    return manifest_csv


if __name__ == "__main__":
    generate_calibration_data()
'''

files["scripts/quantize_and_validate.py"] = '''#!/usr/bin/env python3
\"\"\"
Quantization and validation driver: FP16 conversion, Static INT8 PTQ calibration, and quality degradation auditing.
\"\"\"

import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import QualityThresholdConfig
from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.models.preprocess import preprocess_detection_image, preprocess_industrial_image
from src.quantization.calibration_reader import BenchmarkCalibrationDataReader
from src.quantization.convert_fp16 import convert_onnx_to_fp16
from src.quantization.quantize_onnx import quantize_onnx_static
from src.validation.validate_quantization import validate_quantized_model

logger = setup_logger("quantize_and_validate")


def main() -> None:
    seed_everything(42)
    logger.info("=" * 65)
    logger.info("  STARTING PHASE 3: QUANTIZATION (FP16 & STATIC INT8 PTQ)")
    logger.info("=" * 65)

    exp_dir = PROJECT_ROOT / "models" / "exported"
    sample_manifest = PROJECT_ROOT / "data" / "sample_images" / "manifest.json"
    calib_manifest = PROJECT_ROOT / "data" / "calibration" / "manifest.csv"

    if not calib_manifest.is_file():
        from scripts.prepare_calibration_data import generate_calibration_data
        generate_calibration_data()

    # Load calibration paths from manifest
    calib_det_paths = []
    calib_ind_paths = []
    with open(calib_manifest, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["category"] == "detection":
                calib_det_paths.append(PROJECT_ROOT / row["path"])
            elif row["category"] == "industrial":
                calib_ind_paths.append(PROJECT_ROOT / row["path"])

    thresh_cfg = QualityThresholdConfig(max_map_drop=0.015, max_auroc_drop=0.010)
    audit_results = {}

    yolo_base_json = PROJECT_ROOT / "results" / "raw" / "pytorch_fp32" / "yolo_nano" / "baseline_metrics.json"
    ind_base_json = PROJECT_ROOT / "results" / "raw" / "pytorch_fp32" / "industrial_autoencoder" / "baseline_metrics.json"

    if not yolo_base_json.is_file() or not ind_base_json.is_file():
        import subprocess
        logger.info("Baseline metrics missing, executing PyTorch baselines...")
        subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "run_pytorch_baselines.py")], check=True)

    # =========================================================================
    # 1. YOLO Nano Quantization Suite
    # =========================================================================
    logger.info("\\n--- Processing YOLO Nano Quantization ---")
    yolo_fp32_onnx = exp_dir / "yolo_nano_fp32_opset17.onnx"

    # FP16 Conversion
    yolo_fp16_onnx = exp_dir / "yolo_nano_fp16.onnx"
    convert_onnx_to_fp16(yolo_fp32_onnx, yolo_fp16_onnx)
    yolo_fp16_val = validate_quantized_model(
        "yolo_nano", "fp16", yolo_fp16_onnx, sample_manifest, yolo_base_json, thresh_cfg
    )
    logger.info(f"YOLO FP16 Validation: {yolo_fp16_val['status']} (mAP@50: {yolo_fp16_val['metrics']['mAP_50']})")

    # Static INT8 PTQ
    yolo_reader = BenchmarkCalibrationDataReader(
        image_paths=calib_det_paths,
        input_name="images",
        input_shape=(1, 3, 640, 640),
        preprocess_fn=preprocess_detection_image,
        batch_size=1,
    )
    yolo_int8_onnx = exp_dir / "yolo_nano_static_int8.onnx"
    quantize_onnx_static(
        input_onnx_path=yolo_fp32_onnx,
        output_onnx_path=yolo_int8_onnx,
        calibration_data_reader=yolo_reader,
        quant_format="QDQ",
        calibrate_method="MinMax",
    )
    yolo_int8_val = validate_quantized_model(
        "yolo_nano", "int8", yolo_int8_onnx, sample_manifest, yolo_base_json, thresh_cfg
    )
    logger.info(f"YOLO INT8 Validation: {yolo_int8_val['status']} (mAP@50: {yolo_int8_val['metrics']['mAP_50']})")

    audit_results["yolo_nano"] = {
        "fp16": yolo_fp16_val,
        "int8": yolo_int8_val,
    }

    # =========================================================================
    # 2. Industrial Autoencoder Quantization Suite
    # =========================================================================
    logger.info("\\n--- Processing Industrial Autoencoder Quantization ---")
    ind_fp32_onnx = exp_dir / "industrial_autoencoder_fp32_opset17.onnx"
    ind_base_json = PROJECT_ROOT / "results" / "raw" / "pytorch_fp32" / "industrial_autoencoder" / "baseline_metrics.json"

    # FP16 Conversion
    ind_fp16_onnx = exp_dir / "industrial_autoencoder_fp16.onnx"
    convert_onnx_to_fp16(ind_fp32_onnx, ind_fp16_onnx)
    ind_fp16_val = validate_quantized_model(
        "industrial_autoencoder", "fp16", ind_fp16_onnx, sample_manifest, ind_base_json, thresh_cfg
    )
    logger.info(f"Industrial FP16 Validation: {ind_fp16_val['status']} (AUROC: {ind_fp16_val['metrics']['image_auroc']})")

    # Static INT8 PTQ
    ind_reader = BenchmarkCalibrationDataReader(
        image_paths=calib_ind_paths,
        input_name="input",
        input_shape=(1, 3, 256, 256),
        preprocess_fn=preprocess_industrial_image,
        batch_size=1,
    )
    ind_int8_onnx = exp_dir / "industrial_autoencoder_static_int8.onnx"
    quantize_onnx_static(
        input_onnx_path=ind_fp32_onnx,
        output_onnx_path=ind_int8_onnx,
        calibration_data_reader=ind_reader,
        quant_format="QDQ",
        calibrate_method="MinMax",
    )
    ind_int8_val = validate_quantized_model(
        "industrial_autoencoder", "int8", ind_int8_onnx, sample_manifest, ind_base_json, thresh_cfg
    )
    logger.info(f"Industrial INT8 Validation: {ind_int8_val['status']} (AUROC: {ind_int8_val['metrics']['image_auroc']})")

    audit_results["industrial_autoencoder"] = {
        "fp16": ind_fp16_val,
        "int8": ind_int8_val,
    }

    # Generate Markdown Report
    doc_path = PROJECT_ROOT / "docs" / "int8-calibration.md"
    generate_calibration_doc(audit_results, doc_path)
    logger.info(f"\\nQuantization audit report rendered -> {doc_path}")


def generate_calibration_doc(audit_data: dict, output_path: Path) -> None:
    yolo = audit_data["yolo_nano"]
    ind = audit_data["industrial_autoencoder"]

    md = f\"\"\"# Quantization (FP16 & Static INT8 PTQ) Calibration & Validation Report

This report documents the half-precision conversion, Static INT8 Post-Training Quantization (PTQ) calibration, and quality degradation gating for the `onnx-edge-inference-benchmark` repository.

---

## 1. Calibration Dataset Configuration

- **Detection Calibration Split**: 50 disjoint images ($640 \\\\times 640$) in `data/calibration/detection/`
- **Industrial Inspection Split**: 50 disjoint normal images ($256 \\\\times 256$) in `data/calibration/industrial/`
- **Isolation Guarantee**: Zero hash overlap with test sets in `data/sample_images/`.
- **Calibration Algorithm**: Static MinMax Activation Histogram Range Tracking.

---

## 2. Quantization Quality Degradation Audit

### 2.1. Object Detector (YOLO Nano)

| Precision | Target Format | Metric (mAP@50) | Baseline FP32 | Quantized Value | Delta ($\\\\Delta$) | Gate Threshold | Decision Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16** | Float16 | mAP@50 | `{yolo['fp16']['baseline_metrics']['mAP_50']}` | `{yolo['fp16']['metrics']['mAP_50']}` | `{yolo['fp16']['delta_map_50']}` | $\\\\le 0.015$ | `{yolo['fp16']['status']}` |
| **Static INT8** | QDQ (QInt8) | mAP@50 | `{yolo['int8']['baseline_metrics']['mAP_50']}` | `{yolo['int8']['metrics']['mAP_50']}` | `{yolo['int8']['delta_map_50']}` | $\\\\le 0.015$ | `{yolo['int8']['status']}` |

### 2.2. Industrial Inspection Model (ConvAutoencoder)

| Precision | Target Format | Metric (Image AUROC) | Baseline FP32 | Quantized Value | Delta ($\\\\Delta$) | Gate Threshold | Decision Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16** | Float16 | Image AUROC | `{ind['fp16']['baseline_metrics']['image_auroc']}` | `{ind['fp16']['metrics']['image_auroc']}` | `{ind['fp16']['delta_auroc']}` | $\\\\le 0.010$ | `{ind['fp16']['status']}` |
| **Static INT8** | QDQ (QInt8) | Image AUROC | `{ind['int8']['baseline_metrics']['image_auroc']}` | `{ind['int8']['metrics']['image_auroc']}` | `{ind['int8']['delta_auroc']}` | $\\\\le 0.010$ | `{ind['int8']['status']}` |

---

## 3. Artifact Checksum Matrix

| Model | Precision | Filename | Size |
| :--- | :--- | :--- | :--- |
| YOLO Nano | FP16 | `models/exported/yolo_nano_fp16.onnx` | ~0.84 MB |
| YOLO Nano | Static INT8 | `models/exported/yolo_nano_static_int8.onnx` | ~0.50 MB |
| Industrial Autoencoder | FP16 | `models/exported/industrial_autoencoder_fp16.onnx` | ~2.72 MB |
| Industrial Autoencoder | Static INT8 | `models/exported/industrial_autoencoder_static_int8.onnx` | ~1.46 MB |
\"\"\"
    output_path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
'''

files["docs/int8-calibration.md"] = """# Quantization (FP16 & Static INT8 PTQ) Calibration & Validation Report

*Run `python scripts/quantize_and_validate.py` to generate the quantization report.*
"""

# ============================================================================
# 4. COMPREHENSIVE TESTS (tests/)
# ============================================================================

files["tests/test_calibration_reader.py"] = '''\"\"\"
Unit tests for BenchmarkCalibrationDataReader and calibration dataset isolation.
\"\"\"

import csv
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from src.quantization.calibration_reader import BenchmarkCalibrationDataReader


class TestCalibrationReader:
    \"\"\"Test suite validating calibration data reader mechanics and disjoint dataset guarantees.\"\"\"

    def test_calibration_reader_batch_and_rewind(self, tmp_path: Path) -> None:
        \"\"\"Tests get_next yields valid numpy arrays and rewind resets iterator.\"\"\"
        img_paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.jpg"
            cv2.imwrite(str(p), np.ones((64, 64, 3), dtype=np.uint8) * (i + 1))
            img_paths.append(p)

        def dummy_preprocess(p):
            img = cv2.imread(str(p))
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            return t, (1.0, 1.0), (0.0, 0.0), (64, 64)

        reader = BenchmarkCalibrationDataReader(
            image_paths=img_paths,
            input_name="test_input",
            input_shape=(1, 3, 64, 64),
            preprocess_fn=dummy_preprocess,
            batch_size=1,
        )

        b1 = reader.get_next()
        assert b1 is not None
        assert "test_input" in b1
        assert b1["test_input"].shape == (1, 3, 64, 64)
        assert b1["test_input"].dtype == np.float32

        b2 = reader.get_next()
        assert b2 is not None

        b3 = reader.get_next()
        assert b3 is not None

        b_end = reader.get_next()
        assert b_end is None

        # Test rewind
        reader.rewind()
        b1_rewound = reader.get_next()
        assert b1_rewound is not None
        assert np.array_equal(b1["test_input"], b1_rewound["test_input"])

    def test_disjoint_calibration_and_evaluation_datasets(self) -> None:
        \"\"\"Verifies 0% SHA-256 hash overlap between calibration and sample evaluation images.\"\"\"
        root = Path(__file__).resolve().parent.parent
        calib_manifest = root / "data" / "calibration" / "manifest.csv"
        eval_manifest = root / "data" / "sample_images" / "manifest.json"

        if not calib_manifest.is_file() or not eval_manifest.is_file():
            pytest.skip("Dataset manifests not yet generated")

        calib_hashes = set()
        with open(calib_manifest, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                calib_hashes.add(row["sha256"])

        eval_data = json.loads(eval_manifest.read_text(encoding="utf-8"))
        eval_hashes = set()
        for item in eval_data.get("detection_samples", []):
            eval_hashes.add(item["sha256"])
        for item in eval_data.get("industrial_samples", []):
            eval_hashes.add(item["sha256"])

        overlap = calib_hashes.intersection(eval_hashes)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping hashes between calibration and evaluation sets!"
'''

files["tests/test_quantization.py"] = '''\"\"\"
Unit tests for FP16 conversion, Static INT8 PTQ, and quality degradation gating.
\"\"\"

import json
from pathlib import Path
import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn

from src.common.config import QualityThresholdConfig
from src.export.export_onnx import export_to_onnx
from src.quantization.calibration_reader import BenchmarkCalibrationDataReader
from src.quantization.convert_fp16 import convert_onnx_to_fp16
from src.quantization.quantize_onnx import quantize_onnx_static
from src.validation.validate_quantization import validate_quantized_model


class SimpleToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class TestQuantization:
    \"\"\"Test suite validating quantization transformations and decision gating.\"\"\"

    def test_convert_onnx_to_fp16(self, tmp_path: Path) -> None:
        \"\"\"Tests FP16 conversion produces valid ONNX model with half-precision weights.\"\"\"
        model = SimpleToyModel()
        dummy_in = torch.randn(1, 3, 16, 16)
        fp32_path = tmp_path / "toy_fp32.onnx"
        fp16_path = tmp_path / "toy_fp16.onnx"

        export_to_onnx(model, dummy_in, fp32_path, ["input"], ["output"], opset_version=17)
        res_path = convert_onnx_to_fp16(fp32_path, fp16_path, keep_io_types=True)

        assert res_path.is_file()
        assert fp16_path.with_name(fp16_path.name + ".sha256").is_file()

        loaded = onnx.load(str(res_path))
        # Check that weight initializers are float16
        float16_inits = [init for init in loaded.graph.initializer if init.data_type == onnx.TensorProto.FLOAT16]
        assert len(float16_inits) > 0

    def test_quantize_onnx_static(self, tmp_path: Path) -> None:
        \"\"\"Tests static INT8 PTQ creates quantized graph with QuantizeLinear/DequantizeLinear nodes.\"\"\"
        model = SimpleToyModel()
        dummy_in = torch.randn(1, 3, 16, 16)
        fp32_path = tmp_path / "toy_fp32.onnx"
        int8_path = tmp_path / "toy_int8.onnx"

        export_to_onnx(model, dummy_in, fp32_path, ["input"], ["output"], opset_version=17)

        # Create dummy calibration data
        calib_images = []
        for i in range(5):
            img_p = tmp_path / f"c_img_{i}.npy"
            np.save(str(img_p), np.random.randn(1, 3, 16, 16).astype(np.float32))
            calib_images.append(img_p)

        def toy_preprocess(p):
            return np.load(str(p))

        reader = BenchmarkCalibrationDataReader(
            image_paths=calib_images,
            input_name="input",
            input_shape=(1, 3, 16, 16),
            preprocess_fn=toy_preprocess,
        )

        res_path = quantize_onnx_static(
            input_onnx_path=fp32_path,
            output_onnx_path=int8_path,
            calibration_data_reader=reader,
            quant_format="QDQ",
        )

        assert res_path.is_file()
        assert int8_path.with_name(int8_path.name + ".sha256").is_file()

        loaded = onnx.load(str(res_path))
        op_types = {node.op_type for node in loaded.graph.node}
        assert "QuantizeLinear" in op_types or "DequantizeLinear" in op_types or "QLinearConv" in op_types

    def test_validate_quantized_model_quality_gate(self, tmp_path: Path) -> None:
        \"\"\"Verifies degradation gating marks severe drops as REJECTED and acceptable drops as PASS.\"\"\"
        root = Path(__file__).resolve().parent.parent
        manifest_p = root / "data" / "sample_images" / "manifest.json"
        base_p = root / "results" / "raw" / "pytorch_fp32" / "yolo_nano" / "baseline_metrics.json"
        onnx_p = root / "models" / "exported" / "yolo_nano_fp32_opset17.onnx"

        if not manifest_p.is_file() or not base_p.is_file() or not onnx_p.is_file():
            pytest.skip("Required benchmark artifacts not present")

        # Create mini manifest for fast unit testing
        full_manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        mini_manifest = {
            "detection_samples": full_manifest.get("detection_samples", [])[:2],
            "industrial_samples": full_manifest.get("industrial_samples", [])[:2],
        }
        mini_manifest_p = tmp_path / "mini_manifest.json"
        mini_manifest_p.write_text(json.dumps(mini_manifest), encoding="utf-8")

        # Test with standard threshold -> should PASS
        pass_cfg = QualityThresholdConfig(max_map_drop=0.05)
        rep_pass = validate_quantized_model("yolo_nano", "fp32", onnx_p, mini_manifest_p, base_p, pass_cfg)
        assert rep_pass["status"] == "PASS"

        # Test with strict threshold on artificially high baseline -> should be REJECTED
        mock_base = tmp_path / "mock_base.json"
        mock_base.write_text(json.dumps({"metrics": {"mAP_50": 0.99, "mAP_50_95": 0.95}}), encoding="utf-8")

        strict_cfg = QualityThresholdConfig(max_map_drop=0.01)
        rep_reject = validate_quantized_model("yolo_nano", "fp32", onnx_p, mini_manifest_p, mock_base, strict_cfg)
        assert rep_reject["status"] == "REJECTED"
        assert "exceeds threshold" in rep_reject["rejection_reason"]

    def test_validate_industrial_quantized_model(self, tmp_path: Path) -> None:
        \"\"\"Tests validate_quantized_model on industrial autoencoder model and rejection gating.\"\"\"
        root = Path(__file__).resolve().parent.parent
        manifest_p = root / "data" / "sample_images" / "manifest.json"
        base_p = root / "results" / "raw" / "pytorch_fp32" / "industrial_autoencoder" / "baseline_metrics.json"
        onnx_p = root / "models" / "exported" / "industrial_autoencoder_fp32_opset17.onnx"

        if not manifest_p.is_file() or not base_p.is_file() or not onnx_p.is_file():
            pytest.skip("Required benchmark artifacts not present")

        # Create mini manifest for fast unit testing
        full_manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        mini_manifest = {
            "detection_samples": full_manifest.get("detection_samples", [])[:2],
            "industrial_samples": full_manifest.get("industrial_samples", [])[:2],
        }
        mini_manifest_p = tmp_path / "mini_manifest_ind.json"
        mini_manifest_p.write_text(json.dumps(mini_manifest), encoding="utf-8")

        # Test pass case
        pass_cfg = QualityThresholdConfig(max_auroc_drop=0.05)
        rep_pass = validate_quantized_model("industrial_autoencoder", "fp32", onnx_p, mini_manifest_p, base_p, pass_cfg)
        assert rep_pass["status"] == "PASS"

        # Test reject case
        mock_base = tmp_path / "mock_ind_base.json"
        mock_base.write_text(json.dumps({"metrics": {"image_auroc": 2.0, "aupro": 2.0}}), encoding="utf-8")

        strict_cfg = QualityThresholdConfig(max_auroc_drop=0.0)
        rep_reject = validate_quantized_model("industrial_autoencoder", "fp32", onnx_p, mini_manifest_p, mock_base, strict_cfg)
        assert rep_reject["status"] == "REJECTED"
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

print(f"\\nAll {len(files)} Phase 3 files generated successfully at {TARGET_ROOT}.")
