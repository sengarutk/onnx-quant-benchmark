#!/usr/bin/env python3
"""
FP16 conversion and static INT8 post-training quantization of the exported FP32 ONNX graphs, followed by
test-split quality gating against the PyTorch FP32 baseline. Renders docs/int8-calibration.md and
models/exported/quantization_manifest.json.
"""

import csv
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.memory import get_artifact_size_mb
from src.common.hashes import compute_file_sha256
from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.models.preprocess import preprocess_detection_image, preprocess_industrial_image
from src.quantization.calibration_reader import BenchmarkCalibrationDataReader
from src.quantization.convert_fp16 import convert_onnx_to_fp16
from src.quantization.quantize_onnx import DEFAULT_QUANTIZED_OPS, quantize_onnx_static
from src.validation.validate_quantization import validate_quantized_model

logger = setup_logger("quantize_and_validate")

QUANT_SETTINGS = {
    "format": "QDQ",
    "calibrate_method": "selected per model on the calibration split from CALIBRATION_METHODS",
    "per_channel": True,
    "weight_type": "QInt8",
    "activation_type": "QUInt8",
    "symmetric_weights": True,
    "symmetric_activations": True,
    "op_types": DEFAULT_QUANTIZED_OPS,
}


CALIBRATION_METHODS = ["MinMax", "Percentile", "Entropy"]


def calibration_split_score(model: str, onnx_path: Path, fp32_path: Path, calib_json: Path) -> float:
    """
    Selection criterion computed on the CALIBRATION split only (never the test split):
    detector -> mAP@50 on the labeled calibration images; autoencoder -> negative mean relative error of the
    anomaly map vs the FP32 graph on the (normal) calibration textures.
    """
    import numpy as np
    import onnxruntime as ort
    from src.data.writers import load_manifest
    from src.validation.numerical_equivalence import evaluate_onnx_dataset

    if model == "yolo_nano":
        return float(evaluate_onnx_dataset(model, onnx_path, calib_json)["metrics"]["mAP_50"])
    cand = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ref = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    errs = []
    for s in load_manifest(calib_json)["industrial_samples"]:
        x = preprocess_industrial_image(PROJECT_ROOT / s["image_path"]).numpy()
        a = ref.run(None, {"input": x})[1]
        b = cand.run(None, {"input": x})[1]
        errs.append(float(np.abs(a - b).mean() / (np.abs(a).mean() + 1e-12)))
    return -float(np.mean(errs))


def main() -> None:
    seed_everything(42)
    exp = PROJECT_ROOT / "models" / "exported"
    test_manifest = PROJECT_ROOT / "data" / "sample_images" / "manifest.json"
    calib_csv = PROJECT_ROOT / "data" / "calibration" / "manifest.csv"
    calib_json = PROJECT_ROOT / "data" / "calibration" / "manifest.json"
    if not calib_csv.is_file():
        raise FileNotFoundError("Calibration split missing: run scripts/prepare_calibration_data.py")

    calib = {"detection": [], "industrial": []}
    with open(calib_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            calib[row["category"]].append(PROJECT_ROOT / row["path"])

    specs = {
        "yolo_nano": ("images", (1, 3, 640, 640), preprocess_detection_image, calib["detection"]),
        "industrial_autoencoder": ("input", (1, 3, 256, 256), preprocess_industrial_image, calib["industrial"]),
    }
    report = {"settings": QUANT_SETTINGS, "models": {}}

    for model, (in_name, shape, pre_fn, paths) in specs.items():
        fp32 = exp / f"{model}_fp32_opset17.onnx"
        baseline = PROJECT_ROOT / "results" / "baselines" / f"{model}.json"
        fp16 = convert_onnx_to_fp16(fp32, exp / f"{model}_fp16.onnx")
        # INT8: quantize with every calibration method, select on the calibration split, keep all for transparency
        candidates = {}
        for method in CALIBRATION_METHODS:
            path = quantize_onnx_static(
                fp32, exp / f"{model}_int8_{method.lower()}.onnx",
                BenchmarkCalibrationDataReader(paths, in_name, shape, pre_fn),
                quant_format=QUANT_SETTINGS["format"], calibrate_method=method,
                per_channel=QUANT_SETTINGS["per_channel"], weight_type=QUANT_SETTINGS["weight_type"],
                activation_type=QUANT_SETTINGS["activation_type"],
            )
            score = calibration_split_score(model, path, fp32, calib_json)
            test_v = validate_quantized_model(model, "int8", path, test_manifest, baseline)
            candidates[method] = {"file": path.name, "calibration_split_score": score,
                                  "test_metric": test_v["metrics"][test_v["gated_metric"]], "test_delta": test_v["delta"],
                                  "test_status": test_v["status"]}
            logger.info(f"{model} INT8/{method}: calibration-split score {score:.4f} | test delta {test_v['delta']:+.4f}")
        selected = max(CALIBRATION_METHODS, key=lambda m: candidates[m]["calibration_split_score"])
        int8 = exp / f"{model}_static_int8.onnx"
        shutil.copy2(exp / candidates[selected]["file"], int8)
        (int8.with_name(int8.name + ".sha256")).write_text(f"{compute_file_sha256(int8)}  {int8.name}\n", encoding="utf-8")
        entry = {"calibration_images": len(paths), "fp32_size_mb": round(get_artifact_size_mb(fp32), 3),
                 "int8_calibration_method": selected, "int8_method_ablation": candidates}
        for prec, path in (("fp16", fp16), ("int8", int8)):
            v = validate_quantized_model(model, prec, path, test_manifest, baseline)
            entry[prec] = {"file": path.name, "size_mb": round(get_artifact_size_mb(path), 3), "status": v["status"],
                           "metric": v["gated_metric"], "value": v["metrics"][v["gated_metric"]],
                           "baseline": v["baseline_metrics"][v["gated_metric"]], "delta": v["delta"],
                           "max_allowed_drop": v["max_allowed_drop"], "metrics": v["metrics"]}
            logger.info(f"{model} {prec.upper()}: {v['status']} ({v['gated_metric']} {entry[prec]['value']:.4f}, delta {v['delta']:+.4f})")
        report["models"][model] = entry

    (exp / "quantization_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (PROJECT_ROOT / "results" / "quantization_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    render_doc(report, PROJECT_ROOT / "docs" / "int8-calibration.md")


def render_doc(report: dict, path: Path) -> None:
    s = report["settings"]
    md = ["# FP16 Conversion & Static INT8 PTQ Report", "",
          "Generated by `scripts/quantize_and_validate.py`. Quality is measured with ONNX Runtime (CPU) on the full "
          "synthetic test split and compared with the PyTorch FP32 baseline (`results/baselines/`).", "",
          "## 1. Settings", "",
          f"- **INT8**: ONNX Runtime `quantize_static`, {s['format']} format, calibration method selected per model on the "
          f"calibration split among {', '.join(CALIBRATION_METHODS)} (detector: calibration-split mAP@50; autoencoder: "
          "anomaly-map relative error vs FP32), "
          f"{'per-channel' if s['per_channel'] else 'per-tensor'} {s['weight_type']} weights, {s['activation_type']} activations "
          f"(symmetric), quantized op types: {', '.join(s['op_types'])}.",
          "- **FP16**: `onnxconverter_common.float16.convert_float_to_float16` with FP32 graph inputs/outputs.",
          f"- **Calibration data**: the labeled calibration split (`data/calibration/`, "
          f"{report['models']['yolo_nano']['calibration_images']} detection / "
          f"{report['models']['industrial_autoencoder']['calibration_images']} normal industrial images), generated from a "
          "seed range disjoint from the test split (SHA-256 overlap verified to be 0).", "",
          "## 2. Results", "",
          "| Model | Precision | File | Size (MB) | vs FP32 | Gated metric | Value | FP32 baseline | Delta | Allowed drop | Status |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for model, e in report["models"].items():
        for prec in ("fp16", "int8"):
            x = e[prec]
            md.append(f"| `{model}` | {prec.upper()} | `{x['file']}` | {x['size_mb']:.2f} | "
                      f"{(x['size_mb'] / e['fp32_size_mb'] - 1) * 100:+.1f}% | {x['metric']} | {x['value']:.4f} | "
                      f"{x['baseline']:.4f} | {x['delta']:+.4f} | {x['max_allowed_drop']} | {x['status']} |")
    md += ["", "## 3. INT8 calibration-method ablation", "",
           "The method is selected on the calibration split (column 3); test-split results of every method are listed "
           "for transparency but were not used for the choice.", "",
           "| Model | Method | Calibration-split score | Test metric | Test delta vs FP32 | Test gate | Selected |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for model, e in report["models"].items():
        for method, c in e["int8_method_ablation"].items():
            md.append(f"| `{model}` | {method} | {c['calibration_split_score']:.4f} | {c['test_metric']:.4f} | "
                      f"{c['test_delta']:+.4f} | {c['test_status']} | {'yes' if method == e['int8_calibration_method'] else ''} |")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
