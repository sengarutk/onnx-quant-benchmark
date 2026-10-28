"""
Pytest shared fixtures providing configuration samples and test files.
"""

from pathlib import Path
from typing import Any, Dict
import numpy as np
import pytest
import yaml


@pytest.fixture
def valid_config_dict() -> Dict[str, Any]:
    """Returns a valid dictionary matching the MasterConfig schema."""
    return {
        "model": {
            "name": "yolo_nano",
            "task": "detection",
            "input_shape": [1, 3, 640, 640],
            "weights_path": "models/weights/yolo_nano_trained.pt",
            "opset": 17,
            "class_names": ["rectangle", "ellipse"],
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
        },
        "runtime": {
            "engine": "ort_cuda",
            "precision": "fp32",
            "provider": "CUDAExecutionProvider",
            "device_id": 0,
            "intra_op_num_threads": 8,
            "inter_op_num_threads": 1,
            "io_binding": True,
            "workspace_gb": 4.0,
            "dynamic_shapes": False,
        },
        "benchmark": {
            "batch_size": 1,
            "warmup_iterations": 50,
            "timed_iterations": 300,
            "stability_sessions": 3,
            "cooldown_seconds": 2.0,
            "device_id": 0,
        },
        "quality_thresholds": {
            "max_rel_error": 1e-4,
            "min_cosine_similarity": 0.99999,
            "max_map_drop": 0.005,
            "max_auroc_drop": 0.005,
        },
        "paths": {
            "weights_dir": "models/weights",
            "exported_dir": "models/exported",
            "engines_dir": "models/engines",
            "results_raw": "results/raw",
            "results_manifests": "results/manifests",
            "results_tables": "results/tables",
            "results_figures": "results/figures",
            "results_profiles": "results/profiles",
            "calibration_data": "data/calibration",
            "sample_data": "data/sample_images",
        },
    }


@pytest.fixture
def temp_yaml_file(tmp_path: Path, valid_config_dict: Dict[str, Any]) -> Path:
    """Writes a valid configuration to a temporary YAML file."""
    yaml_file = tmp_path / "test_config.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(valid_config_dict, f)
    return yaml_file


@pytest.fixture
def temp_dummy_file(tmp_path: Path) -> Path:
    """Creates a temporary file with a known byte sequence for hashing tests."""
    file_path = tmp_path / "dummy.bin"
    file_path.write_bytes(b"Benchmark test data for SHA-256 verification\n")
    return file_path


# ---------------------------------------------------------------------------------------------------------------
# Synthetic benchmark session (run.json records in the exact schema written by BenchmarkSuite)
# ---------------------------------------------------------------------------------------------------------------
import json  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def make_run_record(model: str, runtime: str, provider: str, precision: str, p50: float, session_id: str,
                    size_mb: float, quality: float, ref_quality: float, stable: bool = True,
                    numerical_ok: bool = True, cuda: bool = False) -> Dict[str, Any]:
    metric = "mAP_50" if model == "yolo_nano" else "image_auroc"
    file_src = runtime != "PyTorch"
    p50s = [p50 * f for f in (0.98, 1.0, 1.02, 0.99, 1.01)] if stable else [p50 * f for f in (0.8, 1.0, 1.25, 0.9, 1.1)]
    cv = float(np.std(p50s) / np.mean(p50s))
    return {
        "run_id": f"{model}_{runtime}_{precision}_{provider}", "session_id": session_id,
        "timestamp": "2026-10-26T12:00:00+00:00",
        "status": "PASS" if (stable and numerical_ok) else "UNSTABLE_LATENCY",
        "model_metadata": {"model": model, "task": "detection" if model == "yolo_nano" else "anomaly_detection",
                           "precision": precision, "batch_size": 1, "input_shape": [1, 3, 640, 640],
                           "model_size_mb": size_mb, "model_size_source": "serialized_file" if file_src else "parameters_and_buffers",
                           "model_file": None, "model_sha256": None},
        "runtime_metadata": {"runtime": runtime, "provider": provider},
        "protocol": {},
        "environment_manifest": {},
        "model_path_latency_ms": {"p50_ms": p50, "p90_ms": p50 * 1.1, "p95_ms": p50 * 1.2, "p99_ms": p50 * 1.5,
                                  "mean_ms": p50 * 1.05, "std_ms": p50 * 0.05, "min_ms": p50 * 0.9, "max_ms": p50 * 2,
                                  "throughput_fps": 1000.0 / p50, "p50_ci95_ms": [p50 * 0.98, p50 * 1.02],
                                  "p95_ci95_ms": [p50 * 1.15, p50 * 1.25], "samples": 1500},
        "end_to_end_latency_ms": {"p50": p50 + 6, "p90": p50 + 7, "p95": p50 + 8, "p99": p50 + 9, "mean": p50 + 6.5,
                                  "std": 0.5, "throughput_fps": 1000.0 / (p50 + 6), "iterations": 200,
                                  "stages_p50": {"preprocess": 4.0, "inference": p50 + 0.5, "postprocess": 1.5},
                                  "stages_mean": {"preprocess": 4.1, "inference": p50 + 0.6, "postprocess": 1.6}},
        "memory_utilization": {"peak_vram_allocated_mb": 30.0 if cuda else 0.0, "peak_vram_reserved_mb": 44.0 if cuda else 0.0,
                               "process_rss_mb": 900.0, "nvml_gpu_memory_used_mb": 0.0},
        "stability_audit": {"session_p50_values": p50s, "session_p95_values": [p * 1.2 for p in p50s], "cv_p50": cv,
                            "cv_p50_mad_filtered": cv, "initial_cv_p50": cv, "supplemental_sessions_run": False,
                            "stable": stable, "stability_warning": None, "total_sessions": len(p50s)},
        "numerical_checks": {"max_abs_error": 1e-5 if numerical_ok else 1.0, "mean_abs_error": 1e-6,
                             "max_rel_error": 1e-6 if numerical_ok else 0.5, "cosine_similarity": 1.0 if numerical_ok else 0.9,
                             "per_output": {}, "nan_count": 0, "inf_count": 0, "thresholds": {}, "passed": numerical_ok},
        "quality_audit": {"metric": metric, "value": quality, "reference_value": ref_quality, "delta": quality - ref_quality,
                          "max_allowed_drop": 0.02, "all_metrics": {metric: quality}, "reference_metrics": {metric: ref_quality},
                          "passed": quality - ref_quality >= -0.02},
    }


@pytest.fixture
def fake_session(tmp_path: Path):
    """Writes a 10-configuration session to tmp_path/raw/<session_id>/ and returns (raw_dir, session_id, records)."""
    sid = "session_20261026T120000Z"
    specs = [
        ("yolo_nano", "PyTorch", "PyTorch_CPU", "fp32", 20.0, 7.3, 0.80, True, True, False),
        ("yolo_nano", "PyTorch", "PyTorch_CUDA:0", "fp32", 1.0, 7.3, 0.80, False, True, True),
        ("yolo_nano", "ORT_CPU", "CPUExecutionProvider", "fp32", 9.0, 7.3, 0.80, True, True, False),
        ("yolo_nano", "ORT_CPU", "CPUExecutionProvider", "fp16", 10.0, 3.7, 0.80, True, True, False),
        ("yolo_nano", "ORT_CPU", "CPUExecutionProvider", "int8", 7.0, 1.9, 0.79, True, True, False),
        ("industrial_autoencoder", "PyTorch", "PyTorch_CPU", "fp32", 8.0, 5.4, 0.90, True, True, False),
        ("industrial_autoencoder", "PyTorch", "PyTorch_CUDA:0", "fp32", 1.2, 5.4, 0.90, True, True, True),
        ("industrial_autoencoder", "ORT_CPU", "CPUExecutionProvider", "fp32", 7.0, 5.4, 0.90, True, True, False),
        ("industrial_autoencoder", "ORT_CPU", "CPUExecutionProvider", "fp16", 9.0, 2.7, 0.90, True, True, False),
        ("industrial_autoencoder", "ORT_CPU", "CPUExecutionProvider", "int8", 6.5, 1.4, 0.85, True, False, False),
    ]
    raw = tmp_path / "raw"
    records = []
    for model, rt, prov, prec, p50, size, q, stable, num_ok, cuda in specs:
        ref = 0.80 if model == "yolo_nano" else 0.90
        rec = make_run_record(model, rt, prov, prec, p50, sid, size, q, ref, stable, num_ok, cuda)
        d = raw / sid / rec["run_id"]
        d.mkdir(parents=True)
        (d / "run.json").write_text(json.dumps(rec), encoding="utf-8")
        records.append(rec)
    (raw / sid / "session.json").write_text(json.dumps({"session_id": sid, "quick": False, "executed": [], "skipped": []}))
    return raw, sid, records


def artifacts_available() -> bool:
    return all(p.is_file() for p in (
        PROJECT_ROOT / "models" / "weights" / "yolo_nano_trained.pt",
        PROJECT_ROOT / "models" / "weights" / "industrial_autoencoder_trained.pt",
        PROJECT_ROOT / "models" / "exported" / "yolo_nano_static_int8.onnx",
        PROJECT_ROOT / "data" / "sample_images" / "manifest.json",
        PROJECT_ROOT / "data" / "calibration" / "manifest.json",
    ))


requires_artifacts = pytest.mark.skipif(not artifacts_available(), reason="run `make pipeline` (or data/train/export/quantize) first")
