"""
Pytest shared fixtures providing configuration samples and test files.
"""

from pathlib import Path
from typing import Any, Dict
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
            "weights_path": "models/weights/yolov8n.pt",
            "opset": 17,
            "class_names": ["person", "car"],
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
            "collect_nvml": True,
            "device_id": 0,
        },
        "quality_thresholds": {
            "max_abs_error": 1e-4,
            "mean_abs_error": 1e-5,
            "max_map_drop": 0.005,
            "max_auroc_drop": 0.005,
            "max_aupro_drop": 0.010,
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
