"""
Unit tests for the Pydantic v2 configuration subsystem.
"""

from pathlib import Path
import pytest
from pydantic import ValidationError

from src.common.config import (
    BenchmarkConfig,
    MasterConfig,
    ModelConfig,
    PathConfig,
    QualityThresholdConfig,
    RuntimeConfig,
    load_config,
    save_config,
)


class TestConfig:
    """Test suite validating configuration schemas and serialization."""

    def test_load_all_existing_yaml_configs(self) -> None:
        """Verifies all default YAML configuration files in configs/ parse and validate cleanly."""
        root_dir = Path(__file__).resolve().parent.parent
        configs_dir = root_dir / "configs"
        yaml_files = list(configs_dir.rglob("*.yaml"))

        assert len(yaml_files) >= 7, f"Expected at least 7 YAML config files, found {len(yaml_files)}"

        for yf in yaml_files:
            cfg = load_config(yf)
            assert isinstance(cfg, MasterConfig), f"Failed to load {yf}"
            assert isinstance(cfg.benchmark, BenchmarkConfig)
            assert isinstance(cfg.paths, PathConfig)

    def test_model_config_validation(self) -> None:
        """Validates ModelConfig fields and input_shape constraints."""
        # Valid model config
        m = ModelConfig(
            name="yolo_nano",
            task="detection",
            input_shape=[1, 3, 640, 640],
            weights_path="models/weights/yolov8n.pt",
        )
        assert m.opset == 17
        assert m.input_shape == [1, 3, 640, 640]

        # Invalid shape: 3 dimensions instead of 4
        with pytest.raises(ValidationError):
            ModelConfig(
                name="invalid",
                task="detection",
                input_shape=[3, 640, 640],
                weights_path="test.pt",
            )

        # Invalid shape: non-positive dimension
        with pytest.raises(ValidationError):
            ModelConfig(
                name="invalid",
                task="detection",
                input_shape=[1, 3, -640, 640],
                weights_path="test.pt",
            )

        # Invalid task name
        with pytest.raises(ValidationError):
            ModelConfig(
                name="invalid",
                task="segmentation",  # type: ignore
                input_shape=[1, 3, 640, 640],
                weights_path="test.pt",
            )

    def test_runtime_config_validation(self) -> None:
        """Validates RuntimeConfig allowable backends and precisions."""
        r = RuntimeConfig(engine="ort_cuda", precision="fp16")
        assert r.provider == "CPUExecutionProvider"
        assert r.io_binding is True

        # Invalid engine
        with pytest.raises(ValidationError):
            RuntimeConfig(engine="vllm", precision="fp16")  # type: ignore

        # Invalid precision
        with pytest.raises(ValidationError):
            RuntimeConfig(engine="tensorrt", precision="int4")  # type: ignore

    def test_benchmark_config_validation(self) -> None:
        """Validates BenchmarkConfig bounds."""
        b = BenchmarkConfig(batch_size=4, warmup_iterations=10, timed_iterations=100)
        assert b.batch_size == 4

        # Zero or negative batch size
        with pytest.raises(ValidationError):
            BenchmarkConfig(batch_size=0)

        # Negative cooldown
        with pytest.raises(ValidationError):
            BenchmarkConfig(cooldown_seconds=-1.0)

    def test_config_serialization_roundtrip(self, tmp_path: Path, temp_yaml_file: Path) -> None:
        """Tests load -> save -> load roundtrip fidelity."""
        cfg1 = load_config(temp_yaml_file)

        saved_path = tmp_path / "roundtrip.yaml"
        save_config(cfg1, saved_path)

        cfg2 = load_config(saved_path)
        assert cfg1.model_dump() == cfg2.model_dump()

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        """Ensures load_config raises FileNotFoundError for missing paths."""
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.yaml")
