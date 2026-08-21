"""
Unit tests for BenchmarkSuite execution, JSON schema conformity, and CSV updating.
"""

import csv
import json
from pathlib import Path
import pytest
import torch

from src.benchmarking.benchmark_suite import BenchmarkSuite, init_master_csv
from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.pytorch_runtime import PyTorchRuntime


class TestBenchmarkSuite:
    """Test suite validating single run execution and manifest serialization."""

    def test_benchmark_suite_single_run_and_manifest(self, tmp_path: Path) -> None:
        """Tests run_single_configuration produces valid run.json and appends to runs.csv."""
        root = Path(__file__).resolve().parent.parent
        sample_dir = root / "data" / "sample_images" / "detection"
        samples = list(sample_dir.rglob("*.jpg"))

        if not samples:
            pytest.skip("Detection sample images not found")

        suite = BenchmarkSuite(
            results_dir=tmp_path / "results",
            warmup_model=2,
            timed_model=5,
            warmup_e2e=2,
            timed_e2e=3,
            stability_sessions=3,
        )

        adapter = YOLOAdapter()
        runtime = PyTorchRuntime(adapter.get_pytorch_model(), device="cpu")
        runtime.load()

        dummy_in = torch.randn(1, 3, 640, 640)
        rec = suite.run_single_configuration(
            model_name="yolo_nano",
            runtime_name="PyTorch",
            precision="fp32",
            runtime=runtime,
            input_tensor=dummy_in,
            sample_image_paths=samples[:2],
            adapter=adapter,
            task="detection",
            quality_metric="mAP_50",
            quality_value=0.0099,
        )

        assert rec["status"] in ["PASS", "UNSTABLE_LATENCY"]
        assert "model_path_latency_ms" in rec
        assert "end_to_end_latency_ms" in rec
        assert rec["model_metadata"]["model_size_mb"] > 0.0
        assert rec["model_path_latency_ms"]["model_throughput_fps"] > 0.0
        assert rec["end_to_end_latency_ms"]["e2e_throughput_fps"] > 0.0

        # Check run.json exists
        run_json = tmp_path / "results" / "raw" / rec["run_id"] / "run.json"
        assert run_json.is_file()
        data = json.loads(run_json.read_text(encoding="utf-8"))
        assert data["run_id"] == rec["run_id"]
        assert data["model_metadata"]["model_size_mb"] > 0.0

        # Check runs.csv
        runs_csv = tmp_path / "results" / "runs.csv"
        assert runs_csv.is_file()
        with open(runs_csv, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            assert len(reader) >= 2  # Header + 1 row
            assert reader[1][0] == rec["run_id"]
            # Check non-zero model size in CSV column 19 (index 19 in 0-indexed)
            assert float(reader[1][19]) > 0.0

        runtime.cleanup()
