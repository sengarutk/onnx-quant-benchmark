"""
Unit tests for ModelPathTimer and EndToEndTimer dual-regime latency timing.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.benchmarking.timer import EndToEndTimer, ModelPathTimer
from src.models.industrial_model_adapter import IndustrialModelAdapter
from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.pytorch_runtime import PyTorchRuntime


class TestTimer:
    """Test suite validating timer accuracy, percentile monotonicity, and pipeline execution."""

    def test_model_path_timer_percentile_monotonicity(self) -> None:
        """Tests that model-path timer percentiles obey p50 <= p90 <= p95 <= p99."""
        adapter = YOLOAdapter()
        runtime = PyTorchRuntime(adapter.get_pytorch_model(), device="cpu")
        runtime.load()

        timer = ModelPathTimer(warmup_iterations=5, timed_iterations=20, device="cpu")
        dummy_in = torch.randn(1, 3, 640, 640)
        res = timer.benchmark_device(runtime, dummy_in)

        assert res["p50_ms"] > 0.0
        assert res["p50_ms"] <= res["p90_ms"]
        assert res["p90_ms"] <= res["p95_ms"]
        assert res["p95_ms"] <= res["p99_ms"]
        assert res["throughput_fps"] > 0.0
        assert res["iterations"] == 20.0
        runtime.cleanup()

    def test_end_to_end_timer_detection(self) -> None:
        """Tests EndToEndTimer on detection sample images."""
        root = Path(__file__).resolve().parent.parent
        sample_dir = root / "data" / "sample_images" / "detection"
        samples = list(sample_dir.rglob("*.jpg"))

        if not samples:
            pytest.skip("Detection sample images not found")

        adapter = YOLOAdapter()
        runtime = PyTorchRuntime(adapter.get_pytorch_model(), device="cpu")
        runtime.load()

        timer = EndToEndTimer(warmup_iterations=2, timed_iterations=5)
        res = timer.benchmark_e2e_detection(runtime, samples, adapter)

        assert res["p50_e2e_ms"] > 0.0
        assert res["p50_e2e_ms"] <= res["p95_e2e_ms"]
        assert res["throughput_e2e_fps"] > 0.0
        runtime.cleanup()

    def test_end_to_end_timer_industrial(self) -> None:
        """Tests EndToEndTimer on industrial sample images."""
        root = Path(__file__).resolve().parent.parent
        sample_dir = root / "data" / "sample_images" / "industrial"
        samples = list(sample_dir.rglob("*.png"))

        if not samples:
            pytest.skip("Industrial sample images not found")

        adapter = IndustrialModelAdapter()
        runtime = PyTorchRuntime(adapter.get_pytorch_model(), device="cpu")
        runtime.load()

        timer = EndToEndTimer(warmup_iterations=2, timed_iterations=5)
        res = timer.benchmark_e2e_industrial(runtime, samples, adapter)

        assert res["p50_e2e_ms"] > 0.0
        assert res["p50_e2e_ms"] <= res["p95_e2e_ms"]
        assert res["throughput_e2e_fps"] > 0.0
        runtime.cleanup()

    def test_yolo_postprocessing_latency_sub_10ms(self) -> None:
        """Asserts that YOLO detection postprocessing completes in under 10 ms for batch size 1."""
        adapter = YOLOAdapter()
        model = adapter.get_pytorch_model().eval()
        with torch.no_grad():
            dummy_raw = model(torch.randn(1, 3, 640, 640))
        orig_shape = (640, 640)
        ratio = (1.0, 1.0)
        pad = (0.0, 0.0)

        # Warmup
        for _ in range(5):
            _ = adapter.postprocess(dummy_raw, orig_shape, ratio, pad)

        import time
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            _ = adapter.postprocess(dummy_raw, orig_shape, ratio, pad)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        p50_postprocess_ms = float(np.median(latencies))
        assert p50_postprocess_ms < 10.0, f"YOLO postprocessing p50={p50_postprocess_ms:.2f}ms exceeds 10ms limit"
