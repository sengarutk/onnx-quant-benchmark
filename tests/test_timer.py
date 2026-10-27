"""Unit tests for ModelPathTimer and EndToEndTimer (self-contained: images are generated into tmp_path)."""

from pathlib import Path
import cv2
import pytest
import torch

from src.benchmarking.timer import EndToEndTimer, ModelPathTimer, summarize_latencies
from src.data.synthetic import generate_detection_sample, generate_industrial_sample
from src.export.export_onnx import AutoencoderExportWrapper
from src.models.industrial_model_adapter import IndustrialModelAdapter
from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.pytorch_runtime import PyTorchRuntime


@pytest.fixture
def det_images(tmp_path: Path):
    paths = []
    for i in range(3):
        img, _ = generate_detection_sample("test", i)
        p = tmp_path / f"d{i}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


@pytest.fixture
def ind_images(tmp_path: Path):
    paths = []
    for i in range(3):
        img, _ = generate_industrial_sample("test", i, anomalous=bool(i % 2))
        p = tmp_path / f"i{i}.png"
        cv2.imwrite(str(p), img)
        paths.append(p)
    return paths


class TestTimer:
    def test_model_path_timer_percentiles(self) -> None:
        rt = PyTorchRuntime(YOLOAdapter().get_pytorch_model(), device="cpu")
        rt.load()
        res = ModelPathTimer(warmup_iterations=2, timed_iterations=12, device="cpu").benchmark_device(rt, torch.randn(1, 3, 320, 320))
        assert 0 < res["p50_ms"] <= res["p90_ms"] <= res["p95_ms"] <= res["p99_ms"]
        assert res["iterations"] == 12 and len(res["latencies_ms"]) == 12

    def test_summarize_latencies_batch_throughput(self) -> None:
        s = summarize_latencies([2.0, 2.0, 2.0], batch_size=4)
        assert s["p50_ms"] == 2.0 and s["throughput_fps"] == 2000.0

    def test_end_to_end_detection_stage_breakdown(self, det_images) -> None:
        adapter = YOLOAdapter()
        rt = PyTorchRuntime(adapter.get_pytorch_model(), device="cpu")
        rt.load()
        res = EndToEndTimer(warmup_iterations=1, timed_iterations=4).benchmark_e2e_detection(rt, det_images, adapter)
        assert 0 < res["e2e_p50_ms"] <= res["e2e_p95_ms"]
        stages = res["stage_preprocess_mean_ms"] + res["stage_inference_mean_ms"] + res["stage_postprocess_mean_ms"]
        assert abs(stages - res["e2e_mean_ms"]) < 0.05 * res["e2e_mean_ms"] + 0.1  # stages tile the total

    def test_end_to_end_industrial(self, ind_images) -> None:
        model = AutoencoderExportWrapper(IndustrialModelAdapter().get_pytorch_model()).eval()
        rt = PyTorchRuntime(model, device="cpu", input_names=["input"], output_names=["reconstruction", "anomaly_map"])
        rt.load()
        res = EndToEndTimer(warmup_iterations=1, timed_iterations=4).benchmark_e2e_industrial(rt, ind_images)
        assert res["e2e_p50_ms"] > 0 and res["stage_inference_p50_ms"] > 0
