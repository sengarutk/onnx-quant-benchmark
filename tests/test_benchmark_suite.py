"""Integration test for BenchmarkSuite.run_single_configuration (runs on CPU with tiny iteration counts)."""

import json
from pathlib import Path
import cv2
import numpy as np
import torch

from src.benchmarking.benchmark_suite import BenchmarkSuite
from src.common.config import QualityThresholdConfig
from src.data.synthetic import generate_detection_sample
from src.models.preprocess import preprocess_detection_image
from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.pytorch_runtime import PyTorchRuntime


def test_single_configuration_record(tmp_path: Path) -> None:
    img, _ = generate_detection_sample("test", 0)
    p = tmp_path / "img.jpg"
    cv2.imwrite(str(p), img)
    adapter = YOLOAdapter()
    rt = PyTorchRuntime(adapter.get_pytorch_model(), device="cpu")
    rt.load()
    x = preprocess_detection_image(p)[0]
    with torch.inference_mode():
        ref = [adapter.get_pytorch_model()(x).numpy()]

    suite = BenchmarkSuite(results_dir=tmp_path / "results", session_id="session_test", warmup_model=1, timed_model=4,
                           warmup_e2e=1, timed_e2e=2, stability_sessions=2, cooldown_seconds=0.0)
    calls = []

    def quality(runtime):
        calls.append(runtime)
        return {"mAP_50": 0.40}

    rec = suite.run_single_configuration(
        model_name="yolo_nano", runtime_name="PyTorch", precision="fp32", runtime=rt, input_tensor=x,
        sample_image_paths=[p], adapter=adapter, task="detection", reference_outputs=ref, quality_fn=quality,
        reference_quality={"mAP_50": 0.45}, quality_metric="mAP_50",
        thresholds=QualityThresholdConfig(max_map_drop=0.01),
    )
    assert calls == [rt]  # quality is measured with the runtime under test
    assert rec["numerical_checks"]["passed"] is True and rec["numerical_checks"]["max_rel_error"] == 0.0
    assert rec["quality_audit"]["passed"] is False and np.isclose(rec["quality_audit"]["delta"], -0.05)
    assert "QUALITY_GATE_FAIL" in rec["status"]
    assert set(rec["end_to_end_latency_ms"]["stages_p50"]) == {"preprocess", "inference", "postprocess"}
    assert rec["model_path_latency_ms"]["samples"] >= 8
    saved = json.loads((tmp_path / "results" / "raw" / "session_test" / rec["run_id"] / "run.json").read_text())
    assert saved["session_id"] == "session_test" and saved["run_id"] == rec["run_id"]
