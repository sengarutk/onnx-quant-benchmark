"""
Dual-regime latency timing.

* ModelPathTimer: isolated model-path latency. On CUDA, each iteration is bracketed by CUDA events recorded on the
  current stream (the PyTorch runtime does not synchronize inside ``predict_device``), so the measured interval is
  the device-side span of the forward pass including any launch gaps. On CPU, ``time.perf_counter_ns`` wall time.
* EndToEndTimer: full single-image pipeline wall time (decode + letterbox/normalize, inference including host<->device
  copies, post-processing), with every stage timed separately so the stage breakdown is measured, not estimated.
"""

import gc
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Union
import numpy as np
import torch

from src.benchmarking.throughput import compute_throughput
from src.common.logging import setup_logger
from src.models.industrial_model_adapter import top_k_anomaly_score
from src.models.preprocess import preprocess_detection_image, preprocess_industrial_image
from src.runtimes.base import BaseRuntime

logger = setup_logger("timer")


def summarize_latencies(latencies_ms: List[float], batch_size: int = 1, prefix: str = "") -> Dict[str, Any]:
    """Percentile summary of a latency sample (throughput is derived as batch_size / p50)."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    p50 = float(np.percentile(arr, 50))
    return {
        f"{prefix}p50_ms": p50,
        f"{prefix}p90_ms": float(np.percentile(arr, 90)),
        f"{prefix}p95_ms": float(np.percentile(arr, 95)),
        f"{prefix}p99_ms": float(np.percentile(arr, 99)),
        f"{prefix}mean_ms": float(np.mean(arr)),
        f"{prefix}std_ms": float(np.std(arr)),
        f"{prefix}min_ms": float(np.min(arr)),
        f"{prefix}max_ms": float(np.max(arr)),
        f"{prefix}throughput_fps": compute_throughput(p50, batch_size=batch_size),
        f"{prefix}iterations": int(arr.size),
    }


class ModelPathTimer:
    """Isolated model-path latency timer (CUDA events on GPU, perf_counter_ns on CPU)."""

    def __init__(self, warmup_iterations: int = 50, timed_iterations: int = 300, device: str = "cpu") -> None:
        self.device = device
        self.warmup_iterations = max(1, warmup_iterations)
        self.timed_iterations = max(1, timed_iterations)

    def benchmark_device(self, runtime: BaseRuntime, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Runs `warmup_iterations` untimed and `timed_iterations` timed forward passes.

        Returns:
            Percentile summary plus the raw per-iteration ``latencies_ms`` list.
        """
        is_cuda = "cuda" in self.device.lower() and torch.cuda.is_available()

        for _ in range(self.warmup_iterations):
            runtime.predict_device(input_tensor)

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            if is_cuda:
                stream = torch.cuda.current_stream()
                torch.cuda.synchronize()
                starts = [torch.cuda.Event(enable_timing=True) for _ in range(self.timed_iterations)]
                ends = [torch.cuda.Event(enable_timing=True) for _ in range(self.timed_iterations)]
                for i in range(self.timed_iterations):
                    starts[i].record(stream)
                    runtime.predict_device(input_tensor)
                    ends[i].record(stream)
                torch.cuda.synchronize()
                latencies = [float(s.elapsed_time(e)) for s, e in zip(starts, ends)]
            else:
                latencies = []
                for _ in range(self.timed_iterations):
                    t0 = time.perf_counter_ns()
                    runtime.predict_device(input_tensor)
                    latencies.append((time.perf_counter_ns() - t0) / 1e6)
        finally:
            if gc_was_enabled:
                gc.enable()

        res = summarize_latencies(latencies, batch_size=int(input_tensor.shape[0]))
        res["latencies_ms"] = latencies
        return res


class EndToEndTimer:
    """End-to-end single-image pipeline profiler with per-stage timing."""

    def __init__(self, warmup_iterations: int = 20, timed_iterations: int = 200) -> None:
        self.warmup_iterations = max(1, warmup_iterations)
        self.timed_iterations = max(1, timed_iterations)

    def _run(
        self,
        image_paths: List[Union[str, Path]],
        preprocess: Callable[[Path], Any],
        infer: Callable[[Any], Any],
        postprocess: Callable[[Any, Any], Any],
    ) -> Dict[str, Any]:
        if not image_paths:
            raise ValueError("image_paths cannot be empty")
        n = len(image_paths)
        for i in range(self.warmup_iterations):
            pre = preprocess(Path(image_paths[i % n]))
            postprocess(infer(pre), pre)

        total, t_pre, t_inf, t_post = [], [], [], []
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for i in range(self.timed_iterations):
                path = Path(image_paths[i % n])
                t0 = time.perf_counter_ns()
                pre = preprocess(path)
                t1 = time.perf_counter_ns()
                out = infer(pre)
                t2 = time.perf_counter_ns()
                postprocess(out, pre)
                t3 = time.perf_counter_ns()
                t_pre.append((t1 - t0) / 1e6)
                t_inf.append((t2 - t1) / 1e6)
                t_post.append((t3 - t2) / 1e6)
                total.append((t3 - t0) / 1e6)
        finally:
            if gc_was_enabled:
                gc.enable()

        res = summarize_latencies(total, batch_size=1, prefix="e2e_")
        res.update({
            "stage_preprocess_p50_ms": float(np.median(t_pre)),
            "stage_inference_p50_ms": float(np.median(t_inf)),
            "stage_postprocess_p50_ms": float(np.median(t_post)),
            "stage_preprocess_mean_ms": float(np.mean(t_pre)),
            "stage_inference_mean_ms": float(np.mean(t_inf)),
            "stage_postprocess_mean_ms": float(np.mean(t_post)),
        })
        return res

    def benchmark_e2e_detection(
        self,
        runtime: BaseRuntime,
        image_paths: List[Union[str, Path]],
        yolo_adapter: Any,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ) -> Dict[str, Any]:
        """Decode + letterbox -> inference (host arrays in/out) -> confidence filter + class-aware NMS."""
        in_name = next(iter(runtime.get_input_spec()))

        def pre(p: Path):
            return preprocess_detection_image(p)

        def infer(x):
            return runtime.predict({in_name: x[0].numpy()})

        def post(out, x):
            raw = list(out.values())[0]
            return yolo_adapter.postprocess(raw, x[3], x[1], x[2], conf_thresh, iou_thresh)

        return self._run(image_paths, pre, infer, post)

    def benchmark_e2e_industrial(
        self,
        runtime: BaseRuntime,
        image_paths: List[Union[str, Path]],
        industrial_adapter: Any = None,
    ) -> Dict[str, Any]:
        """Decode + resize/normalize -> inference -> top-1% anomaly score aggregation."""
        in_name = next(iter(runtime.get_input_spec()))

        def pre(p: Path):
            return preprocess_industrial_image(p).numpy()

        def infer(x):
            return runtime.predict({in_name: x})

        def post(out, x):
            vals = list(out.values())
            a_map = vals[1] if len(vals) > 1 else np.mean(np.abs(x - vals[0]), axis=1, keepdims=True)
            return top_k_anomaly_score(a_map)

        return self._run(image_paths, pre, infer, post)
