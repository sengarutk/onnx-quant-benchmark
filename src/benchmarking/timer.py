"""
Dual-Regime Latency Timing: Stream-synchronized CUDA Events & End-to-End Pipeline Profiler.
"""

import gc
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.throughput import compute_throughput
from src.common.logging import setup_logger
from src.models.industrial_model_adapter import IndustrialModelAdapter
from src.models.preprocess import preprocess_detection_image, preprocess_industrial_image
from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.base import BaseRuntime

logger = setup_logger("timer")


class ModelPathTimer:
    """
    Isolated Model-Path Compute Latency Timer using CUDA Events on GPU and perf_counter_ns on CPU.
    """

    def __init__(
        self,
        warmup_iterations: int = 50,
        timed_iterations: int = 300,
        device: str = "cpu",
    ) -> None:
        self.device = device
        self.warmup_iterations = max(1, warmup_iterations)
        self.timed_iterations = max(1, timed_iterations)

    def benchmark_device(
        self,
        runtime: BaseRuntime,
        input_tensor: torch.Tensor,
    ) -> Dict[str, float]:
        """
        Measures pure model-path forward execution with adaptive burn-in warmup and stream-synchronized timing.

        Args:
            runtime: Initialized BaseRuntime engine.
            input_tensor: Contiguous input tensor.

        Returns:
            Dictionary with statistical latency metrics (mean, std, p50, p90, p95, p99, throughput_fps).
        """
        is_cuda = "cuda" in self.device.lower() and torch.cuda.is_available()

        if is_cuda:
            stream = torch.cuda.current_stream()
            stream.synchronize()

            # Adaptive Warmup on CUDA with stream synchronization
            window_size = min(10, max(2, self.warmup_iterations))
            min_warmup = min(10, max(1, self.warmup_iterations))
            max_warmup = max(min_warmup, self.warmup_iterations)
            warmup_lats = []

            w_starts = [torch.cuda.Event(enable_timing=True) for _ in range(max_warmup)]
            w_ends = [torch.cuda.Event(enable_timing=True) for _ in range(max_warmup)]

            for w_idx in range(max_warmup):
                w_starts[w_idx].record(stream)
                _ = runtime.predict_device(input_tensor)
                w_ends[w_idx].record(stream)
                w_ends[w_idx].synchronize()
                warmup_lats.append(float(w_starts[w_idx].elapsed_time(w_ends[w_idx])))

                if len(warmup_lats) >= window_size and w_idx >= min_warmup:
                    window = warmup_lats[-window_size:]
                    w_mean = float(np.mean(window))
                    w_std = float(np.std(window))
                    w_cv = float(w_std / w_mean) if w_mean > 1e-9 else 0.0
                    if w_cv < 0.01:
                        break

            stream.synchronize()

            # Timed iterations on CUDA
            start_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.timed_iterations)]
            end_events = [torch.cuda.Event(enable_timing=True) for _ in range(self.timed_iterations)]

            torch.cuda.synchronize()
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                for i in range(self.timed_iterations):
                    start_events[i].record(stream)
                    _ = runtime.predict_device(input_tensor)
                    end_events[i].record(stream)
                torch.cuda.synchronize()
            finally:
                if gc_was_enabled:
                    gc.enable()

            latencies_ms = [float(s.elapsed_time(e)) for s, e in zip(start_events, end_events)]
        else:
            # High-resolution monotonic CPU clock - completely free of CUDA calls
            window_size = min(10, max(2, self.warmup_iterations))
            min_warmup = min(10, max(1, self.warmup_iterations))
            max_warmup = max(min_warmup, self.warmup_iterations)
            warmup_lats = []

            for w_idx in range(max_warmup):
                t0 = time.perf_counter_ns()
                _ = runtime.predict_device(input_tensor)
                t1 = time.perf_counter_ns()
                warmup_lats.append((t1 - t0) / 1_000_000.0)

                if len(warmup_lats) >= window_size and w_idx >= min_warmup:
                    window = warmup_lats[-window_size:]
                    w_mean = float(np.mean(window))
                    w_std = float(np.std(window))
                    w_cv = float(w_std / w_mean) if w_mean > 1e-9 else 0.0
                    if w_cv < 0.01:
                        break

            latencies_ms = []
            gc_was_enabled = gc.isenabled()
            gc.disable()
            try:
                for _ in range(self.timed_iterations):
                    t0 = time.perf_counter_ns()
                    _ = runtime.predict_device(input_tensor)
                    t1 = time.perf_counter_ns()
                    latencies_ms.append((t1 - t0) / 1_000_000.0)
            finally:
                if gc_was_enabled:
                    gc.enable()

        lat_arr = np.array(latencies_ms, dtype=np.float64)
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        mean_lat = float(np.mean(lat_arr))
        std_lat = float(np.std(lat_arr))
        min_lat = float(np.min(lat_arr))
        max_lat = float(np.max(lat_arr))
        fps = compute_throughput(p50, batch_size=int(input_tensor.shape[0]))

        return {
            "p50_ms": p50,
            "p90_ms": p90,
            "p95_ms": p95,
            "p99_ms": p99,
            "mean_ms": mean_lat,
            "std_ms": std_lat,
            "min_ms": min_lat,
            "max_ms": max_lat,
            "throughput_fps": fps,
            "iterations": float(self.timed_iterations),
        }


class EndToEndTimer:
    """
    End-to-End pipeline wall-clock profiler including Decode, Preprocess, Inference, and Postprocess.
    """

    def __init__(
        self,
        warmup_iterations: int = 10,
        timed_iterations: int = 30,
    ) -> None:
        self.warmup_iterations = max(1, warmup_iterations)
        self.timed_iterations = max(1, timed_iterations)

    def benchmark_e2e_detection(
        self,
        runtime: BaseRuntime,
        image_paths: List[Union[str, Path]],
        yolo_adapter: YOLOAdapter,
        conf_thresh: float = 0.25,
        iou_thresh: float = 0.45,
    ) -> Dict[str, float]:
        """
        Measures total wall-clock time for detection inspection:
        Read/Decode -> Preprocess -> Inference -> Vectorized NMS Postprocess.
        """
        if not image_paths:
            raise ValueError("image_paths cannot be empty")

        num_images = len(image_paths)

        # Warmup loop
        for i in range(self.warmup_iterations):
            img_p = image_paths[i % num_images]
            inp_t, ratio, pad, orig_shape = preprocess_detection_image(img_p)
            out_dict = runtime.predict({"images": inp_t.numpy()})
            raw_out = out_dict.get("output0", list(out_dict.values())[0])
            _ = yolo_adapter.postprocess(raw_out, orig_shape, ratio, pad)

        latencies_ms: List[float] = []

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for i in range(self.timed_iterations):
                img_p = image_paths[i % num_images]
                t0 = time.perf_counter_ns()

                # 1. Decode & Preprocess
                inp_t, ratio, pad, orig_shape = preprocess_detection_image(img_p)

                # 2. Inference
                out_dict = runtime.predict({"images": inp_t.numpy()})
                raw_out = out_dict.get("output0", list(out_dict.values())[0])

                # 3. Vectorized Postprocess & NMS
                _ = yolo_adapter.postprocess(raw_out, orig_shape, ratio, pad)

                t1 = time.perf_counter_ns()
                latencies_ms.append((t1 - t0) / 1_000_000.0)
        finally:
            if gc_was_enabled:
                gc.enable()

        lat_arr = np.array(latencies_ms, dtype=np.float64)
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        mean_lat = float(np.mean(lat_arr))
        std_lat = float(np.std(lat_arr))
        fps = compute_throughput(p50, batch_size=1)

        return {
            "p50_e2e_ms": p50,
            "p90_e2e_ms": p90,
            "p95_e2e_ms": p95,
            "p99_e2e_ms": p99,
            "mean_e2e_ms": mean_lat,
            "std_e2e_ms": std_lat,
            "throughput_e2e_fps": fps,
        }

    def benchmark_e2e_industrial(
        self,
        runtime: BaseRuntime,
        image_paths: List[Union[str, Path]],
        industrial_adapter: IndustrialModelAdapter,
    ) -> Dict[str, float]:
        """
        Measures total wall-clock time for industrial inspection:
        Read/Decode -> Resize/Normalize -> Inference -> Anomaly Map Aggregation.
        """
        if not image_paths:
            raise ValueError("image_paths cannot be empty")

        num_images = len(image_paths)

        # Warmup loop
        for i in range(self.warmup_iterations):
            img_p = image_paths[i % num_images]
            inp_t = preprocess_industrial_image(img_p)
            out_dict = runtime.predict({"input": inp_t.numpy()})
            _ = list(out_dict.values())[0]

        latencies_ms: List[float] = []

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for i in range(self.timed_iterations):
                img_p = image_paths[i % num_images]
                t0 = time.perf_counter_ns()

                # 1. Read & Preprocess
                inp_t = preprocess_industrial_image(img_p)

                # 2. Inference
                out_dict = runtime.predict({"input": inp_t.numpy()})

                # 3. Postprocess anomaly score
                if "anomaly_map" in out_dict:
                    am = out_dict["anomaly_map"]
                else:
                    recon = out_dict.get("reconstruction", list(out_dict.values())[0])
                    am = np.mean(np.abs(inp_t.numpy() - recon), axis=1, keepdims=True)
                _ = float(np.percentile(am, 99.0))

                t1 = time.perf_counter_ns()
                latencies_ms.append((t1 - t0) / 1_000_000.0)
        finally:
            if gc_was_enabled:
                gc.enable()

        lat_arr = np.array(latencies_ms, dtype=np.float64)
        p50 = float(np.percentile(lat_arr, 50))
        p90 = float(np.percentile(lat_arr, 90))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))
        mean_lat = float(np.mean(lat_arr))
        std_lat = float(np.std(lat_arr))
        fps = compute_throughput(p50, batch_size=1)

        return {
            "p50_e2e_ms": p50,
            "p90_e2e_ms": p90,
            "p95_e2e_ms": p95,
            "p99_e2e_ms": p99,
            "mean_e2e_ms": mean_lat,
            "std_e2e_ms": std_lat,
            "throughput_e2e_fps": fps,
        }
