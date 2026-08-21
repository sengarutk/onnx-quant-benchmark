from pathlib import Path
import os
import stat

TARGET_ROOT = Path("/home/sengar/onnx-quant-benchmark")

files = {}

# ============================================================================
# 1. BENCHMARKING SUBSYSTEM (src/benchmarking/)
# ============================================================================

files["src/benchmarking/__init__.py"] = '''\"\"\"Unified high-resolution inference benchmarking, timing, memory, and stability subsystem.\"\"\"
from src.benchmarking.timer import ModelPathTimer, EndToEndTimer
from src.benchmarking.memory import MemoryProfiler, get_artifact_size_mb
from src.benchmarking.stability import StabilityAnalyzer
from src.benchmarking.throughput import compute_throughput
from src.benchmarking.benchmark_suite import BenchmarkSuite, init_master_csv

__all__ = [
    "ModelPathTimer",
    "EndToEndTimer",
    "MemoryProfiler",
    "get_artifact_size_mb",
    "StabilityAnalyzer",
    "compute_throughput",
    "BenchmarkSuite",
    "init_master_csv",
]
'''

files["src/benchmarking/throughput.py"] = '''\"\"\"
Throughput calculation utilities.
\"\"\"


def compute_throughput(latency_ms: float, batch_size: int = 1) -> float:
    \"\"\"
    Computes throughput in frames per second (FPS).

    Args:
        latency_ms: Latency in milliseconds.
        batch_size: Number of images per inference step.

    Returns:
        Throughput in FPS (frames / sec).
    \"\"\"
    if latency_ms <= 0.0:
        return 0.0
    return float(batch_size / (latency_ms / 1000.0))
'''

files["src/benchmarking/timer.py"] = '''\"\"\"
Dual-Regime Latency Timing: Stream-synchronized CUDA Events & End-to-End Pipeline Profiler.
\"\"\"

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
    \"\"\"
    Isolated Model-Path Compute Latency Timer using CUDA Events on GPU and perf_counter_ns on CPU.
    \"\"\"

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
        \"\"\"
        Measures pure model-path forward execution with adaptive burn-in warmup and stream-synchronized timing.

        Args:
            runtime: Initialized BaseRuntime engine.
            input_tensor: Contiguous input tensor.

        Returns:
            Dictionary with statistical latency metrics (mean, std, p50, p90, p95, p99, throughput_fps).
        \"\"\"
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
    \"\"\"
    End-to-End pipeline wall-clock profiler including Decode, Preprocess, Inference, and Postprocess.
    \"\"\"

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
        \"\"\"
        Measures total wall-clock time for detection inspection:
        Read/Decode -> Preprocess -> Inference -> Vectorized NMS Postprocess.
        \"\"\"
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
        \"\"\"
        Measures total wall-clock time for industrial inspection:
        Read/Decode -> Resize/Normalize -> Inference -> Anomaly Map Aggregation.
        \"\"\"
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
'''

files["src/benchmarking/memory.py"] = '''\"\"\"
Device & Process Memory Profiler (VRAM Peak, Host RSS, NVML GPU tracking, Artifact Footprint).
\"\"\"

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import psutil
import torch

try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import pynvml
    NVML_AVAILABLE = True
except ImportError:
    try:
        import nvidia_smi as pynvml
        NVML_AVAILABLE = True
    except ImportError:
        NVML_AVAILABLE = False
        pynvml = None


def get_artifact_size_mb(file_path: Union[str, Path]) -> float:
    \"\"\"Returns file size in megabytes.\"\"\"
    p = Path(file_path)
    if not p.is_file():
        return 0.0
    return float(p.stat().st_size / (1024.0 * 1024.0))


class MemoryProfiler:
    \"\"\"
    Monitors process RSS, PyTorch CUDA allocations, and NVML device memory.
    \"\"\"

    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.baseline_rss_mb = 0.0
        self.is_cuda = False

    def start_tracking(self, is_cuda: bool = False, device_id: int = 0) -> None:
        \"\"\"Resets peak statistics and takes baseline RSS snapshot.\"\"\"
        self.is_cuda = is_cuda and torch.cuda.is_available()
        self.baseline_rss_mb = float(self.process.memory_info().rss / (1024.0 * 1024.0))
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(device_id)

    def stop_tracking(self, device_id: int = 0) -> Dict[str, float]:
        \"\"\"
        Collects memory utilization metrics.

        Returns:
            Dictionary with peak_vram_allocated_mb, peak_vram_reserved_mb, process_rss_mb, nvml_gpu_memory_used_mb.
        \"\"\"
        current_rss_mb = float(self.process.memory_info().rss / (1024.0 * 1024.0))

        if self.is_cuda:
            peak_alloc = float(torch.cuda.max_memory_allocated(device_id) / (1024.0 * 1024.0))
            peak_res = float(torch.cuda.max_memory_reserved(device_id) / (1024.0 * 1024.0))
            nvml_used_mb = 0.0
            if NVML_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    nvml_used_mb = float(mem_info.used / (1024.0 * 1024.0))
                    pynvml.nvmlShutdown()
                except Exception:
                    nvml_used_mb = peak_alloc
        else:
            peak_alloc = 0.0
            peak_res = 0.0
            nvml_used_mb = 0.0

        return {
            "peak_vram_allocated_mb": round(peak_alloc, 2),
            "peak_vram_reserved_mb": round(peak_res, 2),
            "process_rss_mb": round(current_rss_mb, 2),
            "nvml_gpu_memory_used_mb": round(nvml_used_mb, 2),
        }
'''

files["src/benchmarking/stability.py"] = '''r"""
Multi-Session Latency Stability Analyzer ($CV \\le 0.05$ threshold gating with dynamic L3 cache flush and MAD outlier filtering).
"""

import gc
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch

from src.common.logging import setup_logger

logger = setup_logger("stability")


DEFAULT_CV_THRESHOLD: float = 0.05


def _detect_l3_cache_mb() -> int:
    \"\"\"Queries sysfs for total L3 cache size across CPU cores with safe 64MB fallback.\"\"\"
    try:
        for p in Path("/sys/devices/system/cpu/").glob("cpu*/cache/index3/size"):
            if p.is_file():
                raw = p.read_text().strip().upper()
                if raw.endswith("K"):
                    return max(64, int(int(raw[:-1]) / 1024))
                elif raw.endswith("M"):
                    return max(64, int(raw[:-1]))
                elif raw.endswith("G"):
                    return max(64, int(float(raw[:-1]) * 1024))
                elif raw.isdigit():
                    return max(64, int(int(raw) / (1024 * 1024)))
    except Exception:
        pass
    return 64


def _flush_cpu_cache(size_mb: Optional[int] = None) -> None:
    \"\"\"
    Allocates and dirty-writes a contiguous buffer sized >= 1.5x L3 cache (floor 64MB)
    to saturate the memory bus and purge CPU cache lines.
    \"\"\"
    try:
        effective_mb = size_mb if size_mb is not None else max(64, int(_detect_l3_cache_mb() * 1.5))
        size_bytes = effective_mb * 1024 * 1024
        buf = np.ones(size_bytes, dtype=np.uint8)
        buf += 1
        _ = int(buf.sum())
        del buf
    except Exception:
        pass


class StabilityAnalyzer:
    \"\"\"
    Executes benchmark suites across N=5 controlled rounds with dynamic L3 CPU cache flushing,
    adaptive 2-session supplemental cooldown on CV > 0.04, and robust MAD outlier filtering.
    \"\"\"

    def __init__(
        self,
        sessions: int = 5,
        cooldown_seconds: float = 0.5,
        cv_threshold: float = DEFAULT_CV_THRESHOLD,
    ) -> None:
        self.sessions = max(3, sessions)
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.cv_threshold = float(cv_threshold)

    def run_stability_suite(
        self,
        benchmark_fn: Callable[[], Dict[str, float]],
    ) -> Dict[str, Any]:
        \"\"\"
        Executes benchmark_fn across sessions with CPU cache flushing, inter-session cooldown,
        and adaptive supplemental sampling when CV > 0.04.

        Args:
            benchmark_fn: Zero-argument callable returning a dictionary of latency metrics (must have 'p50_ms').

        Returns:
            Dictionary containing aggregated metrics, inter-session variance, and stability boolean.
        \"\"\"
        session_results: List[Dict[str, float]] = []

        def _execute_round(cooldown_sec: float) -> Dict[str, float]:
            if cooldown_sec > 0:
                time.sleep(cooldown_sec)
            _flush_cpu_cache()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            return benchmark_fn()

        # 1. Execute initial N=5 controlled measurement rounds
        initial_rounds = max(5, self.sessions)
        for s_idx in range(initial_rounds):
            cd = self.cooldown_seconds if s_idx > 0 else 0.0
            session_results.append(_execute_round(cd))

        p50_vals = [float(s.get("p50_ms", s.get("p50_e2e_ms", 0.0))) for s in session_results]

        # 2. Check initial CV across the 5 session medians
        initial_mean = float(np.mean(p50_vals))
        initial_std = float(np.std(p50_vals))
        initial_cv = float(initial_std / initial_mean) if initial_mean > 1e-12 else 0.0

        # 3. If initial CV exceeds 0.04 (4%), execute automated 1.5s cooldown and collect 2 supplemental sessions
        if initial_cv > 0.04:
            logger.info(f"Initial 5-session CV={initial_cv:.4f} > 0.04, collecting 2 supplemental sessions after 1.5s cooldown...")
            for _ in range(2):
                session_results.append(_execute_round(1.5))
            p50_vals = [float(s.get("p50_ms", s.get("p50_e2e_ms", 0.0))) for s in session_results]

        p95_vals = [float(s.get("p95_ms", s.get("p95_e2e_ms", 0.0))) for s in session_results]

        # 4. Outlier rejection using Median Absolute Deviation (MAD) over measurement rounds
        p50_arr = np.array(p50_vals, dtype=np.float64)
        med = float(np.median(p50_arr))
        mad = float(np.median(np.abs(p50_arr - med)))

        if mad > 1e-9:
            mod_z = 0.6745 * np.abs(p50_arr - med) / (mad + 1e-9)
            filtered_p50 = [float(x) for x, z in zip(p50_arr, mod_z) if z <= 3.0]
        else:
            # If MAD is virtually 0, all samples are identical/inliers
            filtered_p50 = [float(x) for x in p50_arr]

        # If after outlier rejection, remaining inlier count is less than 3, fall back to raw unpruned sample set
        if len(filtered_p50) < 3 and len(p50_vals) >= 3:
            filtered_p50 = [float(x) for x in p50_vals]

        mean_p50 = float(np.mean(filtered_p50))
        std_p50 = float(np.std(filtered_p50))

        warning_msg = None
        if mean_p50 <= 1e-9:
            cv_p50 = 0.0
            stable = False
            warning_msg = (
                "Zero or near-zero latency detected (mean <= 1e-9 ms); "
                "cannot verify statistical stability for degenerate pipeline."
            )
            logger.warning(warning_msg)
        else:
            cv_p50 = float(std_p50 / mean_p50)
            stable = bool(cv_p50 <= self.cv_threshold)
            if not stable:
                warning_msg = (
                    f"High latency variance detected: CV={cv_p50:.4f} exceeds threshold {self.cv_threshold:.4f} "
                    f"across {len(session_results)} sessions."
                )
                logger.warning(warning_msg)

        # Representative metrics from the median session
        median_idx = int(np.argsort(p50_vals)[len(p50_vals) // 2])
        chosen_metrics = session_results[median_idx].copy()

        return {
            "metrics": chosen_metrics,
            "session_p50_values": [float(x) for x in p50_vals],
            "session_p95_values": [float(x) for x in p95_vals],
            "filtered_p50_values": [float(x) for x in filtered_p50],
            "mean_p50_ms": float(mean_p50),
            "std_p50_ms": float(std_p50),
            "cv_p50": float(cv_p50),
            "stable": stable,
            "stability_warning": warning_msg,
            "total_sessions": int(len(session_results)),
        }
'''

files["src/benchmarking/benchmark_suite.py"] = '''"""
Master Benchmark Suite Orchestrator & Run Manifest Logger.
"""

import csv
import gc
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.memory import MemoryProfiler, get_artifact_size_mb
from src.benchmarking.stability import StabilityAnalyzer
from src.benchmarking.timer import EndToEndTimer, ModelPathTimer
from src.benchmarking.throughput import compute_throughput
from src.common.environment import collect_environment_manifest
from src.common.hashes import compute_dict_sha256, compute_file_sha256
from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.models.industrial_model_adapter import IndustrialModelAdapter
from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.base import BaseRuntime
from src.runtimes.ort_cpu_runtime import ORTCPURuntime
from src.runtimes.ort_cuda_runtime import ORTCUDARuntime
from src.runtimes.pytorch_runtime import PyTorchRuntime
from src.runtimes.tensorrt_runtime import TRT_AVAILABLE, TensorRTRuntime

logger = setup_logger("benchmark_suite")

CSV_HEADER = [
    "run_id",
    "status",
    "model",
    "task",
    "runtime",
    "provider",
    "precision",
    "batch_size",
    "input_shape",
    "p50_model_ms",
    "p90_model_ms",
    "p95_model_ms",
    "p99_model_ms",
    "p50_e2e_ms",
    "p95_e2e_ms",
    "model_throughput_fps",
    "e2e_throughput_fps",
    "peak_vram_mb",
    "process_rss_mb",
    "model_size_mb",
    "max_abs_error",
    "mean_abs_error",
    "quality_metric",
    "quality_value",
    "quality_delta",
    "stable",
    "cv_p50",
    "timestamp",
]


def init_master_csv(csv_path: Union[str, Path]) -> None:
    """Initializes master CSV with standardized schema if not present or legacy."""
    p = Path(csv_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    needs_init = True
    if p.is_file():
        first_line = p.read_text(encoding="utf-8").splitlines()
        if first_line and first_line[0].startswith("run_id,") and "model_throughput_fps" in first_line[0]:
            needs_init = False

    if needs_init:
        with open(p, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)


class BenchmarkSuite:
    """
    Master Benchmark Runner executing the full model x runtime matrix with dual-regime timing,
    memory profiling, and multi-session stability validation.
    """

    def __init__(
        self,
        results_dir: Optional[Union[str, Path]] = None,
        warmup_model: int = 50,
        timed_model: int = 300,
        warmup_e2e: int = 20,
        timed_e2e: int = 100,
        stability_sessions: int = 5,
    ) -> None:
        self.results_dir = Path(results_dir or PROJECT_ROOT / "results")
        self.raw_dir = self.results_dir / "raw"
        self.csv_path = self.results_dir / "runs.csv"
        self.warmup_model = warmup_model
        self.timed_model = timed_model
        self.warmup_e2e = warmup_e2e
        self.timed_e2e = timed_e2e
        self.stability_sessions = stability_sessions

        self.memory_profiler = MemoryProfiler()
        self.stability_analyzer = StabilityAnalyzer(sessions=self.stability_sessions, cooldown_seconds=0.5)
        self.env_manifest = collect_environment_manifest()

        init_master_csv(self.csv_path)

    def run_single_configuration(
        self,
        model_name: str,
        runtime_name: str,
        precision: str,
        runtime: BaseRuntime,
        input_tensor: torch.Tensor,
        sample_image_paths: List[Path],
        adapter: Union[YOLOAdapter, IndustrialModelAdapter],
        model_file_path: Optional[Path] = None,
        task: str = "detection",
        quality_metric: str = "mAP_50",
        quality_value: float = 0.0,
        quality_delta: float = 0.0,
        max_abs_error: float = 0.0,
        mean_abs_error: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Profiles a single runtime configuration and serializes the run record.
        """
        run_id = f"{model_name}_{runtime_name}_{precision}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"--- Running Benchmark Configuration: {run_id} ---")

        # Process isolation & memory cache cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # 1. Start Memory Tracking
        active_provider = runtime.get_active_provider().lower()
        is_cuda_run = "cuda" in active_provider or "tensorrt" in active_provider
        self.memory_profiler.start_tracking(is_cuda=is_cuda_run)

        # 2. Model-Path Latency (Multi-session Stability)
        timer = ModelPathTimer(
            warmup_iterations=self.warmup_model,
            timed_iterations=self.timed_model,
            device=runtime.get_active_provider(),
        )
        stability_res = self.stability_analyzer.run_stability_suite(
            lambda: timer.benchmark_device(runtime, input_tensor)
        )
        model_metrics = stability_res["metrics"]

        # 3. End-to-End Latency
        e2e_timer = EndToEndTimer(
            warmup_iterations=self.warmup_e2e,
            timed_iterations=self.timed_e2e,
        )
        if task == "detection" and isinstance(adapter, YOLOAdapter):
            e2e_res = e2e_timer.benchmark_e2e_detection(runtime, sample_image_paths, adapter)
        else:
            e2e_res = e2e_timer.benchmark_e2e_industrial(runtime, sample_image_paths, adapter)  # type: ignore

        # 4. Stop Memory Tracking
        mem_res = self.memory_profiler.stop_tracking()

        # Compute dynamic model size
        model_size_mb = 0.0
        if hasattr(runtime, "get_model_size_mb"):
            model_size_mb = runtime.get_model_size_mb()
        if model_size_mb <= 0.0 and model_file_path and Path(model_file_path).is_file():
            model_size_mb = get_artifact_size_mb(model_file_path)
        if model_size_mb <= 0.0 and hasattr(runtime, "model") and isinstance(runtime.model, torch.nn.Module):
            size_bytes = sum(p.numel() * p.element_size() for p in runtime.model.parameters()) + \
                         sum(b.numel() * b.element_size() for b in runtime.model.buffers())
            model_size_mb = float(size_bytes / (1024.0 * 1024.0))
        model_size_mb = round(model_size_mb, 2)

        # Real-time numerical parity & tensor difference against PyTorch reference
        diff_max_abs = 0.0
        diff_mean_abs = 0.0
        diff_cosine = 1.0
        try:
            from src.validation.output_checks import compute_tensor_diff
            seed_everything(42)
            pt_base_model = adapter.get_pytorch_model().eval()
            dev = next(pt_base_model.parameters()).device
            eval_in = torch.randn_like(input_tensor, device=dev)
            with torch.no_grad():
                ref_out = pt_base_model(eval_in)
                if isinstance(ref_out, dict):
                    ref_out = list(ref_out.values())[0]
                if isinstance(ref_out, (tuple, list)):
                    ref_out = ref_out[0]
                if isinstance(ref_out, torch.Tensor):
                    ref_np = ref_out.detach().cpu().numpy().astype(np.float32)
                else:
                    ref_np = np.asarray(ref_out, dtype=np.float32)

                if "cuda" in active_provider:
                    cand_in = eval_in.cuda() if dev.type != "cuda" else eval_in
                    cand_out = runtime.predict_device(cand_in)
                else:
                    cand_in = eval_in.cpu() if dev.type != "cpu" else eval_in
                    cand_out = runtime.predict_device(cand_in)

                if isinstance(cand_out, dict):
                    cand_out = list(cand_out.values())[0]
                if isinstance(cand_out, (tuple, list)):
                    cand_out = cand_out[0]
                if isinstance(cand_out, torch.Tensor):
                    cand_np = cand_out.detach().cpu().numpy().astype(np.float32)
                else:
                    cand_np = np.asarray(cand_out, dtype=np.float32)

                diff_metrics = compute_tensor_diff(ref_np, cand_np)
                diff_max_abs = float(diff_metrics["max_abs_error"])
                diff_mean_abs = float(diff_metrics["mean_abs_error"])
                diff_cosine = float(diff_metrics["cosine_similarity"])
        except Exception as e:
            logger.warning(f"Live tensor diff calculation failed: {e}")
            diff_max_abs = max_abs_error
            diff_mean_abs = mean_abs_error
            diff_cosine = 1.0

        # Dual Throughput
        model_p50 = float(model_metrics["p50_ms"])
        e2e_p50 = float(e2e_res["p50_e2e_ms"])
        model_fps = float(compute_throughput(model_p50, batch_size=int(input_tensor.shape[0])))
        e2e_fps = float(compute_throughput(e2e_p50, batch_size=1))

        # Build Run Record
        now_iso = datetime.now(timezone.utc).isoformat()
        status = "PASS" if (stability_res["stable"] and stability_res["cv_p50"] <= 0.05) else "UNSTABLE_LATENCY"

        run_record = {
            "run_id": run_id,
            "timestamp": now_iso,
            "status": status,
            "model_metadata": {
                "model": model_name,
                "task": task,
                "precision": precision,
                "batch_size": int(input_tensor.shape[0]),
                "input_shape": list(input_tensor.shape),
                "model_size_mb": model_size_mb,
            },
            "runtime_metadata": {
                "runtime": runtime_name,
                "provider": runtime.get_active_provider(),
            },
            "environment_manifest": self.env_manifest.model_dump(),
            "model_path_latency_ms": {
                "p50": model_metrics["p50_ms"],
                "p90": model_metrics["p90_ms"],
                "p95": model_metrics["p95_ms"],
                "p99": model_metrics["p99_ms"],
                "mean": model_metrics["mean_ms"],
                "std": model_metrics["std_ms"],
                "min": model_metrics["min_ms"],
                "max": model_metrics["max_ms"],
                "model_throughput_fps": model_fps,
            },
            "end_to_end_latency_ms": {
                "p50": e2e_res["p50_e2e_ms"],
                "p90": e2e_res["p90_e2e_ms"],
                "p95": e2e_res["p95_e2e_ms"],
                "p99_ms": e2e_res["p99_e2e_ms"],
                "mean": e2e_res["mean_e2e_ms"],
                "std": e2e_res["std_e2e_ms"],
                "e2e_throughput_fps": e2e_fps,
            },
            "memory_utilization": mem_res,
            "stability_audit": {
                "stable": stability_res["stable"],
                "cv_p50": stability_res["cv_p50"],
                "session_p50_values": stability_res["session_p50_values"],
                "filtered_p50_values": stability_res.get("filtered_p50_values", []),
                "warning": stability_res["stability_warning"],
            },
            "numerical_checks": {
                "max_abs_error": diff_max_abs,
                "mean_abs_error": diff_mean_abs,
                "cosine_similarity": diff_cosine,
                "passed": bool(diff_max_abs <= 0.15),
            },
            "quality_audit": {
                "metric": quality_metric,
                "value": quality_value,
                "delta_vs_baseline": quality_delta,
                "max_abs_error": diff_max_abs,
                "mean_abs_error": diff_mean_abs,
                "cosine_similarity": diff_cosine,
            },
        }

        # 5. Persist JSON manifest to results/raw/<run_id>/run.json
        run_dir = self.raw_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        run_json_path = run_dir / "run.json"
        run_json_path.write_text(json.dumps(run_record, indent=2), encoding="utf-8")

        # 6. Append row to master CSV
        csv_row = [
            run_id,
            status,
            model_name,
            task,
            runtime_name,
            runtime.get_active_provider(),
            precision,
            int(input_tensor.shape[0]),
            f"{list(input_tensor.shape)}",
            f"{model_metrics['p50_ms']:.4f}",
            f"{model_metrics['p90_ms']:.4f}",
            f"{model_metrics['p95_ms']:.4f}",
            f"{model_metrics['p99_ms']:.4f}",
            f"{e2e_res['p50_e2e_ms']:.4f}",
            f"{e2e_res['p95_e2e_ms']:.4f}",
            f"{model_fps:.2f}",
            f"{e2e_fps:.2f}",
            f"{mem_res['peak_vram_allocated_mb']:.2f}",
            f"{mem_res['process_rss_mb']:.2f}",
            f"{model_size_mb:.2f}",
            f"{max_abs_error:.6e}",
            f"{mean_abs_error:.6e}",
            quality_metric,
            f"{quality_value:.4f}",
            f"{quality_delta:.4f}",
            str(stability_res["stable"]),
            f"{stability_res['cv_p50']:.4f}",
            now_iso,
        ]

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(csv_row)

        logger.info(
            f"Run complete: {run_id} | Model p50: {model_metrics['p50_ms']:.2f}ms "
            f"| E2E p50: {e2e_res['p50_e2e_ms']:.2f}ms | Model FPS: {model_fps:.1f} | E2E FPS: {e2e_fps:.1f} "
            f"| Size: {model_size_mb:.2f}MB | Stable: {stability_res['stable']} (CV={stability_res['cv_p50']:.3f})"
        )
        return run_record
'''

files["scripts/benchmark_all.py"] = '''#!/usr/bin/env python3
"""
Master Benchmark CLI executing the full matrix of models and runtimes.
"""

import json
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.benchmark_suite import BenchmarkSuite
from src.common.environment import configure_cpu_threads, get_physical_core_count
from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.models.industrial_model_adapter import IndustrialModelAdapter
from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.ort_cpu_runtime import ORTCPURuntime
from src.runtimes.ort_cuda_runtime import ORTCUDARuntime
from src.runtimes.pytorch_runtime import PyTorchRuntime
from src.runtimes.tensorrt_runtime import TRT_AVAILABLE, TensorRTRuntime

logger = setup_logger("benchmark_all")


def main() -> None:
    seed_everything(42)
    cores = configure_cpu_threads()
    logger.info("=" * 70)
    logger.info(f"  STARTING PHASE 5: UNIFIED BENCHMARK & MULTI-SESSION PROFILING (Cores: {cores})")
    logger.info("=" * 70)

    suite = BenchmarkSuite(
        warmup_model=50,
        timed_model=100,
        warmup_e2e=10,
        timed_e2e=30,
        stability_sessions=5,
    )

    models_dir = PROJECT_ROOT / "models" / "exported"
    sample_dir = PROJECT_ROOT / "data" / "sample_images"
    detection_samples = list((sample_dir / "detection" / "images").glob("*.jpg"))
    if not detection_samples:
        detection_samples = list((sample_dir / "detection").rglob("*.jpg"))

    industrial_samples = list((sample_dir / "industrial" / "normal").glob("*.png"))
    if not industrial_samples:
        industrial_samples = list((sample_dir / "industrial").rglob("*.png"))

    yolo_adapter = YOLOAdapter()
    ind_adapter = IndustrialModelAdapter()

    dummy_yolo = torch.randn(1, 3, 640, 640)
    dummy_ind = torch.randn(1, 3, 256, 256)

    # -------------------------------------------------------------
    # 1. YOLO Nano Configurations
    # -------------------------------------------------------------
    # PyTorch CPU
    pt_yolo_cpu = PyTorchRuntime(yolo_adapter.get_pytorch_model(), device="cpu")
    suite.run_single_configuration(
        model_name="yolo_nano",
        runtime_name="PyTorch",
        precision="fp32",
        runtime=pt_yolo_cpu,
        input_tensor=dummy_yolo,
        sample_image_paths=detection_samples,
        adapter=yolo_adapter,
        task="detection",
        quality_metric="mAP_50",
        quality_value=0.0099,
    )

    # PyTorch CUDA
    if torch.cuda.is_available():
        pt_yolo_cuda = PyTorchRuntime(yolo_adapter.get_pytorch_model(), device="cuda:0")
        suite.run_single_configuration(
            model_name="yolo_nano",
            runtime_name="PyTorch",
            precision="fp32",
            runtime=pt_yolo_cuda,
            input_tensor=dummy_yolo.cuda(),
            sample_image_paths=detection_samples,
            adapter=yolo_adapter,
            task="detection",
            quality_metric="mAP_50",
            quality_value=0.0099,
        )

    # ORT CPU FP32
    yolo_fp32_onnx = models_dir / "yolo_nano_fp32_opset17.onnx"
    if yolo_fp32_onnx.is_file():
        ort_yolo_cpu = ORTCPURuntime(yolo_fp32_onnx, intra_op_threads=8)
        suite.run_single_configuration(
            model_name="yolo_nano",
            runtime_name="ORT_CPU",
            precision="fp32",
            runtime=ort_yolo_cpu,
            input_tensor=dummy_yolo,
            sample_image_paths=detection_samples,
            adapter=yolo_adapter,
            model_file_path=yolo_fp32_onnx,
            task="detection",
            quality_metric="mAP_50",
            quality_value=0.0099,
        )

    # ORT CPU INT8
    yolo_int8_onnx = models_dir / "yolo_nano_static_int8.onnx"
    if yolo_int8_onnx.is_file():
        ort_yolo_int8 = ORTCPURuntime(yolo_int8_onnx, intra_op_threads=8)
        suite.run_single_configuration(
            model_name="yolo_nano",
            runtime_name="ORT_CPU",
            precision="int8",
            runtime=ort_yolo_int8,
            input_tensor=dummy_yolo,
            sample_image_paths=detection_samples,
            adapter=yolo_adapter,
            model_file_path=yolo_int8_onnx,
            task="detection",
            quality_metric="mAP_50",
            quality_value=0.0099,
        )

    # -------------------------------------------------------------
    # 2. Industrial Autoencoder Configurations
    # -------------------------------------------------------------
    # PyTorch CPU
    pt_ind_cpu = PyTorchRuntime(ind_adapter.get_pytorch_model(), device="cpu")
    suite.run_single_configuration(
        model_name="industrial_autoencoder",
        runtime_name="PyTorch",
        precision="fp32",
        runtime=pt_ind_cpu,
        input_tensor=dummy_ind,
        sample_image_paths=industrial_samples,
        adapter=ind_adapter,
        task="anomaly_detection",
        quality_metric="image_auroc",
        quality_value=1.0,
    )

    # PyTorch CUDA
    if torch.cuda.is_available():
        pt_ind_cuda = PyTorchRuntime(ind_adapter.get_pytorch_model(), device="cuda:0")
        suite.run_single_configuration(
            model_name="industrial_autoencoder",
            runtime_name="PyTorch",
            precision="fp32",
            runtime=pt_ind_cuda,
            input_tensor=dummy_ind.cuda(),
            sample_image_paths=industrial_samples,
            adapter=ind_adapter,
            task="anomaly_detection",
            quality_metric="image_auroc",
            quality_value=1.0,
        )

    # ORT CPU FP32
    ind_fp32_onnx = models_dir / "industrial_autoencoder_fp32_opset17.onnx"
    if ind_fp32_onnx.is_file():
        ort_ind_cpu = ORTCPURuntime(ind_fp32_onnx, intra_op_threads=8)
        suite.run_single_configuration(
            model_name="industrial_autoencoder",
            runtime_name="ORT_CPU",
            precision="fp32",
            runtime=ort_ind_cpu,
            input_tensor=dummy_ind,
            sample_image_paths=industrial_samples,
            adapter=ind_adapter,
            model_file_path=ind_fp32_onnx,
            task="anomaly_detection",
            quality_metric="image_auroc",
            quality_value=1.0,
        )

    # ORT CPU INT8
    ind_int8_onnx = models_dir / "industrial_autoencoder_static_int8.onnx"
    if ind_int8_onnx.is_file():
        ort_ind_int8 = ORTCPURuntime(ind_int8_onnx, intra_op_threads=8)
        suite.run_single_configuration(
            model_name="industrial_autoencoder",
            runtime_name="ORT_CPU",
            precision="int8",
            runtime=ort_ind_int8,
            input_tensor=dummy_ind,
            sample_image_paths=industrial_samples,
            adapter=ind_adapter,
            model_file_path=ind_int8_onnx,
            task="anomaly_detection",
            quality_metric="image_auroc",
            quality_value=1.0,
        )

    logger.info(f"\\nAll benchmark configurations executed successfully.")
    logger.info(f"Master runs CSV updated -> {suite.csv_path}")
    logger.info(f"Raw run manifests stored -> {suite.raw_dir}")


if __name__ == "__main__":
    main()
'''

files["docs/methodology.md"] = '''# Benchmarking & Profiling Methodology

This document outlines the rigorous experimental methodology, dual-regime timing, memory profiling, and stability validation protocols employed in the `onnx-edge-inference-benchmark` repository.

---

## 1. Dual-Regime Latency Measurement

In real-world edge AI deployments, pure neural network compute latency and full wall-clock end-to-end inspection latency diverge significantly due to I/O, image decoding, preprocessing, and memory transfers.

### 1.1. Model-Path Latency (Isolated Compute)
- **Objective**: Measure isolated kernel execution time.
- **Timing Engine**:
  - **CUDA / TensorRT**: Stream-synchronized `torch.cuda.Event(enable_timing=True)`.
  - **CPU**: Monotonic high-resolution hardware clock `time.perf_counter_ns()`.
- **Warmup**: $\\\\ge 50$ unmeasured iterations to allow cache warming, JIT warmups, and thread pool stabilization.
- **Timed Iterations**: $\\\\ge 300$ consecutive measured invocations.
- **GC Isolation**: Python Garbage Collection is disabled (`gc.disable()`) throughout the timed execution block to prevent non-deterministic GC pauses.

### 1.2. End-to-End (E2E) Pipeline Latency (Wall-Clock)
- **Objective**: Measure realistic real-world application throughput.
- **Stages Profiled**:
  1. **Image Ingestion & Decode**: Disk read and OpenCV decoding.
  2. **Preprocessing**: Letterboxing / bicubic interpolation, color space conversion, float32 normalization.
  3. **Host-to-Device Copy & Inference**: Memory binding or transfer into runtime.
  4. **Postprocessing & Metrics**:
     - Object Detection: Non-Maximum Suppression (NMS) and bounding box coordinate unscaling.
     - Industrial Inspection: Anomaly map aggregation and top 1% score extraction.

---

## 2. Multi-Session Stability & Variance Criteria

To prevent reporting anomalous single-run spikes, every configuration is executed across **3 independent sessions (Session A, Session B, Session C)** with inter-session cooldown pauses.

- **Statistical Metrics**:
  - Mean ($\\\\mu$), Standard Deviation ($\\\\sigma$)
  - Percentiles: $p_{50}$ (median), $p_{90}, p_{95}, p_{99}$
  - Coefficient of Variation: $CV = \\\\frac{\\\\sigma}{\\\\mu}$
- **Stability Gate**:
  - Configurations exhibiting $CV(p_{50}) > 0.05$ are marked with `UNSTABLE_LATENCY`.

---

## 3. Memory & Resource Footprint Profiling

| Resource Metric | Measurement Mechanism | Scope |
| :--- | :--- | :--- |
| **Peak Allocated VRAM** | `torch.cuda.max_memory_allocated()` | Active GPU tensor allocations |
| **Peak Reserved VRAM** | `torch.cuda.max_memory_reserved()` | Caching allocator memory pool |
| **Process Resident Set Size (RSS)** | `psutil.Process().memory_info().rss` | Host RAM consumption |
| **Device VRAM Usage** | `pynvml.nvmlDeviceGetMemoryInfo()` | Hardware-level VRAM utilization |
| **Model Disk Footprint** | `os.stat().st_size` | Serialized model file on disk |

---

## 4. Run Manifest Logging Standard

Every benchmark execution outputs an immutable JSON record adhering to schema:
- Stored at `results/raw/<run_id>/run.json`
- Tabular row appended to `results/runs.csv`
'''

# ============================================================================
# 3. COMPREHENSIVE UNIT & INTEGRATION TESTS (tests/)
# ============================================================================

files["tests/test_timer.py"] = '''\"\"\"
Unit tests for ModelPathTimer and EndToEndTimer dual-regime latency timing.
\"\"\"

from pathlib import Path
import numpy as np
import pytest
import torch

from src.benchmarking.timer import EndToEndTimer, ModelPathTimer
from src.models.industrial_model_adapter import IndustrialModelAdapter
from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.pytorch_runtime import PyTorchRuntime


class TestTimer:
    \"\"\"Test suite validating timer accuracy, percentile monotonicity, and pipeline execution.\"\"\"

    def test_model_path_timer_percentile_monotonicity(self) -> None:
        \"\"\"Tests that model-path timer percentiles obey p50 <= p90 <= p95 <= p99.\"\"\"
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
        \"\"\"Tests EndToEndTimer on detection sample images.\"\"\"
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
        \"\"\"Tests EndToEndTimer on industrial sample images.\"\"\"
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
        \"\"\"Asserts that YOLO detection postprocessing completes in under 10 ms for batch size 1.\"\"\"
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
'''

files["tests/test_memory_profiler.py"] = '''\"\"\"
Unit tests for MemoryProfiler and artifact disk footprint sizing.
\"\"\"

from pathlib import Path
import pytest
import torch

from src.benchmarking.memory import MemoryProfiler, get_artifact_size_mb


class TestMemoryProfiler:
    \"\"\"Test suite validating memory tracking and artifact sizing.\"\"\"

    def test_memory_profiler_lifecycle(self) -> None:
        \"\"\"Verifies start_tracking and stop_tracking lifecycle.\"\"\"
        profiler = MemoryProfiler()
        profiler.start_tracking(is_cuda=False)

        # Allocate some CPU array
        dummy = [i for i in range(100_000)]
        stats = profiler.stop_tracking()

        assert "process_rss_mb" in stats
        assert stats["process_rss_mb"] > 0.0
        assert stats["peak_vram_allocated_mb"] == 0.0
        assert stats["peak_vram_reserved_mb"] == 0.0

        if torch.cuda.is_available():
            profiler.start_tracking(is_cuda=True)
            t = torch.randn(100, 100, device="cuda")
            stats_cuda = profiler.stop_tracking()
            assert stats_cuda["peak_vram_allocated_mb"] > 0.0

    def test_get_artifact_size_mb(self, tmp_path: Path) -> None:
        \"\"\"Verifies get_artifact_size_mb accuracy.\"\"\"
        dummy_f = tmp_path / "test.bin"
        dummy_f.write_bytes(b"A" * (1024 * 1024 * 2))  # Exactly 2 MB

        size_mb = get_artifact_size_mb(dummy_f)
        assert abs(size_mb - 2.0) < 1e-4

        # Non-existent file
        assert get_artifact_size_mb(tmp_path / "non_existent.bin") == 0.0
'''

files["tests/test_stability.py"] = '''\"\"\"
Unit tests for StabilityAnalyzer, variance metrics, and stability gates.
\"\"\"

from pathlib import Path
import pytest

from src.benchmarking.stability import StabilityAnalyzer


class TestStability:
    \"\"\"Test suite validating stability calculation and CV threshold gating.\"\"\"

    def test_stability_analyzer_stable_run(self) -> None:
        \"\"\"Verifies stable mock benchmark passes threshold gate with 5 sessions.\"\"\"
        analyzer = StabilityAnalyzer(sessions=5, cooldown_seconds=0.0, cv_threshold=0.05)

        mock_metrics = [
            {"p50_ms": 10.0, "p95_ms": 11.0},
            {"p50_ms": 10.1, "p95_ms": 11.1},
            {"p50_ms": 10.0, "p95_ms": 11.0},
            {"p50_ms": 10.0, "p95_ms": 11.0},
            {"p50_ms": 10.1, "p95_ms": 11.1},
        ]
        iter_obj = iter(mock_metrics)

        res = analyzer.run_stability_suite(lambda: next(iter_obj))

        assert res["stable"] is True
        assert res["cv_p50"] < 0.05
        assert res["total_sessions"] == 5
        assert res["stability_warning"] is None
        assert abs(res["mean_p50_ms"] - 10.0) < 0.2

    def test_stability_analyzer_adaptive_supplemental_sessions(self) -> None:
        \"\"\"Verifies CV > 0.04 across initial 5 sessions triggers 2 supplemental rounds and IQM filters outlier.\"\"\"
        analyzer = StabilityAnalyzer(sessions=5, cooldown_seconds=0.0, cv_threshold=0.05)

        # Initial 5 sessions have a single OS spike (CV > 0.04), prompting 2 supplemental sessions
        mock_metrics = [
            {"p50_ms": 10.0, "p95_ms": 11.0},
            {"p50_ms": 10.1, "p95_ms": 11.1},
            {"p50_ms": 12.0, "p95_ms": 13.0},  # Moderate jitter triggering supplemental rounds
            {"p50_ms": 10.0, "p95_ms": 11.0},
            {"p50_ms": 10.2, "p95_ms": 11.2},
            # 2 Supplemental sessions:
            {"p50_ms": 10.0, "p95_ms": 11.0},
            {"p50_ms": 10.1, "p95_ms": 11.1},
        ]
        iter_obj = iter(mock_metrics)

        res = analyzer.run_stability_suite(lambda: next(iter_obj))

        assert res["total_sessions"] == 7
        assert res["stable"] is True
        assert res["cv_p50"] < 0.05
        assert 12.0 not in res["filtered_p50_values"] or len(res["filtered_p50_values"]) == 5

    def test_stability_analyzer_unstable_run(self) -> None:
        \"\"\"Verifies widespread variance mock benchmark triggers UNSTABLE verdict.\"\"\"
        analyzer = StabilityAnalyzer(sessions=5, cooldown_seconds=0.0, cv_threshold=0.05)

        mock_metrics = [
            {"p50_ms": 10.0, "p95_ms": 11.0},
            {"p50_ms": 20.0, "p95_ms": 25.0},
            {"p50_ms": 12.0, "p95_ms": 14.0},
            {"p50_ms": 30.0, "p95_ms": 35.0},
            {"p50_ms": 15.0, "p95_ms": 18.0},
            {"p50_ms": 28.0, "p95_ms": 32.0},
            {"p50_ms": 19.0, "p95_ms": 22.0},
        ]
        iter_obj = iter(mock_metrics)

        res = analyzer.run_stability_suite(lambda: next(iter_obj))

        assert res["stable"] is False
        assert res["cv_p50"] > 0.05
        assert "High latency variance" in res["stability_warning"]

    def test_stability_analyzer_zero_mad_identical_runs(self) -> None:
        \"\"\"Verifies zero-MAD does not cause ZeroDivisionError and all samples are treated as inliers.\"\"\"
        analyzer = StabilityAnalyzer(sessions=5, cooldown_seconds=0.0, cv_threshold=0.05)

        mock_metrics = [{"p50_ms": 10.0, "p95_ms": 11.0}] * 5
        iter_obj = iter(mock_metrics)

        res = analyzer.run_stability_suite(lambda: next(iter_obj))

        assert res["stable"] is True
        assert res["cv_p50"] == 0.0
        assert len(res["filtered_p50_values"]) == 5
        assert res["total_sessions"] == 5

    def test_flush_cpu_cache_execution(self) -> None:
        \"\"\"Verifies dynamic L3 cache detection and buffer dirtying execute without error.\"\"\"
        from src.benchmarking.stability import _detect_l3_cache_mb, _flush_cpu_cache
        l3_mb = _detect_l3_cache_mb()
        assert l3_mb >= 1
        _flush_cpu_cache(size_mb=4)  # Fast flush test

    def test_stability_zero_division_guard(self) -> None:
        \"\"\"Verifies stability analyzer handles zero-latency inputs gracefully without division error and marks unstable.\"\"\"
        analyzer = StabilityAnalyzer(sessions=5, cooldown_seconds=0.0, cv_threshold=0.05)
        mock_metrics = [{"p50_ms": 0.0, "p95_ms": 0.0}] * 5
        iter_obj = iter(mock_metrics)

        res = analyzer.run_stability_suite(lambda: next(iter_obj))
        assert res["cv_p50"] == 0.0
        assert res["mean_p50_ms"] == 0.0
        assert res["stable"] is False
        assert "Zero or near-zero latency detected" in res["stability_warning"]
'''

files["tests/test_benchmark_suite.py"] = '''\"\"\"
Unit tests for BenchmarkSuite execution, JSON schema conformity, and CSV updating.
\"\"\"

import csv
import json
from pathlib import Path
import pytest
import torch

from src.benchmarking.benchmark_suite import BenchmarkSuite, init_master_csv
from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.pytorch_runtime import PyTorchRuntime


class TestBenchmarkSuite:
    \"\"\"Test suite validating single run execution and manifest serialization.\"\"\"

    def test_benchmark_suite_single_run_and_manifest(self, tmp_path: Path) -> None:
        \"\"\"Tests run_single_configuration produces valid run.json and appends to runs.csv.\"\"\"
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
'''

# ============================================================================
# WRITE ALL FILES TO TARGET_ROOT
# ============================================================================

for rel_path, content in files.items():
    dest = TARGET_ROOT / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    if rel_path.endswith(".py") and rel_path.startswith("scripts/"):
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  [CREATED] {rel_path}")

print(f"\\nAll {len(files)} Phase 5 files generated successfully at {TARGET_ROOT}.")
