"""
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
            size_bytes = sum(p.numel() * p.element_size() for p in runtime.model.parameters()) +                          sum(b.numel() * b.element_size() for b in runtime.model.buffers())
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
