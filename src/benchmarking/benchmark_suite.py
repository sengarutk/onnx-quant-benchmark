"""
Benchmark Suite: profiles one (model, runtime, precision) configuration and serializes an auditable run record.

Every record contains:
  * model-path latency pooled over all stability sessions (+ bootstrap 95% CIs) and the session-median CV gate,
  * end-to-end latency with a measured per-stage breakdown,
  * memory usage,
  * numerical parity of ALL output tensors against the PyTorch FP32 CPU reference on a real test image,
  * task quality measured with this runtime on the full test split, and the delta vs the PyTorch FP32 reference,
  * the acceptance thresholds that were applied (from configs/<model>/<precision>.yaml).

Records are written to results/raw/<session_id>/<run_id>/run.json; results/runs.csv is derived from them by
src/analysis/aggregate_results.py (single source of truth).
"""

import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch

from src.benchmarking.memory import MemoryProfiler, get_artifact_size_mb
from src.benchmarking.stability import StabilityAnalyzer
from src.benchmarking.timer import EndToEndTimer, ModelPathTimer
from src.common.config import QualityThresholdConfig
from src.common.environment import collect_environment_manifest
from src.common.hashes import compute_file_sha256
from src.common.logging import setup_logger
from src.runtimes.base import BaseRuntime
from src.validation.output_checks import compare_output_sets

logger = setup_logger("benchmark_suite")
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def new_session_id() -> str:
    return datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%SZ")


class BenchmarkSuite:
    """Runs the full measurement protocol for one configuration at a time."""

    def __init__(
        self,
        results_dir: Optional[Union[str, Path]] = None,
        session_id: Optional[str] = None,
        warmup_model: int = 50,
        timed_model: int = 300,
        warmup_e2e: int = 20,
        timed_e2e: int = 200,
        stability_sessions: int = 5,
        cooldown_seconds: float = 0.5,
        cv_threshold: float = 0.05,
    ) -> None:
        self.results_dir = Path(results_dir or PROJECT_ROOT / "results")
        self.session_id = session_id or new_session_id()
        self.session_dir = self.results_dir / "raw" / self.session_id
        self.protocol = {
            "warmup_iterations": warmup_model,
            "timed_iterations": timed_model,
            "stability_sessions": stability_sessions,
            "cooldown_seconds": cooldown_seconds,
            "cv_threshold": cv_threshold,
            "e2e_warmup_iterations": warmup_e2e,
            "e2e_timed_iterations": timed_e2e,
        }
        self.memory_profiler = MemoryProfiler()
        self.stability_analyzer = StabilityAnalyzer(
            sessions=stability_sessions, cooldown_seconds=cooldown_seconds, cv_threshold=cv_threshold
        )
        self.env_manifest = collect_environment_manifest()

    def run_single_configuration(
        self,
        model_name: str,
        runtime_name: str,
        precision: str,
        runtime: BaseRuntime,
        input_tensor: torch.Tensor,
        sample_image_paths: List[Path],
        adapter: Any,
        task: str,
        reference_outputs: List[np.ndarray],
        quality_fn: Callable[[BaseRuntime], Dict[str, float]],
        reference_quality: Dict[str, float],
        quality_metric: str,
        thresholds: QualityThresholdConfig,
        model_file_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Profiles one configuration.

        Args:
            input_tensor: Preprocessed real test image [1, C, H, W] on the runtime's device.
            reference_outputs: PyTorch FP32 CPU outputs for ``input_tensor`` (all output tensors, in order).
            quality_fn: Callable evaluating task metrics with ``runtime`` on the full test split.
            reference_quality: Metrics of the PyTorch FP32 CPU reference on the same split.
            quality_metric: Key of the gated metric ('mAP_50' or 'image_auroc').
            thresholds: Acceptance gates for this precision tier.
        """
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_id = f"{model_name}_{runtime_name}_{precision}_{stamp}"
        logger.info(f"--- Benchmark configuration: {run_id} ---")

        gc.collect()
        provider = runtime.get_active_provider()
        is_cuda = any(k in provider.lower() for k in ("cuda", "tensorrt"))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        self.memory_profiler.start_tracking(is_cuda=is_cuda)

        # 1. Model-path latency across stability sessions
        timer = ModelPathTimer(self.protocol["warmup_iterations"], self.protocol["timed_iterations"], device=provider)
        stab = self.stability_analyzer.run_stability_suite(
            lambda: timer.benchmark_device(runtime, input_tensor), batch_size=int(input_tensor.shape[0])
        )
        lat = stab["metrics"]

        # 2. End-to-end latency with stage breakdown
        e2e_timer = EndToEndTimer(self.protocol["e2e_warmup_iterations"], self.protocol["e2e_timed_iterations"])
        if task == "detection":
            e2e = e2e_timer.benchmark_e2e_detection(runtime, sample_image_paths, adapter)
        else:
            e2e = e2e_timer.benchmark_e2e_industrial(runtime, sample_image_paths, adapter)

        mem = self.memory_profiler.stop_tracking()

        # 3. Numerical parity (all outputs) on the same real input
        in_name = next(iter(runtime.get_input_spec()))
        cand = runtime.predict({in_name: input_tensor.detach().cpu().numpy()})
        out_names = list(cand.keys())
        parity = compare_output_sets(reference_outputs, [np.asarray(v, dtype=np.float32) for v in cand.values()], out_names)
        numerical_gate = bool(
            parity["max_rel_error"] <= thresholds.max_rel_error
            and parity["cosine_similarity"] >= thresholds.min_cosine_similarity
            and parity["nan_count"] == 0
            and parity["inf_count"] == 0
        )

        # 4. Task quality on the full test split
        quality = quality_fn(runtime)
        q_value = float(quality[quality_metric])
        q_ref = float(reference_quality[quality_metric])
        q_delta = q_value - q_ref
        max_drop = thresholds.max_map_drop if quality_metric.startswith("mAP") else thresholds.max_auroc_drop
        quality_passed = bool(q_delta >= -max_drop)

        # 5. Model size
        if model_file_path and Path(model_file_path).is_file():
            size_mb, size_source = get_artifact_size_mb(model_file_path), "serialized_file"
            model_sha = compute_file_sha256(model_file_path)
        else:
            size_mb, size_source = (runtime.get_model_size_mb() if hasattr(runtime, "get_model_size_mb") else 0.0), "parameters_and_buffers"
            model_sha = None

        reasons = []
        if not stab["stable"]:
            reasons.append("UNSTABLE_LATENCY")
        if not numerical_gate:
            reasons.append("NUMERICAL_GATE_FAIL")
        if not quality_passed:
            reasons.append("QUALITY_GATE_FAIL")
        status = "PASS" if not reasons else "+".join(reasons)

        record = {
            "run_id": run_id,
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "model_metadata": {
                "model": model_name,
                "task": task,
                "precision": precision,
                "batch_size": int(input_tensor.shape[0]),
                "input_shape": list(input_tensor.shape),
                "model_size_mb": round(float(size_mb), 4),
                "model_size_source": size_source,
                "model_file": str(Path(model_file_path).relative_to(PROJECT_ROOT)) if model_file_path else None,
                "model_sha256": model_sha,
            },
            "runtime_metadata": {"runtime": runtime_name, "provider": provider},
            "protocol": self.protocol,
            "environment_manifest": self.env_manifest.model_dump(),
            "model_path_latency_ms": lat,
            "end_to_end_latency_ms": {
                "p50": e2e["e2e_p50_ms"], "p90": e2e["e2e_p90_ms"], "p95": e2e["e2e_p95_ms"], "p99": e2e["e2e_p99_ms"],
                "mean": e2e["e2e_mean_ms"], "std": e2e["e2e_std_ms"], "throughput_fps": e2e["e2e_throughput_fps"],
                "iterations": e2e["e2e_iterations"],
                "stages_p50": {
                    "preprocess": e2e["stage_preprocess_p50_ms"],
                    "inference": e2e["stage_inference_p50_ms"],
                    "postprocess": e2e["stage_postprocess_p50_ms"],
                },
                "stages_mean": {
                    "preprocess": e2e["stage_preprocess_mean_ms"],
                    "inference": e2e["stage_inference_mean_ms"],
                    "postprocess": e2e["stage_postprocess_mean_ms"],
                },
            },
            "memory_utilization": mem,
            "stability_audit": {k: v for k, v in stab.items() if k != "metrics"},
            "numerical_checks": {
                **{k: v for k, v in parity.items()},
                "thresholds": {"max_rel_error": thresholds.max_rel_error,
                               "min_cosine_similarity": thresholds.min_cosine_similarity},
                "passed": numerical_gate,
            },
            "quality_audit": {
                "metric": quality_metric,
                "value": q_value,
                "reference_value": q_ref,
                "delta": q_delta,
                "max_allowed_drop": max_drop,
                "all_metrics": quality,
                "reference_metrics": reference_quality,
                "passed": quality_passed,
            },
        }

        run_dir = self.session_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run.json").write_text(json.dumps(record, indent=2), encoding="utf-8")

        ci = lat["p50_ci95_ms"]
        logger.info(
            f"{run_id}: model p50 {lat['p50_ms']:.3f} ms [{ci[0]:.3f}, {ci[1]:.3f}] | e2e p50 {e2e['e2e_p50_ms']:.2f} ms | "
            f"CV {stab['cv_p50']:.3f} | rel.err {parity['max_rel_error']:.2e} | {quality_metric} {q_value:.4f} "
            f"({q_delta:+.4f}) | {status}"
        )
        return record
