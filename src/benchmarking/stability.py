"""
Multi-session latency stability analysis.

Protocol (per configuration):
  1. Run ``sessions`` independent measurement sessions (each: cache-flush, GC, optional cooldown, then the
     benchmark function, which performs its own warmup + timed loop).
  2. If the coefficient of variation (CV) of the per-session medians exceeds ``supplemental_cv_trigger`` (0.04),
     run ``supplemental_sessions`` (2) extra sessions after a longer cooldown. This happens at most once.
  3. Report percentiles over the POOLED per-iteration latencies of all sessions, with percentile bootstrap 95% CIs,
     and gate stability on the CV of the per-session medians over ALL sessions (``cv_threshold`` = 0.05).
     A MAD-filtered CV (modified z-score <= 3.5) is reported alongside for diagnosis but is not used for gating.
"""

import gc
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import numpy as np
import torch

from src.analysis.stats import bootstrap_confidence_interval
from src.common.logging import setup_logger

logger = setup_logger("stability")

DEFAULT_CV_THRESHOLD: float = 0.05
SUPPLEMENTAL_CV_TRIGGER: float = 0.04


def _detect_l3_cache_mb() -> int:
    """Queries sysfs for the L3 cache size (MB) with a 64 MB fallback."""
    try:
        for p in Path("/sys/devices/system/cpu/").glob("cpu*/cache/index3/size"):
            raw = p.read_text().strip().upper()
            if raw.endswith("K"):
                return max(1, int(raw[:-1]) // 1024)
            if raw.endswith("M"):
                return int(raw[:-1])
    except Exception:
        pass
    return 64


def _flush_cpu_cache(size_mb: Optional[int] = None) -> None:
    """Writes and reads a buffer of max(64 MB, 1.5 x L3) to evict cache lines between sessions."""
    effective_mb = size_mb if size_mb is not None else max(64, int(_detect_l3_cache_mb() * 1.5))
    buf = np.ones(effective_mb * 1024 * 1024, dtype=np.uint8)
    buf += 1
    _ = int(buf.sum())
    del buf


def coefficient_of_variation(values: List[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(arr)) if arr.size else 0.0
    return float(np.std(arr) / mean) if mean > 1e-12 else 0.0


def mad_filter(values: List[float], z_max: float = 3.5) -> List[float]:
    """Removes outliers with modified z-score > z_max (Iglewicz & Hoaglin); returns the input if MAD == 0."""
    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    if mad <= 1e-12:
        return [float(v) for v in arr]
    z = 0.6745 * np.abs(arr - med) / mad
    return [float(v) for v, zi in zip(arr, z) if zi <= z_max]


class StabilityAnalyzer:
    """Runs a benchmark function across sessions and reports pooled percentiles, CIs and a CV stability gate."""

    def __init__(
        self,
        sessions: int = 5,
        cooldown_seconds: float = 0.5,
        cv_threshold: float = DEFAULT_CV_THRESHOLD,
        supplemental_sessions: int = 2,
        supplemental_cv_trigger: float = SUPPLEMENTAL_CV_TRIGGER,
        bootstrap_resamples: int = 2000,
    ) -> None:
        self.sessions = max(2, int(sessions))
        self.cooldown_seconds = max(0.0, cooldown_seconds)
        self.cv_threshold = float(cv_threshold)
        self.supplemental_sessions = max(0, int(supplemental_sessions))
        self.supplemental_cv_trigger = float(supplemental_cv_trigger)
        self.bootstrap_resamples = int(bootstrap_resamples)

    def run_stability_suite(self, benchmark_fn: Callable[[], Dict[str, Any]], batch_size: int = 1) -> Dict[str, Any]:
        """
        Args:
            benchmark_fn: Zero-argument callable returning a dict with ``p50_ms``, ``p95_ms`` and ``latencies_ms``.
            batch_size: Images per forward pass (for throughput = batch_size / p50).
        """
        session_results: List[Dict[str, Any]] = []

        def _round(cooldown: float) -> Dict[str, Any]:
            if cooldown > 0:
                time.sleep(cooldown)
            _flush_cpu_cache()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return benchmark_fn()

        for i in range(self.sessions):
            session_results.append(_round(self.cooldown_seconds if i > 0 else 0.0))

        initial_cv = coefficient_of_variation([s["p50_ms"] for s in session_results])
        supplemental_run = False
        if self.supplemental_sessions and initial_cv > self.supplemental_cv_trigger:
            logger.info(f"Session-median CV={initial_cv:.4f} > {self.supplemental_cv_trigger}; "
                        f"running {self.supplemental_sessions} supplemental sessions")
            for _ in range(self.supplemental_sessions):
                session_results.append(_round(max(1.5, self.cooldown_seconds)))
            supplemental_run = True

        p50s = [float(s["p50_ms"]) for s in session_results]
        cv = coefficient_of_variation(p50s)
        cv_mad = coefficient_of_variation(mad_filter(p50s))
        stable = bool(cv <= self.cv_threshold)
        warning = None
        if float(np.mean(p50s)) <= 1e-9:
            stable = False
            warning = "Zero or near-zero latency detected; cannot assess stability of a degenerate benchmark"
            logger.warning(warning)
        elif not stable:
            warning = f"Session-median CV={cv:.4f} exceeds {self.cv_threshold:.2f} across {len(p50s)} sessions"
            logger.warning(warning)

        pooled = np.concatenate([np.asarray(s["latencies_ms"], dtype=np.float64) for s in session_results])
        pooled_p50 = float(np.percentile(pooled, 50))
        _, p50_lo, p50_hi = bootstrap_confidence_interval(pooled, np.median, n_boot=self.bootstrap_resamples)
        _, p95_lo, p95_hi = bootstrap_confidence_interval(
            pooled, lambda a: np.percentile(a, 95), n_boot=self.bootstrap_resamples
        )

        metrics = {
            "p50_ms": pooled_p50,
            "p90_ms": float(np.percentile(pooled, 90)),
            "p95_ms": float(np.percentile(pooled, 95)),
            "p99_ms": float(np.percentile(pooled, 99)),
            "mean_ms": float(np.mean(pooled)),
            "std_ms": float(np.std(pooled)),
            "min_ms": float(np.min(pooled)),
            "max_ms": float(np.max(pooled)),
            "throughput_fps": float(batch_size * 1000.0 / pooled_p50) if pooled_p50 > 0 else 0.0,
            "p50_ci95_ms": [p50_lo, p50_hi],
            "p95_ci95_ms": [p95_lo, p95_hi],
            "samples": int(pooled.size),
        }

        return {
            "metrics": metrics,
            "session_p50_values": p50s,
            "session_p95_values": [float(s["p95_ms"]) for s in session_results],
            "cv_p50": cv,
            "cv_p50_mad_filtered": cv_mad,
            "initial_cv_p50": initial_cv,
            "supplemental_sessions_run": supplemental_run,
            "stable": stable,
            "stability_warning": warning,
            "total_sessions": len(session_results),
        }
