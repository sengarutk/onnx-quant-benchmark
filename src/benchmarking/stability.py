r"""
Multi-Session Latency Stability Analyzer ($CV \le 0.05$ threshold gating with dynamic L3 cache flush and MAD outlier filtering).
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
    """Queries sysfs for total L3 cache size across CPU cores with safe 64MB fallback."""
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
    """
    Allocates and dirty-writes a contiguous buffer sized >= 1.5x L3 cache (floor 64MB)
    to saturate the memory bus and purge CPU cache lines.
    """
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
    """
    Executes benchmark suites across N=5 controlled rounds with dynamic L3 CPU cache flushing,
    adaptive 2-session supplemental cooldown on CV > 0.04, and robust MAD outlier filtering.
    """

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
        """
        Executes benchmark_fn across sessions with CPU cache flushing, inter-session cooldown,
        and adaptive supplemental sampling when CV > 0.04.

        Args:
            benchmark_fn: Zero-argument callable returning a dictionary of latency metrics (must have 'p50_ms').

        Returns:
            Dictionary containing aggregated metrics, inter-session variance, and stability boolean.
        """
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
