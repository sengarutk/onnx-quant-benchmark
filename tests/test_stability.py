"""Unit tests for the multi-session StabilityAnalyzer."""

import numpy as np

from src.benchmarking.stability import (
    StabilityAnalyzer,
    _detect_l3_cache_mb,
    _flush_cpu_cache,
    coefficient_of_variation,
    mad_filter,
)


def _session(p50: float, n: int = 50, jitter: float = 0.01, seed: int = 0) -> dict:
    lat = list(p50 * (1 + jitter * np.random.default_rng(seed).standard_normal(n)))
    return {"p50_ms": float(np.median(lat)), "p95_ms": float(np.percentile(lat, 95)), "latencies_ms": lat}


def _runner(p50s):
    it = iter([_session(p, seed=i) for i, p in enumerate(p50s)])
    return lambda: next(it)


class TestStability:
    def test_stable_run_pooled_metrics_and_ci(self) -> None:
        an = StabilityAnalyzer(sessions=5, cooldown_seconds=0.0, bootstrap_resamples=200)
        res = an.run_stability_suite(_runner([10.0, 10.05, 10.0, 9.98, 10.02]))
        assert res["stable"] is True and res["total_sessions"] == 5
        assert res["supplemental_sessions_run"] is False
        m = res["metrics"]
        assert m["samples"] == 250
        assert m["p50_ci95_ms"][0] <= m["p50_ms"] <= m["p50_ci95_ms"][1]
        assert np.isclose(m["throughput_fps"], 1000.0 / m["p50_ms"])

    def test_supplemental_sessions_triggered_once(self) -> None:
        an = StabilityAnalyzer(sessions=5, cooldown_seconds=0.0, bootstrap_resamples=100)
        res = an.run_stability_suite(_runner([10.0, 10.1, 12.0, 10.0, 10.2, 10.0, 10.1]))
        assert res["supplemental_sessions_run"] is True
        assert res["total_sessions"] == 7
        # The gate uses ALL sessions; the MAD-filtered CV is diagnostic only
        assert np.isclose(res["cv_p50"], coefficient_of_variation(res["session_p50_values"]))
        assert res["cv_p50_mad_filtered"] < res["cv_p50"]
        assert res["stable"] is (res["cv_p50"] <= 0.05)

    def test_unstable_run(self) -> None:
        an = StabilityAnalyzer(sessions=5, cooldown_seconds=0.0, bootstrap_resamples=100)
        res = an.run_stability_suite(_runner([10, 20, 12, 30, 15, 28, 19]))
        assert res["stable"] is False
        assert "exceeds" in res["stability_warning"]

    def test_configured_session_count_is_respected(self) -> None:
        an = StabilityAnalyzer(sessions=3, cooldown_seconds=0.0, supplemental_sessions=0, bootstrap_resamples=50)
        assert an.run_stability_suite(_runner([5.0, 5.0, 5.0]))["total_sessions"] == 3

    def test_zero_latency_is_flagged(self) -> None:
        an = StabilityAnalyzer(sessions=2, cooldown_seconds=0.0, bootstrap_resamples=20)
        z = {"p50_ms": 0.0, "p95_ms": 0.0, "latencies_ms": [0.0] * 10}
        res = an.run_stability_suite(lambda: z)
        assert res["stable"] is False and "near-zero" in res["stability_warning"]

    def test_mad_filter_and_cv(self) -> None:
        assert mad_filter([10, 10, 10, 10, 50]) == [10, 10, 10, 10, 50]  # MAD = 0 -> unchanged
        assert 50.0 not in mad_filter([10, 10.2, 9.8, 10.1, 50])
        assert coefficient_of_variation([0.0, 0.0]) == 0.0

    def test_flush_cpu_cache_execution(self) -> None:
        assert _detect_l3_cache_mb() >= 1
        _flush_cpu_cache(size_mb=4)
