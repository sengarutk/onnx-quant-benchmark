"""
Unit tests for StabilityAnalyzer, variance metrics, and stability gates.
"""

from pathlib import Path
import pytest

from src.benchmarking.stability import StabilityAnalyzer


class TestStability:
    """Test suite validating stability calculation and CV threshold gating."""

    def test_stability_analyzer_stable_run(self) -> None:
        """Verifies stable mock benchmark passes threshold gate with 5 sessions."""
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
        """Verifies CV > 0.04 across initial 5 sessions triggers 2 supplemental rounds and IQM filters outlier."""
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
        """Verifies widespread variance mock benchmark triggers UNSTABLE verdict."""
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
        """Verifies zero-MAD does not cause ZeroDivisionError and all samples are treated as inliers."""
        analyzer = StabilityAnalyzer(sessions=5, cooldown_seconds=0.0, cv_threshold=0.05)

        mock_metrics = [{"p50_ms": 10.0, "p95_ms": 11.0}] * 5
        iter_obj = iter(mock_metrics)

        res = analyzer.run_stability_suite(lambda: next(iter_obj))

        assert res["stable"] is True
        assert res["cv_p50"] == 0.0
        assert len(res["filtered_p50_values"]) == 5
        assert res["total_sessions"] == 5

    def test_flush_cpu_cache_execution(self) -> None:
        """Verifies dynamic L3 cache detection and buffer dirtying execute without error."""
        from src.benchmarking.stability import _detect_l3_cache_mb, _flush_cpu_cache
        l3_mb = _detect_l3_cache_mb()
        assert l3_mb >= 1
        _flush_cpu_cache(size_mb=4)  # Fast flush test

    def test_stability_zero_division_guard(self) -> None:
        """Verifies stability analyzer handles zero-latency inputs gracefully without division error and marks unstable."""
        analyzer = StabilityAnalyzer(sessions=5, cooldown_seconds=0.0, cv_threshold=0.05)
        mock_metrics = [{"p50_ms": 0.0, "p95_ms": 0.0}] * 5
        iter_obj = iter(mock_metrics)

        res = analyzer.run_stability_suite(lambda: next(iter_obj))
        assert res["cv_p50"] == 0.0
        assert res["mean_p50_ms"] == 0.0
        assert res["stable"] is False
        assert "Zero or near-zero latency detected" in res["stability_warning"]
