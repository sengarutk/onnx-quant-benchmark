"""
Unit tests for Non-Parametric Bootstrap Confidence Intervals & Wilcoxon Paired Tests.
"""

import numpy as np
import pytest

from src.analysis.stats import bootstrap_confidence_interval, wilcoxon_paired_test


class TestStatsCI:
    """Test suite asserting statistical bootstrap coverage and hypothesis test properties."""

    def test_bootstrap_ci_coverage_normal(self) -> None:
        """Asserts that 95% bootstrap CI contains true parameter on synthetic normal data."""
        np.random.seed(42)
        data = np.random.normal(loc=10.0, scale=1.0, size=500)

        point, ci_low, ci_high = bootstrap_confidence_interval(data, stat_fn=np.mean, n_boot=2000, alpha=0.05, seed=42)

        assert np.isclose(point, np.mean(data))
        assert ci_low < point < ci_high
        assert 9.8 <= ci_low <= 10.0
        assert 10.0 <= ci_high <= 10.2

    def test_bootstrap_ci_reproducibility(self) -> None:
        """Asserts exact numerical reproducibility under identical seeds."""
        data = np.array([1.2, 2.3, 3.4, 4.5, 5.6, 6.7])
        p1, l1, h1 = bootstrap_confidence_interval(data, seed=123)
        p2, l2, h2 = bootstrap_confidence_interval(data, seed=123)
        assert p1 == p2 and l1 == l2 and h1 == h2

    def test_bootstrap_ci_constant_and_single(self) -> None:
        """Asserts constant and single-element arrays return degenerate zero-width intervals."""
        single = np.array([42.0])
        p, l, h = bootstrap_confidence_interval(single)
        assert p == l == h == 42.0

        constant = np.array([5.0, 5.0, 5.0, 5.0])
        p, l, h = bootstrap_confidence_interval(constant)
        assert p == l == h == 5.0

    def test_bootstrap_ci_empty_raises(self) -> None:
        """Asserts empty array raises ValueError."""
        with pytest.raises(ValueError, match="empty data array"):
            bootstrap_confidence_interval(np.array([]))

    def test_wilcoxon_paired_test_identical_and_different(self) -> None:
        """Asserts Wilcoxon returns p=1.0 for identical data and p<0.05 for strongly shifted data."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        res_ident = wilcoxon_paired_test(a, a)
        assert res_ident["p_value"] == 1.0
        assert not res_ident["significant_05"]

        b = a + np.array([10.0, 12.0, 11.0, 15.0, 14.0, 13.0, 16.0, 12.0, 14.0, 15.0])
        res_diff = wilcoxon_paired_test(b, a)
        assert res_diff["p_value"] < 0.05
        assert res_diff["significant_05"]

    def test_wilcoxon_mismatched_length_raises(self) -> None:
        """Asserts mismatched array lengths raise ValueError."""
        with pytest.raises(ValueError, match="equal length"):
            wilcoxon_paired_test([1, 2], [1, 2, 3])


def test_holm_bonferroni_step_down() -> None:
    from src.analysis.stats import holm_bonferroni
    res = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.03})
    # sorted p: a=0.01 (x3=0.03), c=0.03 (x2=0.06), b=0.04 (x1=0.04 -> monotone 0.06)
    assert res["a"]["reject"] is True and abs(res["a"]["p_adjusted"] - 0.03) < 1e-12
    assert res["c"]["reject"] is False and abs(res["c"]["p_adjusted"] - 0.06) < 1e-12
    assert res["b"]["reject"] is False and abs(res["b"]["p_adjusted"] - 0.06) < 1e-12
