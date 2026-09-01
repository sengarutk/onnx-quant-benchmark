"""
Non-Parametric Statistical Utilities: Bootstrap Confidence Intervals & Paired Hypothesis Tests.
"""

from typing import Any, Callable, Dict, Tuple, Union
import numpy as np
import scipy.stats


def bootstrap_confidence_interval(
    data: Union[np.ndarray, list],
    stat_fn: Callable[[np.ndarray], float] = np.median,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Computes a non-parametric empirical bootstrap confidence interval for any statistical estimator.

    Args:
        data: 1D numerical array or list of observations.
        stat_fn: Summary statistic function mapping 1D array to a scalar float (default: np.median).
        n_boot: Number of bootstrap resamples with replacement (default: 2000).
        alpha: Significance level (default: 0.05 for 95% two-sided confidence interval).
        seed: Random seed for reproducible pseudo-random sampling.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper) as float values.

    Raises:
        ValueError: If data is empty or invalid.
    """
    arr = np.asarray(data, dtype=np.float64).ravel()
    if arr.size == 0:
        raise ValueError("Cannot compute bootstrap confidence interval on empty data array.")

    point_estimate = float(stat_fn(arr))

    if arr.size == 1 or np.all(arr == arr[0]):
        return point_estimate, point_estimate, point_estimate

    rng = np.random.default_rng(seed)
    n = arr.size

    # Vectorized bootstrap resampling
    indices = rng.integers(0, n, size=(n_boot, n))
    resamples = arr[indices]

    # Compute estimator on each resample
    boot_stats = np.apply_along_axis(stat_fn, axis=1, arr=resamples)

    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(boot_stats, lower_pct))
    ci_upper = float(np.percentile(boot_stats, upper_pct))

    return point_estimate, ci_lower, ci_upper


def wilcoxon_paired_test(
    data_a: Union[np.ndarray, list],
    data_b: Union[np.ndarray, list],
    alternative: str = "two-sided",
) -> Dict[str, Any]:
    """
    Performs the Wilcoxon signed-rank paired non-parametric test on paired observations.

    Args:
        data_a: First array of paired observations (e.g. baseline FP32 latency or F1).
        data_b: Second array of paired observations (e.g. quantized INT8 latency or calibrated F1).
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        Dictionary containing statistic, p_value, significance flag, and delta summaries.

    Raises:
        ValueError: If inputs have different lengths or are empty.
    """
    arr_a = np.asarray(data_a, dtype=np.float64).ravel()
    arr_b = np.asarray(data_b, dtype=np.float64).ravel()

    if arr_a.size == 0 or arr_b.size == 0:
        raise ValueError("Inputs to wilcoxon_paired_test cannot be empty.")
    if arr_a.size != arr_b.size:
        raise ValueError(f"Input arrays must have equal length, got {arr_a.size} vs {arr_b.size}")

    diff = arr_a - arr_b
    mean_diff = float(np.mean(diff))
    median_diff = float(np.median(diff))

    # If all paired differences are zero, p-value is trivially 1.0
    if np.allclose(arr_a, arr_b) or np.all(diff == 0):
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_05": False,
            "mean_diff": mean_diff,
            "median_diff": median_diff,
            "n_pairs": int(arr_a.size),
        }

    try:
        stat_res = scipy.stats.wilcoxon(arr_a, arr_b, alternative=alternative)
        stat_val = float(stat_res.statistic)
        p_val = float(stat_res.pvalue)
    except Exception:
        stat_val = 0.0
        p_val = 1.0

    return {
        "statistic": stat_val,
        "p_value": p_val,
        "significant_05": bool(p_val < 0.05),
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "n_pairs": int(arr_a.size),
    }
