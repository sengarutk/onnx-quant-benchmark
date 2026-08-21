"""
2D Non-Dominated Pareto Frontier Extraction Engine.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


def identify_pareto_frontier(
    df: pd.DataFrame,
    objective_x: str,
    objective_y: str,
    minimize_x: bool = True,
    maximize_y: bool = True,
) -> pd.DataFrame:
    """
    Identifies the non-dominated Pareto frontier points for a 2-objective optimization space.

    A point (x_i, y_i) dominates (x_j, y_j) if:
      - x_i is better than or equal to x_j
      - y_i is better than or equal to y_j
      - At least one objective is strictly better

    Args:
        df: Input DataFrame containing candidate benchmark configurations.
        objective_x: Name of column for X objective (e.g. latency, VRAM).
        objective_y: Name of column for Y objective (e.g. mAP, AUROC, throughput).
        minimize_x: True if lower X is better (e.g. latency).
        maximize_y: True if higher Y is better (e.g. quality, throughput).

    Returns:
        Sub-DataFrame containing non-dominated Pareto optimal records, sorted by objective_x.
    """
    if df.empty or objective_x not in df.columns or objective_y not in df.columns:
        return df.copy()

    valid_df = df.dropna(subset=[objective_x, objective_y]).copy()
    if valid_df.empty:
        return valid_df

    x_vals = valid_df[objective_x].to_numpy(dtype=np.float64)
    y_vals = valid_df[objective_y].to_numpy(dtype=np.float64)
    n = len(valid_df)

    is_dominated = np.zeros(n, dtype=bool)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            # Check if j dominates i
            x_better_or_equal = (x_vals[j] <= x_vals[i]) if minimize_x else (x_vals[j] >= x_vals[i])
            y_better_or_equal = (y_vals[j] >= y_vals[i]) if maximize_y else (y_vals[j] <= y_vals[i])

            x_strictly_better = (x_vals[j] < x_vals[i]) if minimize_x else (x_vals[j] > x_vals[i])
            y_strictly_better = (y_vals[j] > y_vals[i]) if maximize_y else (y_vals[j] < y_vals[i])

            if x_better_or_equal and y_better_or_equal and (x_strictly_better or y_strictly_better):
                is_dominated[i] = True
                break

    pareto_df = valid_df[~is_dominated].copy()
    pareto_df = pareto_df.sort_values(by=objective_x, ascending=minimize_x).reset_index(drop=True)
    return pareto_df


def get_model_pareto_summary(df: pd.DataFrame, model_name: str) -> Dict[str, Any]:
    """
    Extracts Pareto optimal operating points for a specific model family.

    Args:
        df: Consolidated runs DataFrame.
        model_name: Target model identifier.

    Returns:
        Dictionary summarizing Pareto configurations for quality vs latency and latency vs VRAM.
    """
    model_df = df[df["model"] == model_name].copy()
    if model_df.empty:
        return {"model": model_name, "quality_vs_latency": [], "latency_vs_vram": []}

    # 1. Quality vs E2E Latency (Min Latency, Max Quality)
    p_quality = identify_pareto_frontier(
        model_df,
        objective_x="p50_e2e_ms",
        objective_y="quality_value",
        minimize_x=True,
        maximize_y=True,
    )

    # 2. Latency vs VRAM Footprint (Min Latency, Min VRAM)
    cuda_df = model_df[model_df["peak_vram_mb"] > 0].copy()
    p_vram = identify_pareto_frontier(
        cuda_df,
        objective_x="peak_vram_mb",
        objective_y="p50_model_ms",
        minimize_x=True,
        maximize_y=False,
    )

    return {
        "model": model_name,
        "quality_vs_latency": p_quality.to_dict(orient="records"),
        "latency_vs_vram": p_vram.to_dict(orient="records"),
    }
