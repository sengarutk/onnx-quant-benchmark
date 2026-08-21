"""
Unit tests for 2D Non-Dominated Pareto Frontier Extraction.
"""

import pandas as pd
import pytest

from src.analysis.pareto import identify_pareto_frontier, get_model_pareto_summary


class TestParetoFrontier:
    """Test suite validating mathematical Pareto optimality."""

    def test_pareto_frontier_synthetic_points(self) -> None:
        """
        Tests Pareto dominance on known synthetic coordinates:
          Points (Latency [Min], Accuracy [Max]):
            A: (10, 0.90) -> Pareto optimal (fast, good acc)
            B: (20, 0.95) -> Pareto optimal (slower, best acc)
            C: (5, 0.80)  -> Pareto optimal (fastest, lowest acc)
            D: (15, 0.85) -> Dominated by A (A is faster AND more accurate)
            E: (25, 0.90) -> Dominated by A and B
        """
        df = pd.DataFrame([
            {"model": "test", "name": "A", "latency": 10.0, "accuracy": 0.90},
            {"model": "test", "name": "B", "latency": 20.0, "accuracy": 0.95},
            {"model": "test", "name": "C", "latency": 5.0, "accuracy": 0.80},
            {"model": "test", "name": "D", "latency": 15.0, "accuracy": 0.85},
            {"model": "test", "name": "E", "latency": 25.0, "accuracy": 0.90},
        ])

        pareto = identify_pareto_frontier(
            df,
            objective_x="latency",
            objective_y="accuracy",
            minimize_x=True,
            maximize_y=True,
        )

        assert len(pareto) == 3
        pareto_names = set(pareto["name"].tolist())
        assert pareto_names == {"A", "B", "C"}
        assert "D" not in pareto_names
        assert "E" not in pareto_names

    def test_pareto_empty_and_single_point(self) -> None:
        """Verifies edge cases: empty DataFrame and single point."""
        empty_df = pd.DataFrame()
        assert identify_pareto_frontier(empty_df, "x", "y").empty

        single_df = pd.DataFrame([{"model": "test", "x": 1.0, "y": 2.0}])
        pareto = identify_pareto_frontier(single_df, "x", "y")
        assert len(pareto) == 1

    def test_get_model_pareto_summary(self) -> None:
        """Verifies model-specific Pareto summary extraction."""
        df = pd.DataFrame([
            {"model": "yolo_nano", "p50_e2e_ms": 10.0, "quality_value": 0.35, "peak_vram_mb": 100.0, "p50_model_ms": 8.0},
            {"model": "yolo_nano", "p50_e2e_ms": 5.0, "quality_value": 0.34, "peak_vram_mb": 50.0, "p50_model_ms": 4.0},
        ])
        summary = get_model_pareto_summary(df, "yolo_nano")
        assert summary["model"] == "yolo_nano"
        assert len(summary["quality_vs_latency"]) >= 1
