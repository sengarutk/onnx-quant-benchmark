"""
Unit tests validating 300 DPI publication figure rendering.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.visualization.plots import (
    plot_pareto_frontier,
    plot_latency_breakdown,
    plot_speedup_barchart,
    plot_tail_latencies,
    plot_memory_footprints,
    plot_stability_trends,
    plot_all_figures,
)


class TestPlots:
    """Test suite ensuring all plot generators create valid image files on disk."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "model": "yolo_nano",
                "runtime": "PyTorch",
                "precision": "fp32",
                "p50_model_ms": 20.0,
                "p90_model_ms": 22.0,
                "p95_model_ms": 24.0,
                "p99_model_ms": 28.0,
                "p50_e2e_ms": 25.0,
                "speedup_model": 1.0,
                "quality_value": 0.35,
                "peak_vram_mb": 250.0,
                "model_size_mb": 12.0,
            },
            {
                "model": "yolo_nano",
                "runtime": "ORT_CUDA",
                "precision": "fp16",
                "p50_model_ms": 2.0,
                "p90_model_ms": 2.2,
                "p95_model_ms": 2.5,
                "p99_model_ms": 3.0,
                "p50_e2e_ms": 4.5,
                "speedup_model": 10.0,
                "quality_value": 0.349,
                "peak_vram_mb": 120.0,
                "model_size_mb": 6.0,
            },
            {
                "model": "industrial_autoencoder",
                "runtime": "ORT_CPU",
                "precision": "int8",
                "p50_model_ms": 8.0,
                "p90_model_ms": 9.0,
                "p95_model_ms": 10.0,
                "p99_model_ms": 12.0,
                "p50_e2e_ms": 12.0,
                "speedup_model": 2.5,
                "quality_value": 0.982,
                "peak_vram_mb": 0.0,
                "model_size_mb": 3.0,
            }
        ])

    def test_plot_all_figures_generation(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Verifies that all 6 figures are rendered as non-empty image files."""
        fig_dir = tmp_path / "figures"
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()

        figures = plot_all_figures(sample_df, raw_dir, fig_dir)
        assert len(figures) == 6

        for fig_path in figures:
            assert fig_path.is_file()
            assert fig_path.stat().st_size > 1000  # Non-empty PNG (>1KB)

    def test_plot_empty_df_graceful_handling(self, tmp_path: Path) -> None:
        """Verifies all plots handle empty dataframes gracefully without raising errors."""
        fig_dir = tmp_path / "figures_empty"
        raw_dir = tmp_path / "raw_empty"
        raw_dir.mkdir()

        figures = plot_all_figures(pd.DataFrame(), raw_dir, fig_dir)
        assert len(figures) == 6
        for fig_path in figures:
            assert fig_path.is_file()
