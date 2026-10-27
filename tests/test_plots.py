"""Tests for session figures."""

from pathlib import Path
import pandas as pd

from src.analysis.aggregate_results import aggregate_benchmark_runs
from src.visualization.plots import config_label, plot_all_figures


class TestPlots:
    def test_plot_all_figures_generation(self, fake_session, tmp_path: Path) -> None:
        raw, _, records = fake_session
        df = aggregate_benchmark_runs(raw)
        paths = plot_all_figures(df, records, tmp_path)
        assert len(paths) == 6
        for p in paths:
            assert p.is_file() and p.stat().st_size > 10_000

    def test_plot_empty_df_graceful_handling(self, tmp_path: Path) -> None:
        paths = plot_all_figures(pd.DataFrame(), [], tmp_path)
        assert all(p.is_file() for p in paths)

    def test_config_label(self) -> None:
        row = pd.Series({"model": "yolo_nano", "runtime": "ORT_CPU", "provider": "CPUExecutionProvider", "precision": "int8"})
        assert config_label(row) == "ORT CPU INT8"
        assert config_label(row, with_model=True) == "YOLO | ORT CPU INT8"
