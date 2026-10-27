"""Tests for the data-driven deployment decision matrix."""

from pathlib import Path
import pandas as pd

from src.analysis.aggregate_results import aggregate_benchmark_runs
from src.analysis.decision_matrix import synthesize_decision_matrix


class TestDecisionMatrix:
    def test_matrix_uses_only_gated_configurations(self, fake_session, tmp_path: Path) -> None:
        raw, sid, _ = fake_session
        df = aggregate_benchmark_runs(raw)
        out = tmp_path / "matrix.md"
        md = synthesize_decision_matrix(df, out)
        assert out.is_file() and sid in md
        # The numerically failing autoencoder INT8 run is excluded from recommendations and listed as rejected
        rec_lines = [l for l in md.splitlines() if l.startswith("| D.") and "industrial_autoencoder" in l]
        assert rec_lines and "INT8" not in rec_lines[0]
        assert "numerical gate FAIL" in md
        # Scenario A picks the minimum E2E latency detector config and states the budget verdict
        assert "PyTorch (PyTorch_CUDA:0) FP32" in [l for l in md.splitlines() if l.startswith("| A.")][0]
        assert "meets a 10 ms budget" in md
        # No unmeasured marketing claims
        assert "10-30x" not in md and "TensorRT and ORT CUDA deliver" not in md

    def test_empty_dataframe(self) -> None:
        assert "No benchmark data" in synthesize_decision_matrix(pd.DataFrame())
