"""Tests for session-scoped aggregation of run records."""

import json
from pathlib import Path
import numpy as np
import pytest

from src.analysis.aggregate_results import COLUMNS, aggregate_benchmark_runs, list_sessions
from tests.conftest import make_run_record


class TestAggregateResults:
    def test_missing_session_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            aggregate_benchmark_runs(tmp_path / "raw")

    def test_aggregate_fake_session(self, fake_session, tmp_path: Path) -> None:
        raw, sid, records = fake_session
        out = tmp_path / "runs.csv"
        df = aggregate_benchmark_runs(raw, out)
        assert out.is_file()
        assert list(df.columns) == COLUMNS
        assert len(df) == len(records)
        assert set(df["session_id"]) == {sid}
        # PyTorch CPU FP32 row is the speedup baseline for its model
        base = df[(df["model"] == "yolo_nano") & (df["provider"] == "PyTorch_CPU")].iloc[0]
        assert np.isclose(base["speedup_model"], 1.0)
        int8 = df[(df["model"] == "yolo_nano") & (df["precision"] == "int8")].iloc[0]
        assert np.isclose(int8["speedup_model"], 20.0 / 7.0)
        assert np.isclose(int8["speedup_vs_runtime_fp32"], 9.0 / 7.0)
        assert np.isclose(int8["storage_compression_pct"], (1 - 1.9 / 7.3) * 100)
        assert np.isnan(base["storage_compression_pct"])  # parameter size, not a serialized file

    def test_latest_full_session_selected_and_quick_ignored(self, fake_session) -> None:
        raw, sid, _ = fake_session
        newer = "session_20261027T000000Z"
        rec = make_run_record("yolo_nano", "PyTorch", "PyTorch_CPU", "fp32", 5.0, newer, 7.3, 0.8, 0.8)
        (raw / newer / rec["run_id"]).mkdir(parents=True)
        (raw / newer / rec["run_id"] / "run.json").write_text(json.dumps(rec))
        (raw / newer / "session.json").write_text(json.dumps({"session_id": newer, "quick": True}))
        assert [p.name for p in list_sessions(raw)] == [sid]
        assert aggregate_benchmark_runs(raw)["session_id"].iloc[0] == sid
        df_quick = aggregate_benchmark_runs(raw, include_quick=True)
        assert len(df_quick) == 1 and df_quick["session_id"].iloc[0] == newer
        assert aggregate_benchmark_runs(raw, session_id=sid)["session_id"].iloc[0] == sid
