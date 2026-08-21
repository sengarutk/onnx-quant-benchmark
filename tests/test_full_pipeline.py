"""
Unit tests validating Master Pipeline Orchestrator, Deduplication, and Repository Metadata.
"""

import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest

from src.analysis.aggregate_results import aggregate_benchmark_runs


class TestFullPipeline:
    """Test suite for pipeline orchestration and packaging integrity."""

    def test_aggregator_multi_run_deduplication(self, tmp_path: Path) -> None:
        """Verifies that older runs for the same configuration are cleanly deduplicated."""
        raw_dir = tmp_path / "raw"
        out_csv = tmp_path / "runs.csv"

        # Manifest 1: Older run (2026-08-20T10:00:00)
        run1_dir = raw_dir / "run_older"
        run1_dir.mkdir(parents=True)
        m1 = {
            "run_id": "run_older",
            "model_name": "yolo_nano",
            "runtime_name": "ORT_CPU",
            "provider": "CPUExecutionProvider",
            "precision": "fp32",
            "timestamp": "2026-08-20T10:00:00",
            "model_path_latency_ms": {"p50_ms": 10.0},
            "end_to_end_latency_ms": {"p50_e2e_ms": 15.0},
        }
        (run1_dir / "run.json").write_text(json.dumps(m1))

        # Manifest 2: Newer run (2026-08-20T12:00:00) with updated latency
        run2_dir = raw_dir / "run_newer"
        run2_dir.mkdir(parents=True)
        m2 = {
            "run_id": "run_newer",
            "model_name": "yolo_nano",
            "runtime_name": "ORT_CPU",
            "provider": "CPUExecutionProvider",
            "precision": "fp32",
            "timestamp": "2026-08-20T12:00:00",
            "model_path_latency_ms": {"p50_ms": 4.5},
            "end_to_end_latency_ms": {"p50_e2e_ms": 8.0},
        }
        (run2_dir / "run.json").write_text(json.dumps(m2))

        df = aggregate_benchmark_runs(raw_dir, out_csv)
        assert len(df) == 1, "Aggregator must deduplicate to exactly one record per unique config"
        assert df.iloc[0]["run_id"] == "run_newer"
        assert df.iloc[0]["p50_model_ms"] == 4.5

    def test_run_full_pipeline_dry_run(self) -> None:
        """Verifies scripts/run_full_pipeline.py dry-run mode completes with exit code 0."""
        root = Path(__file__).resolve().parent.parent
        script = root / "scripts" / "run_full_pipeline.py"
        assert script.is_file()

        res = subprocess.run([sys.executable, str(script), "--dry-run"], cwd=str(root))
        assert res.returncode == 0

    def test_metadata_files_exist_and_non_empty(self) -> None:
        """Verifies README.md, LICENSE, CITATION.cff, and Makefile exist and are non-empty."""
        root = Path(__file__).resolve().parent.parent
        readme = root / "README.md"
        license_f = root / "LICENSE"
        citation = root / "CITATION.cff"
        makefile = root / "Makefile"

        assert readme.is_file() and len(readme.read_text(encoding="utf-8")) > 500
        assert license_f.is_file() and "Apache License" in license_f.read_text(encoding="utf-8")
        assert citation.is_file() and "cff-version" in citation.read_text(encoding="utf-8")
        assert makefile.is_file() and "pipeline:" in makefile.read_text(encoding="utf-8")
