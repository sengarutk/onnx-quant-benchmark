"""
Unit tests for Result Aggregation & Metrics Normalization Engine.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

from src.analysis.aggregate_results import aggregate_benchmark_runs


class TestAggregateResults:
    """Test suite validating raw manifest aggregation and CSV consolidation."""

    def test_aggregate_empty_dir(self, tmp_path: Path) -> None:
        """Verifies graceful handling of empty raw results directory."""
        empty_dir = tmp_path / "raw"
        empty_dir.mkdir()
        out_csv = tmp_path / "runs.csv"

        df = aggregate_benchmark_runs(empty_dir, out_csv)
        assert isinstance(df, pd.DataFrame)
        assert df.empty
        assert out_csv.is_file()

    def test_aggregate_valid_manifests(self, tmp_path: Path) -> None:
        """Verifies proper ingestion, flattening, and derivation computation."""
        raw_dir = tmp_path / "raw"
        out_csv = tmp_path / "runs.csv"

        # Create mock baseline manifest (PyTorch FP32 CPU)
        run1_dir = raw_dir / "yolo_nano_PyTorch_fp32"
        run1_dir.mkdir(parents=True)
        manifest1 = {
            "run_id": "yolo_nano_PyTorch_fp32",
            "model_name": "yolo_nano",
            "task": "detection",
            "runtime_name": "PyTorch",
            "provider": "PyTorch_CPU",
            "precision": "fp32",
            "model_path_latency_ms": {"p50_ms": 20.0, "p90_ms": 22.0, "p95_ms": 25.0, "p99_ms": 30.0, "model_throughput_fps": 50.0},
            "end_to_end_latency_ms": {"p50_e2e_ms": 25.0, "p90_e2e_ms": 28.0, "p95_e2e_ms": 32.0, "p99_e2e_ms": 35.0, "e2e_throughput_fps": 40.0},
            "memory_profile": {"peak_vram_allocated_mb": 0.0, "process_rss_mb": 200.0},
            "model_metadata": {"model_size_mb": 10.0},
            "quality_evaluation": {"metric_name": "mAP_50", "metric_value": 0.35, "metric_delta": 0.0, "passed": True},
            "stability_assessment": {"is_stable": True, "cv_p50": 0.02, "total_sessions": 5},
        }
        (run1_dir / "run.json").write_text(json.dumps(manifest1))

        # Create mock candidate manifest (ORT CPU INT8)
        run2_dir = raw_dir / "yolo_nano_ORT_CPU_int8"
        run2_dir.mkdir(parents=True)
        manifest2 = {
            "run_id": "yolo_nano_ORT_CPU_int8",
            "model_name": "yolo_nano",
            "task": "detection",
            "runtime_name": "ORT_CPU",
            "provider": "CPUExecutionProvider",
            "precision": "int8",
            "model_path_latency_ms": {"p50_ms": 5.0, "p90_ms": 6.0, "p95_ms": 7.0, "p99_ms": 8.0, "model_throughput_fps": 200.0},
            "end_to_end_latency_ms": {"p50_e2e_ms": 10.0, "p90_e2e_ms": 12.0, "p95_e2e_ms": 14.0, "p99_e2e_ms": 16.0, "e2e_throughput_fps": 100.0},
            "memory_profile": {"peak_vram_allocated_mb": 0.0, "process_rss_mb": 150.0},
            "model_metadata": {"model_size_mb": 2.5},
            "quality_evaluation": {"metric_name": "mAP_50", "metric_value": 0.34, "metric_delta": -0.01, "passed": True},
            "stability_assessment": {"is_stable": True, "cv_p50": 0.01, "total_sessions": 5},
        }
        (run2_dir / "run.json").write_text(json.dumps(manifest2))

        df = aggregate_benchmark_runs(raw_dir, out_csv)
        assert len(df) == 2
        assert out_csv.is_file()

        # Check calculated speedup and storage compression
        cand_row = df[df["run_id"] == "yolo_nano_ORT_CPU_int8"].iloc[0]
        assert cand_row["speedup_model"] == 4.0  # 20.0 / 5.0
        assert cand_row["storage_compression_pct"] == 75.0  # (1 - 2.5/10.0) * 100
