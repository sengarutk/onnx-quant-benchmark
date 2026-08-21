"""
Unit tests validating Markdown summary table generators.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.visualization.report_tables import (
    generate_table1_numerical_correctness,
    generate_table2_latency_throughput,
    generate_table3_memory_footprint,
    generate_table4_quality_retention,
    generate_table5_int8_audit,
    generate_all_tables,
)


class TestReportTables:
    """Test suite validating report table generation."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "model": "yolo_nano",
                "runtime": "PyTorch",
                "provider": "PyTorch_CPU",
                "precision": "fp32",
                "max_abs_error": 0.0,
                "mean_abs_error": 0.0,
                "cosine_similarity": 1.0,
                "quality_delta": 0.0,
                "numerical_gate": True,
                "p50_model_ms": 20.0,
                "p90_model_ms": 22.0,
                "p95_model_ms": 24.0,
                "p99_model_ms": 28.0,
                "p50_e2e_ms": 25.0,
                "model_throughput_fps": 50.0,
                "e2e_throughput_fps": 40.0,
                "peak_vram_mb": 250.0,
                "peak_vram_reserved_mb": 300.0,
                "process_rss_mb": 200.0,
                "model_size_mb": 12.0,
                "storage_compression_pct": 0.0,
                "quality_metric": "mAP_50",
                "quality_value": 0.35,
                "quality_passed": True,
                "speedup_model": 1.0,
            },
            {
                "model": "yolo_nano",
                "runtime": "ORT_CPU",
                "provider": "CPUExecutionProvider",
                "precision": "int8",
                "max_abs_error": 0.02,
                "mean_abs_error": 0.005,
                "cosine_similarity": 0.999,
                "quality_delta": -0.005,
                "numerical_gate": True,
                "p50_model_ms": 5.0,
                "p90_model_ms": 6.0,
                "p95_model_ms": 7.0,
                "p99_model_ms": 8.0,
                "p50_e2e_ms": 10.0,
                "model_throughput_fps": 200.0,
                "e2e_throughput_fps": 100.0,
                "peak_vram_mb": 0.0,
                "peak_vram_reserved_mb": 0.0,
                "process_rss_mb": 150.0,
                "model_size_mb": 3.0,
                "storage_compression_pct": 75.0,
                "quality_metric": "mAP_50",
                "quality_value": 0.345,
                "quality_passed": True,
                "speedup_model": 4.0,
            }
        ])

    def test_generate_all_tables(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Verifies all 5 table markdown files are created and populated."""
        out_dir = tmp_path / "tables"
        generate_all_tables(sample_df, out_dir)

        t1 = out_dir / "table1_numerical_correctness.md"
        t2 = out_dir / "table2_latency_throughput.md"
        t3 = out_dir / "table3_memory_footprint.md"
        t4 = out_dir / "table4_quality_retention.md"
        t5 = out_dir / "table5_int8_quantization_audit.md"

        assert t1.is_file() and "Max Abs Error" in t1.read_text()
        assert t2.is_file() and "Model $p_{50}$" in t2.read_text()
        assert t3.is_file() and "Peak VRAM" in t3.read_text()
        assert t4.is_file() and "Quality" in t4.read_text()
        assert t5.is_file() and "75.0%" in t5.read_text()
