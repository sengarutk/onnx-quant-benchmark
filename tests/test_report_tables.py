"""Tests for Markdown report tables generated from a session DataFrame."""

from pathlib import Path

from src.analysis.aggregate_results import aggregate_benchmark_runs
from src.visualization.report_tables import generate_all_tables


class TestReportTables:
    def test_generate_all_tables(self, fake_session, tmp_path: Path) -> None:
        raw, _, records = fake_session
        df = aggregate_benchmark_runs(raw)
        generate_all_tables(df, tmp_path)
        names = ["table1_numerical_correctness", "table2_latency_throughput", "table3_memory_footprint",
                 "table4_quality_retention", "table5_int8_quantization_audit"]
        for n in names:
            text = (tmp_path / f"{n}.md").read_text()
            assert "| --- |" in text and "None" not in text
        t1 = (tmp_path / "table1_numerical_correctness.md").read_text()
        assert t1.count("FAIL") == 1  # verdicts come from the run records, not re-thresholding
        t2 = (tmp_path / "table2_latency_throughput.md").read_text()
        assert t2.count("\n") == len(records) + 2
        t5 = (tmp_path / "table5_int8_quantization_audit.md").read_text()
        precisions = {line.split("|")[3].strip() for line in t5.splitlines()[2:]}
        assert precisions == {"FP16", "INT8"}
