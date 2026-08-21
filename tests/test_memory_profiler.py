"""
Unit tests for MemoryProfiler and artifact disk footprint sizing.
"""

from pathlib import Path
import pytest
import torch

from src.benchmarking.memory import MemoryProfiler, get_artifact_size_mb


class TestMemoryProfiler:
    """Test suite validating memory tracking and artifact sizing."""

    def test_memory_profiler_lifecycle(self) -> None:
        """Verifies start_tracking and stop_tracking lifecycle."""
        profiler = MemoryProfiler()
        profiler.start_tracking(is_cuda=False)

        # Allocate some CPU array
        dummy = [i for i in range(100_000)]
        stats = profiler.stop_tracking()

        assert "process_rss_mb" in stats
        assert stats["process_rss_mb"] > 0.0
        assert stats["peak_vram_allocated_mb"] == 0.0
        assert stats["peak_vram_reserved_mb"] == 0.0

        if torch.cuda.is_available():
            profiler.start_tracking(is_cuda=True)
            t = torch.randn(100, 100, device="cuda")
            stats_cuda = profiler.stop_tracking()
            assert stats_cuda["peak_vram_allocated_mb"] > 0.0

    def test_get_artifact_size_mb(self, tmp_path: Path) -> None:
        """Verifies get_artifact_size_mb accuracy."""
        dummy_f = tmp_path / "test.bin"
        dummy_f.write_bytes(b"A" * (1024 * 1024 * 2))  # Exactly 2 MB

        size_mb = get_artifact_size_mb(dummy_f)
        assert abs(size_mb - 2.0) < 1e-4

        # Non-existent file
        assert get_artifact_size_mb(tmp_path / "non_existent.bin") == 0.0
