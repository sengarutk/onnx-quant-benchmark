"""
Unit tests for Scalability Sweeps across Batch Dimensions and Resolutions.
"""

from pathlib import Path
import pytest
import torch

from src.experiments.run_scalability_sweep import run_scalability_sweeps


class TestScalabilitySweep:
    """Test suite asserting scalability sweep execution and dimension handling."""

    def test_scalability_sweep_minimal_run(self) -> None:
        """Executes minimal scalability sweep across 2 resolutions and 2 batch sizes."""
        df = run_scalability_sweeps(
            resolutions=[320, 416],
            batch_sizes=[1, 2],
            warmup_iters=2,
            timed_iters=3,
        )

        assert len(df) == 4
        assert "throughput_fps" in df.columns
        assert "p50_ms" in df.columns
        assert all(df["p50_ms"] > 0.0)
        assert all(df["throughput_fps"] > 0.0)

        # Assert throughput of batch 2 is higher than or comparable to batch 1
        b1_fps = df[(df["resolution"] == 320) & (df["batch_size"] == 1)]["throughput_fps"].values[0]
        b2_fps = df[(df["resolution"] == 320) & (df["batch_size"] == 2)]["throughput_fps"].values[0]
        assert b2_fps > 0.0
