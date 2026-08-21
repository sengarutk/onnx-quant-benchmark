"""
Unit tests validating Deployment Decision Matrix synthesis.
"""

from pathlib import Path
import pandas as pd
import pytest

from src.analysis.decision_matrix import synthesize_decision_matrix


class TestDecisionMatrix:
    """Test suite validating deployment recommendations logic."""

    def test_synthesize_decision_matrix(self, tmp_path: Path) -> None:
        """Verifies Markdown table structure and scenario recommendations."""
        df = pd.DataFrame([
            {
                "model": "yolo_nano",
                "runtime": "ORT_CUDA",
                "provider": "CUDAExecutionProvider",
                "precision": "fp16",
                "p50_model_ms": 2.0,
                "p50_e2e_ms": 4.5,
                "model_throughput_fps": 500.0,
                "quality_value": 0.35,
                "quality_delta": -0.001,
                "peak_vram_mb": 120.0,
            },
            {
                "model": "yolo_nano",
                "runtime": "ORT_CPU",
                "provider": "CPUExecutionProvider",
                "precision": "int8",
                "p50_model_ms": 6.0,
                "p50_e2e_ms": 10.5,
                "model_throughput_fps": 160.0,
                "quality_value": 0.345,
                "quality_delta": -0.005,
                "peak_vram_mb": 0.0,
            },
            {
                "model": "industrial_autoencoder",
                "runtime": "PyTorch",
                "provider": "PyTorch_CPU",
                "precision": "fp32",
                "p50_model_ms": 15.0,
                "p50_e2e_ms": 18.0,
                "model_throughput_fps": 66.0,
                "quality_value": 0.995,
                "quality_delta": 0.0,
                "peak_vram_mb": 0.0,
            }
        ])

        out_path = tmp_path / "deployment_decision_matrix.md"
        doc = synthesize_decision_matrix(df, out_path)

        assert out_path.is_file()
        assert "Scenario A: Low-Latency Target" in doc
        assert "Scenario B: Edge Gateway / IPC" in doc
        assert "Scenario C: High-Fidelity Anomaly Inspection" in doc
        assert "Scenario D: High-Throughput Offline Batch" in doc
        assert "ORT_CUDA" in doc
