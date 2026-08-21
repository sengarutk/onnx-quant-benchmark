"""
Unit tests for FallbackAuditor and provider placement validation.
"""

from pathlib import Path
import pytest
import onnxruntime as ort

from src.runtimes.fallback_audit import FallbackAuditor, audit_ort_session, assert_zero_fallback
from src.runtimes.pytorch_runtime import PyTorchRuntime
from src.models.yolo_adapter import YOLOAdapter


class TestFallbackAudit:
    """Test suite validating execution provider auditing and exception trapping."""

    def test_audit_ort_session_cpu_pass(self) -> None:
        """Verifies audit_ort_session passes for valid CPU execution provider."""
        root = Path(__file__).resolve().parent.parent
        onnx_p = root / "models" / "exported" / "yolo_nano_fp32_opset17.onnx"

        if not onnx_p.is_file():
            pytest.skip("Exported ONNX model not found")

        session = ort.InferenceSession(str(onnx_p), providers=["CPUExecutionProvider"])
        report = audit_ort_session(session, "CPUExecutionProvider")

        assert report["requested_provider"] == "CPUExecutionProvider"
        assert report["primary_provider"] == "CPUExecutionProvider"
        assert report["fallback_occurred"] is False

    def test_audit_ort_session_fallback_failure_trapped(self) -> None:
        """Verifies audit_ort_session raises RuntimeError when requested provider is missing."""
        root = Path(__file__).resolve().parent.parent
        onnx_p = root / "models" / "exported" / "yolo_nano_fp32_opset17.onnx"

        if not onnx_p.is_file():
            pytest.skip("Exported ONNX model not found")

        session = ort.InferenceSession(str(onnx_p), providers=["CPUExecutionProvider"])
        with pytest.raises(RuntimeError) as excinfo:
            audit_ort_session(session, "NonExistentGPUExecutionProvider")

        assert "CRITICAL FALLBACK DETECTED" in str(excinfo.value)

    def test_assert_zero_fallback(self) -> None:
        """Tests assert_zero_fallback helper on active runtimes."""
        adapter = YOLOAdapter()
        runtime = PyTorchRuntime(adapter.get_pytorch_model(), device="cpu")
        assert_zero_fallback(runtime)
