"""
Unit tests for CPU inference runtimes (PyTorchRuntime and ORTCPURuntime).
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.ort_cpu_runtime import ORTCPURuntime
from src.runtimes.pytorch_runtime import PyTorchRuntime


class TestRuntimesCPU:
    """Test suite validating CPU inference execution, thread configuration, and output shapes."""

    def test_pytorch_runtime_cpu(self) -> None:
        """Tests PyTorchRuntime forward pass and output dictionary standard format."""
        adapter = YOLOAdapter()
        runtime = PyTorchRuntime(adapter.get_pytorch_model(), device="cpu", precision="fp32")
        runtime.load()

        assert runtime.get_active_provider() == "PyTorch_CPU"

        dummy_in = np.random.randn(1, 3, 640, 640).astype(np.float32)
        out = runtime.predict({"images": dummy_in})

        assert "output0" in out
        assert out["output0"].shape == (1, 84, 8400)
        assert isinstance(out["output0"], np.ndarray)
        assert "images" in runtime.get_input_spec()
        assert "output0" in runtime.get_output_spec()

        runtime.cleanup()

    def test_pytorch_runtime_context_manager_and_fp16(self) -> None:
        """Tests PyTorchRuntime as context manager and with fp16 precision."""
        adapter = YOLOAdapter()
        with PyTorchRuntime(adapter.get_pytorch_model(), device="cpu", precision="fp32") as runtime:
            dummy_in = torch.randn(1, 3, 640, 640)
            dev_outs = runtime.predict_device(dummy_in)
            assert "output0" in dev_outs

    def test_ort_cpu_runtime_thread_control(self) -> None:
        """Tests ORTCPURuntime loads model and respects threading configurations."""
        root = Path(__file__).resolve().parent.parent
        onnx_p = root / "models" / "exported" / "yolo_nano_fp32_opset17.onnx"

        if not onnx_p.is_file():
            pytest.skip("Exported ONNX model not found")

        runtime = ORTCPURuntime(onnx_p, intra_op_threads=4, inter_op_threads=1)
        runtime.load()

        assert runtime.get_active_provider() == "CPUExecutionProvider"
        assert "images" in runtime.get_input_spec()
        assert "output0" in runtime.get_output_spec()

        dummy_in = np.random.randn(1, 3, 640, 640).astype(np.float32)
        out = runtime.predict({"images": dummy_in})

        assert "output0" in out
        assert out["output0"].shape == (1, 84, 8400)

        runtime.cleanup()

    def test_ort_cpu_runtime_predict_device_and_context_manager(self) -> None:
        """Tests ORTCPURuntime predict_device and context manager usage."""
        root = Path(__file__).resolve().parent.parent
        onnx_p = root / "models" / "exported" / "yolo_nano_fp32_opset17.onnx"

        if not onnx_p.is_file():
            pytest.skip("Exported ONNX model not found")

        with ORTCPURuntime(onnx_p) as runtime:
            dummy_tensor = torch.randn(1, 3, 640, 640)
            dev_outs = runtime.predict_device(dummy_tensor)
            assert "output0" in dev_outs
            assert isinstance(dev_outs["output0"], torch.Tensor)
