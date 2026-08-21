"""
Unit tests for GPU inference runtimes (PyTorch CUDA and ORTCUDARuntime with IOBinding).
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch
import onnxruntime as ort

from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.ort_cuda_runtime import ORTCUDARuntime
from src.runtimes.pytorch_runtime import PyTorchRuntime


class TestRuntimesCUDA:
    """Test suite validating CUDA runtime execution, device tensors, and IOBinding."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available on host")
    def test_pytorch_runtime_cuda(self) -> None:
        """Tests PyTorchRuntime on CUDA device."""
        adapter = YOLOAdapter()
        runtime = PyTorchRuntime(adapter.get_pytorch_model(), device="cuda:0", precision="fp32")
        runtime.load()

        assert "CUDA" in runtime.get_active_provider()

        gpu_in = torch.randn(1, 3, 640, 640, device="cuda:0")
        out_dev = runtime.predict_device(gpu_in)

        assert "output0" in out_dev
        assert out_dev["output0"].is_cuda
        assert out_dev["output0"].shape == (1, 84, 8400)

        # Test predict with host array
        host_arr = np.random.randn(1, 3, 640, 640).astype(np.float32)
        host_out = runtime.predict({"images": host_arr})
        assert "output0" in host_out

        runtime.cleanup()

    def test_ort_cuda_runtime_unavailable_or_iobinding(self) -> None:
        """Tests ORTCUDARuntime raises RuntimeError if provider is missing, or tests IOBinding if available."""
        root = Path(__file__).resolve().parent.parent
        onnx_p = root / "models" / "exported" / "yolo_nano_fp32_opset17.onnx"

        if not onnx_p.is_file():
            pytest.skip("Exported ONNX model not found")

        available = ort.get_available_providers()
        runtime = ORTCUDARuntime(onnx_p, device_id=0)

        if "CUDAExecutionProvider" not in available:
            with pytest.raises(RuntimeError) as excinfo:
                runtime.load()
            assert "CUDAExecutionProvider is not available" in str(excinfo.value)
        else:
            runtime.load()
            assert runtime.get_active_provider() == "CUDAExecutionProvider"
            gpu_in = torch.randn(1, 3, 640, 640, device="cuda:0")
            dev_out = runtime.predict_device(gpu_in)
            assert "output0" in dev_out
            runtime.cleanup()

    def test_ort_cuda_runtime_mocked_iobinding_execution(self, tmp_path: Path) -> None:
        """Tests ORTCUDARuntime IOBinding and zero-copy dispatch using mocked ORT session."""
        dummy_onnx = tmp_path / "dummy.onnx"
        dummy_onnx.write_bytes(b"ONNX_DUMMY_GRAPH")

        mock_session = MagicMock()
        mock_binding = MagicMock()
        mock_session.io_binding.return_value = mock_binding
        mock_session.get_providers.return_value = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        inp_mock = MagicMock()
        inp_mock.name = "images"
        inp_mock.shape = [1, 3, 640, 640]
        inp_mock.type = "tensor(float)"

        out_mock = MagicMock()
        out_mock.name = "output0"
        out_mock.shape = [1, 84, 8400]
        out_mock.type = "tensor(float)"

        mock_session.get_inputs.return_value = [inp_mock]
        mock_session.get_outputs.return_value = [out_mock]

        with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]),              patch("onnxruntime.InferenceSession", return_value=mock_session):

            runtime = ORTCUDARuntime(dummy_onnx, device_id=0)
            runtime.load()

            assert runtime.get_active_provider() == "CUDAExecutionProvider"
            assert "images" in runtime.get_input_spec()
            assert "output0" in runtime.get_output_spec()

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dummy_in = torch.randn(1, 3, 640, 640, device=device)
            dev_outs = runtime.predict_device(dummy_in)

            assert "output0" in dev_outs
            mock_session.run_with_iobinding.assert_called_once()

            # Host predict test
            host_in = np.random.randn(1, 3, 640, 640).astype(np.float32)
            host_outs = runtime.predict({"images": host_in})
            assert "output0" in host_outs

            runtime.cleanup()
