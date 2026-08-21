"""
Unit tests for TensorRTRuntime and build_tensorrt_engine error handling and interfaces.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest
import torch

from src.quantization.build_trt_engine import TRT_AVAILABLE, build_tensorrt_engine
import src.quantization.build_trt_engine as trt_builder_mod
from src.runtimes.tensorrt_runtime import TensorRTRuntime
import src.runtimes.tensorrt_runtime as trt_runtime_mod


class TestTensorRTRuntime:
    """Test suite validating TensorRT runtime interfaces and availability guards."""

    def test_tensorrt_unavailable_raises_runtime_error(self, tmp_path: Path) -> None:
        """Ensures informative RuntimeError is raised when TensorRT bindings are not present."""
        if TRT_AVAILABLE:
            pytest.skip("TensorRT is installed; skipping unavailable test")

        dummy_engine = tmp_path / "dummy.engine"
        dummy_engine.write_bytes(b"DUMMY_ENGINE_BYTES")

        runtime = TensorRTRuntime(dummy_engine)
        with pytest.raises(RuntimeError) as excinfo:
            runtime.load()

        assert "TensorRT Python bindings" in str(excinfo.value)

    def test_build_tensorrt_engine_unavailable_raises(self, tmp_path: Path) -> None:
        """Tests build_tensorrt_engine raises when TRT is not installed."""
        if TRT_AVAILABLE:
            pytest.skip("TensorRT is installed; skipping test")

        with pytest.raises(RuntimeError) as excinfo:
            build_tensorrt_engine(tmp_path / "dummy.onnx", tmp_path / "out.engine")
        assert "TensorRT is not available" in str(excinfo.value)

    def test_tensorrt_runtime_mocked_execution(self, tmp_path: Path) -> None:
        """Tests TensorRTRuntime initialization and predict dispatch with mocked TensorRT API."""
        dummy_engine = tmp_path / "mock.engine"
        dummy_engine.write_bytes(b"TRT_MOCK_BYTES")

        mock_trt = MagicMock()
        mock_engine = MagicMock()
        mock_context = MagicMock()
        mock_runtime = MagicMock()
        mock_runtime.deserialize_cuda_engine.return_value = mock_engine
        mock_trt.Runtime.return_value = mock_runtime
        mock_engine.create_execution_context.return_value = mock_context
        mock_engine.num_io_tensors = 2
        mock_engine.get_tensor_name.side_effect = lambda idx: ["images", "output0"][idx]
        mock_engine.get_tensor_shape.side_effect = lambda name: [1, 3, 640, 640] if name == "images" else [1, 84, 8400]
        mock_engine.get_tensor_dtype.return_value = mock_trt.DataType.FLOAT
        mock_engine.get_tensor_mode.side_effect = lambda name: mock_trt.TensorIOMode.INPUT if name == "images" else mock_trt.TensorIOMode.OUTPUT

        with patch.object(trt_runtime_mod, "TRT_AVAILABLE", True),              patch.object(trt_runtime_mod, "trt", mock_trt):

            runtime = TensorRTRuntime(dummy_engine, device_id=0)
            runtime.load()

            assert runtime.get_active_provider() == "TensorRT"
            assert "images" in runtime.get_input_spec()
            assert "output0" in runtime.get_output_spec()

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            dummy_in = torch.randn(1, 3, 640, 640, device=device)
            dev_outs = runtime.predict_device(dummy_in)
            assert "output0" in dev_outs

            # Host predict test
            host_in = np.random.randn(1, 3, 640, 640).astype(np.float32)
            host_outs = runtime.predict({"images": host_in})
            assert "output0" in host_outs

            runtime.cleanup()

    def test_build_tensorrt_engine_mocked_build(self, tmp_path: Path) -> None:
        """Tests build_tensorrt_engine flow with mocked TensorRT builder."""
        dummy_onnx = tmp_path / "model.onnx"
        dummy_onnx.write_bytes(b"ONNX_BYTES")
        engine_out = tmp_path / "model.engine"

        mock_trt = MagicMock()
        mock_trt.__version__ = "10.0.1"
        mock_builder = MagicMock()
        mock_network = MagicMock()
        mock_parser = MagicMock()
        mock_parser.parse.return_value = True
        mock_config = MagicMock()

        mock_trt.Builder.return_value = mock_builder
        mock_builder.create_network.return_value = mock_network
        mock_trt.OnnxParser.return_value = mock_parser
        mock_builder.create_builder_config.return_value = mock_config
        mock_builder.build_serialized_network.return_value = b"MOCK_SERIALIZED_PLAN"

        with patch.object(trt_builder_mod, "TRT_AVAILABLE", True),              patch.object(trt_builder_mod, "trt", mock_trt):

            res = build_tensorrt_engine(
                dummy_onnx,
                engine_out,
                precision="fp16",
                workspace_gb=2.0,
                static_shapes={"images": (1, 3, 640, 640)},
            )
            assert res.is_file()
            assert engine_out.read_bytes() == b"MOCK_SERIALIZED_PLAN"
