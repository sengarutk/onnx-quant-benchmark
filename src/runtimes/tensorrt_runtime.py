"""
Direct TensorRT Execution Engine with CUDA streams and TensorRT 10.x / 8.x compatibility.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging import setup_logger
from src.runtimes.base import BaseRuntime

logger = setup_logger("tensorrt_runtime")


def _trt_to_torch_dtype(dtype_trt: Any) -> Tuple[torch.dtype, Any]:
    """Maps a TensorRT DataType to (torch dtype, numpy dtype)."""
    mapping = {
        "FLOAT": (torch.float32, np.float32),
        "HALF": (torch.float16, np.float16),
        "INT8": (torch.int8, np.int8),
        "INT32": (torch.int32, np.int32),
        "INT64": (torch.int64, np.int64),
        "BOOL": (torch.bool, np.bool_),
        "UINT8": (torch.uint8, np.uint8),
    }
    for name, pair in mapping.items():
        if hasattr(trt.DataType, name) and dtype_trt == getattr(trt.DataType, name):
            return pair
    raise TypeError(f"Unsupported TensorRT tensor dtype: {dtype_trt}")


def _static_shape(shape: Tuple[int, ...], name: str) -> Tuple[int, ...]:
    if any(d < 0 for d in shape):
        raise ValueError(f"TensorRT tensor '{name}' has dynamic shape {shape}; build the engine with static shapes")
    return shape

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    trt = None


class TensorRTRuntime(BaseRuntime):
    """
    Native TensorRT engine using IExecutionContext and dedicated CUDA streams.
    """

    def __init__(
        self,
        engine_path: Union[str, Path],
        device_id: int = 0,
    ) -> None:
        super().__init__()
        self.engine_path = Path(engine_path)
        self.device_id = device_id
        self.engine = None
        self.context = None
        self.stream = None
        self._input_specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
        self._output_specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
        self._buffers: Dict[str, torch.Tensor] = {}

    def load(self) -> None:
        """Deserializes CUDA engine and allocates CUDA stream and buffer pointers."""
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT Python bindings ('tensorrt') are not installed in the environment.")

        if not self.engine_path.is_file():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        engine_bytes = self.engine_path.read_bytes()
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {self.engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = torch.cuda.Stream(device=self.device_id)

        # Inspect I/O bindings (TRT 10.x API with TRT 8.x fallback)
        if hasattr(self.engine, "num_io_tensors"):
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                shape = _static_shape(tuple(self.engine.get_tensor_shape(name)), name)
                dtype_trt = self.engine.get_tensor_dtype(name)
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
                torch_dtype, np_dtype = _trt_to_torch_dtype(dtype_trt)

                buf = torch.empty(shape, dtype=torch_dtype, device=f"cuda:{self.device_id}")
                self._buffers[name] = buf
                if is_input:
                    self._input_specs[name] = (shape, np_dtype)
                else:
                    self._output_specs[name] = (shape, np_dtype)
        else:
            # TRT 8.x fallback
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                shape = _static_shape(tuple(self.engine.get_binding_shape(i)), name)
                dtype_trt = self.engine.get_binding_dtype(i)
                is_input = self.engine.binding_is_input(i)
                torch_dtype, np_dtype = _trt_to_torch_dtype(dtype_trt)

                buf = torch.empty(shape, dtype=torch_dtype, device=f"cuda:{self.device_id}")
                self._buffers[name] = buf
                if is_input:
                    self._input_specs[name] = (shape, np_dtype)
                else:
                    self._output_specs[name] = (shape, np_dtype)

        logger.info(f"Loaded TensorRTRuntime: {self.engine_path.name} on cuda:{self.device_id}")

    def predict_device(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Dispatches execution on CUDA stream without host interaction."""
        if self.context is None:
            self.load()

        inp_name = list(self._input_specs.keys())[0]
        # Order the input copy and the engine execution on the same dedicated stream; the stream first waits for
        # any producer work queued on the caller's current stream.
        self.stream.wait_stream(torch.cuda.current_stream(self.device_id))
        with torch.cuda.stream(self.stream):
            self._buffers[inp_name].copy_(input_tensor, non_blocking=True)

        # TRT 10.x execution vs 8.x execution
        if hasattr(self.context, "execute_async_v3"):
            for name, buf in self._buffers.items():
                self.context.set_tensor_address(name, buf.data_ptr())
            self.context.execute_async_v3(self.stream.cuda_stream)
        else:
            bindings = [buf.data_ptr() for buf in self._buffers.values()]
            self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.cuda_stream)

        self.stream.synchronize()
        return {name: self._buffers[name] for name in self._output_specs.keys()}

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        inp_name = list(self._input_specs.keys())[0]
        gpu_in = torch.from_numpy(inputs[inp_name]).to(f"cuda:{self.device_id}")
        dev_outs = self.predict_device(gpu_in)
        return {k: v.detach().cpu().numpy() for k, v in dev_outs.items()}

    def get_input_spec(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        if not self._input_specs:
            self.load()
        return self._input_specs

    def get_output_spec(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        if not self._output_specs:
            self.load()
        return self._output_specs

    def get_active_provider(self) -> str:
        return "TensorRT"

    def cleanup(self) -> None:
        self.context = None
        self.engine = None
        self._buffers = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
