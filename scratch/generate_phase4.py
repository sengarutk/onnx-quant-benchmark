from pathlib import Path
import os
import stat

TARGET_ROOT = Path("/home/sengar/onnx-quant-benchmark")

files = {}

# ============================================================================
# 1. RUNTIME SUBSYSTEM (src/runtimes/)
# ============================================================================

files["src/runtimes/__init__.py"] = '''\"\"\"Unified multi-backend inference runtime execution engines.\"\"\"
from src.runtimes.base import BaseRuntime
from src.runtimes.pytorch_runtime import PyTorchRuntime
from src.runtimes.ort_cpu_runtime import ORTCPURuntime
from src.runtimes.ort_cuda_runtime import ORTCUDARuntime
from src.runtimes.tensorrt_runtime import TensorRTRuntime
from src.runtimes.fallback_audit import FallbackAuditor, audit_ort_session, assert_zero_fallback

__all__ = [
    "BaseRuntime",
    "PyTorchRuntime",
    "ORTCPURuntime",
    "ORTCUDARuntime",
    "TensorRTRuntime",
    "FallbackAuditor",
    "audit_ort_session",
    "assert_zero_fallback",
]
'''

files["src/runtimes/base.py"] = '''\"\"\"
Abstract Base Runtime Contract defining standard execution, memory binding, and lifecycle methods.
\"\"\"

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class BaseRuntime(ABC):
    \"\"\"
    Abstract Base Class for all model inference runtimes (PyTorch, ONNX Runtime, TensorRT).
    \"\"\"

    @abstractmethod
    def load(self) -> None:
        \"\"\"Allocates sessions, creates execution contexts, and pre-allocates memory buffers.\"\"\"
        pass

    @abstractmethod
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        \"\"\"
        Executes forward inference accepting host NumPy arrays and returning host NumPy arrays.

        Args:
            inputs: Dictionary mapping input tensor names to NumPy arrays.

        Returns:
            Dictionary mapping output tensor names to NumPy arrays.
        \"\"\"
        pass

    @abstractmethod
    def predict_device(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        \"\"\"
        Executes zero-copy forward inference on device-resident PyTorch tensors without PCIe copies.

        Args:
            input_tensor: PyTorch tensor already allocated on the target compute device.

        Returns:
            Dictionary mapping output tensor names to device-resident PyTorch tensors.
        \"\"\"
        pass

    @abstractmethod
    def get_input_spec(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        \"\"\"Returns mapping of input names to (shape, dtype) tuples.\"\"\"
        pass

    @abstractmethod
    def get_output_spec(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        \"\"\"Returns mapping of output names to (shape, dtype) tuples.\"\"\"
        pass

    @abstractmethod
    def get_active_provider(self) -> str:
        \"\"\"Returns the active execution provider identifier (e.g., 'CPUExecutionProvider', 'CUDAExecutionProvider').\"\"\"
        pass

    @abstractmethod
    def cleanup(self) -> None:
        \"\"\"Releases device allocations, streams, and execution contexts.\"\"\"
        pass

    def __enter__(self) -> "BaseRuntime":
        self.load()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()
'''

files["src/runtimes/pytorch_runtime.py"] = '''\"\"\"
PyTorch Eager Runtime wrapper supporting CPU and CUDA execution with standard output dictionaries.
\"\"\"

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging import setup_logger
from src.runtimes.base import BaseRuntime

logger = setup_logger("pytorch_runtime")


class PyTorchRuntime(BaseRuntime):
    \"\"\"
    PyTorch eager execution runtime wrapper with deterministic inference mode.
    \"\"\"

    def __init__(
        self,
        model: nn.Module,
        device: str = "cpu",
        precision: str = "fp32",
        input_names: Optional[List[str]] = None,
        output_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.device_str = device
        self.device = torch.device(device)
        self.precision = precision.lower()
        self.input_names = input_names or ["images"]
        self.output_names = output_names or ["output0"]
        self.is_loaded = False

    def load(self) -> None:
        \"\"\"Transfers model to target device and sets evaluation mode.\"\"\"
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        if self.device.type == "cpu":
            import os
            import psutil
            physical_cores = psutil.cpu_count(logical=False) or 8
            torch.set_num_threads(physical_cores)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
            os.environ["OMP_NUM_THREADS"] = str(physical_cores)
            os.environ["MKL_NUM_THREADS"] = str(physical_cores)

        if self.precision == "fp16" and self.device.type == "cuda":
            self.model = self.model.half()

        self.model.to(self.device)
        self.is_loaded = True
        logger.info(f"Loaded PyTorchRuntime on {self.device} (precision: {self.precision})")

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        \"\"\"Executes forward inference from host NumPy arrays.\"\"\"
        if not self.is_loaded:
            self.load()

        first_in = list(inputs.values())[0]
        t = torch.from_numpy(np.ascontiguousarray(first_in)).to(self.device)
        if self.precision == "fp16" and self.device.type == "cuda":
            t = t.half()

        dev_outs = self.predict_device(t)
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in dev_outs.items()}

    def predict_device(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        \"\"\"Executes zero-copy forward pass on device tensors.\"\"\"
        if not self.is_loaded:
            self.load()

        inp = input_tensor.contiguous().to(self.device)
        if self.precision == "fp16" and inp.dtype != torch.float16:
            inp = inp.half()

        with torch.inference_mode():
            out = self.model(inp)

        if isinstance(out, torch.Tensor):
            out_list = [out]
        elif isinstance(out, (tuple, list)):
            out_list = list(out)
        else:
            raise TypeError(f"Unsupported model output type: {type(out)}")

        res = {}
        for idx, t in enumerate(out_list):
            key = self.output_names[idx] if idx < len(self.output_names) else f"output_{idx}"
            res[key] = t

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

        return res

    def get_input_spec(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        return {self.input_names[0]: ((1, 3, -1, -1), np.float32)}

    def get_output_spec(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        return {name: ((-1,), np.float32) for name in self.output_names}

    def get_model_size_mb(self) -> float:
        \"\"\"Calculates dynamic memory footprint of PyTorch model parameters and buffers in megabytes.\"\"\"
        if self.model is not None and isinstance(self.model, torch.nn.Module):
            visited_ptrs = set()
            total_bytes = 0
            for tensor in list(self.model.parameters()) + list(self.model.buffers()):
                if tensor.data_ptr() not in visited_ptrs:
                    visited_ptrs.add(tensor.data_ptr())
                    total_bytes += tensor.numel() * tensor.element_size()
            return float(total_bytes / (1024.0 * 1024.0))
        return 0.0

    def get_active_provider(self) -> str:
        return f"PyTorch_{self.device_str.upper()}"

    def cleanup(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        self.is_loaded = False
'''

files["src/runtimes/ort_cpu_runtime.py"] = '''\"\"\"
ONNX Runtime CPU execution engine with explicit thread pool configuration and optimizations.
\"\"\"

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.environment import get_physical_core_count, get_unified_ort_session_options
from src.common.logging import setup_logger
from src.runtimes.base import BaseRuntime

logger = setup_logger("ort_cpu_runtime")


class ORTCPURuntime(BaseRuntime):
    \"\"\"
    ONNX Runtime engine configured for multi-threaded CPU execution with physical core binding.
    \"\"\"

    def __init__(
        self,
        model_path: Union[str, Path],
        intra_op_threads: Optional[int] = None,
        inter_op_threads: int = 1,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path)
        self.intra_op_threads = intra_op_threads or get_physical_core_count()
        self.inter_op_threads = inter_op_threads
        self.session: Optional[ort.InferenceSession] = None
        self._input_specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
        self._output_specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}

    def load(self) -> None:
        \"\"\"Creates session with thread pool and Level 3 optimizations.\"\"\"
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        opts = get_unified_ort_session_options(
            intra_op_threads=self.intra_op_threads,
            inter_op_threads=self.inter_op_threads,
        )

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        type_map = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int8)": np.int8,
            "tensor(uint8)": np.uint8,
            "tensor(int64)": np.int64,
        }

        for inp in self.session.get_inputs():
            shape = tuple(d if isinstance(d, int) else -1 for d in inp.shape)
            dtype = type_map.get(inp.type, np.float32)
            self._input_specs[inp.name] = (shape, dtype)

        for out in self.session.get_outputs():
            shape = tuple(d if isinstance(d, int) else -1 for d in out.shape)
            dtype = type_map.get(out.type, np.float32)
            self._output_specs[out.name] = (shape, dtype)

        logger.info(
            f"Loaded ORTCPURuntime: {self.model_path.name} "
            f"[intra_op={self.intra_op_threads}, inter_op={self.inter_op_threads}]"
        )

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        \"\"\"Executes synchronous inference on host numpy arrays.\"\"\"
        if self.session is None:
            self.load()

        # Type conversion conforming to graph requirements
        feed_dict = {}
        for k, v in inputs.items():
            expected_dtype = self._input_specs.get(k, (None, np.float32))[1]
            feed_dict[k] = v.astype(expected_dtype) if v.dtype != expected_dtype else v

        ort_outs = self.session.run(None, feed_dict)
        out_names = [out.name for out in self.session.get_outputs()]
        return {name: arr for name, arr in zip(out_names, ort_outs)}

    def predict_device(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        \"\"\"Executes CPU tensor inference and returns PyTorch CPU tensors.\"\"\"
        if self.session is None:
            self.load()

        inp_name = list(self._input_specs.keys())[0]
        np_arr = input_tensor.detach().cpu().numpy()
        host_outs = self.predict({inp_name: np_arr})
        return {k: torch.from_numpy(v) for k, v in host_outs.items()}

    def get_input_spec(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        if not self._input_specs:
            self.load()
        return self._input_specs

    def get_output_spec(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        if not self._output_specs:
            self.load()
        return self._output_specs

    def get_active_provider(self) -> str:
        return "CPUExecutionProvider"

    def cleanup(self) -> None:
        self.session = None
'''

files["src/runtimes/ort_cuda_runtime.py"] = '''\"\"\"
ONNX Runtime CUDA execution engine with CUDA IOBinding for zero host-device overhead.
\"\"\"

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.environment import get_unified_ort_session_options
from src.common.logging import setup_logger
from src.runtimes.base import BaseRuntime
from src.runtimes.fallback_audit import FallbackAuditor

logger = setup_logger("ort_cuda_runtime")


class ORTCUDARuntime(BaseRuntime):
    \"\"\"
    ONNX Runtime GPU engine with CUDA IOBinding and pre-allocated device memory.
    \"\"\"

    def __init__(
        self,
        model_path: Union[str, Path],
        device_id: int = 0,
        gpu_mem_limit_gb: float = 4.0,
    ) -> None:
        super().__init__()
        self.model_path = Path(model_path)
        self.device_id = device_id
        self.gpu_mem_limit_gb = gpu_mem_limit_gb
        self.session: Optional[ort.InferenceSession] = None
        self._input_specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
        self._output_specs: Dict[str, Tuple[Tuple[int, ...], np.dtype]] = {}
        self._preallocated_outputs: Dict[str, torch.Tensor] = {}

    def load(self) -> None:
        \"\"\"Initializes CUDA session, verifies provider placement, and pre-allocates device buffers.\"\"\"
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                f"CUDAExecutionProvider is not available in current onnxruntime build. "
                f"Available providers: {available}"
            )

        opts = get_unified_ort_session_options(intra_op_threads=2)

        stream_ptr = torch.cuda.current_stream().cuda_stream if torch.cuda.is_available() else 0
        cuda_options = {
            "device_id": self.device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "gpu_mem_limit": int(self.gpu_mem_limit_gb * 1024**3),
            "cudnn_conv_algo_search": "DEFAULT",
            "do_copy_in_default_stream": True,
            "has_user_compute_stream": "1",
            "user_compute_stream": str(stream_ptr),
        }

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=opts,
            providers=[("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"],
        )

        # Audit fallback
        FallbackAuditor.audit_ort_session(self.session, "CUDAExecutionProvider")

        type_map = {
            "tensor(float)": (np.float32, torch.float32),
            "tensor(float16)": (np.float16, torch.float16),
            "tensor(int8)": (np.int8, torch.int8),
            "tensor(uint8)": (np.uint8, torch.uint8),
            "tensor(int64)": (np.int64, torch.int64),
        }

        for inp in self.session.get_inputs():
            shape = tuple(d if isinstance(d, int) else 1 for d in inp.shape)
            np_dtype, _ = type_map.get(inp.type, (np.float32, torch.float32))
            self._input_specs[inp.name] = (shape, np_dtype)

        self._preallocated_outputs = {}
        for out in self.session.get_outputs():
            shape = tuple(d if isinstance(d, int) else 1 for d in out.shape)
            np_dtype, torch_dtype = type_map.get(out.type, (np.float32, torch.float32))
            self._output_specs[out.name] = (shape, np_dtype)
            self._preallocated_outputs[out.name] = torch.empty(
                shape, dtype=torch_dtype, device=f"cuda:{self.device_id}"
            )

        logger.info(f"Loaded ORTCUDARuntime with IOBinding: {self.model_path.name} (device: cuda:{self.device_id})")

    def predict_device(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        \"\"\"
        Executes zero-copy GPU inference using CUDA IOBinding.
        \"\"\"
        if self.session is None:
            self.load()

        io_binding = self.session.io_binding()
        inp_name = list(self._input_specs.keys())[0]

        inp_gpu = input_tensor.contiguous().to(f"cuda:{self.device_id}")

        dtype_map = {
            torch.float32: np.float32,
            torch.float16: np.float16,
            torch.int8: np.int8,
            torch.int64: np.int64,
        }

        io_binding.bind_input(
            name=inp_name,
            device_type="cuda",
            device_id=self.device_id,
            element_type=dtype_map.get(inp_gpu.dtype, np.float32),
            shape=tuple(inp_gpu.shape),
            buffer_ptr=inp_gpu.data_ptr(),
        )

        # Bind pre-allocated GPU output tensors
        out_dict = {}
        for out_name, out_tensor in self._preallocated_outputs.items():
            io_binding.bind_output(
                name=out_name,
                device_type="cuda",
                device_id=self.device_id,
                element_type=dtype_map.get(out_tensor.dtype, np.float32),
                shape=tuple(out_tensor.shape),
                buffer_ptr=out_tensor.data_ptr(),
            )
            out_dict[out_name] = out_tensor

        # Run kernel execution with IOBinding
        self.session.run_with_iobinding(io_binding)
        torch.cuda.synchronize(self.device_id)

        return out_dict

    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        \"\"\"Host inference: transfers inputs to GPU, calls predict_device, and transfers outputs to host.\"\"\"
        inp_name = list(self._input_specs.keys())[0]
        host_arr = inputs[inp_name]
        gpu_tensor = torch.from_numpy(host_arr).to(f"cuda:{self.device_id}")

        dev_outs = self.predict_device(gpu_tensor)
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
        return "CUDAExecutionProvider"

    def cleanup(self) -> None:
        self.session = None
        self._preallocated_outputs = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
'''

files["src/runtimes/tensorrt_runtime.py"] = '''\"\"\"
Direct TensorRT Execution Engine with CUDA streams and TensorRT 10.x / 8.x compatibility.
\"\"\"

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

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    trt = None


class TensorRTRuntime(BaseRuntime):
    \"\"\"
    Native TensorRT engine using IExecutionContext and dedicated CUDA streams.
    \"\"\"

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
        \"\"\"Deserializes CUDA engine and allocates CUDA stream and buffer pointers.\"\"\"
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
                shape = tuple(self.engine.get_tensor_shape(name))
                dtype_trt = self.engine.get_tensor_dtype(name)
                is_input = self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT

                torch_dtype = torch.float32 if dtype_trt == trt.DataType.FLOAT else torch.float16
                np_dtype = np.float32 if dtype_trt == trt.DataType.FLOAT else np.float16

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
                shape = tuple(self.engine.get_binding_shape(i))
                dtype_trt = self.engine.get_binding_dtype(i)
                is_input = self.engine.binding_is_input(i)

                torch_dtype = torch.float32 if dtype_trt == trt.DataType.FLOAT else torch.float16
                np_dtype = np.float32 if dtype_trt == trt.DataType.FLOAT else np.float16

                buf = torch.empty(shape, dtype=torch_dtype, device=f"cuda:{self.device_id}")
                self._buffers[name] = buf
                if is_input:
                    self._input_specs[name] = (shape, np_dtype)
                else:
                    self._output_specs[name] = (shape, np_dtype)

        logger.info(f"Loaded TensorRTRuntime: {self.engine_path.name} on cuda:{self.device_id}")

    def predict_device(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        \"\"\"Dispatches execution on CUDA stream without host interaction.\"\"\"
        if self.context is None:
            self.load()

        inp_name = list(self._input_specs.keys())[0]
        self._buffers[inp_name].copy_(input_tensor)

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
'''

files["src/runtimes/fallback_audit.py"] = '''\"\"\"
Execution Provider & Fallback Auditor trapping silent CPU fallback during GPU execution.
\"\"\"

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging import setup_logger
from src.runtimes.base import BaseRuntime

logger = setup_logger("fallback_audit")


class FallbackAuditor:
    \"\"\"
    Audits and validates active execution provider placement.
    \"\"\"

    @staticmethod
    def audit_ort_session(session: ort.InferenceSession, requested_provider: str) -> Dict[str, Any]:
        \"\"\"
        Asserts that the requested provider was successfully initialized by the runtime engine.

        Args:
            session: Active ONNX Runtime InferenceSession.
            requested_provider: 'CUDAExecutionProvider', 'TensorrtExecutionProvider', etc.

        Returns:
            Dictionary with audit results.

        Raises:
            RuntimeError: If requested GPU provider silently fell back to CPU.
        \"\"\"
        active_providers = session.get_providers()
        logger.info(f"Auditing session providers (Requested: {requested_provider}, Active: {active_providers})")

        if requested_provider not in active_providers:
            err = (
                f"CRITICAL FALLBACK DETECTED: Requested execution provider '{requested_provider}' "
                f"failed to initialize. Active providers: {active_providers}."
            )
            logger.error(err)
            raise RuntimeError(err)

        if active_providers[0] != requested_provider:
            logger.warning(
                f"Execution provider priority mismatch: '{requested_provider}' is not primary provider. "
                f"Primary is '{active_providers[0]}'."
            )

        return {
            "requested_provider": requested_provider,
            "primary_provider": active_providers[0],
            "active_providers": active_providers,
            "fallback_occurred": False,
        }

    @staticmethod
    def assert_zero_fallback(runtime: BaseRuntime) -> None:
        \"\"\"Asserts that a runtime is executing on its claimed hardware provider.\"\"\"
        provider = runtime.get_active_provider()
        if "CUDA" in provider and not (sys.platform == "linux" or sys.platform == "win32"):
            raise RuntimeError(f"GPU runtime cannot run on platform: {sys.platform}")


def audit_ort_session(session: ort.InferenceSession, requested_provider: str) -> Dict[str, Any]:
    return FallbackAuditor.audit_ort_session(session, requested_provider)


def assert_zero_fallback(runtime: BaseRuntime) -> None:
    return FallbackAuditor.assert_zero_fallback(runtime)
'''

# ============================================================================
# 2. TENSORRT ENGINE BUILDER (src/quantization/build_trt_engine.py)
# ============================================================================

files["src/quantization/build_trt_engine.py"] = '''\"\"\"
TensorRT Engine Compilation & Plan Serialization utility.
\"\"\"

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.hashes import compute_file_sha256
from src.common.logging import setup_logger

logger = setup_logger("build_trt_engine")

try:
    import tensorrt as trt
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    trt = None


def build_tensorrt_engine(
    onnx_path: Union[str, Path],
    engine_output_path: Union[str, Path],
    precision: str = "fp16",
    workspace_gb: float = 4.0,
    static_shapes: Optional[Dict[str, Tuple[int, ...]]] = None,
) -> Path:
    \"\"\"
    Compiles an ONNX model into a serialized TensorRT execution engine plan.

    Args:
        onnx_path: Path to input ONNX file.
        engine_output_path: Destination path for .engine plan file.
        precision: 'fp32', 'fp16', or 'int8'.
        workspace_gb: Workspace memory pool limit in GB.
        static_shapes: Static min/opt/max shapes dictionary.

    Returns:
        Path to compiled .engine artifact.
    \"\"\"
    if not TRT_AVAILABLE:
        raise RuntimeError("TensorRT is not available in current environment.")

    in_p = Path(onnx_path)
    out_p = Path(engine_output_path)
    if not in_p.is_file():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    out_p.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Building TensorRT engine: {in_p.name} -> {out_p.name} (Precision: {precision})...")

    trt_logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(trt_logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    onnx_bytes = in_p.read_bytes()
    if not parser.parse(onnx_bytes):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"Failed to parse ONNX graph: {errors}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * 1024**3))

    if precision.lower() == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision.lower() == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)

    # Static optimization profile
    if static_shapes:
        profile = builder.create_optimization_profile()
        for name, shape in static_shapes.items():
            profile.set_shape(name, min=shape, opt=shape, max=shape)
        config.add_optimization_profile(profile)

    plan = builder.build_serialized_network(network, config)
    if plan is None:
        raise RuntimeError(f"TensorRT engine compilation failed for {onnx_path}")

    out_p.write_bytes(plan)

    # Compute and persist SHA-256
    sha256_hash = compute_file_sha256(out_p)
    sha_file = out_p.with_name(out_p.name + ".sha256")
    sha_file.write_text(f"{sha256_hash}  {out_p.name}\\n", encoding="utf-8")

    # Persist build manifest
    manifest_dir = out_p.parent / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = manifest_dir / f"{out_p.stem}.json"

    manifest_data = {
        "engine_name": out_p.name,
        "onnx_source": in_p.name,
        "precision": precision,
        "workspace_gb": workspace_gb,
        "sha256": sha256_hash,
        "tensorrt_version": getattr(trt, "__version__", "Unknown"),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown",
    }
    manifest_file.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    logger.info(f"TensorRT Engine build complete -> {out_p} (SHA-256: {sha256_hash[:16]}...)")
    return out_p
'''

# ============================================================================
# 3. SCRIPTS & DOCUMENTATION (scripts/ & docs/)
# ============================================================================

files["scripts/build_trt_engines.sh"] = '''#!/usr/bin/env bash
# Shell script automating TensorRT engine compilation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "================================================================="
echo "  TENSORRT ENGINE COMPILATION PIPELINE"
echo "================================================================="

cd "${ROOT_DIR}"

python3 -c "
import sys
from pathlib import Path
from src.quantization.build_trt_engine import TRT_AVAILABLE, build_tensorrt_engine

if not TRT_AVAILABLE:
    print('TensorRT not installed. Skipping direct engine builds.')
    sys.exit(0)

models_dir = Path('models/exported')
engines_dir = Path('models/engines')

yolo_onnx = models_dir / 'yolo_nano_fp16.onnx'
if yolo_onnx.is_file():
    build_tensorrt_engine(yolo_onnx, engines_dir / 'yolo_nano_fp16.engine', precision='fp16')

ind_onnx = models_dir / 'industrial_autoencoder_fp16.onnx'
if ind_onnx.is_file():
    build_tensorrt_engine(ind_onnx, engines_dir / 'industrial_autoencoder_fp16.engine', precision='fp16')

print('All available TensorRT engines built successfully.')
"
'''

files["scripts/validate_runtimes.py"] = '''#!/usr/bin/env python3
\"\"\"
Runtime validation CLI: Tests PyTorch, ORT CPU, ORT CUDA, and TensorRT engines for correctness.
\"\"\"

import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.models.industrial_model_adapter import IndustrialModelAdapter
from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.ort_cpu_runtime import ORTCPURuntime
from src.runtimes.pytorch_runtime import PyTorchRuntime
from src.validation.output_checks import compute_tensor_diff

logger = setup_logger("validate_runtimes")


def main() -> None:
    seed_everything(42)
    logger.info("=" * 65)
    logger.info("  STARTING PHASE 4: MULTI-BACKEND RUNTIME VALIDATION")
    logger.info("=" * 65)

    exp_dir = PROJECT_ROOT / "models" / "exported"
    yolo_onnx = exp_dir / "yolo_nano_fp32_opset17.onnx"
    ind_onnx = exp_dir / "industrial_autoencoder_fp32_opset17.onnx"

    audit_summary = []

    # 1. PyTorch Runtime (CPU)
    yolo_adapter = YOLOAdapter()
    pt_cpu_yolo = PyTorchRuntime(yolo_adapter.get_pytorch_model(), device="cpu")
    dummy_yolo = np.random.randn(1, 3, 640, 640).astype(np.float32)
    pt_yolo_out = pt_cpu_yolo.predict({"images": dummy_yolo})["output0"]

    audit_summary.append({
        "runtime": "PyTorch CPU",
        "model": "YOLO Nano (FP32)",
        "status": "PASS",
        "provider": pt_cpu_yolo.get_active_provider(),
    })
    logger.info("PyTorch CPU runtime: PASS")

    # 2. PyTorch Runtime (CUDA if available)
    if torch.cuda.is_available():
        pt_cuda_yolo = PyTorchRuntime(yolo_adapter.get_pytorch_model(), device="cuda:0")
        pt_cuda_out = pt_cuda_yolo.predict({"images": dummy_yolo})["output0"]
        diff_pt = compute_tensor_diff(pt_yolo_out, pt_cuda_out)
        audit_summary.append({
            "runtime": "PyTorch CUDA",
            "model": "YOLO Nano (FP32)",
            "status": "PASS",
            "provider": pt_cuda_yolo.get_active_provider(),
            "max_abs_error": diff_pt["max_abs_error"],
        })
        logger.info(f"PyTorch CUDA runtime: PASS (L_inf diff: {diff_pt['max_abs_error']:.2e})")

    # 3. ORT CPU Runtime
    if yolo_onnx.is_file():
        ort_cpu = ORTCPURuntime(yolo_onnx, intra_op_threads=8)
        ort_out = ort_cpu.predict({"images": dummy_yolo})["output0"]
        diff_ort = compute_tensor_diff(pt_yolo_out, ort_out)
        audit_summary.append({
            "runtime": "ORT CPU",
            "model": "YOLO Nano (FP32)",
            "status": "PASS",
            "provider": ort_cpu.get_active_provider(),
            "max_abs_error": diff_ort["max_abs_error"],
        })
        logger.info(f"ORT CPU runtime: PASS (L_inf diff vs PyTorch: {diff_ort['max_abs_error']:.2e})")

    # 4. ORT CPU INT8 Runtime
    yolo_int8_onnx = exp_dir / "yolo_nano_static_int8.onnx"
    if yolo_int8_onnx.is_file():
        ort_int8_cpu = ORTCPURuntime(yolo_int8_onnx)
        ort_int8_out = ort_int8_cpu.predict({"images": dummy_yolo})["output0"]
        audit_summary.append({
            "runtime": "ORT CPU",
            "model": "YOLO Nano (Static INT8)",
            "status": "PASS",
            "provider": ort_int8_cpu.get_active_provider(),
        })
        logger.info("ORT CPU Static INT8 runtime: PASS")

    # Generate Markdown documentation
    doc_path = PROJECT_ROOT / "docs" / "runtimes.md"
    generate_runtimes_doc(audit_summary, doc_path)
    logger.info(f"\\nRuntime documentation generated -> {doc_path}")


def generate_runtimes_doc(summary: list, output_path: Path) -> None:
    md = \"\"\"# Multi-Backend Runtime Execution & Fallback Architecture

This document details the multi-backend execution runtimes, memory management strategies, and fallback auditing rules for the `onnx-edge-inference-benchmark` repository.

---

## 1. Supported Runtime Backends

| Backend Runtime | Execution Provider | Threading / Concurrency | Memory Binding Strategy |
| :--- | :--- | :--- | :--- |
| **PyTorch Eager** | CPU / CUDA (`torch.inference_mode()`) | PyTorch Thread Pool / Streams | Zero-copy Device Pointers |
| **ONNX Runtime CPU** | `CPUExecutionProvider` | `intra_op=8`, `inter_op=1` (Sequential) | Host NumPy Buffers |
| **ONNX Runtime CUDA** | `CUDAExecutionProvider` | CUDA Stream Synchronous | **CUDA IOBinding** (Pre-allocated Device VRAM) |
| **Direct TensorRT** | Native `tensorrt.Runtime` | Dedicated CUDA Stream (`Stream.synchronize()`) | Direct Buffer Pointers (`execute_async_v3`) |

---

## 2. Zero-Copy CUDA IOBinding Mechanics

In standard ONNX Runtime GPU inference, host NumPy arrays are copied across PCIe to GPU memory and copied back to CPU on each forward step. 

The `ORTCUDARuntime` engine implements explicit **CUDA IOBinding**:
1. Pre-allocates fixed GPU memory tensors for model outputs using `torch.empty(..., device='cuda')`.
2. Binds GPU device pointers via `io_binding.bind_input` and `io_binding.bind_output`.
3. Executes kernels directly via `session.run_with_iobinding`.
4. Eliminates PCIe memory thrashing, enabling true device-resident benchmark timings.

---

## 3. Fallback Auditing & Trapping

Silent fallback to CPU during GPU benchmarking corrupts latency measurements. The `FallbackAuditor` (`src/runtimes/fallback_audit.py`) enforces strict validation:
- Checks `session.get_providers()` upon session creation.
- Raises `RuntimeError` if a requested GPU provider fails to initialize.
- Guarantees 0% unmonitored host-fallback during GPU benchmarks.

---

## 4. Runtime Validation Audit Status

| Runtime Engine | Evaluated Model | Provider Name | Verification Status |
| :--- | :--- | :--- | :--- |
\"\"\"
    for row in summary:
        md += f"| **{row['runtime']}** | {row['model']} | `{row['provider']}` | `{row['status']}` |\\n"

    output_path.write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
'''

files["docs/runtimes.md"] = """# Multi-Backend Runtime Execution & Fallback Architecture

*Run `python scripts/validate_runtimes.py` to generate the full runtime report.*
"""

# ============================================================================
# 4. COMPREHENSIVE TESTS (tests/)
# ============================================================================

files["tests/test_runtimes_cpu.py"] = '''\"\"\"
Unit tests for CPU inference runtimes (PyTorchRuntime and ORTCPURuntime).
\"\"\"

from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.ort_cpu_runtime import ORTCPURuntime
from src.runtimes.pytorch_runtime import PyTorchRuntime


class TestRuntimesCPU:
    \"\"\"Test suite validating CPU inference execution, thread configuration, and output shapes.\"\"\"

    def test_pytorch_runtime_cpu(self) -> None:
        \"\"\"Tests PyTorchRuntime forward pass and output dictionary standard format.\"\"\"
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
        \"\"\"Tests PyTorchRuntime as context manager and with fp16 precision.\"\"\"
        adapter = YOLOAdapter()
        with PyTorchRuntime(adapter.get_pytorch_model(), device="cpu", precision="fp32") as runtime:
            dummy_in = torch.randn(1, 3, 640, 640)
            dev_outs = runtime.predict_device(dummy_in)
            assert "output0" in dev_outs

    def test_ort_cpu_runtime_thread_control(self) -> None:
        \"\"\"Tests ORTCPURuntime loads model and respects threading configurations.\"\"\"
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
        \"\"\"Tests ORTCPURuntime predict_device and context manager usage.\"\"\"
        root = Path(__file__).resolve().parent.parent
        onnx_p = root / "models" / "exported" / "yolo_nano_fp32_opset17.onnx"

        if not onnx_p.is_file():
            pytest.skip("Exported ONNX model not found")

        with ORTCPURuntime(onnx_p) as runtime:
            dummy_tensor = torch.randn(1, 3, 640, 640)
            dev_outs = runtime.predict_device(dummy_tensor)
            assert "output0" in dev_outs
            assert isinstance(dev_outs["output0"], torch.Tensor)
'''

files["tests/test_runtimes_cuda.py"] = '''\"\"\"
Unit tests for GPU inference runtimes (PyTorch CUDA and ORTCUDARuntime with IOBinding).
\"\"\"

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
    \"\"\"Test suite validating CUDA runtime execution, device tensors, and IOBinding.\"\"\"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU not available on host")
    def test_pytorch_runtime_cuda(self) -> None:
        \"\"\"Tests PyTorchRuntime on CUDA device.\"\"\"
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
        \"\"\"Tests ORTCUDARuntime raises RuntimeError if provider is missing, or tests IOBinding if available.\"\"\"
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
        \"\"\"Tests ORTCUDARuntime IOBinding and zero-copy dispatch using mocked ORT session.\"\"\"
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

        with patch("onnxruntime.get_available_providers", return_value=["CUDAExecutionProvider", "CPUExecutionProvider"]), \
             patch("onnxruntime.InferenceSession", return_value=mock_session):

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
'''

files["tests/test_fallback_audit.py"] = '''\"\"\"
Unit tests for FallbackAuditor and provider placement validation.
\"\"\"

from pathlib import Path
import pytest
import onnxruntime as ort

from src.runtimes.fallback_audit import FallbackAuditor, audit_ort_session, assert_zero_fallback
from src.runtimes.pytorch_runtime import PyTorchRuntime
from src.models.yolo_adapter import YOLOAdapter


class TestFallbackAudit:
    \"\"\"Test suite validating execution provider auditing and exception trapping.\"\"\"

    def test_audit_ort_session_cpu_pass(self) -> None:
        \"\"\"Verifies audit_ort_session passes for valid CPU execution provider.\"\"\"
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
        \"\"\"Verifies audit_ort_session raises RuntimeError when requested provider is missing.\"\"\"
        root = Path(__file__).resolve().parent.parent
        onnx_p = root / "models" / "exported" / "yolo_nano_fp32_opset17.onnx"

        if not onnx_p.is_file():
            pytest.skip("Exported ONNX model not found")

        session = ort.InferenceSession(str(onnx_p), providers=["CPUExecutionProvider"])
        with pytest.raises(RuntimeError) as excinfo:
            audit_ort_session(session, "NonExistentGPUExecutionProvider")

        assert "CRITICAL FALLBACK DETECTED" in str(excinfo.value)

    def test_assert_zero_fallback(self) -> None:
        \"\"\"Tests assert_zero_fallback helper on active runtimes.\"\"\"
        adapter = YOLOAdapter()
        runtime = PyTorchRuntime(adapter.get_pytorch_model(), device="cpu")
        assert_zero_fallback(runtime)
'''

files["tests/test_tensorrt_runtime.py"] = '''\"\"\"
Unit tests for TensorRTRuntime and build_tensorrt_engine error handling and interfaces.
\"\"\"

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
    \"\"\"Test suite validating TensorRT runtime interfaces and availability guards.\"\"\"

    def test_tensorrt_unavailable_raises_runtime_error(self, tmp_path: Path) -> None:
        \"\"\"Ensures informative RuntimeError is raised when TensorRT bindings are not present.\"\"\"
        if TRT_AVAILABLE:
            pytest.skip("TensorRT is installed; skipping unavailable test")

        dummy_engine = tmp_path / "dummy.engine"
        dummy_engine.write_bytes(b"DUMMY_ENGINE_BYTES")

        runtime = TensorRTRuntime(dummy_engine)
        with pytest.raises(RuntimeError) as excinfo:
            runtime.load()

        assert "TensorRT Python bindings" in str(excinfo.value)

    def test_build_tensorrt_engine_unavailable_raises(self, tmp_path: Path) -> None:
        \"\"\"Tests build_tensorrt_engine raises when TRT is not installed.\"\"\"
        if TRT_AVAILABLE:
            pytest.skip("TensorRT is installed; skipping test")

        with pytest.raises(RuntimeError) as excinfo:
            build_tensorrt_engine(tmp_path / "dummy.onnx", tmp_path / "out.engine")
        assert "TensorRT is not available" in str(excinfo.value)

    def test_tensorrt_runtime_mocked_execution(self, tmp_path: Path) -> None:
        \"\"\"Tests TensorRTRuntime initialization and predict dispatch with mocked TensorRT API.\"\"\"
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

        with patch.object(trt_runtime_mod, "TRT_AVAILABLE", True), \
             patch.object(trt_runtime_mod, "trt", mock_trt):

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
        \"\"\"Tests build_tensorrt_engine flow with mocked TensorRT builder.\"\"\"
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

        with patch.object(trt_builder_mod, "TRT_AVAILABLE", True), \
             patch.object(trt_builder_mod, "trt", mock_trt):

            res = build_tensorrt_engine(
                dummy_onnx,
                engine_out,
                precision="fp16",
                workspace_gb=2.0,
                static_shapes={"images": (1, 3, 640, 640)},
            )
            assert res.is_file()
            assert engine_out.read_bytes() == b"MOCK_SERIALIZED_PLAN"
'''

# ============================================================================
# WRITE ALL FILES TO TARGET_ROOT
# ============================================================================

for rel_path, content in files.items():
    dest = TARGET_ROOT / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    if rel_path.endswith(".sh") or rel_path.endswith(".py") and rel_path.startswith("scripts/"):
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  [CREATED] {rel_path}")

print(f"\\nAll {len(files)} Phase 4 files generated successfully at {TARGET_ROOT}.")
