"""
ONNX Runtime CUDA execution engine with CUDA IOBinding for zero host-device overhead.
"""

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
    """
    ONNX Runtime GPU engine with CUDA IOBinding and pre-allocated device memory.
    """

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
        """Initializes CUDA session, verifies provider placement, and pre-allocates device buffers."""
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
        """
        Executes zero-copy GPU inference using CUDA IOBinding.
        """
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
        """Host inference: transfers inputs to GPU, calls predict_device, and transfers outputs to host."""
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
