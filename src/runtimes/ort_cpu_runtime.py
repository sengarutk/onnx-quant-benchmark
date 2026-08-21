"""
ONNX Runtime CPU execution engine with explicit thread pool configuration and optimizations.
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

from src.common.environment import get_physical_core_count, get_unified_ort_session_options
from src.common.logging import setup_logger
from src.runtimes.base import BaseRuntime

logger = setup_logger("ort_cpu_runtime")


class ORTCPURuntime(BaseRuntime):
    """
    ONNX Runtime engine configured for multi-threaded CPU execution with physical core binding.
    """

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
        """Creates session with thread pool and Level 3 optimizations."""
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
        """Executes synchronous inference on host numpy arrays."""
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
        """Executes CPU tensor inference and returns PyTorch CPU tensors."""
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
