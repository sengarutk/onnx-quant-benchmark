"""
PyTorch Eager Runtime wrapper supporting CPU and CUDA execution with standard output dictionaries.
"""

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
    """
    PyTorch eager execution runtime wrapper with deterministic inference mode.
    """

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
        """Transfers model to target device and sets evaluation mode."""
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
        """Executes forward inference from host NumPy arrays."""
        if not self.is_loaded:
            self.load()

        first_in = list(inputs.values())[0]
        t = torch.from_numpy(np.ascontiguousarray(first_in)).to(self.device)
        if self.precision == "fp16" and self.device.type == "cuda":
            t = t.half()

        dev_outs = self.predict_device(t)
        return {k: v.detach().cpu().numpy().astype(np.float32) for k, v in dev_outs.items()}

    def predict_device(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Executes zero-copy forward pass on device tensors."""
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
        """Calculates dynamic memory footprint of PyTorch model parameters and buffers in megabytes."""
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
