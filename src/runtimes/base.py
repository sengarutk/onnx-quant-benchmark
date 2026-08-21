"""
Abstract Base Runtime Contract defining standard execution, memory binding, and lifecycle methods.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch


class BaseRuntime(ABC):
    """
    Abstract Base Class for all model inference runtimes (PyTorch, ONNX Runtime, TensorRT).
    """

    @abstractmethod
    def load(self) -> None:
        """Allocates sessions, creates execution contexts, and pre-allocates memory buffers."""
        pass

    @abstractmethod
    def predict(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Executes forward inference accepting host NumPy arrays and returning host NumPy arrays.

        Args:
            inputs: Dictionary mapping input tensor names to NumPy arrays.

        Returns:
            Dictionary mapping output tensor names to NumPy arrays.
        """
        pass

    @abstractmethod
    def predict_device(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Executes zero-copy forward inference on device-resident PyTorch tensors without PCIe copies.

        Args:
            input_tensor: PyTorch tensor already allocated on the target compute device.

        Returns:
            Dictionary mapping output tensor names to device-resident PyTorch tensors.
        """
        pass

    @abstractmethod
    def get_input_spec(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        """Returns mapping of input names to (shape, dtype) tuples."""
        pass

    @abstractmethod
    def get_output_spec(self) -> Dict[str, Tuple[Tuple[int, ...], np.dtype]]:
        """Returns mapping of output names to (shape, dtype) tuples."""
        pass

    @abstractmethod
    def get_active_provider(self) -> str:
        """Returns the active execution provider identifier (e.g., 'CPUExecutionProvider', 'CUDAExecutionProvider')."""
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Releases device allocations, streams, and execution contexts."""
        pass

    def __enter__(self) -> "BaseRuntime":
        self.load()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.cleanup()
