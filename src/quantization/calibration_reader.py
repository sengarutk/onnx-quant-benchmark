"""
Calibration data reader implementing onnxruntime.quantization.CalibrationDataReader interface.
"""

import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from onnxruntime.quantization import CalibrationDataReader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging import setup_logger

logger = setup_logger("calibration_reader")


class BenchmarkCalibrationDataReader(CalibrationDataReader):
    """
    Deterministic calibration data reader feeding representative activation tensors to ONNX Runtime quantizer.
    """

    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        input_name: str,
        input_shape: Tuple[int, int, int, int],
        preprocess_fn: Callable[[Union[str, Path]], Any],
        batch_size: int = 1,
    ) -> None:
        """
        Initializes the calibration data reader.

        Args:
            image_paths: List of absolute or relative paths to calibration images.
            input_name: Graph input tensor name (e.g., 'images' or 'input').
            input_shape: 4D input dimensions [B, C, H, W].
            preprocess_fn: Callable mapping image path to PyTorch Tensor or (Tensor, ...).
            batch_size: Batch size for calibration steps (default: 1).
        """
        super().__init__()
        self.image_paths = [Path(p) for p in image_paths]
        self.input_name = input_name
        self.input_shape = input_shape
        self.preprocess_fn = preprocess_fn
        self.batch_size = max(1, batch_size)
        self._current_index = 0

        logger.info(
            f"Initialized BenchmarkCalibrationDataReader: {len(self.image_paths)} images, "
            f"input='{input_name}', shape={input_shape}, batch_size={self.batch_size}"
        )

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Returns the next batch of preprocessed input data, or None when dataset is exhausted.

        Returns:
            Dictionary {input_name: numpy_ndarray} or None.
        """
        if self._current_index >= len(self.image_paths):
            return None

        batch_paths = self.image_paths[self._current_index : self._current_index + self.batch_size]
        batch_tensors = []

        for p in batch_paths:
            res = self.preprocess_fn(p)
            # Handle preprocess_detection_image returning (tensor, ratio, pad, orig_shape)
            if isinstance(res, (tuple, list)):
                tensor = res[0]
            elif isinstance(res, torch.Tensor):
                tensor = res
            elif isinstance(res, np.ndarray):
                tensor = torch.from_numpy(res)
            else:
                raise TypeError(f"Unexpected preprocessing return type: {type(res)}")

            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            batch_tensors.append(tensor)

        self._current_index += len(batch_paths)
        full_batch = torch.cat(batch_tensors, dim=0).detach().cpu().numpy().astype(np.float32)

        return {self.input_name: full_batch}

    def rewind(self) -> None:
        """Resets the iteration pointer to the beginning of the dataset."""
        self._current_index = 0
