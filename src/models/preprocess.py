"""
Deterministic, memory-efficient image preprocessing pipeline for object detection and industrial anomaly models.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import numpy as np
import torch


def letterbox_image(
    image: np.ndarray,
    target_shape: Tuple[int, int] = (640, 640),
    stride: int = 32,
    auto: bool = False,
    scale_fill: bool = False,
    scale_up: bool = True,
    pad_value: float = 114.0,
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[float, float]]:
    """
    Resizes and pads an image while maintaining aspect ratio with border padding.

    Args:
        image: Input image array in HWC format.
        target_shape: Target (height, width) dimensions (default: 640x640).
        stride: Padding constraint stride.
        auto: Minimum rectangle padding mode.
        scale_fill: Stretch to target shape without letterboxing.
        scale_up: Allow scaling up image if smaller than target_shape.
        pad_value: Constant border padding fill value.

    Returns:
        Tuple of (padded_image, (ratio_w, ratio_h), (pad_left, pad_top)).
    """
    shape = image.shape[:2]  # current shape [height, width]
    target_h, target_w = target_shape

    # Scale ratio (new / old)
    r = min(target_h / shape[0], target_w / shape[1])
    if not scale_up:  # only scale down, do not scale up
        r = min(r, 1.0)

    # Compute padding
    ratio = (r, r)
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw, dh = target_w - new_unpad[0], target_h - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scale_fill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (target_w, target_h)
        ratio = (target_w / shape[1], target_h / shape[0])

    dw /= 2.0  # divide padding into 2 sides
    dh /= 2.0

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    color = (int(pad_value), int(pad_value), int(pad_value))
    padded_img = cv2.copyMakeBorder(
        image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded_img, ratio, (dw, dh)


def preprocess_detection_image(
    image_input: Union[str, Path, np.ndarray],
    target_shape: Tuple[int, int] = (640, 640),
    device: Optional[Union[str, torch.device]] = None,
    pin_memory: bool = False,
) -> Tuple[torch.Tensor, Tuple[float, float], Tuple[float, float], Tuple[int, int]]:
    """
    Ingests an image, applies letterbox transformation, normalizes to [0, 1], and formats to NCHW tensor.

    Args:
        image_input: File path or NumPy image array (BGR or RGB).
        target_shape: Target (height, width) tuple (default: 640x640).
        device: Target execution device.
        pin_memory: Whether to pin tensor in host memory for fast CUDA transfers.

    Returns:
        Tuple of (torch_tensor [1, 3, H, W], (ratio_w, ratio_h), (pad_left, pad_top), (orig_h, orig_w)).
    """
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.is_file():
            raise FileNotFoundError(f"Detection image not found: {image_input}")
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to decode image at path: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    orig_shape = (img.shape[0], img.shape[1])  # (H, W)

    # Convert BGR to RGB if 3 channels
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Apply letterbox padding
    padded_img, ratio, pad = letterbox_image(img, target_shape=target_shape, auto=False)

    # Convert HWC -> CHW, normalize to [0.0, 1.0]
    tensor = padded_img.transpose((2, 0, 1)).astype(np.float32) / 255.0
    tensor = np.ascontiguousarray(tensor)

    # Add batch dimension -> [1, 3, H, W]
    torch_tensor = torch.from_numpy(tensor).unsqueeze(0)
    if device is not None and "cuda" in str(device).lower() and torch.cuda.is_available():
        torch_tensor = torch_tensor.to(device, non_blocking=True)
    elif pin_memory and torch.cuda.is_available():
        torch_tensor = torch_tensor.pin_memory()

    return torch_tensor, ratio, pad, orig_shape


def preprocess_industrial_image(
    image_input: Union[str, Path, np.ndarray],
    target_shape: Tuple[int, int] = (256, 256),
    normalize_mean: Optional[Tuple[float, ...]] = None,
    normalize_std: Optional[Tuple[float, ...]] = None,
    device: Optional[Union[str, torch.device]] = None,
    pin_memory: bool = False,
) -> torch.Tensor:
    """
    Preprocesses an industrial inspection image to exact dimensions [1, 3, target_h, target_w].

    Args:
        image_input: Image file path or NumPy array.
        target_shape: Target (height, width) tuple (default: 256x256).
        normalize_mean: Optional channel-wise mean tuple.
        normalize_std: Optional channel-wise standard deviation tuple.
        device: Target execution device.
        pin_memory: Whether to pin tensor in host memory.

    Returns:
        Normalized PyTorch Tensor [1, 3, target_h, target_w].
    """
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.is_file():
            raise FileNotFoundError(f"Industrial image not found: {image_input}")
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to decode image at path: {image_input}")
    elif isinstance(image_input, np.ndarray):
        img = image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    # Convert to RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Resize to exact target geometry
    target_h, target_w = target_shape
    if (img.shape[0], img.shape[1]) != (target_h, target_w):
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Convert HWC -> CHW, normalize to [0.0, 1.0]
    tensor = img.transpose((2, 0, 1)).astype(np.float32) / 255.0

    # Apply mean/std normalization if provided
    if normalize_mean is not None and normalize_std is not None:
        mean_arr = np.array(normalize_mean, dtype=np.float32).reshape(3, 1, 1)
        std_arr = np.array(normalize_std, dtype=np.float32).reshape(3, 1, 1)
        tensor = (tensor - mean_arr) / std_arr

    tensor = np.ascontiguousarray(tensor)
    torch_tensor = torch.from_numpy(tensor).unsqueeze(0)
    if device is not None and "cuda" in str(device).lower() and torch.cuda.is_available():
        torch_tensor = torch_tensor.to(device, non_blocking=True)
    elif pin_memory and torch.cuda.is_available():
        torch_tensor = torch_tensor.pin_memory()

    return torch_tensor
