"""
Memory-safe streaming hash verification utilities for models, datasets, and runtime tensors.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union
import numpy as np


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """
    Computes the SHA-256 hex digest of a file using buffered streaming.

    Args:
        file_path: Path to the target file.
        chunk_size: Byte buffer size for chunked reading (default: 64 KB).

    Returns:
        Hexadecimal SHA-256 digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the target path is a directory.
    """
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hash calculation: {file_path}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_tensor_sha256(tensor: Any) -> str:
    """
    Computes a deterministic SHA-256 hash of a PyTorch Tensor or NumPy ndarray byte buffer.

    Enforces memory contiguity and canonical byte representation.

    Args:
        tensor: PyTorch tensor or NumPy ndarray.

    Returns:
        Hexadecimal SHA-256 digest of the raw contiguous tensor memory.

    Raises:
        TypeError: If tensor is not a recognized tensor type.
    """
    if hasattr(tensor, "detach") and hasattr(tensor, "cpu") and hasattr(tensor, "numpy"):
        # PyTorch Tensor: force contiguous host memory then extract bytes
        contiguous_tensor = tensor.detach().contiguous().cpu()
        np_arr = contiguous_tensor.numpy()
        byte_data = np.ascontiguousarray(np_arr).tobytes()
    elif isinstance(tensor, np.ndarray):
        # NumPy Array: ensure contiguous C-order layout
        byte_data = np.ascontiguousarray(tensor).tobytes()
    else:
        raise TypeError(f"Unsupported tensor type for hashing: {type(tensor)}")

    return hashlib.sha256(byte_data).hexdigest()


def compute_dict_sha256(data: Dict[str, Any]) -> str:
    """
    Computes a deterministic SHA-256 hash of a dictionary by canonicalizing key order.

    Args:
        data: Arbitrary dictionary to hash.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    canonical_json = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
