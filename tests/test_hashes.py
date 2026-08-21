"""
Unit tests for memory-safe streaming hashing utilities.
"""

import hashlib
from pathlib import Path
import numpy as np
import pytest

from src.common.hashes import compute_dict_sha256, compute_file_sha256, compute_tensor_sha256


class TestHashes:
    """Test suite verifying SHA-256 integrity functions."""

    def test_compute_file_sha256_known_string(self, temp_dummy_file: Path) -> None:
        """Tests compute_file_sha256 against known raw bytes digest."""
        expected = hashlib.sha256(b"Benchmark test data for SHA-256 verification\n").hexdigest()
        actual = compute_file_sha256(temp_dummy_file)
        assert actual == expected
        assert len(actual) == 64

    def test_compute_file_sha256_empty_file(self, tmp_path: Path) -> None:
        """Tests compute_file_sha256 on an empty file against standard SHA-256 empty digest."""
        empty_file = tmp_path / "empty.bin"
        empty_file.write_bytes(b"")
        expected_empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert compute_file_sha256(empty_file) == expected_empty

    def test_compute_file_sha256_missing_file_raises(self, tmp_path: Path) -> None:
        """Ensures compute_file_sha256 raises FileNotFoundError for nonexistent paths."""
        with pytest.raises(FileNotFoundError):
            compute_file_sha256(tmp_path / "nonexistent.bin")

    def test_compute_tensor_sha256_numpy(self) -> None:
        """Tests hashing of NumPy ndarrays ensuring consistency and value sensitivity."""
        arr1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        arr2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        arr3 = np.array([1.0, 2.0, 3.0, 4.0001], dtype=np.float32)

        hash1 = compute_tensor_sha256(arr1)
        hash2 = compute_tensor_sha256(arr2)
        hash3 = compute_tensor_sha256(arr3)

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64

    def test_compute_tensor_sha256_torch(self) -> None:
        """Tests hashing of PyTorch tensors across memory views and non-contiguous layouts."""
        try:
            import torch

            t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
            t2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
            t_permuted = t1.t()  # Non-contiguous view

            hash1 = compute_tensor_sha256(t1)
            hash2 = compute_tensor_sha256(t2)
            hash_perm = compute_tensor_sha256(t_permuted)

            assert hash1 == hash2
            assert hash1 != hash_perm
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_compute_tensor_sha256_invalid_type_raises(self) -> None:
        """Tests compute_tensor_sha256 raises TypeError on invalid types."""
        with pytest.raises(TypeError):
            compute_tensor_sha256("not a tensor")  # type: ignore

    def test_compute_dict_sha256_key_order_invariance(self) -> None:
        """Tests compute_dict_sha256 yields identical digests regardless of key insertion order."""
        d1 = {"model": "yolo", "batch_size": 1, "precision": "fp16", "threads": 8}
        d2 = {"threads": 8, "precision": "fp16", "batch_size": 1, "model": "yolo"}
        d3 = {"threads": 8, "precision": "fp32", "batch_size": 1, "model": "yolo"}

        h1 = compute_dict_sha256(d1)
        h2 = compute_dict_sha256(d2)
        h3 = compute_dict_sha256(d3)

        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 64
