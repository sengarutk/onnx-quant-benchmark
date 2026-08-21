"""
Unit tests for deterministic image preprocessing routines.
"""

from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.preprocess import (
    letterbox_image,
    preprocess_detection_image,
    preprocess_industrial_image,
)


class TestPreprocess:
    """Test suite validating image resizing, letterboxing, padding, and normalization."""

    def test_letterbox_image_dimensions(self) -> None:
        """Tests letterbox output matches exact target shape while preserving aspect ratio."""
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        padded, ratio, pad = letterbox_image(img, target_shape=(640, 640), auto=False)

        assert padded.shape == (640, 640, 3)
        assert ratio[0] == 1.0
        assert ratio[1] == 1.0
        assert pad[1] == 80.0  # (640 - 480) / 2

    def test_preprocess_detection_image_numpy_and_file(self, tmp_path: Path) -> None:
        """Tests detection preprocessing from both file paths and memory arrays."""
        import cv2

        dummy_img = np.ones((300, 400, 3), dtype=np.uint8) * 128
        img_path = tmp_path / "sample_det.jpg"
        cv2.imwrite(str(img_path), dummy_img)

        # Test from file path
        t_file, ratio, pad, orig_shape = preprocess_detection_image(img_path, target_shape=(640, 640))
        assert isinstance(t_file, torch.Tensor)
        assert t_file.shape == (1, 3, 640, 640)
        assert t_file.dtype == torch.float32
        assert 0.0 <= t_file.min() <= t_file.max() <= 1.0
        assert orig_shape == (300, 400)

        # Test from NumPy array
        t_arr, ratio2, pad2, orig_shape2 = preprocess_detection_image(dummy_img, target_shape=(640, 640))
        assert torch.equal(t_file, t_arr)
        assert ratio == ratio2
        assert pad == pad2

    def test_preprocess_industrial_image(self, tmp_path: Path) -> None:
        """Tests industrial inspection preprocessing normalization and shapes."""
        dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
        t = preprocess_industrial_image(dummy_img, target_shape=(256, 256))

        assert isinstance(t, torch.Tensor)
        assert t.shape == (1, 3, 256, 256)
        assert t.dtype == torch.float32
        assert 0.0 <= t.min() <= t.max() <= 1.0

    def test_preprocess_detection_missing_file_raises(self, tmp_path: Path) -> None:
        """Ensures preprocess_detection_image raises FileNotFoundError for missing images."""
        with pytest.raises(FileNotFoundError):
            preprocess_detection_image(tmp_path / "missing.jpg")
