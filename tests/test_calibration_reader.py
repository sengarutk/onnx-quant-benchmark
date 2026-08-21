"""
Unit tests for BenchmarkCalibrationDataReader and calibration dataset isolation.
"""

import csv
import json
from pathlib import Path
import cv2
import numpy as np
import pytest
import torch

from src.quantization.calibration_reader import BenchmarkCalibrationDataReader


class TestCalibrationReader:
    """Test suite validating calibration data reader mechanics and disjoint dataset guarantees."""

    def test_calibration_reader_batch_and_rewind(self, tmp_path: Path) -> None:
        """Tests get_next yields valid numpy arrays and rewind resets iterator."""
        img_paths = []
        for i in range(3):
            p = tmp_path / f"img_{i}.jpg"
            cv2.imwrite(str(p), np.ones((64, 64, 3), dtype=np.uint8) * (i + 1))
            img_paths.append(p)

        def dummy_preprocess(p):
            img = cv2.imread(str(p))
            t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            return t, (1.0, 1.0), (0.0, 0.0), (64, 64)

        reader = BenchmarkCalibrationDataReader(
            image_paths=img_paths,
            input_name="test_input",
            input_shape=(1, 3, 64, 64),
            preprocess_fn=dummy_preprocess,
            batch_size=1,
        )

        b1 = reader.get_next()
        assert b1 is not None
        assert "test_input" in b1
        assert b1["test_input"].shape == (1, 3, 64, 64)
        assert b1["test_input"].dtype == np.float32

        b2 = reader.get_next()
        assert b2 is not None

        b3 = reader.get_next()
        assert b3 is not None

        b_end = reader.get_next()
        assert b_end is None

        # Test rewind
        reader.rewind()
        b1_rewound = reader.get_next()
        assert b1_rewound is not None
        assert np.array_equal(b1["test_input"], b1_rewound["test_input"])

    def test_disjoint_calibration_and_evaluation_datasets(self) -> None:
        """Verifies 0% SHA-256 hash overlap between calibration and sample evaluation images."""
        root = Path(__file__).resolve().parent.parent
        calib_manifest = root / "data" / "calibration" / "manifest.csv"
        eval_manifest = root / "data" / "sample_images" / "manifest.json"

        if not calib_manifest.is_file() or not eval_manifest.is_file():
            pytest.skip("Dataset manifests not yet generated")

        calib_hashes = set()
        with open(calib_manifest, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                calib_hashes.add(row["sha256"])

        eval_data = json.loads(eval_manifest.read_text(encoding="utf-8"))
        eval_hashes = set()
        for item in eval_data.get("detection_samples", []):
            eval_hashes.add(item["sha256"])
        for item in eval_data.get("industrial_samples", []):
            eval_hashes.add(item["sha256"])

        overlap = calib_hashes.intersection(eval_hashes)
        assert len(overlap) == 0, f"Found {len(overlap)} overlapping hashes between calibration and evaluation sets!"
