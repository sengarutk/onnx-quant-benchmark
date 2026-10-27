"""Tests for the procedural synthetic datasets."""

import hashlib
import numpy as np
import pytest

from src.data.synthetic import SPLIT_SEEDS, generate_detection_sample, generate_industrial_sample, sample_seed


def _h(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()


def test_detection_samples_deterministic_and_valid() -> None:
    a, boxes = generate_detection_sample("test", 7)
    b, boxes2 = generate_detection_sample("test", 7)
    assert np.array_equal(a, b) and boxes == boxes2
    h, w = a.shape[:2]
    assert 1 <= len(boxes) <= 6
    for bx in boxes:
        x1, y1, x2, y2 = bx["bbox"]
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h and bx["class_id"] in (0, 1)


def test_splits_are_disjoint() -> None:
    seeds = {split: {sample_seed("detection", split, i) for i in range(200)} for split in SPLIT_SEEDS}
    splits = list(seeds)
    for i, s1 in enumerate(splits):
        for s2 in splits[i + 1:]:
            assert not seeds[s1] & seeds[s2]
    train = {_h(generate_detection_sample("train", i)[0]) for i in range(10)}
    test = {_h(generate_detection_sample("test", i)[0]) for i in range(10)}
    assert not train & test


def test_industrial_masks() -> None:
    img, mask = generate_industrial_sample("test", 3, anomalous=False)
    assert img.shape == (256, 256, 3) and mask.max() == 0
    _, mask = generate_industrial_sample("test", 3, anomalous=True)
    assert mask.max() == 255 and (mask > 0).sum() > 10


def test_invalid_split_rejected() -> None:
    with pytest.raises(ValueError):
        sample_seed("detection", "holdout", 0)
