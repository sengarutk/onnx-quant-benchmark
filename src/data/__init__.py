"""Deterministic synthetic datasets for detection and industrial anomaly inspection."""
from src.data.synthetic import (
    DETECTION_CLASS_NAMES,
    SPLIT_SEEDS,
    generate_detection_sample,
    generate_industrial_sample,
    sample_seed,
)

__all__ = [
    "DETECTION_CLASS_NAMES",
    "SPLIT_SEEDS",
    "generate_detection_sample",
    "generate_industrial_sample",
    "sample_seed",
]
