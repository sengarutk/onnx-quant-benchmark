"""
Procedural synthetic datasets used throughout the benchmark.

Every sample is a pure function of ``(task, split, index)``: the per-sample RNG seed is derived from a
fixed split offset, so the train / calibration / test splits are reproducible and disjoint by construction
(and the preparation scripts additionally verify zero SHA-256 overlap between the files they write).

Detection task (2 classes):
    class 0 ``rectangle`` and class 1 ``ellipse`` drawn with random colours, sizes and aspect ratios on noisy,
    textured backgrounds of varying resolution, with unlabeled line/blob distractors and mild blur.

Industrial anomaly task:
    normal samples are randomly parameterised sinusoidal surface textures with sensor noise; anomalous samples
    add a scratch, stain or texture-dropout defect with a pixel-accurate ground-truth mask.
"""

from typing import Dict, List, Tuple
import cv2
import numpy as np

DETECTION_CLASS_NAMES: List[str] = ["rectangle", "ellipse"]

# Disjoint seed offsets per split (indices are always < 1_000_000).
SPLIT_SEEDS: Dict[str, int] = {
    "train": 10_000_000,
    "calibration": 20_000_000,
    "test": 30_000_000,
    "validation": 40_000_000,  # hyper-parameter selection only (never used for reported metrics)
}

_TASK_OFFSETS: Dict[str, int] = {"detection": 0, "industrial": 100_000_000}

DETECTION_IMAGE_SIZES: List[Tuple[int, int]] = [(640, 640), (480, 640), (600, 800), (720, 960)]  # (H, W)


def sample_seed(task: str, split: str, index: int) -> int:
    """Returns the deterministic RNG seed for one sample of a task split."""
    if split not in SPLIT_SEEDS:
        raise ValueError(f"Unknown split '{split}', expected one of {sorted(SPLIT_SEEDS)}")
    if not 0 <= index < 1_000_000:
        raise ValueError(f"Sample index out of range: {index}")
    return _TASK_OFFSETS[task] + SPLIT_SEEDS[split] + index


def _gaussian_noise(rng: np.random.Generator, shape: Tuple[int, ...], sigma: float) -> np.ndarray:
    """Fast seeded Gaussian noise (OpenCV RNG seeded from the sample RNG)."""
    noise = np.empty(shape, dtype=np.float32)
    cv2.setRNGSeed(int(rng.integers(0, 2**31 - 1)))
    cv2.randn(noise, 0.0, float(sigma))
    return noise


def _textured_background(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    base = rng.uniform(60, 200, size=3).astype(np.float32)
    angle = rng.uniform(0, np.pi)
    amp = rng.uniform(-40, 40)
    gx = (np.cos(angle) * amp * np.arange(w, dtype=np.float32) / w)[None, :]
    gy = (np.sin(angle) * amp * np.arange(h, dtype=np.float32) / h)[:, None]
    img = (gx + gy)[..., None] + base[None, None, :]
    img = img + _gaussian_noise(rng, (h, w, 3), rng.uniform(4.0, 14.0))
    if rng.random() < 0.5:
        step = int(rng.integers(24, 64))
        tone = float(rng.uniform(-30, 30))
        img[::step, :, :] += tone
        img[:, ::step, :] += tone
    return img


def _box_iou(a: List[int], b: List[int]) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(union, 1)


def generate_detection_sample(split: str, index: int) -> Tuple[np.ndarray, List[Dict]]:
    """
    Generates one detection sample.

    Returns:
        (BGR uint8 image of shape [H, W, 3], list of {"bbox": [x1, y1, x2, y2], "class_id": int}) in
        original-image pixel coordinates.
    """
    rng = np.random.default_rng(sample_seed("detection", split, index))
    h, w = DETECTION_IMAGE_SIZES[int(rng.integers(len(DETECTION_IMAGE_SIZES)))]
    img = _textured_background(rng, h, w)

    # Unlabeled distractors: thin lines and small blobs (sources of potential false positives).
    for _ in range(int(rng.integers(0, 6))):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.line(img, p1, p2, tuple(float(v) for v in rng.uniform(0, 255, 3)), int(rng.integers(1, 3)))
    for _ in range(int(rng.integers(0, 4))):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(img, c, int(rng.integers(2, 7)), tuple(float(v) for v in rng.uniform(0, 255, 3)), -1)

    boxes: List[Dict] = []
    n_objects = int(rng.integers(1, 7))
    attempts = 0
    while len(boxes) < n_objects and attempts < 60:
        attempts += 1
        bw = int(rng.integers(24, 201))
        bh = int(np.clip(bw * rng.uniform(0.5, 2.0), 24, 220))
        x1 = int(rng.integers(0, max(1, w - bw)))
        y1 = int(rng.integers(0, max(1, h - bh)))
        box = [x1, y1, x1 + bw, y1 + bh]
        if any(_box_iou(box, b["bbox"]) > 0.3 for b in boxes):
            continue
        cls = int(rng.integers(0, 2))
        color = tuple(float(v) for v in rng.uniform(0, 255, 3))
        if cls == 0:
            cv2.rectangle(img, (box[0], box[1]), (box[2] - 1, box[3] - 1), color, -1)
        else:
            center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            cv2.ellipse(img, center, (bw // 2, bh // 2), 0.0, 0.0, 360.0, color, -1)
        if rng.random() < 0.5:
            edge = tuple(float(v) for v in rng.uniform(0, 255, 3))
            if cls == 0:
                cv2.rectangle(img, (box[0], box[1]), (box[2] - 1, box[3] - 1), edge, 2)
            else:
                cv2.ellipse(img, center, (bw // 2, bh // 2), 0.0, 0.0, 360.0, edge, 2)
        boxes.append({"bbox": box, "class_id": cls})

    if rng.random() < 0.5:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    img += _gaussian_noise(rng, img.shape, 3.0)
    return np.clip(img, 0, 255).astype(np.uint8), boxes


def _normal_texture(rng: np.random.Generator, size: int = 256) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    freq = rng.uniform(2.0, 6.0)
    theta = rng.uniform(0, np.pi)
    phase = rng.uniform(0, 2 * np.pi)
    proj = np.cos(theta) * xx + np.sin(theta) * yy
    texture = np.sin(2 * np.pi * freq * proj + phase)
    texture += 0.5 * np.sin(2 * np.pi * freq * 2.1 * (np.sin(theta) * xx - np.cos(theta) * yy) + phase)
    gray = rng.uniform(100, 160) + rng.uniform(15, 30) * texture
    gray = gray.astype(np.float32) + _gaussian_noise(rng, gray.shape, rng.uniform(2.0, 5.0))
    tint = rng.uniform(0.92, 1.08, size=3)
    return np.stack([gray * t for t in tint], axis=-1).astype(np.float32)


def generate_industrial_sample(split: str, index: int, anomalous: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates one 256x256 industrial texture sample.

    Returns:
        (BGR uint8 image [256, 256, 3], uint8 mask [256, 256] with 255 on defect pixels).
    """
    rng = np.random.default_rng(sample_seed("industrial", split, index) * 2 + int(anomalous))
    img = _normal_texture(rng)
    mask = np.zeros((256, 256), dtype=np.uint8)

    if anomalous:
        kind = int(rng.integers(0, 3))
        delta = float(rng.uniform(25, 70)) * (1 if rng.random() < 0.5 else -1)
        defect = np.zeros((256, 256), dtype=np.uint8)
        if kind == 0:  # scratch
            x1, y1 = int(rng.integers(20, 236)), int(rng.integers(20, 236))
            length = rng.uniform(25, 90)
            ang = rng.uniform(0, np.pi)
            x2 = int(np.clip(x1 + length * np.cos(ang), 0, 255))
            y2 = int(np.clip(y1 + length * np.sin(ang), 0, 255))
            cv2.line(defect, (x1, y1), (x2, y2), 255, int(rng.integers(1, 4)))
        elif kind == 1:  # stain
            c = (int(rng.integers(25, 231)), int(rng.integers(25, 231)))
            axes = (int(rng.integers(5, 19)), int(rng.integers(5, 19)))
            cv2.ellipse(defect, c, axes, float(rng.uniform(0, 180)), 0.0, 360.0, 255, -1)
        else:  # texture dropout: region flattened to its local mean
            x, y = int(rng.integers(10, 200)), int(rng.integers(10, 200))
            pw, ph = int(rng.integers(14, 46)), int(rng.integers(14, 46))
            defect[y:y + ph, x:x + pw] = 255
            region = img[y:y + ph, x:x + pw]
            img[y:y + ph, x:x + pw] = region.mean(axis=(0, 1), keepdims=True)
            delta = 0.0
        if delta != 0.0:
            soft = cv2.GaussianBlur(defect.astype(np.float32) / 255.0, (5, 5), 0)
            img += soft[..., None] * delta
        mask = defect

    return np.clip(img, 0, 255).astype(np.uint8), mask
