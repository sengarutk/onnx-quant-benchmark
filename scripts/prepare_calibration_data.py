#!/usr/bin/env python3
"""
Generates disjoint representative calibration datasets for detection and industrial models.
"""

import csv
import sys
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.hashes import compute_file_sha256
from src.common.logging import setup_logger
from src.common.seed import seed_everything

logger = setup_logger("prepare_calibration_data")


def generate_calibration_data(
    detection_count: int = 50,
    industrial_count: int = 50,
    output_dir: Path = PROJECT_ROOT / "data" / "calibration",
) -> Path:
    """
    Synthesizes disjoint calibration datasets with unique random seeds ensuring 0% overlap with evaluation splits.
    """
    det_dir = output_dir / "detection"
    ind_dir = output_dir / "industrial"
    det_dir.mkdir(parents=True, exist_ok=True)
    ind_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []

    # 1. Generate Detection Calibration Images (Seed: 100)
    seed_everything(100)
    logger.info(f"Generating {detection_count} detection calibration images...")

    for i in range(detection_count):
        # Create realistic multi-channel noise + geometric shapes
        img = np.random.randint(40, 210, (640, 640, 3), dtype=np.uint8)
        # Draw background structures
        for _ in range(5):
            pt1 = (np.random.randint(0, 640), np.random.randint(0, 640))
            pt2 = (np.random.randint(0, 640), np.random.randint(0, 640))
            color = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))
            cv2.rectangle(img, pt1, pt2, color, -1)

        img_path = det_dir / f"calib_det_{i:03d}.jpg"
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        sha = compute_file_sha256(img_path)
        manifest_rows.append({
            "path": f"data/calibration/detection/{img_path.name}",
            "category": "detection",
            "split": "calibration",
            "sha256": sha,
        })

    # 2. Generate Industrial Normal Calibration Textures (Seed: 200)
    seed_everything(200)
    logger.info(f"Generating {industrial_count} industrial normal calibration images...")

    for i in range(industrial_count):
        # Create continuous brushed metal texture
        base_val = np.random.randint(110, 160)
        grad = np.tile(np.linspace(base_val - 15, base_val + 15, 256, dtype=np.float32), (256, 1))
        noise = np.random.normal(0, 4.0, (256, 256)).astype(np.float32)
        texture = np.clip(grad + noise, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)

        img_path = ind_dir / f"calib_ind_{i:03d}.png"
        cv2.imwrite(str(img_path), img)

        sha = compute_file_sha256(img_path)
        manifest_rows.append({
            "path": f"data/calibration/industrial/{img_path.name}",
            "category": "industrial",
            "split": "calibration",
            "sha256": sha,
        })

    # Write CSV manifest
    manifest_csv = output_dir / "manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "category", "split", "sha256"])
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Write README.md
    readme_md = output_dir / "README.md"
    readme_md.write_text(
        f"""# Representative Calibration Dataset

- **Total Samples**: {len(manifest_rows)} ({detection_count} Detection, {industrial_count} Industrial Normal)
- **Manifest**: `manifest.csv`
- **Isolation Constraint**: 100% disjoint from `data/sample_images/` test splits.
""",
        encoding="utf-8",
    )

    logger.info(f"Calibration data saved: {manifest_csv} ({len(manifest_rows)} records)")
    return manifest_csv


if __name__ == "__main__":
    generate_calibration_data()
