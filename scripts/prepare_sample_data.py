#!/usr/bin/env python3
"""
Generates reproducible synthetic evaluation datasets for detection (640x640) and industrial inspection (256x256).
"""

import json
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

logger = setup_logger("prepare_sample_data")


def create_synthetic_detection_dataset(output_dir: Path, num_images: int = 30) -> list:
    """Generates synthetic 640x640 images with geometric objects and YOLO format labels."""
    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []

    for i in range(num_images):
        img_name = f"det_{i:03d}.jpg"
        lbl_name = f"det_{i:03d}.txt"
        img_path = img_dir / img_name
        lbl_path = lbl_dir / lbl_name

        # Create canvas with textured background
        img = np.full((640, 640, 3), fill_value=200 + (i % 20), dtype=np.uint8)

        # Draw grid lines for realistic edge features
        for y in range(0, 640, 40):
            cv2.line(img, (0, y), (640, y), (180, 180, 180), 1)
        for x in range(0, 640, 40):
            cv2.line(img, (x, 0), (x, 640), (180, 180, 180), 1)

        # Generate 1 to 3 deterministic target objects
        num_objects = 1 + (i % 3)
        labels = []
        boxes_meta = []

        for obj_idx in range(num_objects):
            w = 80 + (obj_idx * 40)
            h = 60 + (obj_idx * 30)
            cx = 120 + (obj_idx * 160) + (i * 5) % 100
            cy = 150 + (obj_idx * 120) + (i * 7) % 80
            class_id = obj_idx % 2

            x1, y1 = cx - w // 2, cy - h // 2
            x2, y2 = cx + w // 2, cy + h // 2

            # Draw colored bounding box object
            color = (50, 150, 240) if class_id == 0 else (200, 80, 50)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # YOLO label format: class cx cy w h (normalized to [0, 1])
            norm_cx = cx / 640.0
            norm_cy = cy / 640.0
            norm_w = w / 640.0
            norm_h = h / 640.0
            labels.append(f"{class_id} {norm_cx:.6f} {norm_cy:.6f} {norm_w:.6f} {norm_h:.6f}")
            boxes_meta.append({"bbox": [x1, y1, x2, y2], "class_id": class_id})

        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lbl_path.write_text("\n".join(labels), encoding="utf-8")

        manifest_entries.append(
            {
                "image_path": str(img_path.relative_to(PROJECT_ROOT)),
                "label_path": str(lbl_path.relative_to(PROJECT_ROOT)),
                "shape": [640, 640, 3],
                "sha256": compute_file_sha256(img_path),
                "ground_truth_boxes": boxes_meta,
            }
        )

    return manifest_entries


def create_synthetic_industrial_dataset(output_dir: Path, num_per_class: int = 25) -> list:
    """Generates normal and anomalous 256x256 images with pixel ground-truth defect masks."""
    norm_dir = output_dir / "normal"
    anom_dir = output_dir / "anomalous"
    mask_dir = output_dir / "masks"

    norm_dir.mkdir(parents=True, exist_ok=True)
    anom_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries = []

    # 1. Normal Samples (smooth texture without defects)
    for i in range(num_per_class):
        img_name = f"normal_{i:03d}.png"
        img_path = norm_dir / img_name

        # Create smooth patterned surface
        x = np.linspace(0, 4 * np.pi, 256)
        y = np.linspace(0, 4 * np.pi, 256)
        xx, yy = np.meshgrid(x, y)
        pattern = (np.sin(xx + i * 0.1) * np.cos(yy + i * 0.1) * 30 + 128).astype(np.uint8)
        img = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)

        cv2.imwrite(str(img_path), img)
        manifest_entries.append(
            {
                "image_path": str(img_path.relative_to(PROJECT_ROOT)),
                "is_anomalous": False,
                "mask_path": None,
                "shape": [256, 256, 3],
                "sha256": compute_file_sha256(img_path),
            }
        )

    # 2. Anomalous Samples (surface with localized scratch/crack/stain defect)
    for i in range(num_per_class):
        img_name = f"anom_{i:03d}.png"
        mask_name = f"mask_{i:03d}.png"
        img_path = anom_dir / img_name
        mask_path = mask_dir / mask_name

        x = np.linspace(0, 4 * np.pi, 256)
        y = np.linspace(0, 4 * np.pi, 256)
        xx, yy = np.meshgrid(x, y)
        pattern = (np.sin(xx + i * 0.1) * np.cos(yy + i * 0.1) * 30 + 128).astype(np.uint8)
        img = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)

        # Defect mask canvas
        mask = np.zeros((256, 256), dtype=np.uint8)

        # Inject localized defect (scratch or stain)
        cx = 50 + (i * 7) % 150
        cy = 50 + (i * 9) % 150
        radius = 12 + (i % 10)

        if i % 2 == 0:
            # Scratch line defect
            cv2.line(img, (cx - 20, cy - 20), (cx + 20, cy + 20), (250, 30, 30), 4)
            cv2.line(mask, (cx - 20, cy - 20), (cx + 20, cy + 20), 255, 4)
        else:
            # Stain defect
            cv2.circle(img, (cx, cy), radius, (20, 20, 20), -1)
            cv2.circle(mask, (cx, cy), radius, 255, -1)

        cv2.imwrite(str(img_path), img)
        cv2.imwrite(str(mask_path), mask)

        manifest_entries.append(
            {
                "image_path": str(img_path.relative_to(PROJECT_ROOT)),
                "is_anomalous": True,
                "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
                "shape": [256, 256, 3],
                "sha256": compute_file_sha256(img_path),
            }
        )

    return manifest_entries


def main() -> None:
    seed_everything(42)
    logger.info("Generating synthetic evaluation data...")

    det_entries = create_synthetic_detection_dataset(
        PROJECT_ROOT / "data" / "sample_images" / "detection", num_images=30
    )
    logger.info(f"Generated {len(det_entries)} detection sample images.")

    ind_entries = create_synthetic_industrial_dataset(
        PROJECT_ROOT / "data" / "sample_images" / "industrial", num_per_class=25
    )
    logger.info(f"Generated {len(ind_entries)} industrial inspection images (50 total).")

    full_manifest = {
        "dataset_name": "synthetic_edge_benchmark_samples",
        "detection_samples": det_entries,
        "industrial_samples": ind_entries,
    }

    manifest_path = PROJECT_ROOT / "data" / "sample_images" / "manifest.json"
    manifest_path.write_text(json.dumps(full_manifest, indent=2), encoding="utf-8")
    logger.info(f"Sample data manifest saved -> {manifest_path}")


if __name__ == "__main__":
    main()
