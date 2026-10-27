#!/usr/bin/env python3
"""
Generates the labeled synthetic CALIBRATION split, used both for static INT8 PTQ range calibration and for
Q-Aware NMS threshold calibration. Verifies zero SHA-256 overlap with the test split.

Output: data/calibration/{detection,industrial}/, data/calibration/manifest.json and manifest.csv
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging import setup_logger
from src.data.writers import write_detection_split, write_industrial_split, write_manifest

logger = setup_logger("prepare_calibration_data")


def generate_calibration_data(
    detection_count: int = 100,
    industrial_count: int = 100,
    output_dir: Path = PROJECT_ROOT / "data" / "calibration",
) -> Path:
    """Writes the calibration split and returns the path of the CSV manifest."""
    for sub in ("detection", "industrial"):
        shutil.rmtree(output_dir / sub, ignore_errors=True)

    det = write_detection_split(output_dir / "detection", "calibration", detection_count, PROJECT_ROOT)
    ind = write_industrial_split(output_dir / "industrial", "calibration", industrial_count, 0, PROJECT_ROOT)
    write_manifest(output_dir / "manifest.json", "synthetic_edge_benchmark_calibration", "calibration", det, ind)

    manifest_csv = output_dir / "manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "category", "split", "sha256", "label_path"])
        writer.writeheader()
        for e in det:
            writer.writerow({"path": e["image_path"], "category": "detection", "split": "calibration",
                             "sha256": e["sha256"], "label_path": e["label_path"]})
        for e in ind:
            writer.writerow({"path": e["image_path"], "category": "industrial", "split": "calibration",
                             "sha256": e["sha256"], "label_path": ""})

    # Disjointness audit against the test split (if present)
    test_manifest = PROJECT_ROOT / "data" / "sample_images" / "manifest.json"
    overlap = 0
    if test_manifest.is_file():
        test = json.loads(test_manifest.read_text(encoding="utf-8"))
        test_hashes = {s["sha256"] for s in test["detection_samples"] + test["industrial_samples"]}
        overlap = len(test_hashes & {e["sha256"] for e in det + ind})
        if overlap:
            raise RuntimeError(f"Calibration/test leakage detected: {overlap} identical files")

    (output_dir / "README.md").write_text(
        f"""# Calibration Split

- **Detection**: {len(det)} labeled images (YOLO-format labels in `detection/labels/`)
- **Industrial**: {len(ind)} normal textures (no anomalies, matching unsupervised deployment)
- **Manifests**: `manifest.json` (with ground truth), `manifest.csv` (paths + SHA-256)
- **Disjointness**: generated from a separate seed range; verified SHA-256 overlap with the test split = {overlap}
""",
        encoding="utf-8",
    )
    logger.info(f"Calibration split saved: {len(det)} detection + {len(ind)} industrial images (test overlap: {overlap})")
    return manifest_csv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection", type=int, default=100)
    parser.add_argument("--industrial", type=int, default=100)
    args = parser.parse_args()
    generate_calibration_data(args.detection, args.industrial)


if __name__ == "__main__":
    main()
