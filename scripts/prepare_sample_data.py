#!/usr/bin/env python3
"""
Generates the held-out synthetic TEST split for detection and industrial inspection.

Output: data/sample_images/{detection,industrial}/ and data/sample_images/manifest.json
"""

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging import setup_logger
from src.data.writers import write_detection_split, write_industrial_split, write_manifest

logger = setup_logger("prepare_sample_data")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection", type=int, default=200, help="Number of detection test images")
    parser.add_argument("--industrial", type=int, default=100, help="Normal AND anomalous industrial test images (each)")
    args = parser.parse_args()

    out_root = PROJECT_ROOT / "data" / "sample_images"
    for sub in ("detection", "industrial"):
        shutil.rmtree(out_root / sub, ignore_errors=True)

    det = write_detection_split(out_root / "detection", "test", args.detection, PROJECT_ROOT)
    logger.info(f"Generated {len(det)} detection test images ({sum(len(d['ground_truth_boxes']) for d in det)} boxes).")

    ind = write_industrial_split(out_root / "industrial", "test", args.industrial, args.industrial, PROJECT_ROOT)
    logger.info(f"Generated {len(ind)} industrial test images ({args.industrial} normal / {args.industrial} anomalous).")

    manifest = out_root / "manifest.json"
    write_manifest(manifest, "synthetic_edge_benchmark_test", "test", det, ind)
    logger.info(f"Test manifest saved -> {manifest}")


if __name__ == "__main__":
    main()
