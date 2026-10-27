"""
Materializes synthetic dataset splits to disk with YOLO-format labels, defect masks and JSON manifests.
"""

import json
from pathlib import Path
from typing import Dict, List, Union
import cv2

from src.common.hashes import compute_file_sha256
from src.data.synthetic import DETECTION_CLASS_NAMES, generate_detection_sample, generate_industrial_sample


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def write_detection_split(out_dir: Path, split: str, count: int, project_root: Path) -> List[Dict]:
    """Writes `count` detection samples (JPEG + YOLO txt label) and returns manifest entries."""
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for i in range(count):
        img, boxes = generate_detection_sample(split, i)
        h, w = img.shape[:2]
        img_path = img_dir / f"{split}_det_{i:04d}.jpg"
        lbl_path = lbl_dir / f"{split}_det_{i:04d}.txt"
        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lines = []
        for b in boxes:
            x1, y1, x2, y2 = b["bbox"]
            lines.append(
                f"{b['class_id']} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
            )
        lbl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        entries.append({
            "image_path": _rel(img_path, project_root),
            "label_path": _rel(lbl_path, project_root),
            "shape": [h, w, 3],
            "sha256": compute_file_sha256(img_path),
            "ground_truth_boxes": boxes,
        })
    return entries


def write_industrial_split(
    out_dir: Path, split: str, n_normal: int, n_anomalous: int, project_root: Path
) -> List[Dict]:
    """Writes normal and (optionally) anomalous industrial samples with defect masks."""
    norm_dir = out_dir / "normal"
    norm_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for i in range(n_normal):
        img, _ = generate_industrial_sample(split, i, anomalous=False)
        p = norm_dir / f"{split}_normal_{i:04d}.png"
        cv2.imwrite(str(p), img)
        entries.append({
            "image_path": _rel(p, project_root),
            "is_anomalous": False,
            "mask_path": None,
            "shape": [256, 256, 3],
            "sha256": compute_file_sha256(p),
        })

    if n_anomalous:
        anom_dir = out_dir / "anomalous"
        mask_dir = out_dir / "masks"
        anom_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_anomalous):
            img, mask = generate_industrial_sample(split, i, anomalous=True)
            p = anom_dir / f"{split}_anom_{i:04d}.png"
            m = mask_dir / f"{split}_mask_{i:04d}.png"
            cv2.imwrite(str(p), img)
            cv2.imwrite(str(m), mask)
            entries.append({
                "image_path": _rel(p, project_root),
                "is_anomalous": True,
                "mask_path": _rel(m, project_root),
                "shape": [256, 256, 3],
                "sha256": compute_file_sha256(p),
            })
    return entries


def write_manifest(path: Path, name: str, split: str, det: List[Dict], ind: List[Dict]) -> None:
    payload = {
        "dataset_name": name,
        "split": split,
        "class_names": DETECTION_CLASS_NAMES,
        "detection_samples": det,
        "industrial_samples": ind,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_manifest(path: Union[str, Path]) -> Dict:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {p}. Run scripts/prepare_sample_data.py first.")
    return json.loads(p.read_text(encoding="utf-8"))
