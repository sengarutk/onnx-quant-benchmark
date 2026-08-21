#!/usr/bin/env python3
"""
Runs reference PyTorch FP32 evaluations, computes quality metrics, and persists baseline signatures.
"""

import json
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_config
from src.common.hashes import compute_tensor_sha256
from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.models.industrial_model_adapter import IndustrialModelAdapter
from src.models.preprocess import preprocess_detection_image, preprocess_industrial_image
from src.models.yolo_adapter import YOLOAdapter
from src.validation.anomaly_quality import compute_aupro, compute_image_auroc, compute_pixel_auroc
from src.validation.detection_quality import evaluate_detection_dataset

logger = setup_logger("run_pytorch_baselines")


def run_yolo_baseline() -> None:
    logger.info("Executing PyTorch FP32 baseline for YOLO nano detector...")
    cfg = load_config(PROJECT_ROOT / "configs" / "yolo" / "fp32.yaml")
    adapter = YOLOAdapter(conf_threshold=cfg.model.conf_threshold, iou_threshold=cfg.model.iou_threshold)

    # Serialize baseline weights
    weights_dir = PROJECT_ROOT / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    yolo_weights_path = weights_dir / "yolo_nano_baseline.pt"
    torch.save(adapter.model.state_dict(), yolo_weights_path)
    logger.info(f"YOLO baseline weights serialized -> {yolo_weights_path}")

    manifest_path = PROJECT_ROOT / "data" / "sample_images" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    det_samples = manifest["detection_samples"]

    all_preds = []
    all_gts = []
    raw_outputs = []

    for item in det_samples:
        img_path = PROJECT_ROOT / item["image_path"]
        tensor, ratio, pad, orig_shape = preprocess_detection_image(img_path)

        raw_output = adapter.forward(tensor)
        raw_outputs.append(raw_output.detach().cpu())

        detections = adapter.postprocess(raw_output, orig_shape, ratio, pad)
        all_preds.append(detections)
        all_gts.append(item["ground_truth_boxes"])

    metrics = evaluate_detection_dataset(all_preds, all_gts)

    # Compute hash of concatenated output tensors
    stacked_raw = torch.cat(raw_outputs, dim=0).numpy()
    out_hash = compute_tensor_sha256(stacked_raw)

    result_payload = {
        "model_name": "yolo_nano",
        "precision": "fp32",
        "runtime": "pytorch",
        "metrics": metrics,
        "raw_output_sha256": out_hash,
        "sample_count": len(det_samples),
    }

    out_dir = PROJECT_ROOT / "results" / "raw" / "pytorch_fp32" / "yolo_nano"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "baseline_metrics.json"
    out_file.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    logger.info(f"YOLO FP32 baseline saved -> {out_file} (mAP@50: {metrics['mAP_50']})")


def run_industrial_baseline() -> None:
    logger.info("Executing PyTorch FP32 baseline for Industrial Autoencoder...")
    adapter = IndustrialModelAdapter()

    # Serialize baseline weights
    weights_dir = PROJECT_ROOT / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    ind_weights_path = weights_dir / "industrial_autoencoder_baseline.pt"
    torch.save(adapter.model.state_dict(), ind_weights_path)
    logger.info(f"Industrial baseline weights serialized -> {ind_weights_path}")

    manifest_path = PROJECT_ROOT / "data" / "sample_images" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ind_samples = manifest["industrial_samples"]

    y_true = []
    y_scores = []
    gt_masks = []
    anomaly_maps = []
    reconstructed_tensors = []

    for item in ind_samples:
        img_path = PROJECT_ROOT / item["image_path"]
        tensor = preprocess_industrial_image(img_path)

        recon, a_map = adapter.forward(tensor)
        reconstructed_tensors.append(recon.detach().cpu())

        score = adapter.compute_anomaly_score(a_map)
        is_anom = 1 if item["is_anomalous"] else 0

        y_true.append(is_anom)
        y_scores.append(score)

        if item["mask_path"]:
            mask_img = cv2.imread(str(PROJECT_ROOT / item["mask_path"]), cv2.IMREAD_GRAYSCALE)
            mask_arr = (mask_img / 255.0).astype(np.float32)
        else:
            mask_arr = np.zeros((256, 256), dtype=np.float32)

        gt_masks.append(mask_arr)
        anomaly_maps.append(a_map.squeeze().detach().cpu().numpy())

    image_auroc = compute_image_auroc(y_true, y_scores)
    pixel_auroc = compute_pixel_auroc(gt_masks, anomaly_maps)
    aupro = compute_aupro(gt_masks, anomaly_maps)

    stacked_recons = torch.cat(reconstructed_tensors, dim=0).numpy()
    out_hash = compute_tensor_sha256(stacked_recons)

    result_payload = {
        "model_name": "industrial_autoencoder",
        "precision": "fp32",
        "runtime": "pytorch",
        "metrics": {
            "image_auroc": image_auroc,
            "pixel_auroc": pixel_auroc,
            "aupro": aupro,
        },
        "raw_output_sha256": out_hash,
        "sample_count": len(ind_samples),
    }

    out_dir = PROJECT_ROOT / "results" / "raw" / "pytorch_fp32" / "industrial_autoencoder"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "baseline_metrics.json"
    out_file.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    logger.info(f"Industrial FP32 baseline saved -> {out_file} (Image AUROC: {image_auroc}, Pixel AUROC: {pixel_auroc})")


def main() -> None:
    seed_everything(42)
    run_yolo_baseline()
    run_industrial_baseline()
    print("\nPyTorch FP32 Baselines executed and persisted successfully.")


if __name__ == "__main__":
    main()
