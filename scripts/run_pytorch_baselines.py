#!/usr/bin/env python3
"""
Evaluates the trained PyTorch FP32 models on the test split (CPU) and stores the reference metrics used by the
quantization gates: results/baselines/{yolo_nano,industrial_autoencoder}.json
"""

import json
import sys
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.hashes import compute_file_sha256
from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.data.writers import load_manifest
from src.export.export_onnx import AutoencoderExportWrapper
from src.models.industrial_model_adapter import IndustrialModelAdapter
from src.models.yolo_adapter import YOLOAdapter
from src.validation.task_eval import evaluate_autoencoder, evaluate_detector

logger = setup_logger("run_pytorch_baselines")
OUT_DIR = PROJECT_ROOT / "results" / "baselines"


def _predict(model: torch.nn.Module):
    def fn(x: np.ndarray):
        with torch.inference_mode():
            out = model(torch.from_numpy(x))
        return [o.numpy() for o in out] if isinstance(out, (tuple, list)) else out.numpy()
    return fn


def main() -> None:
    seed_everything(42)
    manifest = load_manifest(PROJECT_ROOT / "data" / "sample_images" / "manifest.json")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    yolo = YOLOAdapter(require_weights=True)
    det = evaluate_detector(_predict(yolo.model), manifest["detection_samples"], yolo)
    ae = IndustrialModelAdapter(require_weights=True)
    ind = evaluate_autoencoder(_predict(AutoencoderExportWrapper(ae.model).eval()), manifest["industrial_samples"])

    for name, metrics, weights, n in (
        ("yolo_nano", det, "yolo_nano_trained.pt", len(manifest["detection_samples"])),
        ("industrial_autoencoder", ind, "industrial_autoencoder_trained.pt", len(manifest["industrial_samples"])),
    ):
        payload = {"model_name": name, "precision": "fp32", "runtime": "pytorch_cpu", "split": "test",
                   "sample_count": n, "weights_sha256": compute_file_sha256(PROJECT_ROOT / "models" / "weights" / weights),
                   "metrics": metrics}
        (OUT_DIR / f"{name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(f"{name} PyTorch FP32 baseline: {metrics}")


if __name__ == "__main__":
    main()
