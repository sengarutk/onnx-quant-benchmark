#!/usr/bin/env python3
"""
Trains both benchmark models from scratch on the procedural synthetic TRAIN split and records training logs.

  * YOLO-Nano-style detector: FCOS-style assignment, Quality Focal + GIoU loss, AdamW, cosine LR, FP32, checkpoint every 500 steps (--resume).
  * Convolutional autoencoder: MSE reconstruction on defect-free textures only (unsupervised anomaly setting).

Outputs:
  models/weights/yolo_nano_trained.pt, models/weights/industrial_autoencoder_trained.pt
  results/training/{yolo_nano,industrial_autoencoder}_training.json (loss curve + test-split metrics)
"""

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.data.writers import load_manifest
from src.models.industrial_model_adapter import ConvAutoencoder, IndustrialModelAdapter
from src.models.yolo_adapter import YOLOAdapter, YOLONanoDetector
from src.training import DetectionLoss, SyntheticDetectionDataset, SyntheticTextureDataset, detection_collate
from src.validation.task_eval import evaluate_autoencoder, evaluate_detector

logger = setup_logger("train_models")
WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"
LOG_DIR = PROJECT_ROOT / "results" / "training"


def _cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return base * (0.02 + 0.98 * 0.5 * (1 + math.cos(math.pi * progress)))


def _torch_predict(model: torch.nn.Module, device: torch.device):
    def fn(x: np.ndarray):
        with torch.inference_mode():
            out = model(torch.from_numpy(x).to(device))
        if isinstance(out, (tuple, list)):
            return [o.float().cpu().numpy() for o in out]
        return out.float().cpu().numpy()
    return fn


def train_detector(steps: int, batch: int, lr: float, workers: int, device: torch.device, resume: bool = False) -> dict:
    """Trains the detector; checkpoints every 500 steps so an interrupted run can continue with --resume."""
    seed_everything(0)
    model = YOLONanoDetector().to(device)
    criterion = DetectionLoss(model.num_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    ckpt_path = WEIGHTS_DIR / "yolo_nano_checkpoint.pt"
    start, history, prev_seconds = 0, [], 0.0
    if resume and ckpt_path.is_file():
        ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
        if ckpt["steps"] == steps and ckpt["batch"] == batch:
            model.load_state_dict(ckpt["model"])
            opt.load_state_dict(ckpt["optimizer"])
            start, history, prev_seconds = ckpt["step"] + 1, ckpt["history"], ckpt["train_seconds"]
            logger.info(f"[detector] resuming from step {start}")

    # Sample i of step k is always index k * batch + i, so a resumed run sees exactly the same data stream.
    dataset = Subset(SyntheticDetectionDataset(length=steps * batch), range(start * batch, steps * batch))
    loader = DataLoader(dataset, batch_size=batch, shuffle=False, num_workers=workers,
                        collate_fn=detection_collate, persistent_workers=workers > 0)

    t0 = time.perf_counter()
    model.train()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for step, (imgs, targets) in enumerate(loader, start=start):
        for g in opt.param_groups:
            g["lr"] = _cosine_lr(step, steps, lr, warmup=200)
        raw = model.forward_raw(imgs.to(device, non_blocking=True))
        losses = criterion([r.float() for r in raw], targets)
        opt.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            rec = {k: round(float(v), 4) for k, v in losses.items()}
            rec["step"] = step
            history.append(rec)
            logger.info(f"[detector] step {step:5d}/{steps} loss={rec['loss']:.4f} cls={rec['loss_cls']:.4f} box={rec['loss_box']:.4f}")
        if (step + 1) % 500 == 0:
            torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step, "steps": steps,
                        "batch": batch, "history": history, "train_seconds": prev_seconds + time.perf_counter() - t0},
                       str(ckpt_path))
    train_seconds = prev_seconds + time.perf_counter() - t0
    ckpt_path.unlink(missing_ok=True)

    model.eval()
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    weights = WEIGHTS_DIR / "yolo_nano_trained.pt"
    torch.save(model.cpu().state_dict(), weights)

    adapter = YOLOAdapter(weights_path=str(weights))
    val_metrics = evaluate_detector_on_validation(adapter)
    logger.info(f"[detector] validation-split metrics: {val_metrics}")
    test = load_manifest(PROJECT_ROOT / "data" / "sample_images" / "manifest.json")["detection_samples"]
    metrics = evaluate_detector(_torch_predict(adapter.model, torch.device("cpu")), test, adapter)
    logger.info(f"[detector] test-split metrics: {metrics}")
    return {"steps": steps, "batch_size": batch, "lr": lr, "train_seconds": round(train_seconds, 1),
            "parameters": sum(p.numel() for p in model.parameters()), "history": history,
            "validation_metrics": val_metrics, "test_metrics": metrics}


def evaluate_detector_on_validation(adapter: YOLOAdapter, n: int = 200) -> dict:
    """mAP / F1 on the VALIDATION split (generated in memory); used for training-recipe decisions."""
    import tempfile
    import cv2
    from src.data.synthetic import generate_detection_sample

    with tempfile.TemporaryDirectory() as tmp:
        samples = []
        for i in range(n):
            img, boxes = generate_detection_sample("validation", i)
            path = Path(tmp) / f"val_{i:04d}.jpg"
            cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])  # same encoding as the test split
            samples.append({"image_path": str(path), "ground_truth_boxes": boxes})
        return evaluate_detector(_torch_predict(adapter.model, torch.device("cpu")), samples, adapter)


def train_autoencoder(steps: int, batch: int, lr: float, workers: int, device: torch.device) -> dict:
    seed_everything(0)
    model = ConvAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loader = DataLoader(SyntheticTextureDataset(length=steps * batch), batch_size=batch, shuffle=False,
                        num_workers=workers, persistent_workers=workers > 0)

    history, t0 = [], time.perf_counter()
    model.train()
    for step, x in enumerate(loader):
        for g in opt.param_groups:
            g["lr"] = _cosine_lr(step, steps, lr, warmup=100)
        x = x.to(device, non_blocking=True)
        loss = torch.nn.functional.mse_loss(model(x), x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % 100 == 0 or step == steps - 1:
            history.append({"step": step, "loss": round(float(loss), 6)})
            logger.info(f"[autoencoder] step {step:5d}/{steps} mse={float(loss):.6f}")
    train_seconds = time.perf_counter() - t0

    model.eval()
    weights = WEIGHTS_DIR / "industrial_autoencoder_trained.pt"
    torch.save(model.cpu().state_dict(), weights)
    scoring = select_anomaly_scoring(model)

    from src.export.export_onnx import AutoencoderExportWrapper
    adapter = IndustrialModelAdapter(weights_path=str(weights))
    test = load_manifest(PROJECT_ROOT / "data" / "sample_images" / "manifest.json")["industrial_samples"]
    metrics = evaluate_autoencoder(_torch_predict(AutoencoderExportWrapper(adapter.model).eval(), torch.device("cpu")), test)
    logger.info(f"[autoencoder] test-split metrics: {metrics}")
    return {"steps": steps, "batch_size": batch, "lr": lr, "train_seconds": round(train_seconds, 1),
            "parameters": sum(p.numel() for p in model.parameters()), "history": history,
            "scoring_selection": scoring, "test_metrics": metrics}


def select_anomaly_scoring(model: torch.nn.Module, n_per_class: int = 100) -> dict:
    """
    Selects (blur_sigma, top_k_ratio) of the anomaly score by image AUROC on the VALIDATION split (separate seed
    range; never the test split) and stores them in models/weights/industrial_autoencoder_scoring.json.
    """
    import cv2
    from sklearn.metrics import roc_auc_score
    from src.data.synthetic import generate_industrial_sample
    from src.models.industrial_model_adapter import SCORING_PARAMS_PATH, top_k_anomaly_score

    maps, labels = [], []
    model = model.cpu().eval()
    for anomalous in (False, True):
        for i in range(n_per_class):
            img, _ = generate_industrial_sample("validation", i, anomalous)
            x = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1).copy()).float()[None] / 255.0
            with torch.inference_mode():
                maps.append(torch.mean(torch.abs(x - model(x)), dim=1)[0].numpy())
            labels.append(int(anomalous))
    grid = []
    for sigma in (0.0, 1.0, 2.0, 4.0, 8.0):
        for k in (0.001, 0.005, 0.01, 0.05):
            auc = roc_auc_score(labels, [top_k_anomaly_score(m, k, sigma) for m in maps])
            grid.append({"blur_sigma": sigma, "top_k_ratio": k, "val_image_auroc": round(float(auc), 4)})
    best = max(grid, key=lambda g: g["val_image_auroc"])
    SCORING_PARAMS_PATH.write_text(json.dumps({**best, "selected_on": "validation split", "grid": grid}, indent=2), encoding="utf-8")
    logger.info(f"[autoencoder] anomaly scoring selected on validation split: {best}")
    return {"selected": best, "grid": grid}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--det-steps", type=int, default=12000)
    parser.add_argument("--det-batch", type=int, default=16)
    parser.add_argument("--ae-steps", type=int, default=2500)
    parser.add_argument("--ae-batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--only", choices=["detector", "autoencoder"], default=None)
    parser.add_argument("--resume", action="store_true", help="Continue detector training from its last checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    meta = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__, "timestamp": datetime.now(timezone.utc).isoformat()}

    if args.only in (None, "detector"):
        log = train_detector(args.det_steps, args.det_batch, 2e-3, args.workers, device, resume=args.resume)
        (LOG_DIR / "yolo_nano_training.json").write_text(json.dumps({**meta, **log}, indent=2), encoding="utf-8")
    if args.only in (None, "autoencoder"):
        log = train_autoencoder(args.ae_steps, args.ae_batch, 1e-3, args.workers, device)
        (LOG_DIR / "industrial_autoencoder_training.json").write_text(json.dumps({**meta, **log}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
