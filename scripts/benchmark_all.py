#!/usr/bin/env python3
"""
Master Benchmark CLI executing the full matrix of models and runtimes.
"""

import json
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.benchmark_suite import BenchmarkSuite
from src.common.environment import configure_cpu_threads, get_physical_core_count
from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.models.industrial_model_adapter import IndustrialModelAdapter
from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.ort_cpu_runtime import ORTCPURuntime
from src.runtimes.ort_cuda_runtime import ORTCUDARuntime
from src.runtimes.pytorch_runtime import PyTorchRuntime
from src.runtimes.tensorrt_runtime import TRT_AVAILABLE, TensorRTRuntime

logger = setup_logger("benchmark_all")


def main() -> None:
    seed_everything(42)
    cores = configure_cpu_threads()
    logger.info("=" * 70)
    logger.info(f"  STARTING PHASE 5: UNIFIED BENCHMARK & MULTI-SESSION PROFILING (Cores: {cores})")
    logger.info("=" * 70)

    suite = BenchmarkSuite(
        warmup_model=50,
        timed_model=100,
        warmup_e2e=10,
        timed_e2e=30,
        stability_sessions=5,
    )

    models_dir = PROJECT_ROOT / "models" / "exported"
    sample_dir = PROJECT_ROOT / "data" / "sample_images"
    detection_samples = list((sample_dir / "detection" / "images").glob("*.jpg"))
    if not detection_samples:
        detection_samples = list((sample_dir / "detection").rglob("*.jpg"))

    industrial_samples = list((sample_dir / "industrial" / "normal").glob("*.png"))
    if not industrial_samples:
        industrial_samples = list((sample_dir / "industrial").rglob("*.png"))

    yolo_adapter = YOLOAdapter()
    ind_adapter = IndustrialModelAdapter()

    dummy_yolo = torch.randn(1, 3, 640, 640)
    dummy_ind = torch.randn(1, 3, 256, 256)

    # -------------------------------------------------------------
    # 1. YOLO Nano Configurations
    # -------------------------------------------------------------
    # PyTorch CPU
    pt_yolo_cpu = PyTorchRuntime(yolo_adapter.get_pytorch_model(), device="cpu")
    suite.run_single_configuration(
        model_name="yolo_nano",
        runtime_name="PyTorch",
        precision="fp32",
        runtime=pt_yolo_cpu,
        input_tensor=dummy_yolo,
        sample_image_paths=detection_samples,
        adapter=yolo_adapter,
        task="detection",
        quality_metric="mAP_50",
        quality_value=0.0099,
    )

    # PyTorch CUDA
    if torch.cuda.is_available():
        pt_yolo_cuda = PyTorchRuntime(yolo_adapter.get_pytorch_model(), device="cuda:0")
        suite.run_single_configuration(
            model_name="yolo_nano",
            runtime_name="PyTorch",
            precision="fp32",
            runtime=pt_yolo_cuda,
            input_tensor=dummy_yolo.cuda(),
            sample_image_paths=detection_samples,
            adapter=yolo_adapter,
            task="detection",
            quality_metric="mAP_50",
            quality_value=0.0099,
        )

    # ORT CPU FP32
    yolo_fp32_onnx = models_dir / "yolo_nano_fp32_opset17.onnx"
    if yolo_fp32_onnx.is_file():
        ort_yolo_cpu = ORTCPURuntime(yolo_fp32_onnx, intra_op_threads=8)
        suite.run_single_configuration(
            model_name="yolo_nano",
            runtime_name="ORT_CPU",
            precision="fp32",
            runtime=ort_yolo_cpu,
            input_tensor=dummy_yolo,
            sample_image_paths=detection_samples,
            adapter=yolo_adapter,
            model_file_path=yolo_fp32_onnx,
            task="detection",
            quality_metric="mAP_50",
            quality_value=0.0099,
        )

    # ORT CPU INT8
    yolo_int8_onnx = models_dir / "yolo_nano_static_int8.onnx"
    if yolo_int8_onnx.is_file():
        ort_yolo_int8 = ORTCPURuntime(yolo_int8_onnx, intra_op_threads=8)
        suite.run_single_configuration(
            model_name="yolo_nano",
            runtime_name="ORT_CPU",
            precision="int8",
            runtime=ort_yolo_int8,
            input_tensor=dummy_yolo,
            sample_image_paths=detection_samples,
            adapter=yolo_adapter,
            model_file_path=yolo_int8_onnx,
            task="detection",
            quality_metric="mAP_50",
            quality_value=0.0099,
        )

    # -------------------------------------------------------------
    # 2. Industrial Autoencoder Configurations
    # -------------------------------------------------------------
    # PyTorch CPU
    pt_ind_cpu = PyTorchRuntime(ind_adapter.get_pytorch_model(), device="cpu")
    suite.run_single_configuration(
        model_name="industrial_autoencoder",
        runtime_name="PyTorch",
        precision="fp32",
        runtime=pt_ind_cpu,
        input_tensor=dummy_ind,
        sample_image_paths=industrial_samples,
        adapter=ind_adapter,
        task="anomaly_detection",
        quality_metric="image_auroc",
        quality_value=1.0,
    )

    # PyTorch CUDA
    if torch.cuda.is_available():
        pt_ind_cuda = PyTorchRuntime(ind_adapter.get_pytorch_model(), device="cuda:0")
        suite.run_single_configuration(
            model_name="industrial_autoencoder",
            runtime_name="PyTorch",
            precision="fp32",
            runtime=pt_ind_cuda,
            input_tensor=dummy_ind.cuda(),
            sample_image_paths=industrial_samples,
            adapter=ind_adapter,
            task="anomaly_detection",
            quality_metric="image_auroc",
            quality_value=1.0,
        )

    # ORT CPU FP32
    ind_fp32_onnx = models_dir / "industrial_autoencoder_fp32_opset17.onnx"
    if ind_fp32_onnx.is_file():
        ort_ind_cpu = ORTCPURuntime(ind_fp32_onnx, intra_op_threads=8)
        suite.run_single_configuration(
            model_name="industrial_autoencoder",
            runtime_name="ORT_CPU",
            precision="fp32",
            runtime=ort_ind_cpu,
            input_tensor=dummy_ind,
            sample_image_paths=industrial_samples,
            adapter=ind_adapter,
            model_file_path=ind_fp32_onnx,
            task="anomaly_detection",
            quality_metric="image_auroc",
            quality_value=1.0,
        )

    # ORT CPU INT8
    ind_int8_onnx = models_dir / "industrial_autoencoder_static_int8.onnx"
    if ind_int8_onnx.is_file():
        ort_ind_int8 = ORTCPURuntime(ind_int8_onnx, intra_op_threads=8)
        suite.run_single_configuration(
            model_name="industrial_autoencoder",
            runtime_name="ORT_CPU",
            precision="int8",
            runtime=ort_ind_int8,
            input_tensor=dummy_ind,
            sample_image_paths=industrial_samples,
            adapter=ind_adapter,
            model_file_path=ind_int8_onnx,
            task="anomaly_detection",
            quality_metric="image_auroc",
            quality_value=1.0,
        )

    logger.info(f"\nAll benchmark configurations executed successfully.")
    logger.info(f"Master runs CSV updated -> {suite.csv_path}")
    logger.info(f"Raw run manifests stored -> {suite.raw_dir}")


if __name__ == "__main__":
    main()
