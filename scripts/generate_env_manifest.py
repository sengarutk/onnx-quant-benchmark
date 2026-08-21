#!/usr/bin/env python3
"""
CLI tool to inspect system hardware, runtime environment, and serialize manifests.
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.environment import (
    collect_environment_manifest,
    save_environment_manifest,
    generate_hardware_doc,
)
from src.common.logging import setup_logger

logger = setup_logger("generate_env_manifest")


def main() -> None:
    logger.info("Inspecting hardware topology and runtime environment...")
    manifest = collect_environment_manifest()

    manifest_path = save_environment_manifest(PROJECT_ROOT / "results" / "manifests")
    logger.info(f"Environment manifest saved -> {manifest_path}")

    doc_path = PROJECT_ROOT / "docs" / "hardware.md"
    generate_hardware_doc(doc_path)
    logger.info(f"Hardware documentation updated -> {doc_path}")

    print("\n" + "=" * 65)
    print("        ENVIRONMENT & HARDWARE AUDIT SUMMARY")
    print("=" * 65)
    print(f"Timestamp:          {manifest.timestamp}")
    print(f"Git Commit:         {manifest.git_commit}")
    print(f"OS Info:            {manifest.os_info}")
    print(f"CPU Model:          {manifest.cpu_info}")
    print(f"System RAM:         {manifest.ram_total_gb} GB")
    print(f"GPU Available:      {manifest.gpu_available}")
    print(f"GPU Device:         {manifest.gpu_name or 'N/A'}")
    print(f"GPU VRAM:           {manifest.gpu_vram_total_gb or 'N/A'} GB")
    print(f"CUDA Driver:        {manifest.nvidia_driver_version or 'N/A'}")
    print(f"CUDA Runtime:       {manifest.cuda_runtime_version or 'N/A'}")
    print(f"cuDNN:              {manifest.cudnn_version or 'N/A'}")
    print(f"TensorRT:           {manifest.tensorrt_version}")
    print(f"ONNX Runtime:       {manifest.onnxruntime_version}")
    print(f"PyTorch:            {manifest.torch_version}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
