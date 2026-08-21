#!/usr/bin/env bash
# Shell script automating TensorRT engine compilation
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "================================================================="
echo "  TENSORRT ENGINE COMPILATION PIPELINE"
echo "================================================================="

cd "${ROOT_DIR}"

python3 -c "
import sys
from pathlib import Path
from src.quantization.build_trt_engine import TRT_AVAILABLE, build_tensorrt_engine

if not TRT_AVAILABLE:
    print('TensorRT not installed. Skipping direct engine builds.')
    sys.exit(0)

models_dir = Path('models/exported')
engines_dir = Path('models/engines')

yolo_onnx = models_dir / 'yolo_nano_fp16.onnx'
if yolo_onnx.is_file():
    build_tensorrt_engine(yolo_onnx, engines_dir / 'yolo_nano_fp16.engine', precision='fp16')

ind_onnx = models_dir / 'industrial_autoencoder_fp16.onnx'
if ind_onnx.is_file():
    build_tensorrt_engine(ind_onnx, engines_dir / 'industrial_autoencoder_fp16.engine', precision='fp16')

print('All available TensorRT engines built successfully.')
"
