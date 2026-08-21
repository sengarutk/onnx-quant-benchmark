#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo " Setting up ONNX Edge Inference Benchmark Environment"
echo "=========================================================="

# Check Python version
PYTHON_CMD="python3"
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "ERROR: python3 could not be found."
    exit 1
fi

PY_VER=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python version: $PY_VER"

# Create required directories
mkdir -p models/weights models/exported models/engines/manifests \
         data/sample_images data/calibration \
         results/raw results/manifests results/tables results/figures results/profiles

echo "Directories verified."

# Check NVIDIA GPU support
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA Driver detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Running in CPU-only or container mode."
fi

# Run environment manifest generation
$PYTHON_CMD scripts/generate_env_manifest.py

echo "Environment setup complete."
