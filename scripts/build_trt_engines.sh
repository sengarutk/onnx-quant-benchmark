#!/usr/bin/env bash
# Builds TensorRT engines from the exported ONNX graphs (requires the `tensorrt` Python package).
#   FP32 / FP16 engines are built from the FP32 ONNX graph; INT8 engines from the static INT8 QDQ graph.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."
PY="${PYTHON:-python3}"

"${PY}" - <<'PYEOF'
import sys
from pathlib import Path
sys.path.insert(0, ".")
from src.quantization.build_trt_engine import TRT_AVAILABLE, build_tensorrt_engine

if not TRT_AVAILABLE:
    print("TensorRT Python bindings not installed; skipping engine builds.")
    sys.exit(0)

exp, eng = Path("models/exported"), Path("models/engines")
plan = {
    "yolo_nano": ("images", (1, 3, 640, 640)),
    "industrial_autoencoder": ("input", (1, 3, 256, 256)),
}
for model, (inp, shape) in plan.items():
    for precision, src in (("fp32", f"{model}_fp32_opset17.onnx"), ("fp16", f"{model}_fp32_opset17.onnx"),
                           ("int8", f"{model}_static_int8.onnx")):
        onnx_path = exp / src
        if onnx_path.is_file():
            build_tensorrt_engine(onnx_path, eng / f"{model}_{precision}.engine", precision=precision,
                                  static_shapes={inp: shape})
print("TensorRT engine builds complete.")
PYEOF
