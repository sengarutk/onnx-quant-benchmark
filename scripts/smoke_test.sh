#!/usr/bin/env bash
# Fast end-to-end smoke test. Never modifies tracked files: the quick benchmark writes to a temp directory.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PYTHON:-python3}"

echo "[1/3] Dependency check"
"${PY}" -c "import torch, onnx, onnxruntime, onnxsim, onnxconverter_common, scipy, sklearn, pandas, matplotlib, cv2; print('imports OK')"

echo "[2/3] Unit and integration tests"
"${PY}" -m pytest tests/ -q

echo "[3/3] Quick benchmark + report into a temporary results directory"
if [ -f models/weights/yolo_nano_trained.pt ] && [ -f models/exported/yolo_nano_static_int8.onnx ] && [ -f data/sample_images/manifest.json ]; then
    TMP="$(mktemp -d)"
    trap 'rm -rf "${TMP}"' EXIT
    "${PY}" scripts/benchmark_all.py --quick --results-dir "${TMP}"
    "${PY}" scripts/generate_report.py --results-dir "${TMP}" --include-quick
    test -s "${TMP}/runs.csv"
    for f in pareto_quality_vs_latency latency_breakdown_stacked speedup_comparison tail_latency_p50_p95 memory_vs_footprint stability_variance_trends; do
        test -s "${TMP}/figures/${f}.png"
    done
    echo "quick benchmark OK ($(($(wc -l < "${TMP}/runs.csv") - 1)) configurations)"
else
    echo "trained/exported artifacts not found; run 'make pipeline' for the full benchmark (skipping quick benchmark)"
fi
echo "SMOKE TEST PASSED"
