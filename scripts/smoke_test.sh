#!/usr/bin/env bash
set -e

echo "============================================================"
echo "ONNX Edge Inference Benchmark — End-to-End Smoke Test Pipeline (v1.1)"
echo "============================================================"

# Auto-detect Python binary
if [ -x "/home/sengar/miniconda3/bin/python3" ]; then
    PY="/home/sengar/miniconda3/bin/python3"
    PYTEST="/home/sengar/miniconda3/bin/pytest"
elif command -v python3 &> /dev/null; then
    PY="python3"
    PYTEST="pytest"
else
    PY="python"
    PYTEST="pytest"
fi

echo "[1/5] Running Environment Verification..."
$PY -c "import torch, onnx, onnxruntime, scipy, sklearn; print('Environment dependencies OK')"

echo "[2/5] Executing Master Report Generator & V1.1 Experiments..."
# Ensure sample data and model weights exist for fresh clones
if [ ! -f "data/sample_images/manifest.json" ]; then
    echo "  [PREPARE] Generating sample evaluation data..."
    $PY scripts/prepare_sample_data.py
fi
if [ ! -f "data/calibration/manifest.csv" ]; then
    echo "  [PREPARE] Generating calibration data..."
    $PY scripts/prepare_calibration_data.py
fi
if [ ! -f "models/weights/yolo_nano_baseline.pt" ]; then
    echo "  [PREPARE] Generating model baselines..."
    $PY scripts/run_pytorch_baselines.py
fi
if [ ! -f "models/exported/yolo_nano_fp32_opset17.onnx" ]; then
    echo "  [PREPARE] Exporting ONNX models..."
    $PY scripts/export_and_validate.py
fi
if [ ! -f "models/exported/yolo_nano_static_int8.onnx" ]; then
    echo "  [PREPARE] Quantizing models to FP16 and INT8..."
    $PY scripts/quantize_and_validate.py
fi
$PY scripts/generate_report.py
$PY src/experiments/run_q_aware_ablation.py
# Execute scalability sweep only if artifacts are missing
if [ ! -f "results/scalability_sweep.csv" ] || [ ! -f "results/figures/scalability_batch_resolution.png" ] || [ "${FORCE_SWEEP:-0}" = "1" ]; then
    echo "  [BENCHMARK] Executing scalability sweeps across batch sizes & resolutions..."
    $PY src/experiments/run_scalability_sweep.py
else
    echo "  [CACHE] Scalability sweep artifacts present, skipping live re-benchmarking."
fi

echo "[3/5] Verifying Generated Artifacts & Data Integrity..."
test -f results/runs.csv
test -f results/tables/table1_numerical_correctness.md
test -f results/tables/table2_latency_throughput.md
test -f results/tables/table3_memory_footprint.md
test -f results/tables/table4_quality_retention.md
test -f results/tables/table5_int8_quantization_audit.md
test -f results/tables/table6_q_aware_nms_ablation.md
test -f results/tables/table7_decision_flip_audit.md
test -f results/tables/deployment_decision_matrix.md

test -f results/figures/pareto_quality_vs_latency.png
test -f results/figures/latency_breakdown_stacked.png
test -f results/figures/speedup_comparison.png
test -f results/figures/tail_latency_p50_p95.png
test -f results/figures/memory_vs_footprint.png
test -f results/figures/stability_variance_trends.png
test -f results/figures/q_aware_pareto_recovery.png
test -f results/figures/decision_flip_attribution.png
test -f results/figures/scalability_batch_resolution.png
test -f results/scalability_sweep.csv

# Assert scalability_sweep.csv has 16 data rows (1 header + 16 rows = 17 lines)
SWEEP_LINES=$(wc -l < results/scalability_sweep.csv 2>/dev/null || echo "0")
if [ "$SWEEP_LINES" -lt 17 ]; then
    echo "ERROR: results/scalability_sweep.csv has fewer than 16 sweep rows (found $((SWEEP_LINES - 1)))!"
    exit 1
fi
echo "  [CHECK] results/scalability_sweep.csv grid integrity (16 configurations) -> PASS"

# 1. Assert runs.csv has at least 8 data rows (header + 8 rows = 9 lines minimum)
CSV_TOTAL_LINES=$(wc -l < results/runs.csv)
if [ "$CSV_TOTAL_LINES" -lt 9 ]; then
    echo "ERROR: results/runs.csv has fewer than 8 data rows (found $((CSV_TOTAL_LINES - 1)) rows)!"
    exit 1
fi
echo "  [CHECK] results/runs.csv row count: $((CSV_TOTAL_LINES - 1)) rows (>= 8) -> PASS"

# 2. Assert no table contains | None | or | NONE |
for tbl in results/tables/*.md; do
    if grep -qE "\|\s*(None|NONE)\s*\|" "$tbl"; then
        echo "ERROR: Table $tbl contains unpopulated None/NONE values!"
        exit 1
    fi
done
echo "  [CHECK] Markdown tables integrity (no unpopulated None/NONE values) -> PASS"

# 3. Assert all figures are >= 20KB (20480 bytes)
for fig in results/figures/*.png; do
    fsize=$(stat -c%s "$fig" 2>/dev/null || stat -f%z "$fig")
    if [ "$fsize" -lt 20480 ]; then
        echo "ERROR: Figure $fig size is too small ($fsize bytes < 20KB)!"
        exit 1
    fi
done
echo "  [CHECK] Publication figures size verification (all >= 20KB) -> PASS"

# 4. Verify Overleaf Archive existence and zip integrity
if [ ! -f "overleaf_paper.zip" ]; then
    echo "ERROR: overleaf_paper.zip is missing!"
    exit 1
fi
zip_bytes=$(stat -c%s "overleaf_paper.zip" 2>/dev/null || stat -f%z "overleaf_paper.zip")
if [ "$zip_bytes" -lt 2621440 ]; then
    echo "ERROR: overleaf_paper.zip is smaller than 2.5MB ($zip_bytes bytes)!"
    exit 1
fi
unzip -tq overleaf_paper.zip || { echo "ERROR: overleaf_paper.zip failed integrity check!"; exit 1; }
echo "  [CHECK] overleaf_paper.zip archive integrity & size (>= 2.5MB) -> PASS" 

echo "[4/5] Executing Test Suite..."
$PYTEST tests/ -v --cov=src --cov-report=term-missing

echo "============================================================"
echo ">>> SMOKE TEST PASSED WITH 100% SUCCESS <<<"
echo "============================================================"
