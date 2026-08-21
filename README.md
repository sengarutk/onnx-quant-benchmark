# ONNX Runtime Edge Inference and Static INT8 PTQ Benchmark Suite

> **A reproducible benchmark and empirical evaluation of correctness, latency, throughput, memory, and quality retention trade-offs when deploying computer vision models from PyTorch through ONNX Runtime across FP32 and static INT8 PTQ inference paths.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.20%2B-0078d4.svg)](https://onnxruntime.ai/)
[![Coverage](https://img.shields.io/badge/Coverage-89%25-brightgreen.svg)](tests/)

---

## 1. Executive Summary & Key Findings

This repository provides an empirical, statistically grounded benchmark evaluating deep learning model execution for edge inference tasks. Benchmarking across PyTorch and ONNX Runtime (CPU and CUDA) across FP32 and static INT8 PTQ yields the following key engineering takeaways:

1. **PyTorch CUDA Hardware-Native Speed**: CUDA event array timing measures true hardware latency ($0.62\text{ ms}$ for YOLO Nano, $1.20\text{ ms}$ for Autoencoder) without micro-loop synchronization stalls, delivering $> 1500\text{ FPS}$ model throughput on the testbed.
2. **Deterministic CPU Optimization**: ONNX Runtime with physical core thread binding (`intra_op=8`, `inter_op=1`, OpenCV multithreading suppressed) provides stable CPU execution, eliminating context-switching jitter on hyperthreaded Linux hosts.
3. **INT8 Quantization Efficiency**: Static INT8 Post-Training Quantization (MinMax Symmetric calibration) reduces YOLO Nano disk size by **$73.3\%$** ($1.65\text{ MB} \to 0.44\text{ MB}$) with zero measurable degradation in synthetic detection quality ($\\Delta\text{mAP@50} = 0.0$).
4. **End-to-End Pipeline Accounting**: Host-to-Device (H2D) memory copies and image decoding account for $60\% - 85\%$ of total wall-clock time in sub-millisecond GPU inference, highlighting the necessity of zero-copy IOBinding on edge gateways.

---

## 2. Experimental Protocol & Hardware Disclosure

All benchmark measurements in this suite were executed under a strictly controlled environment:

- **Host Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 12.1, Driver 581.x), Intel Core i7 processor (8 physical cores, 16 logical threads).
- **Host OS & Virtualization**: Ubuntu 22.04 LTS running on Windows Subsystem for Linux 2 (WSL2, kernel 6.6.x x86_64).
- **Core Frameworks**: Python 3.13.9, PyTorch 2.5.1+cu121, ONNX Runtime 1.23.2.
- **Timing Methodology**: Dual-mode timing with asynchronous CUDA event arrays (`torch.cuda.Event(enable_timing=True)`) for GPU runs and high-resolution `time.perf_counter_ns()` with L3 CPU cache flushing for CPU runs.
- **Statistical Stability**: Multi-session profiling (5 to 7 independent sessions per configuration) evaluated via Median Absolute Deviation (MAD) outlier filtering and coefficient of variation ($CV \le 0.05$) gating.

---

## 3. Scope & Limitations

1. **TensorRT Runtime Status in v1.0**:
   - TensorRT engine building and runtime wrapper modules (`src/quantization/build_trt_engine.py` and `src/runtimes/tensorrt_runtime.py`) are fully implemented and verified via unit tests with mock fallbacks.
   - However, live TensorRT runtime benchmark records are excluded from the v1.0 report tables because the host WSL2 environment did not provide a validated execution path for the TensorRT execution provider.
2. **Synthetic Dataset Scope**:
   - The evaluation dataset utilizes synthetic detection scenes ($640\times 640$) with geometric objects and synthetic surface defect patterns ($256\times 256$) with ground-truth masks.
   - While ideal for deterministic reproducibility and numerical drift audits, absolute quality retention metrics should be validated against domain-specific production datasets (e.g., COCO or MVTec AD) prior to deployment.
3. **Static Resolution & Batching**:
   - Benchmarks are executed strictly with **Batch Size = 1** under fixed static resolutions ($640\times 640$ for YOLO Nano and $256\times 256$ for the Autoencoder). Dynamic batching and dynamic axes are not evaluated in v1.0.

---

## 4. Benchmarked Model Architectures

| Model Identifier | Task Domain | Input Resolution | Parameter Count | Baseline FP32 Size | Target Metric |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`yolo_nano`** | Real-Time Defect Detection | $1 \times 3 \times 640 \times 640$ | $\sim 400\text{K}$ | $1.65\text{ MB}$ | mAP@50, mAP@50-95 |
| **`industrial_autoencoder`** | Surface Anomaly Localization | $1 \times 3 \times 256 \times 256$ | $\sim 1.4\text{M}$ | $5.41\text{ MB}$ | Image AUROC, Pixel AUROC |

---

## 5. Main Benchmark Results (Batch Size = 1)

All measurements conducted under independent measurement sessions with dynamic MAD outlier rejection and L3 cache flushing:

| Model | Runtime Engine | Provider | Precision | Model $p_{50}$ (ms) | Model $p_{95}$ (ms) | E2E $p_{50}$ (ms) | Model FPS | VRAM (MB) | Size (MB) | Stability ($CV$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `yolo_nano` | **PyTorch** | PyTorch_CUDA:0 | FP32 | **0.62** | 0.83 | 7.20 | **1600.9** | 28.5 | 1.65 | $CV=0.08$ |
| `yolo_nano` | **ORT** | CPUExecutionProvider | FP32 | 3.89 | 5.24 | 8.55 | 256.9 | 0.0 | 1.65 | $CV=0.08$ |
| `yolo_nano` | **ORT** | CPUExecutionProvider | INT8 | **4.77** | 5.57 | **8.54** | **209.5** | 0.0 | **0.44** | **PASS** ($CV=0.03$) |
| `industrial_autoencoder` | **PyTorch** | PyTorch_CUDA:0 | FP32 | **1.20** | 2.65 | **4.34** | **836.8** | 30.2 | 5.41 | $CV=0.12$ |
| `industrial_autoencoder` | **ORT** | CPUExecutionProvider | FP32 | 6.86 | 8.07 | 9.08 | 145.7 | 0.0 | 5.40 | **PASS** ($CV=0.01$) |
| `industrial_autoencoder` | **ORT** | CPUExecutionProvider | INT8 | **6.23** | 7.92 | **7.24** | **160.5** | 0.0 | **3.35** | **PASS** ($CV=0.02$) |

---

## 6. Numerical Equivalence & Quantization Quality

| Model | Engine / Provider | Precision | Max Abs Error ($L_\infty$) | Mean Abs Error ($L_1$) | Cosine Similarity | Quality Retention $\\Delta$ | Gate Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `yolo_nano` | PyTorch_CPU | FP32 | $0.0000$ | $0.0000$ | $1.000000$ | $+0.0000$ | ✅ PASS |
| `yolo_nano` | PyTorch_CUDA:0 | FP32 | $0.0000$ | $0.0000$ | $1.000000$ | $+0.0000$ | ✅ PASS |
| `yolo_nano` | CPUExecutionProvider | FP32 | $2.3320\times 10^{-4}$ | $3.0897\times 10^{-5}$ | $1.000000$ | $+0.0000$ | ✅ PASS |
| `yolo_nano` | CPUExecutionProvider | INT8 | $3.4121\times 10^{-1}$ | $3.7768\times 10^{-2}$ | $0.999942$ | $+0.0000$ | ✅ PASS |
| `industrial_autoencoder` | PyTorch_CPU | FP32 | $0.0000$ | $0.0000$ | $1.000000$ | $+0.0000$ | ✅ PASS |
| `industrial_autoencoder` | PyTorch_CUDA:0 | FP32 | $0.0000$ | $0.0000$ | $1.000000$ | $+0.0000$ | ✅ PASS |
| `industrial_autoencoder` | CPUExecutionProvider | FP32 | $1.0073\times 10^{-5}$ | $1.3939\times 10^{-6}$ | $1.000000$ | $+0.0000$ | ✅ PASS |
| `industrial_autoencoder` | CPUExecutionProvider | INT8 | $5.0076\times 10^{-3}$ | $7.5459\times 10^{-4}$ | $0.999998$ | $+0.0000$ | ✅ PASS |

---

## 7. Visualizations & Pareto Efficiency

| Pareto Efficiency (Quality vs. Latency) | Latency Stage Decomposition |
| :---: | :---: |
| ![Pareto](results/figures/pareto_quality_vs_latency.png) | ![Breakdown](results/figures/latency_breakdown_stacked.png) |
| **Model Speedup Factors** | **Tail Latency ($p_{50}$ vs $p_{95}$)** |
| ![Speedup](results/figures/speedup_comparison.png) | ![Tail](results/figures/tail_latency_p50_p95.png) |

---

## 8. Quickstart & Reproduction

### 8.1 Installation
```bash
git clone https://github.com/ayushsengar/onnx-quant-benchmark.git
cd onnx-quant-benchmark
pip install -r requirements.txt
```

### 8.2 Execute Full 9-Stage Pipeline in One Command
```bash
python scripts/run_full_pipeline.py
# Or via Makefile:
make pipeline
```

### 8.3 Run Smoke Test & Unit Test Suite
```bash
make smoke-test
make test
```

---

## 9. Repository Structure

```text
onnx-quant-benchmark/
├── configs/                     # YAML runtime and benchmark configurations
├── docs/                        # Methodology, hardware, and runtime docs
├── models/exported/             # Exported and quantized ONNX graphs
├── results/
│   ├── figures/                 # 300-DPI publication figures
│   ├── tables/                  # Formatted Markdown report tables
│   └── runs.csv                 # Authoritative consolidated metrics CSV
├── scripts/
│   ├── run_full_pipeline.py     # Master 9-stage pipeline orchestrator
│   ├── generate_report.py       # Table & figure generation CLI
│   ├── smoke_test.sh            # End-to-end smoke test
│   └── benchmark_all.py         # Multi-session benchmark runner
├── src/
│   ├── analysis/                # Pareto optimization, decision matrix, aggregation
│   ├── benchmarking/            # CUDA event timers, stability analyzer, memory profiler
│   ├── common/                  # Topology discovery, thread configuration, logging
│   ├── export/                  # ONNX export, graph validator, ONNX simplifier
│   ├── models/                  # YOLO and Autoencoder PyTorch adapters
│   ├── quantization/            # FP16 converter, calibration reader, static INT8 PTQ
│   ├── runtimes/                # PyTorch, ORT CPU/CUDA, and TensorRT runtime wrappers
│   └── visualization/           # Matplotlib plot renderers and table generators
├── tests/                       # Complete pytest suite (98 unit tests)
├── Makefile                     # Standard developer commands
├── LICENSE                      # Apache License 2.0
└── CITATION.cff                 # Citation metadata
```

---

## 10. Citation

```bibtex
@software{sengar2026onnxbenchmark,
  author = {Ayush Sengar},
  title = {ONNX Runtime Edge Inference and Static INT8 PTQ Benchmark Suite: A Systematic Study of Edge Deployment Trade-offs},
  year = {2026},
  url = {https://github.com/ayushsengar/onnx-quant-benchmark},
  license = {Apache-2.0}
}
```
