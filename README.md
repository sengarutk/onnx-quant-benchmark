# ONNX Quantization Benchmark — CPU vs CUDA vs TensorRT (DistilBERT)

This project is a **systems-oriented benchmarking study** of **ONNX inference performance** under different **precisions** and **execution providers**.  
The goal is to measure real deployment tradeoffs between:

- **FP32 vs FP16 vs INT8**
- **CPU vs CUDA vs TensorRT**
- latency distributions (**mean / p50 / p90 / p99**)
- how quantization behaves differently on CPU vs GPU

---

## Motivation

Modern ML deployment is not just about training accuracy — inference performance (latency + throughput + stability) is critical.

ONNX Runtime provides multiple execution backends (providers):

- **CPUExecutionProvider**
- **CUDAExecutionProvider**
- **TensorrtExecutionProvider**

Quantization (especially INT8) can provide strong speedups on CPU, but may behave differently on GPU depending on kernel support and graph conversions.

This project answers:

**Which precision + provider gives the best inference latency in practice?**

---

## What This Project Implements

End-to-end pipeline:

1. Load a Transformer model from Hugging Face (DistilBERT)
2. Export ONNX in:
   - FP32
   - FP16
3. Perform INT8 quantization:
   - dynamic INT8 (ONNX quantization)
4. Benchmark inference latency with ONNX Runtime:
   - CPU
   - CUDA
   - TensorRT (if available)
5. Plot results:
   - latency comparison bars
   - speedup vs FP32 baseline

---

## Model

- **DistilBERT base uncased**
- Task head: `DistilBertForSequenceClassification`

> Note: the classifier head is randomly initialized in this benchmark, but this does not affect *systems benchmarking* of runtime and memory behavior.

---

## Benchmark Protocol

### Input
- fixed batch size (default: 1)
- fixed sequence length (configurable)

### Measurement
For each `(format, provider)` configuration:

- Warmup runs (exclude compilation / cache setup)
- Timed runs for stable measurement
- Report:
  - **Mean latency**
  - **p50**
  - **p90**
  - **p99**

---

## Key Findings (from observed results)

- **CUDA FP32/FP16** gives the best raw latency.
- **Dynamic INT8** is **very fast on CPU**, often significantly faster than FP32 CPU.
- INT8 on CUDA may not always be faster due to:
  - kernel support limitations
  - graph memcopy nodes
  - provider fallback
  - quantization path overhead

---

## Repository Structure
````
onnx-quant-benchmark/
├── models/
│ ├── onnx_fp32/
│ │ └── model.onnx
│ ├── onnx_fp16/
│ │ └── model.onnx
│ └── onnx_dynamic_int8/
│ └── model.onnx
├── results/
│ ├── runs.csv
│ ├── runs_with_speedups.csv
│ └── plots/
│ ├── latency_cpu.png
│ ├── latency_cuda.png
│ ├── latency_tensorrt.png
│ ├── speedup_cpu.png
│ ├── speedup_cuda.png
│ └── speedup_tensorrt.png
├── src/
│ ├── export_onnx.py
│ ├── quantize_onnx.py
│ ├── benchmark_onnx.py
│ ├── plot_results.py
│ └── plot_speedups.py
├── README.md
└── requirements.txt
````
---

## Setup

### 1) Create environment
````
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
````
### 2) Install PyTorch (CUDA build)
````
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
````
### 3) Install dependencies
````
pip install transformers onnx onnxruntime onnxruntime-gpu numpy pandas matplotlib
````
## How to Run
### 1) Export ONNX (FP32 + FP16)
````
python3 src/export_onnx.py
````
Expected output:
- models/onnx_fp32/model.onnx
- models/onnx_fp16/model.onnx

### 2) Quantize ONNX (Dynamic INT8)
````
python3 src/quantize_onnx.py
````
Expected output:

models/onnx_dynamic_int8/model.onnx

### 3) Benchmark providers
````
python3 src/benchmark_onnx.py
````
This generates:results/runs.csv

### 4) Generate latency plots
````
python3 src/plot_results.py
````
Outputs:

- results/plots/latency_cpu.png

- results/plots/latency_cuda.png

- results/plots/latency_tensorrt.png

### 5) Generate speedup plots (vs FP32 baseline)
````
python3 src/plot_speedups.py
````
Outputs:
- results/runs_with_speedups.csv
- results/plots/speedup_cpu.png
- results/plots/speedup_cuda.png
- results/plots/speedup_tensorrt.png (only if TRT baseline is valid)

## TensorRT Note (Important)
If you see errors like:

- libnvinfer.so not found
- provider fallback to CPU
then TensorRT libraries are missing.

In that case:
- CUDA provider results remain valid
- TensorRT bars may reflect fallback behavior (CPU execution)
---
## Reproducibility
This project is designed to be reproducible:
- deterministic benchmarking setup
- results saved as CSV
- plotting scripts generate figures directly from CSV
---
## Author

Utkarsh Sengar
B.Tech CSE, IIT Dharwad
Email: cs23bt026@iitdh.ac.in