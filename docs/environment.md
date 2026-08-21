# Environment & Software Stack Architecture

This document specifies the software dependencies, CUDA runtime compatibility, and execution provider toolchain used for the `onnx-edge-inference-benchmark` project.

---

## 1. Supported Runtime Providers

1. **PyTorch (Native):** Baseline eager execution and TorchScript tracing on CPU and CUDA.
2. **ONNX Runtime (CPU):** Multi-threaded CPU execution with AVX2/AVX-512 VNNI vectorization.
3. **ONNX Runtime (CUDA):** GPU inference leveraging cuDNN and cuBLAS execution backends with IOBinding support.
4. **TensorRT:** NVIDIA's high-performance inference optimizer utilizing FP16 and INT8 Tensor Cores with kernel auto-tuning and layer fusion.

---

## 2. Dependency Matrix

| Component | Target Version | Purpose |
| :--- | :--- | :--- |
| **Python** | `>=3.10` | Core language runtime |
| **PyTorch** | `>=2.2.0` | Ingestion, PyTorch reference runs, and model export |
| **ONNX** | `>=1.16.0` | Graph representation and schema validation |
| **ONNX Runtime GPU** | `>=1.18.0` | Multi-backend inference runtime |
| **TensorRT** | `10.x / 8.6+` | Low-latency edge GPU inference engine |
| **Pydantic** | `>=2.5.0` | Type-safe configuration and manifest schemas |
| **PyYAML** | `>=6.0.1` | YAML configuration parsing |
| **pytest** | `>=8.0.0` | Automated test suite and coverage reporting |
