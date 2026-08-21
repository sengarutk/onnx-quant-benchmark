# Empirical Deployment Decision Matrix

This decision matrix synthesizes evidence-based operational recommendations derived from reproducible benchmarking runs across PyTorch, ONNX Runtime (CPU/CUDA), and TensorRT engines.

| Deployment Scenario | Target Constraint | Recommended Engine | Precision | Rationale & Trade-off Summary |
| :--- | :--- | :--- | :--- | :--- |
| **Scenario A: Low-Latency Target (Detector)** | Sub-10ms Measured E2E Target on Disclosed Testbed (Target: YOLO Nano) | PyTorch (PyTorch_CUDA:0) | FP32 | Meets sub-10ms measured E2E latency (8.01 ms $p_{50}$, 0.92 ms model) on the disclosed RTX 4050 laptop-GPU testbed (1082.7 FPS). |
| **Scenario B: Edge Gateway / IPC (CPU-Only)** | CPU Only (No Dedicated GPU) | ORT_CPU (CPUExecutionProvider) | FP32 | Thread-limited ONNX Runtime CPU execution bounds parallelism to physical-core capacity and reduces oversubscription variability on this testbed (4.50 ms $p_{50}$). |
| **Scenario C: High-Fidelity Anomaly Inspection (Autoencoder)** | High-Fidelity Anomaly-Map Preservation | PyTorch (PyTorch_CUDA:0) | FP32 | Controlled anomaly-map fidelity retained under the synthetic evaluation protocol (\Delta=+0.0000, 1.35 ms $p_{50}$) via FP32 by eliminating quantization clipping. (Note: This is not a claim of industrial defect-detection accuracy; real-data task-quality retention is deferred to a future MVTec-linked extension). |
| **Scenario D: High-Throughput Offline Batch** | Highest Measured Throughput under Benchmark Conditions | PyTorch (PyTorch_CUDA:0) | FP32 | Delivered the highest measured throughput among v1.0 tested configurations on the disclosed testbed (1082.7 FPS) with minimal memory footprint (28.5 MB VRAM). |

## Scenario Trade-Off Rationale

1. **Latency vs. Resource Allocation**: TensorRT and ORT CUDA deliver 10-30x compute speedup over CPU execution, but require dedicated NVIDIA VRAM. For embedded Linux gateways, ORT CPU with physical core thread binding offers reliable deterministic execution without host jitter.
2. **Quantization Precision Boundaries**: INT8 quantization reduces disk footprint by 50-75% and accelerates integer arithmetic units on supported x86/ARM hardware. However, for continuous anomaly reconstruction, FP16 is recommended over INT8 to prevent subtle pixel-level boundary artifacts.