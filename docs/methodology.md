# Benchmarking & Profiling Methodology

This document outlines the rigorous experimental methodology, dual-regime timing, memory profiling, and stability validation protocols employed in the `onnx-edge-inference-benchmark` repository.

---

## 1. Dual-Regime Latency Measurement

In real-world edge AI deployments, pure neural network compute latency and full wall-clock end-to-end inspection latency diverge significantly due to I/O, image decoding, preprocessing, and memory transfers.

### 1.1. Model-Path Latency (Isolated Compute)
- **Objective**: Measure isolated kernel execution time.
- **Timing Engine**:
  - **CUDA / TensorRT**: Stream-synchronized `torch.cuda.Event(enable_timing=True)`.
  - **CPU**: Monotonic high-resolution hardware clock `time.perf_counter_ns()`.
- **Warmup**: $\\ge 50$ unmeasured iterations to allow cache warming, JIT warmups, and thread pool stabilization.
- **Timed Iterations**: $\\ge 300$ consecutive measured invocations.
- **GC Isolation**: Python Garbage Collection is disabled (`gc.disable()`) throughout the timed execution block to prevent non-deterministic GC pauses.

### 1.2. End-to-End (E2E) Pipeline Latency (Wall-Clock)
- **Objective**: Measure realistic real-world application throughput.
- **Stages Profiled**:
  1. **Image Ingestion & Decode**: Disk read and OpenCV decoding.
  2. **Preprocessing**: Letterboxing / bicubic interpolation, color space conversion, float32 normalization.
  3. **Host-to-Device Copy & Inference**: Memory binding or transfer into runtime.
  4. **Postprocessing & Metrics**:
     - Object Detection: Non-Maximum Suppression (NMS) and bounding box coordinate unscaling.
     - Industrial Inspection: Anomaly map aggregation and top 1% score extraction.

---

## 2. Multi-Session Stability & Variance Criteria

To prevent reporting anomalous single-run spikes, every configuration is executed across **3 independent sessions (Session A, Session B, Session C)** with inter-session cooldown pauses.

- **Statistical Metrics**:
  - Mean ($\\mu$), Standard Deviation ($\\sigma$)
  - Percentiles: $p_{50}$ (median), $p_{90}, p_{95}, p_{99}$
  - Coefficient of Variation: $CV = \\frac{\\sigma}{\\mu}$
- **Stability Gate**:
  - Configurations exhibiting $CV(p_{50}) > 0.05$ are marked with `UNSTABLE_LATENCY`.

---

## 3. Memory & Resource Footprint Profiling

| Resource Metric | Measurement Mechanism | Scope |
| :--- | :--- | :--- |
| **Peak Allocated VRAM** | `torch.cuda.max_memory_allocated()` | Active GPU tensor allocations |
| **Peak Reserved VRAM** | `torch.cuda.max_memory_reserved()` | Caching allocator memory pool |
| **Process Resident Set Size (RSS)** | `psutil.Process().memory_info().rss` | Host RAM consumption |
| **Device VRAM Usage** | `pynvml.nvmlDeviceGetMemoryInfo()` | Hardware-level VRAM utilization |
| **Model Disk Footprint** | `os.stat().st_size` | Serialized model file on disk |

---

## 4. Run Manifest Logging Standard

Every benchmark execution outputs an immutable JSON record adhering to schema:
- Stored at `results/raw/<run_id>/run.json`
- Tabular row appended to `results/runs.csv`
