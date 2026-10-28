# Methodology

This document describes exactly what the code measures. Every protocol parameter below is read from
`configs/benchmark.yaml` and every acceptance threshold from `configs/<model>/<precision>.yaml`.

## 1. Workloads and data

| | Detector (`yolo_nano`) | Anomaly model (`industrial_autoencoder`) |
| --- | --- | --- |
| Architecture | 5-stage strided Conv-BN-SiLU backbone, 3x3+1x1 heads at strides 8/16/32, FCOS-style (l,t,r,b) box decoding inside the graph | 4-stage strided-conv encoder, 1x1 bottleneck, 4-stage transposed-conv decoder |
| Output | `[1, 4 + 2, 8400]`: (cx, cy, w, h) in input pixels + 2 class logits | reconstruction `[1,3,256,256]`, anomaly map `[1,1,256,256]` = mean absolute error over RGB |
| Input | letterboxed RGB `[1,3,640,640]` in [0,1] | RGB `[1,3,256,256]` in [0,1] |
| Training | from scratch, 12k steps: every in-box location at the matching pyramid level is positive, Quality Focal Loss (IoU targets) + GIoU, AdamW, cosine LR (`scripts/train_models.py`); recipe chosen on the validation split (`results/training/yolo_nano_training.json`) | from scratch, MSE on defect-free textures only (unsupervised) |
| Task metric | mAP@50 (COCO 101-point, conf >= 0.001); P/R/F1 at conf 0.25, IoU 0.45 | image AUROC (Gaussian-smoothed map, mean of top-k% pixels), pixel AUROC, AU-PRO (FPR <= 0.3) |

Both models are deliberately small and are trained on **procedural synthetic data** (`src/data/synthetic.py`):
2-class shapes (rectangles vs ellipses) on noisy textured backgrounds of four resolutions for detection, and
sinusoidal surface textures with scratch / stain / texture-dropout defects (pixel masks) for anomaly detection.
Each sample is a pure function of `(task, split, index)`; the **train**, **validation**, **calibration** and
**test** splits use disjoint seed ranges, and the preparation scripts verify zero SHA-256 overlap between the
calibration and test files.

| Split | Used for | Size |
| --- | --- | --- |
| train | model training (generated on the fly) | 64k detection / 80k texture samples |
| validation | detector training-recipe choice; anomaly-score smoothing sigma and top-k ratio | 200 detection / 100 normal + 100 anomalous |
| calibration | static INT8 range calibration and Q-Aware NMS threshold search (labeled) | 100 detection / 100 normal textures |
| test | every reported quality number | 200 detection images / 100 normal + 100 anomalous |

## 2. Model artifacts

* **FP32 ONNX**: `torch.onnx.export` (opset 17), simplified with onnxsim; parity vs PyTorch checked on a real test
  image (max relative error <= 1e-5, cosine >= 0.999999) and on the full test split (|metric delta| <= 0.001).
* **FP16 ONNX**: `onnxconverter_common.float16`; the detector graph keeps FP32 inputs/outputs, the autoencoder graph
  has FP16 inputs/outputs (the converter could not keep FP32 I/O around its residual `Sub`); runtimes cast inputs.
  PyTorch FP16 uses `model.half()`.
* **INT8 ONNX**: ONNX Runtime `quantize_static`, QDQ format, per-channel symmetric INT8 weights, symmetric UINT8
  activations; Conv, ConvTranspose, MatMul and Gemm quantized. The calibration method (MinMax / Percentile / Entropy)
  is selected per model on the calibration split (detector: calibration mAP@50; autoencoder: anomaly-map relative
  error vs FP32); all three are reported in `docs/int8-calibration.md`.
* **PyTorch CUDA FP32** is strict IEEE FP32: TF32 is disabled for cuDNN convolutions and matmuls (PyTorch enables
  TF32 convolutions by default, which exceeded the FP32 parity gate).

## 3. Latency

**Model path** (`ModelPathTimer`): the preprocessed input tensor already lives on the target device.
Each session runs `warmup_iterations` (50) untimed and `timed_iterations` (300) timed forward passes with Python GC
disabled. On CUDA, every iteration is bracketed by CUDA events on the current stream (the PyTorch runtime does not
synchronize inside the call), so the interval is the device-side span including launch gaps. On CPU,
`time.perf_counter_ns` wall time.

**End to end** (`EndToEndTimer`): image decode (OpenCV) + letterbox/resize + normalization, inference through the
runtime's host-array API (including host-device copies on GPU), and post-processing (confidence filter + class-aware
NMS, or anomaly-score aggregation). Each stage is timed separately; 20 warmup + 200 timed iterations cycling through
the test images. The stage breakdown figure plots these measured per-stage medians.

**Stability** (`StabilityAnalyzer`):

1. `stability_sessions` (5) independent sessions; before each: CPU cache-eviction buffer write
   (max(64 MB, 1.5 x L3)), `gc.collect()`, CUDA synchronize, and a 0.5 s cooldown (not before the first).
2. If the coefficient of variation (CV) of the 5 session medians exceeds 0.04, two supplemental sessions are run once
   after a 1.5 s cooldown.
3. Reported percentiles are computed over the **pooled** iterations of all sessions; the p50 and p95 carry a
   percentile-bootstrap 95% confidence interval (B = 2000, resampling iterations - note this assumes iterations are
   exchangeable and therefore under-states between-session drift).
4. **Stable** means CV(session medians, all sessions) <= 0.05. A MAD-filtered CV is reported for diagnosis only.

Throughput is reported as batch / p50 (not a separately measured saturated throughput).

## 4. Correctness and quality gates (per configuration)

* **Numerical parity**: all output tensors of the runtime vs PyTorch FP32 on CPU for the same real test image:
  max |difference| / max |reference| and minimum cosine similarity, gated per precision tier
  (FP32: 1e-4 / 0.99999, FP16: 5e-3 / 0.9999, INT8: 0.1 / 0.99).
* **Task quality**: the full test split is evaluated through the runtime under test; the delta to the PyTorch FP32
  reference must not fall below -0.001 (FP32), -0.005 (FP16) or -0.02 (INT8).
* A run's `status` is `PASS` only if it is stable and passes both gates; otherwise the failing checks are listed.

## 5. Q-Aware NMS and decision-flip attribution

For each detector configuration, raw outputs on the calibration split are cached and the F1-maximizing
(tau_conf, tau_iou) is searched over {0.10, ..., 0.95} x {0.05, ..., 0.70} (step 0.05, 252 points; ties keep the
default 0.25 / 0.45). The default and calibrated thresholds are then compared on the test split:

* micro-F1 (TP/FP/FN pooled over images), greedy score-ordered class-aware matching at IoU 0.5;
* image-level paired bootstrap 95% CI of delta-F1 (B = 2000);
* two-sided Wilcoxon signed-rank test on per-image error counts (FP + FN), Holm-Bonferroni corrected across
  configurations.

Decision flips compare each configuration's detections with the PyTorch FP32 reference (greedy matching at IoU 0.5,
same class). Unmatched reference boxes are *lost TPs* or *suppressed FPs*, unmatched target boxes are *new TPs* or
*new FPs* (TP/FP judged against ground truth). Flip rate = flips / (matched + flips). Reported both at the default
thresholds and with each model's own calibrated thresholds.

## 6. Memory

`torch.cuda.max_memory_allocated` (PyTorch CUDA runs only; it does not see ONNX Runtime or TensorRT allocations),
process RSS after the run (`psutil`), and serialized model file size. All memory and size figures are in binary
megabytes: MB = 2^20 bytes (MiB).

## 7. Known limitations

* Synthetic data and small models: results characterize the runtimes and the methodology, not real-world accuracy.
* Single workstation testbed under WSL2: OS scheduling noise is visible as between-session drift (see the stability
  figure); this is not an embedded/Jetson measurement.
* ONNX Runtime CUDA and TensorRT are implemented and validated when available, but are reported only if the
  corresponding provider/bindings are installed on the benchmark host (see `docs/runtimes.md` and `session.json`).
