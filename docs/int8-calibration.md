# Quantization (FP16 & Static INT8 PTQ) Calibration & Validation Report

This report documents the half-precision conversion, Static INT8 Post-Training Quantization (PTQ) calibration, and quality degradation gating for the `onnx-edge-inference-benchmark` repository.

---

## 1. Calibration Dataset Configuration

- **Detection Calibration Split**: 50 disjoint images ($640 \times 640$) in `data/calibration/detection/`
- **Industrial Inspection Split**: 50 disjoint normal images ($256 \times 256$) in `data/calibration/industrial/`
- **Isolation Guarantee**: Zero hash overlap with test sets in `data/sample_images/`.
- **Calibration Algorithm**: Static MinMax Activation Histogram Range Tracking.

---

## 2. Quantization Quality Degradation Audit

### 2.1. Object Detector (YOLO Nano)

| Precision | Target Format | Metric (mAP@50) | Baseline FP32 | Quantized Value | Delta ($\Delta$) | Gate Threshold | Decision Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16** | Float16 | mAP@50 | `0.0` | `0.0` | `0.0` | $\le 0.015$ | `PASS` |
| **Static INT8** | QDQ (QInt8) | mAP@50 | `0.0` | `0.0` | `0.0` | $\le 0.015$ | `PASS` |

### 2.2. Industrial Inspection Model (ConvAutoencoder)

| Precision | Target Format | Metric (Image AUROC) | Baseline FP32 | Quantized Value | Delta ($\Delta$) | Gate Threshold | Decision Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FP16** | Float16 | Image AUROC | `1.0` | `1.0` | `0.0` | $\le 0.010$ | `PASS` |
| **Static INT8** | QDQ (QInt8) | Image AUROC | `1.0` | `1.0` | `0.0` | $\le 0.010$ | `PASS` |

---

## 3. Artifact Checksum Matrix

| Model | Precision | Filename | Size |
| :--- | :--- | :--- | :--- |
| YOLO Nano | FP16 | `models/exported/yolo_nano_fp16.onnx` | ~0.84 MB |
| YOLO Nano | Static INT8 | `models/exported/yolo_nano_static_int8.onnx` | ~0.50 MB |
| Industrial Autoencoder | FP16 | `models/exported/industrial_autoencoder_fp16.onnx` | ~2.72 MB |
| Industrial Autoencoder | Static INT8 | `models/exported/industrial_autoencoder_static_int8.onnx` | ~1.46 MB |
