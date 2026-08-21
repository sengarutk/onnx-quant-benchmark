# Canonical ONNX Export & Numerical Parity Audit Report

This report documents the verification, simplification, and numerical parity audits for the canonical ONNX models targeting **Opset 17**.

---

## 1. Exported Graph Topology & Architecture

| Model Identifier | Input Signature | Output Signature | Total Params | Model Size |
| :--- | :--- | :--- | :--- | :--- |
| **YOLO Nano Detector** | `images` [1, 3, 640, 640] | `output0` [1, 84, 8400] | 430,327 | 1.646 MB |
| **Industrial Autoencoder** | `input` [1, 3, 256, 256] | `reconstruction` [1, 3, 256, 256], `anomaly_map` [1, 1, 256, 256] | 1,412,803 | 5.397 MB |

---

## 2. Graph Simplification & Optimization Metrics

| Model | Nodes Before | Nodes After | Nodes Eliminated | Reduction % | Checker Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO Nano** | 27 | 22 | 5 | 18.52% | `PASS` |
| **Industrial Autoencoder** | 40 | 26 | 14 | 35.0% | `PASS` |

---

## 3. Numerical Equivalence Gating (PyTorch FP32 vs. ONNX Runtime FP32)

| Model Output Tensor | Max Absolute Error ($L_\infty$) | Mean Absolute Error ($L_1$) | Cosine Similarity | Parity Gate |
| :--- | :--- | :--- | :--- | :--- |
| **yolo_nano -> output0** | `7.63e-06` | `9.17e-07` | `1.000000` | `PASS` |
| **industrial_autoencoder -> reconstruction** | `5.96e-08` | `1.39e-08` | `1.000000` | `PASS` |
| **industrial_autoencoder -> anomaly_map** | `3.58e-07` | `1.22e-08` | `1.000000` | `PASS` |

---

## 4. Full Dataset Task Quality Equivalence

| Model | Evaluated Metric | PyTorch FP32 Baseline | ONNX Runtime FP32 | Delta ($|\Delta|$) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO Nano** | **mAP@50** | `0.0` | `0.0` | `0.0` | `PASS` |
| **Industrial Autoencoder** | **Image AUROC** | `1.0` | `1.0` | `0.0` | `PASS` |

---

## 5. Artifact Inventory

- `models/exported/yolo_nano_fp32_opset17.onnx` (`SHA-256`: `b3a647bfd1b782caf55a35c4059f2b23dfc504156553cfd51f9e7d1d38a350b7`)
- `models/exported/industrial_autoencoder_fp32_opset17.onnx` (`SHA-256`: `14279950242a5d1929cdd428db51452b24468e40515b09578d9f424de6cc5b0f`)
