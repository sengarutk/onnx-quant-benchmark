| Model | Runtime | Precision | Method | Size vs FP32 ONNX | Speedup vs same-runtime FP32 | Quality delta | Gates |
| --- | --- | --- | --- | --- | --- | --- | --- |
| yolo_nano | PyTorch (PyTorch_CUDA:0) | FP16 | PyTorch `model.half()` | n/a | 7.41x | +0.0035 | PASS |
| yolo_nano | ORT_CPU (CPUExecutionProvider) | FP16 | FP16 ONNX graph (onnxconverter-common) | -49.9% | 0.99x | -0.0003 | PASS |
| yolo_nano | ORT_CPU (CPUExecutionProvider) | INT8 | Static QDQ, MinMax, symmetric, per-channel weights | -73.6% | 1.53x | +0.0763 | FAIL |
| industrial_autoencoder | PyTorch (PyTorch_CUDA:0) | FP16 | PyTorch `model.half()` | n/a | 5.61x | -0.0001 | PASS |
| industrial_autoencoder | ORT_CPU (CPUExecutionProvider) | FP16 | FP16 ONNX graph (onnxconverter-common) | -49.9% | 0.69x | -0.0003 | PASS |
| industrial_autoencoder | ORT_CPU (CPUExecutionProvider) | INT8 | Static QDQ, MinMax, symmetric, per-channel weights | -74.3% | 1.04x | -0.0164 | PASS |
