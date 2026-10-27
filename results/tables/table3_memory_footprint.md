| Model | Runtime | Provider | Precision | Peak PyTorch VRAM alloc. (MB) | Process RSS (MB) | Model size (MB) | Size source | Size vs FP32 ONNX |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo_nano | PyTorch | PyTorch_CPU | FP32 | n/a | 881 | 7.43 | parameters and buffers | n/a |
| yolo_nano | PyTorch | PyTorch_CUDA:0 | FP32 | 35.33 | 1153 | 7.43 | parameters and buffers | n/a |
| yolo_nano | PyTorch | PyTorch_CUDA:0 | FP16 | 17.27 | 1454 | 3.72 | parameters and buffers | n/a |
| yolo_nano | ORT_CPU | CPUExecutionProvider | FP32 | n/a | 1499 | 7.50 | serialized file | -0.0% |
| yolo_nano | ORT_CPU | CPUExecutionProvider | FP16 | n/a | 1507 | 3.76 | serialized file | -49.9% |
| yolo_nano | ORT_CPU | CPUExecutionProvider | INT8 | n/a | 1525 | 1.98 | serialized file | -73.6% |
| industrial_autoencoder | PyTorch | PyTorch_CPU | FP32 | n/a | 1485 | 5.41 | parameters and buffers | n/a |
| industrial_autoencoder | PyTorch | PyTorch_CUDA:0 | FP32 | 44.92 | 1532 | 5.41 | parameters and buffers | n/a |
| industrial_autoencoder | PyTorch | PyTorch_CUDA:0 | FP16 | 7.73 | 1596 | 2.70 | parameters and buffers | n/a |
| industrial_autoencoder | ORT_CPU | CPUExecutionProvider | FP32 | n/a | 1574 | 5.40 | serialized file | -0.0% |
| industrial_autoencoder | ORT_CPU | CPUExecutionProvider | FP16 | n/a | 1587 | 2.70 | serialized file | -49.9% |
| industrial_autoencoder | ORT_CPU | CPUExecutionProvider | INT8 | n/a | 1576 | 1.39 | serialized file | -74.3% |
