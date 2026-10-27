| Model | Runtime | Provider | Precision | Max abs. error | Max rel. error | Min cosine sim. | Parity gate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| yolo_nano | PyTorch | PyTorch_CPU | FP32 | 0.000e+00 | 0.000e+00 | 1.000000 | PASS |
| yolo_nano | PyTorch | PyTorch_CUDA:0 | FP32 | 2.747e-04 | 4.197e-07 | 1.000000 | PASS |
| yolo_nano | PyTorch | PyTorch_CUDA:0 | FP16 | 1.385e+00 | 2.117e-03 | 1.000000 | PASS |
| yolo_nano | ORT_CPU | CPUExecutionProvider | FP32 | 3.586e-04 | 5.479e-07 | 1.000000 | PASS |
| yolo_nano | ORT_CPU | CPUExecutionProvider | FP16 | 3.877e-01 | 5.925e-04 | 1.000000 | PASS |
| yolo_nano | ORT_CPU | CPUExecutionProvider | INT8 | 1.323e+02 | 2.022e-01 | 0.999255 | FAIL |
| industrial_autoencoder | PyTorch | PyTorch_CPU | FP32 | 0.000e+00 | 0.000e+00 | 1.000000 | PASS |
| industrial_autoencoder | PyTorch | PyTorch_CUDA:0 | FP32 | 2.980e-07 | 1.562e-06 | 1.000000 | PASS |
| industrial_autoencoder | PyTorch | PyTorch_CUDA:0 | FP16 | 6.398e-04 | 3.727e-03 | 0.999986 | PASS |
| industrial_autoencoder | ORT_CPU | CPUExecutionProvider | FP32 | 2.384e-07 | 1.206e-06 | 1.000000 | PASS |
| industrial_autoencoder | ORT_CPU | CPUExecutionProvider | FP16 | 3.764e-04 | 1.862e-03 | 0.999995 | PASS |
| industrial_autoencoder | ORT_CPU | CPUExecutionProvider | INT8 | 1.658e-02 | 8.325e-02 | 0.992803 | PASS |
