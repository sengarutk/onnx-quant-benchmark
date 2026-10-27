| Model | Runtime | Provider | Precision | Model p50 (ms) [95% CI] | Model p95 (ms) | Model p99 (ms) | E2E p50 (ms) | Model FPS (1/p50) | Session CV | Stable (CV<=0.05) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo_nano | PyTorch | PyTorch_CPU | FP32 | 13.983 [13.948, 14.050] | 18.088 | 19.797 | 20.60 | 71.5 | 0.011 | yes |
| yolo_nano | PyTorch | PyTorch_CUDA:0 | FP32 | 1.607 [1.597, 1.615] | 3.594 | 4.258 | 10.21 | 622.4 | 0.051 | no |
| yolo_nano | PyTorch | PyTorch_CUDA:0 | FP16 | 1.888 [1.882, 1.892] | 4.568 | 5.124 | 10.15 | 529.6 | 0.013 | yes |
| yolo_nano | ORT_CPU | CPUExecutionProvider | FP32 | 8.976 [8.943, 9.009] | 11.855 | 12.715 | 17.67 | 111.4 | 0.016 | yes |
| yolo_nano | ORT_CPU | CPUExecutionProvider | FP16 | 9.077 [9.047, 9.113] | 12.334 | 13.388 | 18.05 | 110.2 | 0.015 | yes |
| yolo_nano | ORT_CPU | CPUExecutionProvider | INT8 | 5.876 [5.847, 5.911] | 6.999 | 7.624 | 14.62 | 170.2 | 0.008 | yes |
| industrial_autoencoder | PyTorch | PyTorch_CPU | FP32 | 5.675 [5.650, 5.712] | 7.292 | 8.192 | 8.09 | 176.2 | 0.023 | yes |
| industrial_autoencoder | PyTorch | PyTorch_CUDA:0 | FP32 | 3.902 [3.900, 3.905] | 4.140 | 4.491 | 6.50 | 256.2 | 0.002 | yes |
| industrial_autoencoder | PyTorch | PyTorch_CUDA:0 | FP16 | 1.012 [1.008, 1.018] | 1.452 | 3.243 | 3.39 | 988.4 | 0.020 | yes |
| industrial_autoencoder | ORT_CPU | CPUExecutionProvider | FP32 | 6.133 [6.115, 6.159] | 7.832 | 9.117 | 7.90 | 163.0 | 0.008 | yes |
| industrial_autoencoder | ORT_CPU | CPUExecutionProvider | FP16 | 8.921 [8.897, 8.947] | 10.456 | 11.994 | 11.17 | 112.1 | 0.006 | yes |
| industrial_autoencoder | ORT_CPU | CPUExecutionProvider | INT8 | 5.875 [5.831, 5.917] | 7.366 | 9.188 | 7.66 | 170.2 | 0.029 | yes |
