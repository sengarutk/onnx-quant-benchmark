| Model | Runtime | Provider | Precision | Metric | Value | FP32 reference | Delta | Quality gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo_nano | PyTorch | PyTorch_CPU | FP32 | mAP_50 | 0.7011 | 0.7011 | +0.0000 | PASS |
| yolo_nano | PyTorch | PyTorch_CUDA:0 | FP32 | mAP_50 | 0.7011 | 0.7011 | +0.0000 | PASS |
| yolo_nano | PyTorch | PyTorch_CUDA:0 | FP16 | mAP_50 | 0.7046 | 0.7011 | +0.0035 | PASS |
| yolo_nano | ORT_CPU | CPUExecutionProvider | FP32 | mAP_50 | 0.7011 | 0.7011 | +0.0000 | PASS |
| yolo_nano | ORT_CPU | CPUExecutionProvider | FP16 | mAP_50 | 0.7008 | 0.7011 | -0.0003 | PASS |
| yolo_nano | ORT_CPU | CPUExecutionProvider | INT8 | mAP_50 | 0.7774 | 0.7011 | +0.0763 | PASS |
| industrial_autoencoder | PyTorch | PyTorch_CPU | FP32 | image_auroc | 0.9309 | 0.9309 | +0.0000 | PASS |
| industrial_autoencoder | PyTorch | PyTorch_CUDA:0 | FP32 | image_auroc | 0.9309 | 0.9309 | +0.0000 | PASS |
| industrial_autoencoder | PyTorch | PyTorch_CUDA:0 | FP16 | image_auroc | 0.9308 | 0.9309 | -0.0001 | PASS |
| industrial_autoencoder | ORT_CPU | CPUExecutionProvider | FP32 | image_auroc | 0.9309 | 0.9309 | +0.0000 | PASS |
| industrial_autoencoder | ORT_CPU | CPUExecutionProvider | FP16 | image_auroc | 0.9306 | 0.9309 | -0.0003 | PASS |
| industrial_autoencoder | ORT_CPU | CPUExecutionProvider | INT8 | image_auroc | 0.9145 | 0.9309 | -0.0164 | PASS |
