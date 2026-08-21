# Models Directory

This directory stores model checkpoints, exported ONNX graphs, and compiled TensorRT engines.

```
models/
├── weights/           # Base PyTorch checkpoints (.pt / .pth)
├── exported/          # Exported and optimized ONNX models (.onnx)
└── engines/           # Compiled TensorRT plan engines (.engine)
    └── manifests/     # Serialized engine compilation manifests (.json)
```
