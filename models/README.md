# Models

Git-ignored; regenerate with `make train export quantize` (or `make pipeline`).

```
models/
├── weights/    # yolo_nano_trained.pt, industrial_autoencoder_trained.pt, industrial_autoencoder_scoring.json
├── exported/   # *_fp32_opset17.onnx, *_fp16.onnx, *_static_int8.onnx (+ .sha256, metadata, quantization_manifest.json)
└── engines/    # optional TensorRT engines (scripts/build_trt_engines.sh) and build manifests
```
