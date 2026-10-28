# Software Environment

The exact versions used for the published results are recorded automatically in `docs/hardware.md` and in every run
record (`results/runs.json` -> `environment_manifest`).

## Installation

```bash
python -m pip install -r requirements.txt
```

* Python >= 3.10 (results produced with Python 3.13).
* NumPy >= 2.0 is required (`np.trapezoid` is used for AU-PRO).
* `requirements.txt` installs the **CPU** build of ONNX Runtime. For GPU inference, install `onnxruntime-gpu`
  **instead** of `onnxruntime` - never both: when both are present the CPU package shadows the GPU one and
  `CUDAExecutionProvider` silently disappears (check with
  `python -c "import onnxruntime as o; print(o.get_available_providers())"`).
* TensorRT is optional (`pip install tensorrt`); engines are built with `scripts/build_trt_engines.sh`.
* ONNX export uses PyTorch's TorchScript-based exporter (`dynamo=False`), which produced the published graphs.
  PyTorch >= 2.9 defaults to the dynamo exporter and prints a deprecation warning for the TorchScript one; the
  results were produced with PyTorch 2.5.1 and CI runs the test suite against the latest CPU release.

## Components

| Component | Role |
| --- | --- |
| PyTorch | training, FP32 reference, eager CPU/CUDA runtime, ONNX export |
| ONNX / onnxsim | graph export, checking, simplification |
| ONNX Runtime | CPU (and optionally CUDA) inference, static INT8 quantization |
| onnxconverter-common | FP16 conversion |
| OpenCV, NumPy | synthetic data generation, preprocessing |
| scikit-learn, SciPy | AUROC, Wilcoxon test |
| pandas, matplotlib | aggregation, tables, figures |
| pydantic, PyYAML | typed configuration |
