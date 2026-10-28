#!/usr/bin/env python3
"""
Smoke-validates every runtime backend that is usable on this host against the PyTorch FP32 CPU reference and
renders docs/runtimes.md with the observed status (PASS / FAIL / NOT AVAILABLE and the reason).
"""

import copy
import sys
from pathlib import Path
from typing import Any, Dict, List
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.config import load_quality_thresholds
from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.data.writers import load_manifest
from src.models.preprocess import preprocess_detection_image
from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.fallback_audit import assert_zero_fallback
from src.runtimes.ort_cpu_runtime import ORTCPURuntime
from src.runtimes.ort_cuda_runtime import ORTCUDARuntime
from src.runtimes.pytorch_runtime import PyTorchRuntime
from src.runtimes.tensorrt_runtime import TRT_AVAILABLE, TensorRTRuntime
from src.validation.output_checks import compute_tensor_diff

logger = setup_logger("validate_runtimes")


class TF32PyTorchRuntime(PyTorchRuntime):
    """PyTorch CUDA FP32 with PyTorch's default TF32 tensor-core math re-enabled (to quantify its parity impact)."""

    def load(self) -> None:
        super().load()
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
EXP = PROJECT_ROOT / "models" / "exported"


def main() -> None:
    seed_everything(42)
    adapter = YOLOAdapter(require_weights=True)
    sample = load_manifest(PROJECT_ROOT / "data" / "sample_images" / "manifest.json")["detection_samples"][0]
    x = preprocess_detection_image(PROJECT_ROOT / sample["image_path"])[0].numpy()
    ref_rt = PyTorchRuntime(copy.deepcopy(adapter.model), device="cpu")
    ref = ref_rt.predict({"images": x})["output0"]

    rows: List[Dict[str, Any]] = []

    def check(name: str, precision: str, make) -> None:
        try:
            rt = make()
            rt.load()
            assert_zero_fallback(rt)
            out = list(rt.predict({next(iter(rt.get_input_spec())): x}).values())[0]
            d = compute_tensor_diff(ref, out)
            gate = load_quality_thresholds("yolo_nano", precision)
            ok = d["max_rel_error"] <= gate.max_rel_error and d["cosine_similarity"] >= gate.min_cosine_similarity
            rows.append({"runtime": name, "precision": precision, "provider": rt.get_active_provider(),
                         "max_rel_error": d["max_rel_error"], "cosine_similarity": d["cosine_similarity"],
                         "status": "PASS" if ok else "GATE FAIL", "detail": f"max rel. error {d['max_rel_error']:.2e}, cosine {d['cosine_similarity']:.6f}"})
            rt.cleanup()
        except Exception as exc:  # noqa: BLE001 - recorded in the report
            rows.append({"runtime": name, "precision": precision, "provider": "-", "status": "ERROR", "detail": f"{type(exc).__name__}: {exc}"})

    check("PyTorch CPU", "fp32", lambda: PyTorchRuntime(copy.deepcopy(adapter.model), device="cpu"))
    if torch.cuda.is_available():
        check("PyTorch CUDA", "fp32", lambda: PyTorchRuntime(copy.deepcopy(adapter.model), device="cuda:0"))
        check("PyTorch CUDA", "fp16", lambda: PyTorchRuntime(copy.deepcopy(adapter.model), device="cuda:0", precision="fp16"))
        check("PyTorch CUDA (TF32 math, PyTorch default)", "fp32", lambda: TF32PyTorchRuntime(copy.deepcopy(adapter.model), device="cuda:0"))
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    else:
        rows.append({"runtime": "PyTorch CUDA", "precision": "fp32/fp16", "provider": "-", "status": "NOT AVAILABLE",
                     "detail": "torch.cuda.is_available() is False"})
    for prec, f in (("fp32", "yolo_nano_fp32_opset17.onnx"), ("fp16", "yolo_nano_fp16.onnx"), ("int8", "yolo_nano_static_int8.onnx")):
        check("ONNX Runtime CPU", prec, lambda p=EXP / f: ORTCPURuntime(p))
    if "CUDAExecutionProvider" in ort.get_available_providers():
        for prec, f in (("fp32", "yolo_nano_fp32_opset17.onnx"), ("fp16", "yolo_nano_fp16.onnx")):
            check("ONNX Runtime CUDA", prec, lambda p=EXP / f: ORTCUDARuntime(p))
    else:
        rows.append({"runtime": "ONNX Runtime CUDA", "precision": "fp32/fp16", "provider": "-", "status": "NOT AVAILABLE",
                     "detail": f"installed onnxruntime providers: {ort.get_available_providers()}"})
    engines = sorted((PROJECT_ROOT / "models" / "engines").glob("yolo_nano_*.engine"))
    if TRT_AVAILABLE and engines and torch.cuda.is_available():
        for e in engines:
            check("TensorRT", e.stem.split("_")[-1], lambda p=e: TensorRTRuntime(p))
    else:
        rows.append({"runtime": "TensorRT", "precision": "fp32/fp16/int8", "provider": "-", "status": "NOT AVAILABLE",
                     "detail": "tensorrt Python bindings not installed" if not TRT_AVAILABLE else "no engines built (scripts/build_trt_engines.sh)"})

    for r in rows:
        logger.info(f"{r['runtime']} {r['precision']}: {r['status']} ({r['detail']})")
    render_doc(rows, PROJECT_ROOT / "docs" / "runtimes.md")
    import json
    (PROJECT_ROOT / "results" / "runtime_validation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    # Gate failures are reported results (e.g. a lossy INT8 graph); only execution errors fail the stage.
    if any(r["status"] == "ERROR" for r in rows):
        sys.exit(1)


def render_doc(rows: List[Dict[str, Any]], path: Path) -> None:
    md = ["# Runtime Backends", "",
          "Generated by `scripts/validate_runtimes.py` on the benchmark host. Each available backend runs the trained "
          "detector on a real test image; outputs are compared with PyTorch FP32 on CPU using the precision tier's gate "
          "from `configs/yolo/<precision>.yaml`. `GATE FAIL` means the backend runs correctly but its output deviates more than the "
          "precision tier allows; `ERROR` means it could not run.", "",
          "## Implementation summary", "",
          "| Backend | Module | Execution details |", "| --- | --- | --- |",
          "| PyTorch eager | `src/runtimes/pytorch_runtime.py` | `torch.inference_mode()`, CPU threads pinned to the visible core count; CUDA path is asynchronous (no per-call sync) |",
          "| ONNX Runtime CPU | `src/runtimes/ort_cpu_runtime.py` | `ORT_ENABLE_ALL` graph optimizations, sequential execution, intra-op threads = visible cores, inter-op = 1 |",
          "| ONNX Runtime CUDA | `src/runtimes/ort_cuda_runtime.py` | IOBinding with pre-allocated device outputs on the PyTorch stream; provider audited at load |",
          "| TensorRT | `src/runtimes/tensorrt_runtime.py` | Deserialized engine, static-shape device buffers, input copy + `execute_async_v3` ordered on a dedicated stream |",
          "", "`FallbackAuditor` (`src/runtimes/fallback_audit.py`) raises if a requested ORT provider is not the session's primary "
          "provider, or if a CUDA PyTorch runtime holds parameters off-device. It does not inspect per-node placement inside a session.",
          "", "## Status on this host", "", "| Runtime | Precision | Provider | Status | Detail |", "| --- | --- | --- | --- | --- |"]
    md += [f"| {r['runtime']} | {r['precision'].upper()} | `{r['provider']}` | {r['status']} | {r['detail']} |" for r in rows]
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
