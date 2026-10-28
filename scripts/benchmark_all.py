#!/usr/bin/env python3
"""
Runs the full benchmark matrix as ONE measurement session.

Matrix (per model): PyTorch CPU FP32, PyTorch CUDA FP32/FP16, ONNX Runtime CPU FP32/FP16/INT8, and - only when
the corresponding execution provider / bindings are actually available - ONNX Runtime CUDA FP32/FP16 and
TensorRT FP32/FP16/INT8. Unavailable configurations are recorded with the reason in session.json.

Protocol parameters come from configs/benchmark.yaml; acceptance gates from configs/<model>/<precision>.yaml.
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List
import numpy as np
import onnxruntime as ort
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.benchmark_suite import BenchmarkSuite
from src.common.config import load_config, load_quality_thresholds
from src.common.environment import configure_cpu_threads
from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.data.writers import load_manifest
from src.export.export_onnx import AutoencoderExportWrapper
from src.models.industrial_model_adapter import IndustrialModelAdapter
from src.models.preprocess import preprocess_detection_image, preprocess_industrial_image
from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.base import BaseRuntime
from src.runtimes.ort_cpu_runtime import ORTCPURuntime
from src.runtimes.ort_cuda_runtime import ORTCUDARuntime
from src.runtimes.pytorch_runtime import PyTorchRuntime
from src.runtimes.tensorrt_runtime import TRT_AVAILABLE, TensorRTRuntime
from src.validation.task_eval import evaluate_autoencoder, evaluate_detector

logger = setup_logger("benchmark_all")
EXPORTED = PROJECT_ROOT / "models" / "exported"
ENGINES = PROJECT_ROOT / "models" / "engines"


def runtime_predict_fn(runtime: BaseRuntime) -> Callable[[np.ndarray], List[np.ndarray]]:
    in_name = next(iter(runtime.get_input_spec()))
    return lambda x: list(runtime.predict({in_name: x}).values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "benchmark.yaml"))
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--quick", action="store_true", help="Tiny iteration counts (smoke testing only)")
    parser.add_argument("--models", nargs="+", default=["yolo_nano", "industrial_autoencoder"])
    args = parser.parse_args()

    seed_everything(42)
    bench = load_config(args.config).benchmark
    if args.quick:
        bench = bench.model_copy(update={"warmup_iterations": 3, "timed_iterations": 10, "stability_sessions": 2,
                                         "cooldown_seconds": 0.0, "e2e_warmup_iterations": 2, "e2e_timed_iterations": 5})
    cores = configure_cpu_threads()
    cuda = torch.cuda.is_available()
    ort_cuda = "CUDAExecutionProvider" in ort.get_available_providers()

    suite = BenchmarkSuite(
        results_dir=args.results_dir,
        warmup_model=bench.warmup_iterations,
        timed_model=bench.timed_iterations,
        warmup_e2e=bench.e2e_warmup_iterations,
        timed_e2e=bench.e2e_timed_iterations,
        stability_sessions=bench.stability_sessions,
        cooldown_seconds=bench.cooldown_seconds,
        cv_threshold=bench.cv_threshold,
    )
    logger.info(f"Session {suite.session_id} | CPU threads: {cores} | CUDA: {cuda} | ORT CUDA EP: {ort_cuda} | TensorRT: {TRT_AVAILABLE}")

    manifest = load_manifest(PROJECT_ROOT / "data" / "sample_images" / "manifest.json")
    executed: List[str] = []
    skipped: List[Dict[str, str]] = []

    for model_name in args.models:
        if model_name == "yolo_nano":
            adapter = YOLOAdapter(require_weights=True)
            samples = manifest["detection_samples"]
            ref_model = copy.deepcopy(adapter.get_pytorch_model()).eval()
            input_t = preprocess_detection_image(PROJECT_ROOT / samples[0]["image_path"])[0]
            task, metric, in_names, out_names = "detection", "mAP_50", ["images"], ["output0"]
            def quality(rt, _s=samples, _a=adapter):
                return evaluate_detector(runtime_predict_fn(rt), _s, _a)
        else:
            adapter = IndustrialModelAdapter(require_weights=True)
            samples = manifest["industrial_samples"]
            ref_model = AutoencoderExportWrapper(copy.deepcopy(adapter.get_pytorch_model())).eval()
            input_t = preprocess_industrial_image(PROJECT_ROOT / samples[0]["image_path"])
            task, metric, in_names, out_names = "anomaly_detection", "image_auroc", ["input"], ["reconstruction", "anomaly_map"]
            def quality(rt, _s=samples):
                return evaluate_autoencoder(runtime_predict_fn(rt), _s)

        image_paths = [PROJECT_ROOT / s["image_path"] for s in samples]
        with torch.inference_mode():
            ref_out = ref_model(input_t)
        ref_outputs = [t.numpy() for t in (ref_out if isinstance(ref_out, (tuple, list)) else [ref_out])]

        ref_runtime = PyTorchRuntime(ref_model, device="cpu", input_names=in_names, output_names=out_names)
        reference_quality = quality(ref_runtime)
        logger.info(f"[{model_name}] PyTorch FP32 CPU reference quality: {reference_quality}")

        candidates: List[Dict[str, Any]] = [
            {"runtime": "PyTorch", "precision": "fp32", "make": lambda: ref_runtime, "device": "cpu"},
        ]
        for prec in ("fp32", "fp16"):
            if cuda:
                candidates.append({"runtime": "PyTorch", "precision": prec, "device": "cuda",
                                   "make": lambda p=prec: PyTorchRuntime(copy.deepcopy(ref_model), device="cuda:0",
                                                                          precision=p, input_names=in_names,
                                                                          output_names=out_names)})
            else:
                skipped.append({"config": f"{model_name}/PyTorch/{prec}/cuda", "reason": "torch.cuda.is_available() is False"})

        onnx_files = {"fp32": EXPORTED / f"{model_name}_fp32_opset17.onnx",
                      "fp16": EXPORTED / f"{model_name}_fp16.onnx",
                      "int8": EXPORTED / f"{model_name}_static_int8.onnx"}
        for prec, path in onnx_files.items():
            if not path.is_file():
                skipped.append({"config": f"{model_name}/ORT_CPU/{prec}", "reason": f"missing {path.name}"})
                continue
            candidates.append({"runtime": "ORT_CPU", "precision": prec, "device": "cpu", "file": path,
                               "make": lambda p=path: ORTCPURuntime(p)})
            if prec != "int8":
                if ort_cuda:
                    candidates.append({"runtime": "ORT_CUDA", "precision": prec, "device": "cuda", "file": path,
                                       "make": lambda p=path: ORTCUDARuntime(p)})
                else:
                    skipped.append({"config": f"{model_name}/ORT_CUDA/{prec}",
                                    "reason": "CUDAExecutionProvider not available in the installed onnxruntime build"})

        for prec in ("fp32", "fp16", "int8"):
            engine = ENGINES / f"{model_name}_{prec}.engine"
            if TRT_AVAILABLE and engine.is_file() and cuda:
                candidates.append({"runtime": "TensorRT", "precision": prec, "device": "cuda", "file": engine,
                                   "make": lambda e=engine: TensorRTRuntime(e)})
            else:
                skipped.append({"config": f"{model_name}/TensorRT/{prec}",
                                "reason": "tensorrt Python bindings not installed" if not TRT_AVAILABLE else f"missing {engine.name}"})

        for c in candidates:
            rt = c["make"]()
            rt.load()
            x = input_t.to("cuda:0") if c["device"] == "cuda" else input_t
            suite.run_single_configuration(
                model_name=model_name, runtime_name=c["runtime"], precision=c["precision"], runtime=rt,
                input_tensor=x, sample_image_paths=image_paths, adapter=adapter, task=task,
                reference_outputs=ref_outputs, quality_fn=quality, reference_quality=reference_quality,
                quality_metric=metric, thresholds=load_quality_thresholds(model_name, c["precision"]),
                model_file_path=c.get("file"),
            )
            executed.append(f"{model_name}/{c['runtime']}/{c['precision']}")
            if rt is not ref_runtime:
                rt.cleanup()

    session = {"session_id": suite.session_id, "protocol": suite.protocol, "quick": args.quick,
               "executed": executed, "skipped": skipped,
               "environment_manifest": suite.env_manifest.model_dump()}
    suite.session_dir.mkdir(parents=True, exist_ok=True)
    (suite.session_dir / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")
    logger.info(f"Session {suite.session_id}: {len(executed)} configurations executed, {len(skipped)} skipped -> {suite.session_dir}")


if __name__ == "__main__":
    main()
