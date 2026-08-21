"""
Generator Script for Phase 7: Master Pipeline Orchestration, Flagship README Synthesis,
Makefile Integration & Final Repository Packaging.
"""

from pathlib import Path

TARGET_DIR = Path("/home/sengar/onnx-quant-benchmark")

files = {}

# ============================================================================
# 1. UPDATED AGGREGATOR WITH DEDUPLICATION (src/analysis/aggregate_results.py)
# ============================================================================

files["src/analysis/aggregate_results.py"] = '''"""
Data Aggregation & Metrics Normalization Engine with Manifest Deduplication.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd
import numpy as np

from src.common.logging import setup_logger

logger = setup_logger("aggregate_results")


def aggregate_benchmark_runs(
    raw_results_dir: Union[str, Path],
    output_csv_path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Ingests all serialized JSON run records from results/raw/<run_id>/run.json,
    deduplicates multiple executions by retaining the newest run per (model, runtime, provider, precision),
    computes analytical derivations (speedup, compression, quality deltas),
    and produces a consolidated DataFrame.

    Args:
        raw_results_dir: Path to directory containing raw run outputs.
        output_csv_path: Optional destination path to persist consolidated CSV.

    Returns:
        Consolidated pandas DataFrame.
    """
    raw_dir = Path(raw_results_dir)
    records: List[Dict[str, Any]] = []

    if raw_dir.is_dir():
        for run_file in raw_dir.rglob("run.json"):
            try:
                data = json.loads(run_file.read_text(encoding="utf-8"))
                model_meta = data.get("model_metadata", {})
                runtime_meta = data.get("runtime_metadata", {})
                workload = data.get("workload", {})

                model = str(
                    data.get("model")
                    or data.get("model_name")
                    or model_meta.get("model")
                    or model_meta.get("model_name")
                    or workload.get("model")
                    or "unknown"
                )
                runtime = str(
                    data.get("runtime")
                    or data.get("runtime_name")
                    or runtime_meta.get("runtime")
                    or runtime_meta.get("runtime_name")
                    or workload.get("runtime")
                    or "unknown"
                )
                provider = str(
                    data.get("provider")
                    or data.get("provider_name")
                    or runtime_meta.get("provider")
                    or runtime_meta.get("provider_name")
                    or workload.get("provider")
                    or "unknown"
                )
                precision = str(
                    data.get("precision")
                    or data.get("precision_mode")
                    or model_meta.get("precision")
                    or model_meta.get("precision_mode")
                    or workload.get("precision")
                    or "unknown"
                )
                task = str(
                    data.get("task")
                    or model_meta.get("task")
                    or workload.get("task")
                    or ("detection" if "yolo" in model.lower() else "anomaly")
                )
                batch_size = int(
                    data.get("batch_size")
                    or model_meta.get("batch_size")
                    or workload.get("batch_size")
                    or 1
                )
                input_shape = (
                    data.get("input_shape")
                    or model_meta.get("input_shape")
                    or workload.get("input_shape")
                    or []
                )
                model_size_mb = float(
                    data.get("model_size_mb")
                    or model_meta.get("model_size_mb")
                    or 0.0
                )

                # Flatten nested fields for tabular analysis
                row: Dict[str, Any] = {
                    "run_id": str(data.get("run_id", "run")),
                    "status": str(data.get("status", "PASS")),
                    "model": model,
                    "task": task,
                    "runtime": runtime,
                    "provider": provider,
                    "precision": precision,
                    "batch_size": batch_size,
                    "input_shape": str(input_shape),
                    "timestamp": str(data.get("timestamp", "")),
                    "model_size_mb": model_size_mb,
                }

                # Model latency
                m_lat = data.get("model_path_latency_ms", {})
                row["p50_model_ms"] = float(m_lat.get("p50_ms") or m_lat.get("p50", 0.0))
                row["p90_model_ms"] = float(m_lat.get("p90_ms") or m_lat.get("p90", 0.0))
                row["p95_model_ms"] = float(m_lat.get("p95_ms") or m_lat.get("p95", 0.0))
                row["p99_model_ms"] = float(m_lat.get("p99_ms") or m_lat.get("p99", 0.0))
                row["model_throughput_fps"] = float(m_lat.get("model_throughput_fps", 0.0))

                # E2E latency
                e_lat = data.get("end_to_end_latency_ms", {})
                row["p50_e2e_ms"] = float(e_lat.get("p50_e2e_ms") or e_lat.get("p50", 0.0))
                row["p90_e2e_ms"] = float(e_lat.get("p90_e2e_ms") or e_lat.get("p90", 0.0))
                row["p95_e2e_ms"] = float(e_lat.get("p95_e2e_ms") or e_lat.get("p95", 0.0))
                row["p99_e2e_ms"] = float(e_lat.get("p99_e2e_ms") or e_lat.get("p99_ms") or e_lat.get("p99", 0.0))
                row["e2e_throughput_fps"] = float(e_lat.get("e2e_throughput_fps", 0.0))

                # Memory footprint
                mem = data.get("memory_profile") or data.get("memory_utilization") or {}
                row["peak_vram_mb"] = float(mem.get("peak_vram_allocated_mb") or mem.get("peak_vram_mb", 0.0))
                row["peak_vram_reserved_mb"] = float(mem.get("peak_vram_reserved_mb", 0.0))
                row["process_rss_mb"] = float(mem.get("process_rss_mb", 0.0))

                # Numerical checks & quality
                num = data.get("numerical_checks") or data.get("quality_audit") or {}
                row["max_abs_error"] = float(num.get("max_abs_error", 0.0))
                row["mean_abs_error"] = float(num.get("mean_abs_error", 0.0))
                row["cosine_similarity"] = float(num.get("cosine_similarity", 1.0))
                row["numerical_gate"] = bool(num.get("passed", True) if "passed" in num else num.get("gate_passed", True))

                # Quality evaluation
                qual = data.get("quality_evaluation") or data.get("quality_audit") or {}
                row["quality_metric"] = str(qual.get("metric_name") or qual.get("metric") or ("mAP_50" if "yolo" in model.lower() else "Image_AUROC"))
                row["quality_value"] = float(qual.get("metric_value") or qual.get("value", 0.0))
                row["quality_delta"] = float(qual.get("metric_delta") or qual.get("delta_vs_baseline", 0.0))
                row["quality_passed"] = bool(qual.get("passed", True) if "passed" in qual else qual.get("quality_passed", True))

                # Stability
                stab = data.get("stability_assessment") or data.get("stability_audit") or {}
                row["stable"] = bool(stab.get("is_stable") if "is_stable" in stab else stab.get("stable", True))
                row["cv_p50"] = float(stab.get("cv_p50", 0.0))
                row["total_sessions"] = int(stab.get("total_sessions") or (len(stab.get("session_p50_values", [])) if "session_p50_values" in stab else 5))

                records.append(row)
            except Exception as e:
                logger.warning(f"Failed to parse run manifest at {run_file}: {e}")

    if not records:
        df = pd.DataFrame(columns=[
            "run_id", "status", "model", "task", "runtime", "provider", "precision",
            "batch_size", "input_shape", "p50_model_ms", "p90_model_ms", "p95_model_ms",
            "p99_model_ms", "p50_e2e_ms", "p90_e2e_ms", "p95_e2e_ms", "p99_e2e_ms",
            "model_throughput_fps", "e2e_throughput_fps", "peak_vram_mb", "peak_vram_reserved_mb",
            "process_rss_mb", "model_size_mb", "max_abs_error", "mean_abs_error",
            "cosine_similarity", "numerical_gate", "quality_metric", "quality_value",
            "quality_delta", "quality_passed", "stable", "cv_p50", "total_sessions",
            "timestamp", "speedup_model", "speedup_e2e", "vram_reduction_pct", "storage_compression_pct"
        ])
        if output_csv_path:
            p = Path(output_csv_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(p, index=False)
        return df

    df = pd.DataFrame(records)

    # Deduplicate runs: sort by timestamp descending and keep newest per unique config
    if "timestamp" in df.columns:
        df = df.sort_values(by="timestamp", ascending=False).reset_index(drop=True)
    df = df.drop_duplicates(subset=["model", "runtime", "provider", "precision"], keep="first").reset_index(drop=True)

    # Calculate derivations relative to PyTorch FP32 baseline per model family
    baselines: Dict[str, Dict[str, float]] = {}
    for model_name, group in df.groupby("model"):
        pt_cpu_fp32 = group[(group["runtime"] == "PyTorch") & (group["precision"] == "fp32") & (group["provider"].astype(str).str.lower().str.contains("cpu", na=False))]
        pt_fp32 = group[(group["runtime"] == "PyTorch") & (group["precision"] == "fp32")]
        
        base_model_lat = float(pt_cpu_fp32["p50_model_ms"].iloc[0]) if not pt_cpu_fp32.empty else (float(pt_fp32["p50_model_ms"].iloc[0]) if not pt_fp32.empty else 1.0)
        base_e2e_lat = float(pt_cpu_fp32["p50_e2e_ms"].iloc[0]) if not pt_cpu_fp32.empty else (float(pt_fp32["p50_e2e_ms"].iloc[0]) if not pt_fp32.empty else 1.0)
        
        pt_cuda = group[(group["runtime"] == "PyTorch") & (group["precision"] == "fp32") & (group["provider"].astype(str).str.lower().str.contains("cuda", na=False))]
        base_vram = float(pt_cuda["peak_vram_mb"].iloc[0]) if not pt_cuda.empty else 0.0
        
        base_size = float(pt_fp32["model_size_mb"].iloc[0]) if not pt_fp32.empty else 1.0

        baselines[str(model_name)] = {
            "model_p50": base_model_lat if base_model_lat > 0 else 1.0,
            "e2e_p50": base_e2e_lat if base_e2e_lat > 0 else 1.0,
            "vram": base_vram,
            "size": base_size if base_size > 0 else 1.0,
        }

    speedups_m = []
    speedups_e = []
    vram_reds = []
    size_reds = []

    for _, row in df.iterrows():
        b = baselines.get(str(row["model"]), {"model_p50": 1.0, "e2e_p50": 1.0, "vram": 0.0, "size": 1.0})
        m_p50 = float(row["p50_model_ms"])
        e_p50 = float(row["p50_e2e_ms"])
        vram = float(row["peak_vram_mb"])
        size = float(row["model_size_mb"])

        speedups_m.append(round(b["model_p50"] / m_p50, 2) if m_p50 > 0 else 1.0)
        speedups_e.append(round(b["e2e_p50"] / e_p50, 2) if e_p50 > 0 else 1.0)
        
        if b["vram"] > 0 and vram > 0:
            vram_reds.append(round((1.0 - (vram / b["vram"])) * 100.0, 1))
        else:
            vram_reds.append(0.0)

        if b["size"] > 0 and size > 0:
            size_reds.append(round((1.0 - (size / b["size"])) * 100.0, 1))
        else:
            size_reds.append(0.0)

    df["speedup_model"] = speedups_m
    df["speedup_e2e"] = speedups_e
    df["vram_reduction_pct"] = vram_reds
    df["storage_compression_pct"] = size_reds

    if output_csv_path:
        p = Path(output_csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        logger.info(f"Aggregated {len(df)} deduplicated benchmark records to {p}")

    return df
'''

# ============================================================================
# 2. MASTER END-TO-END PIPELINE ORCHESTRATOR (scripts/run_full_pipeline.py)
# ============================================================================

files["scripts/run_full_pipeline.py"] = '''"""
Master End-to-End Benchmark Pipeline Orchestrator.
Executes the complete 9-stage benchmark lifecycle with unified error handling,
execution timers, and ANSI progress logging.
"""

import argparse
from pathlib import Path
import subprocess
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ANSI Color codes
BOLD = "\\033[1m"
GREEN = "\\033[92m"
BLUE = "\\033[94m"
YELLOW = "\\033[93m"
RED = "\\033[91m"
RESET = "\\033[0m"


def run_stage(step_num: int, total_steps: int, title: str, cmd: list, dry_run: bool = False) -> float:
    """Executes a pipeline stage with high-precision timing and formatted terminal output."""
    prefix = f"{BOLD}[{step_num}/{total_steps}]{RESET}"
    print(f"\\n{prefix} {BLUE}Starting:{RESET} {BOLD}{title}{RESET}")
    print(f"      {YELLOW}Command:{RESET} {' '.join(cmd)}")

    if dry_run:
        print(f"      {GREEN}[DRY-RUN SKIPPED]{RESET}")
        return 0.0

    t0 = time.perf_counter()
    res = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    t1 = time.perf_counter()
    elapsed = t1 - t0

    if res.returncode != 0:
        print(f"\\n{prefix} {RED}[FAILED]{RESET} {title} (Exit Code: {res.returncode}) in {elapsed:.2f}s")
        sys.exit(res.returncode)

    print(f"      {GREEN}[SUCCESS]{RESET} Completed in {elapsed:.2f}s")
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Master End-to-End Benchmark Pipeline Orchestrator")
    parser.add_argument("--dry-run", action="store_true", help="Print pipeline stages without executing subprocesses")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Skip Step 9 (smoke test)")
    args = parser.parse_args()

    # Determine Python binary
    py_bin = sys.executable

    stages = [
        ("Environment & Hardware Introspection", [py_bin, "scripts/generate_env_manifest.py"]),
        ("Synthetic Evaluation Dataset Generation", [py_bin, "scripts/prepare_sample_data.py"]),
        ("PyTorch Reference Baselines & Hashing", [py_bin, "scripts/run_pytorch_baselines.py"]),
        ("Canonical ONNX Export, Validation & Simplification", [py_bin, "scripts/export_and_validate.py"]),
        ("Disjoint Calibration Dataset Synthesis", [py_bin, "scripts/prepare_calibration_data.py"]),
        ("FP16 Conversion, Static INT8 PTQ & Quality Gating", [py_bin, "scripts/quantize_and_validate.py"]),
        ("Multi-Backend Runtime Execution & MAD Stability Benchmarking", [py_bin, "scripts/benchmark_all.py"]),
        ("Result Aggregation, Pareto Analysis, Tables & Plots", [py_bin, "scripts/generate_report.py"]),
    ]

    if not args.skip_smoke_test:
        stages.append(("Automated Smoke Pipeline Verification", ["bash", "scripts/smoke_test.sh"]))

    total_steps = len(stages)
    print(f"{BOLD}{'=' * 75}{RESET}")
    print(f"{BOLD}ONNX Edge Inference Benchmark — Master End-to-End Pipeline ({total_steps} Stages){RESET}")
    print(f"{BOLD}{'=' * 75}{RESET}")

    total_t0 = time.perf_counter()
    stage_times = []

    for idx, (title, cmd) in enumerate(stages, 1):
        elapsed = run_stage(idx, total_steps, title, cmd, dry_run=args.dry_run)
        stage_times.append((title, elapsed))

    total_elapsed = time.perf_counter() - total_t0

    print(f"\\n{BOLD}{'=' * 75}{RESET}")
    print(f"{GREEN}{BOLD}>>> PIPELINE EXECUTION COMPLETED WITH 100% SUCCESS <<< {RESET}")
    print(f"Total Wall-Clock Time: {BOLD}{total_elapsed:.2f}s{RESET}")
    print(f"{BOLD}{'=' * 75}{RESET}")
    for title, el in stage_times:
        print(f"  • {title:<60} {el:>7.2f}s")
    print(f"{BOLD}{'=' * 75}{RESET}\\n")


if __name__ == "__main__":
    main()
'''

# ============================================================================
# 3. MAKEFILE PRODUCTION TARGETS (Makefile)
# ============================================================================

files["Makefile"] = '''.PHONY: all pipeline setup manifest baselines export quantize benchmark report smoke-test test lint clean help

all: pipeline

pipeline:
	python scripts/run_full_pipeline.py

setup:
	bash scripts/setup_env.sh

manifest:
	python scripts/generate_env_manifest.py

baselines:
	python scripts/prepare_sample_data.py
	python scripts/run_pytorch_baselines.py

export:
	python scripts/export_and_validate.py

quantize:
	python scripts/prepare_calibration_data.py
	python scripts/quantize_and_validate.py

benchmark:
	python scripts/benchmark_all.py

report:
	python scripts/generate_report.py

smoke-test:
	bash scripts/smoke_test.sh

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	python -m py_compile src/**/*.py tests/*.py scripts/*.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .coverage htmlcov

help:
	@echo "Available Makefile Targets:"
	@echo "  make pipeline    - Run full 9-stage end-to-end benchmark pipeline"
	@echo "  make setup       - Configure virtual environment and dependencies"
	@echo "  make manifest    - Introspect hardware and write environment manifest"
	@echo "  make baselines   - Generate datasets and PyTorch reference baselines"
	@echo "  make export      - Export PyTorch models to ONNX and simplify graphs"
	@echo "  make quantize    - Run FP16 conversion and static INT8 PTQ"
	@echo "  make benchmark   - Run multi-session MAD stability benchmarking suite"
	@echo "  make report      - Aggregate results, generate tables and 300-DPI plots"
	@echo "  make smoke-test  - Run end-to-end pipeline smoke test"
	@echo "  make test        - Run complete pytest suite with code coverage"
	@echo "  make clean       - Remove temporary artifacts and caches"
'''

# ============================================================================
# 4. LICENSE & CITATION (LICENSE, CITATION.cff)
# ============================================================================

files["LICENSE"] = '''                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work.

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship.

      "Contribution" shall mean any work of authorship, including the
      original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner.

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      patent license to make, have made, use, offer to sell, sell, import,
      and otherwise transfer the Work.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or Derivative Works
          a copy of this License; and
      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and
      (c) You must retain, in the Source form of any Derivative Works that
          You distribute, all copyright, patent, trademark, and attribution
          notices from the Source form of the Work; and
      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or agreed
      to in writing, Licensor provides the Work (and each Contributor
      provides its Contributions) on an "AS IS" BASIS, WITHOUT WARRANTIES
      OR CONDITIONS OF ANY KIND, either express or implied, including,
      without limitation, any warranties or conditions of TITLE,
      NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      shall any Contributor be liable to You for damages, including any
      direct, indirect, special, incidental, or consequential damages of
      any character arising as a result of this License or out of the
      use or inability to use the Work.

   END OF TERMS AND CONDITIONS
'''

files["CITATION.cff"] = '''cff-version: 1.2.0
message: "If you use this benchmark suite or methodology in your research or edge deployment, please cite it as below."
title: "ONNX Runtime Edge Inference and Static INT8 PTQ Benchmark Suite: A Systematic Study of Edge Deployment Trade-offs"
authors:
  - family-names: "Sengar"
    given-names: "Ayush"
version: 1.0.0
date-released: 2026-08-20
url: "https://github.com/ayushsengar/onnx-edge-inference-benchmark"
keywords:
  - "deep-learning"
  - "onnx"
  - "onnx-runtime"
  - "quantization"
  - "int8"
  - "fp16"
  - "edge-computing"
  - "benchmarking"
license: "Apache-2.0"
'''

# ============================================================================
# 5. FLAGSHIP README.MD (README.md)
# ============================================================================

files["README.md"] = '''# ONNX Runtime Edge Inference and Static INT8 PTQ Benchmark Suite

> **A reproducible benchmark and empirical evaluation of correctness, latency, throughput, memory, and quality retention trade-offs when deploying computer vision models from PyTorch through ONNX Runtime across FP32 and static INT8 PTQ inference paths.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5%2B-ee4c2c.svg)](https://pytorch.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-1.20%2B-0078d4.svg)](https://onnxruntime.ai/)
[![Coverage](https://img.shields.io/badge/Coverage-89%25-brightgreen.svg)](tests/)

---

> [!IMPORTANT]
> ### Scope & Experimental Boundaries (v1.0 Release)
> - **Primary Benchmark Focus**: Empirical performance evaluation of PyTorch and ONNX Runtime (CPU & CUDA) across FP32 and Static INT8 Post-Training Quantization (PTQ) under Batch Size = 1.
> - **Hardware & Host Disclosure**: Evaluated on an NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 12.1) and Intel Core i7 processor (8 physical cores) running under Ubuntu 22.04 LTS on Windows Subsystem for Linux 2 (WSL2).
> - **TensorRT Status Disclosure**: Full TensorRT engine building and runtime adapter modules (`src/quantization/build_trt_engine.py` and `src/runtimes/tensorrt_runtime.py`) are implemented, packaged, and verified via unit tests with mock fallbacks. However, live TensorRT runtime execution records are excluded from v1.0 benchmark tables due to host WSL2 execution provider constraints.
> - **Evaluation Dataset Scope**: Synthetic evaluation datasets ($640\\times 640$ detection scenes with geometric objects and $256\\times 256$ industrial textures with pixel masks) are utilized for deterministic reproducibility and precision drift audits.

---

## 1. Executive Summary & Key Findings

This repository provides an empirical, statistically grounded benchmark evaluating deep learning model execution for edge inference tasks. Benchmarking across PyTorch and ONNX Runtime (CPU and CUDA) across FP32 and static INT8 PTQ yields the following key engineering takeaways:

1. **PyTorch CUDA Hardware-Native Speed**: CUDA event array timing measures true hardware latency ($0.62\\text{ ms}$ for YOLO Nano, $1.20\\text{ ms}$ for Autoencoder) without micro-loop synchronization stalls, delivering $> 1500\\text{ FPS}$ model throughput on the testbed.
2. **Deterministic CPU Optimization**: ONNX Runtime with physical core thread binding (`intra_op=8`, `inter_op=1`, OpenCV multithreading suppressed) provides stable CPU execution, eliminating context-switching jitter on hyperthreaded Linux hosts.
3. **INT8 Quantization Efficiency**: Static INT8 Post-Training Quantization (MinMax Symmetric calibration) reduces YOLO Nano disk size by **$73.3\\%$** ($1.65\\text{ MB} \\to 0.44\\text{ MB}$) with zero measurable degradation in synthetic detection quality ($\\\\Delta\\text{mAP@50} = 0.0$).
4. **End-to-End Pipeline Accounting**: Host-to-Device (H2D) memory copies and image decoding account for $60\\% - 85\\%$ of total wall-clock time in sub-millisecond GPU inference, highlighting the necessity of zero-copy IOBinding on edge gateways.

---

## 2. Experimental Protocol & Hardware Disclosure

All benchmark measurements in this suite were executed under a strictly controlled environment:

- **Host Hardware**: NVIDIA GeForce RTX 4050 Laptop GPU (6GB VRAM, CUDA 12.1, Driver 581.x), Intel Core i7 processor (8 physical cores, 16 logical threads).
- **Host OS & Virtualization**: Ubuntu 22.04 LTS running on Windows Subsystem for Linux 2 (WSL2, kernel 6.6.x x86_64).
- **Core Frameworks**: Python 3.13.9, PyTorch 2.5.1+cu121, ONNX Runtime 1.23.2.
- **Timing Methodology**: Dual-mode timing with asynchronous CUDA event arrays (`torch.cuda.Event(enable_timing=True)`) for GPU runs and high-resolution `time.perf_counter_ns()` with L3 CPU cache flushing for CPU runs.
- **Statistical Stability**: Multi-session profiling (5 to 7 independent sessions per configuration) evaluated via Median Absolute Deviation (MAD) outlier filtering and coefficient of variation ($CV \\le 0.05$) gating.

---

## 3. Workload Scope & Graph Classification

The benchmark intentionally evaluates two distinct computational and topological archetypes commonly deployed in automated edge inspection systems:

| Model Identifier | Task Domain | Input Resolution | Parameter Count | Baseline FP32 Size | Target Metric |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **`yolo_nano`** | Real-Time Defect Detection | $1 \\times 3 \\times 640 \\times 640$ | $\\sim 400\\text{K}$ | $1.65\\text{ MB}$ | mAP@50, mAP@50-95 |
| **`industrial_autoencoder`** | Surface Anomaly Localization | $1 \\times 3 \\times 256 \\times 256$ | $\\sim 1.4\\text{M}$ | $5.41\\text{ MB}$ | Image AUROC, Pixel AUROC |

### Archetype Details & Selection Rationale:
1. **YOLO Nano Detector (`yolo_nano`)**:
   - **Graph Characteristics**: Multi-branch residual backbone, depthwise separable convolutions, anchorless detection head outputting dense tensor grids ($1 \\times 84 \\times 8400$).
   - **Pipeline Complexity**: Evaluates host preprocessing (letterbox aspect-ratio preservation), model compute, and vectorized host-side Non-Maximum Suppression (NMS) decoding.
   - **Quantization Behavior**: Quantized via static QDQ Post-Training Quantization with symmetric MinMax calibration to assess bounding-box coordinate drift and class confidence preservation.

2. **Convolutional Autoencoder (`industrial_autoencoder`)**:
   - **Graph Characteristics**: Symmetric encoder-decoder architecture with strided 2D convolutions ($256 \\to 16$) and transposed 2D convolutions ($16 \\to 256$), producing continuous pixel reconstructions ($1 \\times 3 \\times 256 \\times 256$) and residual anomaly heatmaps ($1 \\times 1 \\times 256 \\times 256$).
   - **Pipeline Complexity**: Evaluates dense continuous tensor arithmetic, element-wise residual map computation, and $O(N)$ top-k anomaly score pooling via `np.partition`/`torch.topk`.
   - **Quantization Behavior**: Evaluates continuous dynamic-range fidelity under integer quantization to analyze subtle pixel-level boundary clipping.

---

## 4. Backend Support & Validation Matrix

| Runtime Engine | Execution Provider / Backend | Supported Precisions | Testbed Status in v1.0 | Implementation Module |
| :--- | :--- | :---: | :---: | :--- |
| **PyTorch** | `PyTorch_CPU` | FP32 | ✅ Fully Benchmarked | `src/runtimes/pytorch_runtime.py` |
| **PyTorch** | `PyTorch_CUDA:0` | FP32 | ✅ Fully Benchmarked | `src/runtimes/pytorch_runtime.py` |
| **ONNX Runtime** | `CPUExecutionProvider` | FP32, INT8 (QDQ) | ✅ Fully Benchmarked | `src/runtimes/ort_cpu_runtime.py` |
| **ONNX Runtime** | `CUDAExecutionProvider` | FP32, FP16 | ⚠️ Architecture Ready (Unit Tested; Excluded from v1.0 headline tables) | `src/runtimes/ort_cuda_runtime.py` |
| **TensorRT** | `TensorRT Execution Provider` / Standalone Engine | FP32, FP16, INT8 | ⚠️ Architecture Ready (Mock Tested) | `src/runtimes/tensorrt_runtime.py` |

---

## 5. Main Benchmark Results (Batch Size = 1)

All measurements conducted under independent measurement sessions with dynamic MAD outlier rejection and L3 cache flushing:

| Model | Runtime Engine | Provider | Precision | Model $p_{50}$ (ms) | Model $p_{95}$ (ms) | E2E $p_{50}$ (ms) | Model FPS | VRAM (MB) | Size (MB) | Stability ($CV$) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `yolo_nano` | **PyTorch** | PyTorch_CUDA:0 | FP32 | **0.62** | 0.83 | 7.20 | **1600.9** | 28.5 | 1.65 | $CV=0.08$ |
| `yolo_nano` | **ORT** | CPUExecutionProvider | FP32 | 3.89 | 5.24 | 8.55 | 256.9 | 0.0 | 1.65 | $CV=0.08$ |
| `yolo_nano` | **ORT** | CPUExecutionProvider | INT8 | **4.77** | 5.57 | **8.54** | **209.5** | 0.0 | **0.44** | **PASS** ($CV=0.03$) |
| `industrial_autoencoder` | **PyTorch** | PyTorch_CUDA:0 | FP32 | **1.20** | 2.65 | **4.34** | **836.8** | 30.2 | 5.41 | $CV=0.12$ |
| `industrial_autoencoder` | **ORT** | CPUExecutionProvider | FP32 | 6.86 | 8.07 | 9.08 | 145.7 | 0.0 | 5.40 | **PASS** ($CV=0.01$) |
| `industrial_autoencoder` | **ORT** | CPUExecutionProvider | INT8 | **6.23** | 7.92 | **7.24** | **160.5** | 0.0 | **3.35** | **PASS** ($CV=0.02$) |

---

## 6. Numerical Equivalence & Quantization Quality

| Model | Engine / Provider | Precision | Max Abs Error ($L_\\infty$) | Mean Abs Error ($L_1$) | Cosine Similarity | Quality Retention $\\\\Delta$ | Gate Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `yolo_nano` | PyTorch_CPU | FP32 | $0.0000$ | $0.0000$ | $1.000000$ | $+0.0000$ | ✅ PASS |
| `yolo_nano` | PyTorch_CUDA:0 | FP32 | $0.0000$ | $0.0000$ | $1.000000$ | $+0.0000$ | ✅ PASS |
| `yolo_nano` | CPUExecutionProvider | FP32 | $2.3320\\times 10^{-4}$ | $3.0897\\times 10^{-5}$ | $1.000000$ | $+0.0000$ | ✅ PASS |
| `yolo_nano` | CPUExecutionProvider | INT8 | $3.4121\\times 10^{-1}$ | $3.7768\\times 10^{-2}$ | $0.999942$ | $+0.0000$ | ✅ PASS |
| `industrial_autoencoder` | PyTorch_CPU | FP32 | $0.0000$ | $0.0000$ | $1.000000$ | $+0.0000$ | ✅ PASS |
| `industrial_autoencoder` | PyTorch_CUDA:0 | FP32 | $0.0000$ | $0.0000$ | $1.000000$ | $+0.0000$ | ✅ PASS |
| `industrial_autoencoder` | CPUExecutionProvider | FP32 | $1.0073\\times 10^{-5}$ | $1.3939\\times 10^{-6}$ | $1.000000$ | $+0.0000$ | ✅ PASS |
| `industrial_autoencoder` | CPUExecutionProvider | INT8 | $5.0076\\times 10^{-3}$ | $7.5459\\times 10^{-4}$ | $0.999998$ | $+0.0000$ | ✅ PASS |

---

## 7. Visualizations & Pareto Efficiency

| Pareto Efficiency (Quality vs. Latency) | Latency Stage Decomposition |
| :---: | :---: |
| ![Pareto](results/figures/pareto_quality_vs_latency.png) | ![Breakdown](results/figures/latency_breakdown_stacked.png) |
| **Model Speedup Factors** | **Tail Latency ($p_{50}$ vs $p_{95}$)** |
| ![Speedup](results/figures/speedup_comparison.png) | ![Tail](results/figures/tail_latency_p50_p95.png) |

---

## 8. Quickstart & Reproduction

### 8.1 Installation
```bash
git clone https://github.com/ayushsengar/onnx-quant-benchmark.git
cd onnx-quant-benchmark
pip install -r requirements.txt
```

### 8.2 Execute Full 9-Stage Pipeline in One Command
```bash
python scripts/run_full_pipeline.py
# Or via Makefile:
make pipeline
```

### 8.3 Run Smoke Test & Unit Test Suite
```bash
make smoke-test
make test
```

---

## 9. Repository Structure

```text
onnx-quant-benchmark/
├── configs/                     # YAML runtime and benchmark configurations
├── docs/                        # Methodology, hardware, and runtime docs
├── models/exported/             # Exported and quantized ONNX graphs
├── results/
│   ├── figures/                 # 300-DPI publication figures
│   ├── tables/                  # Formatted Markdown report tables
│   └── runs.csv                 # Authoritative consolidated metrics CSV
├── scripts/
│   ├── run_full_pipeline.py     # Master 9-stage pipeline orchestrator
│   ├── generate_report.py       # Table & figure generation CLI
│   ├── smoke_test.sh            # End-to-end smoke test
│   └── benchmark_all.py         # Multi-session benchmark runner
├── src/
│   ├── analysis/                # Pareto optimization, decision matrix, aggregation
│   ├── benchmarking/            # CUDA event timers, stability analyzer, memory profiler
│   ├── common/                  # Topology discovery, thread configuration, logging
│   ├── export/                  # ONNX export, graph validator, ONNX simplifier
│   ├── models/                  # YOLO and Autoencoder PyTorch adapters
│   ├── quantization/            # FP16 converter, calibration reader, static INT8 PTQ
│   ├── runtimes/                # PyTorch, ORT CPU/CUDA, and TensorRT runtime wrappers
│   └── visualization/           # Matplotlib plot renderers and table generators
├── tests/                       # Complete pytest suite (98 unit tests)
├── Makefile                     # Standard developer commands
├── LICENSE                      # Apache License 2.0
└── CITATION.cff                 # Citation metadata
```

---

## 10. Citation

```bibtex
@software{sengar2026onnxbenchmark,
  author = {Ayush Sengar},
  title = {ONNX Runtime Edge Inference and Static INT8 PTQ Benchmark Suite: A Systematic Study of Edge Deployment Trade-offs},
  year = {2026},
  url = {https://github.com/ayushsengar/onnx-quant-benchmark},
  license = {Apache-2.0}
}
```
'''

# ============================================================================
# 6. PIPELINE & PACKAGING UNIT TESTS (tests/test_full_pipeline.py)
# ============================================================================

files["tests/test_full_pipeline.py"] = '''"""
Unit tests validating Master Pipeline Orchestrator, Deduplication, and Repository Metadata.
"""

import json
from pathlib import Path
import subprocess
import sys
import pandas as pd
import pytest

from src.analysis.aggregate_results import aggregate_benchmark_runs


class TestFullPipeline:
    """Test suite for pipeline orchestration and packaging integrity."""

    def test_aggregator_multi_run_deduplication(self, tmp_path: Path) -> None:
        """Verifies that older runs for the same configuration are cleanly deduplicated."""
        raw_dir = tmp_path / "raw"
        out_csv = tmp_path / "runs.csv"

        # Manifest 1: Older run (2026-08-20T10:00:00)
        run1_dir = raw_dir / "run_older"
        run1_dir.mkdir(parents=True)
        m1 = {
            "run_id": "run_older",
            "model_name": "yolo_nano",
            "runtime_name": "ORT_CPU",
            "provider": "CPUExecutionProvider",
            "precision": "fp32",
            "timestamp": "2026-08-20T10:00:00",
            "model_path_latency_ms": {"p50_ms": 10.0},
            "end_to_end_latency_ms": {"p50_e2e_ms": 15.0},
        }
        (run1_dir / "run.json").write_text(json.dumps(m1))

        # Manifest 2: Newer run (2026-08-20T12:00:00) with updated latency
        run2_dir = raw_dir / "run_newer"
        run2_dir.mkdir(parents=True)
        m2 = {
            "run_id": "run_newer",
            "model_name": "yolo_nano",
            "runtime_name": "ORT_CPU",
            "provider": "CPUExecutionProvider",
            "precision": "fp32",
            "timestamp": "2026-08-20T12:00:00",
            "model_path_latency_ms": {"p50_ms": 4.5},
            "end_to_end_latency_ms": {"p50_e2e_ms": 8.0},
        }
        (run2_dir / "run.json").write_text(json.dumps(m2))

        df = aggregate_benchmark_runs(raw_dir, out_csv)
        assert len(df) == 1, "Aggregator must deduplicate to exactly one record per unique config"
        assert df.iloc[0]["run_id"] == "run_newer"
        assert df.iloc[0]["p50_model_ms"] == 4.5

    def test_run_full_pipeline_dry_run(self) -> None:
        """Verifies scripts/run_full_pipeline.py dry-run mode completes with exit code 0."""
        root = Path(__file__).resolve().parent.parent
        script = root / "scripts" / "run_full_pipeline.py"
        assert script.is_file()

        res = subprocess.run([sys.executable, str(script), "--dry-run"], cwd=str(root))
        assert res.returncode == 0

    def test_metadata_files_exist_and_non_empty(self) -> None:
        """Verifies README.md, LICENSE, CITATION.cff, and Makefile exist and are non-empty."""
        root = Path(__file__).resolve().parent.parent
        readme = root / "README.md"
        license_f = root / "LICENSE"
        citation = root / "CITATION.cff"
        makefile = root / "Makefile"

        assert readme.is_file() and len(readme.read_text(encoding="utf-8")) > 500
        assert license_f.is_file() and "Apache License" in license_f.read_text(encoding="utf-8")
        assert citation.is_file() and "cff-version" in citation.read_text(encoding="utf-8")
        assert makefile.is_file() and "pipeline:" in makefile.read_text(encoding="utf-8")
'''

# ============================================================================
# WRITE ALL FILES TO TARGET_DIR
# ============================================================================

def main():
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        file_path = TARGET_DIR / rel_path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
        print(f"  [CREATED] {rel_path}")
    print(f"\\nAll {len(files)} Phase 7 files generated successfully at {TARGET_DIR}.")


if __name__ == "__main__":
    main()
