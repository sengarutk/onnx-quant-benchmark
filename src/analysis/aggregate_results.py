"""
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
        if output_csv_path and Path(output_csv_path).is_file():
            try:
                existing_df = pd.read_csv(output_csv_path)
                if len(existing_df) > 0:
                    logger.info(f"Loaded {len(existing_df)} pre-computed runs from {output_csv_path}")
                    return existing_df
            except Exception as e:
                logger.warning(f"Could not read existing CSV at {output_csv_path}: {e}")

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
