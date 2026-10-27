"""
Aggregates the run records of ONE benchmark session into the tabular results/runs.csv.

Session selection prevents mixing measurements taken at different times: by default the most recent session
directory under results/raw/ that contains a session.json (i.e. a completed `benchmark_all.py` run) is used, and
quick/smoke sessions are ignored unless explicitly requested.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from src.common.logging import setup_logger

logger = setup_logger("aggregate_results")

COLUMNS = [
    "run_id", "session_id", "status", "model", "task", "runtime", "provider", "precision", "batch_size",
    "input_shape", "timestamp", "model_size_mb", "model_size_source",
    "p50_model_ms", "p50_model_ci_low_ms", "p50_model_ci_high_ms", "p90_model_ms", "p95_model_ms", "p99_model_ms",
    "model_throughput_fps",
    "p50_e2e_ms", "p90_e2e_ms", "p95_e2e_ms", "p99_e2e_ms", "e2e_throughput_fps",
    "stage_preprocess_ms", "stage_inference_ms", "stage_postprocess_ms",
    "peak_vram_mb", "peak_vram_reserved_mb", "process_rss_mb",
    "max_abs_error", "max_rel_error", "cosine_similarity", "numerical_gate",
    "quality_metric", "quality_value", "quality_reference", "quality_delta", "quality_passed",
    "stable", "cv_p50", "cv_p50_mad_filtered", "total_sessions",
    "speedup_model", "speedup_e2e", "speedup_vs_runtime_fp32", "storage_compression_pct",
]


def list_sessions(raw_results_dir: Union[str, Path], include_quick: bool = False) -> List[Path]:
    """Completed session directories, oldest first."""
    raw = Path(raw_results_dir)
    out = []
    for d in sorted(p for p in raw.glob("session_*") if p.is_dir()):
        meta = d / "session.json"
        if not meta.is_file():
            continue
        if not include_quick and json.loads(meta.read_text(encoding="utf-8")).get("quick", False):
            continue
        out.append(d)
    return out


def load_session_records(session_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(Path(session_dir).glob("*/run.json"))]


def flatten_record(data: Dict[str, Any]) -> Dict[str, Any]:
    mm, rm = data["model_metadata"], data["runtime_metadata"]
    lat, e2e = data["model_path_latency_ms"], data["end_to_end_latency_ms"]
    mem, num, qual, stab = data["memory_utilization"], data["numerical_checks"], data["quality_audit"], data["stability_audit"]
    return {
        "run_id": data["run_id"],
        "session_id": data["session_id"],
        "status": data["status"],
        "model": mm["model"],
        "task": mm["task"],
        "runtime": rm["runtime"],
        "provider": rm["provider"],
        "precision": mm["precision"],
        "batch_size": mm["batch_size"],
        "input_shape": str(mm["input_shape"]),
        "timestamp": data["timestamp"],
        "model_size_mb": mm["model_size_mb"],
        "model_size_source": mm["model_size_source"],
        "p50_model_ms": lat["p50_ms"],
        "p50_model_ci_low_ms": lat["p50_ci95_ms"][0],
        "p50_model_ci_high_ms": lat["p50_ci95_ms"][1],
        "p90_model_ms": lat["p90_ms"],
        "p95_model_ms": lat["p95_ms"],
        "p99_model_ms": lat["p99_ms"],
        "model_throughput_fps": lat["throughput_fps"],
        "p50_e2e_ms": e2e["p50"],
        "p90_e2e_ms": e2e["p90"],
        "p95_e2e_ms": e2e["p95"],
        "p99_e2e_ms": e2e["p99"],
        "e2e_throughput_fps": e2e["throughput_fps"],
        "stage_preprocess_ms": e2e["stages_p50"]["preprocess"],
        "stage_inference_ms": e2e["stages_p50"]["inference"],
        "stage_postprocess_ms": e2e["stages_p50"]["postprocess"],
        "peak_vram_mb": mem["peak_vram_allocated_mb"],
        "peak_vram_reserved_mb": mem["peak_vram_reserved_mb"],
        "process_rss_mb": mem["process_rss_mb"],
        "max_abs_error": num["max_abs_error"],
        "max_rel_error": num["max_rel_error"],
        "cosine_similarity": num["cosine_similarity"],
        "numerical_gate": bool(num["passed"]),
        "quality_metric": qual["metric"],
        "quality_value": qual["value"],
        "quality_reference": qual["reference_value"],
        "quality_delta": qual["delta"],
        "quality_passed": bool(qual["passed"]),
        "stable": bool(stab["stable"]),
        "cv_p50": stab["cv_p50"],
        "cv_p50_mad_filtered": stab["cv_p50_mad_filtered"],
        "total_sessions": stab["total_sessions"],
    }


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Speedups vs PyTorch FP32 CPU and vs the same runtime's FP32 artifact; storage reduction vs FP32 file."""
    df = df.copy()
    for col in ("speedup_model", "speedup_e2e", "speedup_vs_runtime_fp32", "storage_compression_pct"):
        df[col] = float("nan")
    for model, g in df.groupby("model"):
        base = g[(g["runtime"] == "PyTorch") & (g["precision"] == "fp32") & g["provider"].str.contains("CPU")]
        if not base.empty:
            df.loc[g.index, "speedup_model"] = base["p50_model_ms"].iloc[0] / g["p50_model_ms"]
            df.loc[g.index, "speedup_e2e"] = base["p50_e2e_ms"].iloc[0] / g["p50_e2e_ms"]
        for runtime, gr in g.groupby("runtime"):
            fp32 = gr[gr["precision"] == "fp32"]
            if not fp32.empty:
                df.loc[gr.index, "speedup_vs_runtime_fp32"] = fp32["p50_model_ms"].iloc[0] / gr["p50_model_ms"]
        onnx_fp32 = g[(g["precision"] == "fp32") & (g["model_size_source"] == "serialized_file")]
        if not onnx_fp32.empty:
            ref = onnx_fp32["model_size_mb"].iloc[0]
            files = g["model_size_source"] == "serialized_file"
            df.loc[g.index[files.values], "storage_compression_pct"] = (1.0 - g.loc[files, "model_size_mb"] / ref) * 100.0
    return df


def aggregate_benchmark_runs(
    raw_results_dir: Union[str, Path],
    output_csv_path: Optional[Union[str, Path]] = None,
    session_id: Optional[str] = None,
    include_quick: bool = False,
) -> pd.DataFrame:
    """
    Builds the per-configuration table for one session.

    Args:
        raw_results_dir: results/raw directory containing session_* folders.
        output_csv_path: Optional CSV destination.
        session_id: Session to aggregate (default: latest completed session).
        include_quick: Allow selecting --quick smoke sessions.
    """
    raw = Path(raw_results_dir)
    if session_id:
        session_dir = raw / session_id
        if not session_dir.is_dir():
            raise FileNotFoundError(f"Session not found: {session_dir}")
    else:
        sessions = list_sessions(raw, include_quick=include_quick)
        if not sessions:
            raise FileNotFoundError(f"No completed benchmark session under {raw}. Run scripts/benchmark_all.py first.")
        session_dir = sessions[-1]

    records = load_session_records(session_dir)
    if not records:
        raise FileNotFoundError(f"Session {session_dir.name} contains no run records")
    df = add_derived_columns(pd.DataFrame([flatten_record(r) for r in records]))
    order = {"PyTorch": 0, "ORT_CPU": 1, "ORT_CUDA": 2, "TensorRT": 3}
    prec_order = {"fp32": 0, "fp16": 1, "int8": 2}
    df = df.assign(_m=df["model"].map({"yolo_nano": 0}).fillna(1), _r=df["runtime"].map(order),
                   _cpu=~df["provider"].str.contains("CPU"), _p=df["precision"].map(prec_order))
    df = df.sort_values(["_m", "_r", "_cpu", "_p"]).drop(columns=["_m", "_r", "_cpu", "_p"]).reset_index(drop=True)
    df = df[COLUMNS]

    if output_csv_path:
        p = Path(output_csv_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False, float_format="%.6g")
        logger.info(f"Aggregated {len(df)} runs of {session_dir.name} -> {p}")
    return df
