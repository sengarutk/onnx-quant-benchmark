"""
Markdown report tables generated from the aggregated session DataFrame (results/runs.csv).
Gate verdicts are read from the run records; nothing is re-thresholded here.
"""

from pathlib import Path
from typing import Callable, List
import pandas as pd


def _fmt(v: float, spec: str) -> str:
    return "n/a" if pd.isna(v) else format(float(v), spec)


def _write(headers: List[str], rows: List[List[str]], output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]
    lines += [f"| {' | '.join(r)} |" for r in rows]
    content = "\n".join(lines) + "\n"
    output_path.write_text(content, encoding="utf-8")
    return content


def _cfg(r: pd.Series) -> List[str]:
    return [str(r["model"]), str(r["runtime"]), str(r["provider"]), str(r["precision"]).upper()]


def generate_table1_numerical_correctness(df: pd.DataFrame, output_path: Path) -> str:
    """Table 1: output parity vs PyTorch FP32 CPU (all output tensors, real test image)."""
    headers = ["Model", "Runtime", "Provider", "Precision", "Max abs. error", "Max rel. error", "Min cosine sim.", "Parity gate"]
    rows = [_cfg(r) + [_fmt(r["max_abs_error"], ".3e"), _fmt(r["max_rel_error"], ".3e"), _fmt(r["cosine_similarity"], ".6f"),
                       "PASS" if r["numerical_gate"] else "FAIL"] for _, r in df.iterrows()]
    return _write(headers, rows, output_path)


def generate_table2_latency_throughput(df: pd.DataFrame, output_path: Path) -> str:
    """Table 2: batch-1 latency (pooled over sessions) with bootstrap CI and stability verdict."""
    headers = ["Model", "Runtime", "Provider", "Precision", "Model p50 (ms) [95% CI]", "Model p95 (ms)", "Model p99 (ms)",
               "E2E p50 (ms)", "Model FPS (1/p50)", "Session CV", "Stable (CV<=0.05)"]
    rows = [_cfg(r) + [f"{r['p50_model_ms']:.3f} [{r['p50_model_ci_low_ms']:.3f}, {r['p50_model_ci_high_ms']:.3f}]",
                       _fmt(r["p95_model_ms"], ".3f"), _fmt(r["p99_model_ms"], ".3f"), _fmt(r["p50_e2e_ms"], ".2f"),
                       _fmt(r["model_throughput_fps"], ".1f"), _fmt(r["cv_p50"], ".3f"), "yes" if r["stable"] else "no"]
            for _, r in df.iterrows()]
    return _write(headers, rows, output_path)


def generate_table3_memory_footprint(df: pd.DataFrame, output_path: Path) -> str:
    """Table 3: memory and artifact size."""
    headers = ["Model", "Runtime", "Provider", "Precision", "Peak PyTorch VRAM alloc. (MB)", "Process RSS (MB)",
               "Model size (MB)", "Size source", "Size vs FP32 ONNX"]
    rows = []
    for _, r in df.iterrows():
        vram = _fmt(r["peak_vram_mb"], ".2f") if r["runtime"] == "PyTorch" and "CUDA" in str(r["provider"]) else "n/a"
        red = "n/a" if pd.isna(r["storage_compression_pct"]) else f"{-r['storage_compression_pct']:+.1f}%"
        rows.append(_cfg(r) + [vram, _fmt(r["process_rss_mb"], ".0f"), _fmt(r["model_size_mb"], ".2f"),
                               str(r["model_size_source"]).replace("_", " "), red])
    return _write(headers, rows, output_path)


def generate_table4_quality_retention(df: pd.DataFrame, output_path: Path) -> str:
    """Table 4: task quality measured with each runtime on the full test split."""
    headers = ["Model", "Runtime", "Provider", "Precision", "Metric", "Value", "FP32 reference", "Delta", "Quality gate"]
    rows = [_cfg(r) + [str(r["quality_metric"]), _fmt(r["quality_value"], ".4f"), _fmt(r["quality_reference"], ".4f"),
                       _fmt(r["quality_delta"], "+.4f"), "PASS" if r["quality_passed"] else "FAIL"] for _, r in df.iterrows()]
    return _write(headers, rows, output_path)


def generate_table5_int8_audit(df: pd.DataFrame, output_path: Path, quant_method: str = "Static QDQ, MinMax, symmetric, per-channel weights") -> str:
    """Table 5: reduced-precision artifacts vs the same runtime's FP32."""
    headers = ["Model", "Runtime", "Precision", "Method", "Size vs FP32 ONNX", "Speedup vs same-runtime FP32",
               "Quality delta", "Gates"]
    rows = []
    for _, r in df[df["precision"].isin(["fp16", "int8"])].iterrows():
        if r["precision"] == "int8":
            method = quant_method
        elif r["runtime"] == "PyTorch":
            method = "PyTorch `model.half()`"
        else:
            method = "FP16 ONNX graph (onnxconverter-common)"
        red = "n/a" if pd.isna(r["storage_compression_pct"]) else f"{-r['storage_compression_pct']:+.1f}%"
        gates = "PASS" if (r["numerical_gate"] and r["quality_passed"]) else "FAIL"
        rows.append([str(r["model"]), f"{r['runtime']} ({r['provider']})", str(r["precision"]).upper(), method, red,
                     _fmt(r["speedup_vs_runtime_fp32"], ".2f") + "x", _fmt(r["quality_delta"], "+.4f"), gates])
    if not rows:
        rows.append(["n/a"] * len(headers))
    return _write(headers, rows, output_path)


def generate_all_tables(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates tables 1-5 in output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_table1_numerical_correctness(df, output_dir / "table1_numerical_correctness.md")
    generate_table2_latency_throughput(df, output_dir / "table2_latency_throughput.md")
    generate_table3_memory_footprint(df, output_dir / "table3_memory_footprint.md")
    generate_table4_quality_retention(df, output_dir / "table4_quality_retention.md")
    generate_table5_int8_audit(df, output_dir / "table5_int8_quantization_audit.md")
