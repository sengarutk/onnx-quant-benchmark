"""
Publication Markdown Table Reporting Subsystem.
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd


def generate_table1_numerical_correctness(df: pd.DataFrame, output_path: Path) -> str:
    """Generates Table 1: Numerical Correctness & Equivalence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = ["Model", "Runtime", "Provider", "Precision", "Max Abs Error ($L_\\infty$)", "Mean Abs Error ($L_1$)", "Cosine Sim", "Quality Delta", "Gate Status"]
    rows: List[str] = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]

    for _, r in df.iterrows():
        precision_lower = str(r.get("precision", "")).lower()
        max_abs = float(r.get("max_abs_error", 0.0))
        cos_sim = float(r.get("cosine_similarity", 1.0))
        qual_delta = abs(float(r.get("quality_delta", 0.0)))

        if "int8" in precision_lower:
            # INT8 passes if task quality drop <= 0.015 and cosine similarity >= 0.99
            passed = (qual_delta <= 0.015) and (cos_sim >= 0.99) and (max_abs <= 0.50)
        elif "fp16" in precision_lower:
            passed = (max_abs <= 0.05) and (cos_sim >= 0.999) and (qual_delta <= 0.005)
        else:
            # FP32 strict numerical equivalence
            passed = (max_abs <= 0.02) and (cos_sim >= 0.9999) and (qual_delta <= 0.0005)

        status_icon = "✅ PASS" if passed else "❌ FAIL"
        row_str = (
            f"| {r.get('model', 'N/A')} "
            f"| {r.get('runtime', 'N/A')} "
            f"| {r.get('provider', 'N/A')} "
            f"| {str(r.get('precision', 'N/A')).upper()} "
            f"| {max_abs:.4e} "
            f"| {float(r.get('mean_abs_error', 0.0)):.4e} "
            f"| {cos_sim:.6f} "
            f"| {float(r.get('quality_delta', 0.0)):+.4f} "
            f"| {status_icon} |"
        )
        rows.append(row_str)

    content = "\n".join(rows)
    output_path.write_text(content, encoding="utf-8")
    return content


def generate_table2_latency_throughput(df: pd.DataFrame, output_path: Path) -> str:
    """Generates Table 2: Batch-1 Latency & Throughput."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = ["Model", "Runtime", "Provider", "Precision", "Model $p_{50}$ (ms)", "Model $p_{90}$ (ms)", "Model $p_{95}$ (ms)", "Model $p_{99}$ (ms)", "E2E $p_{50}$ (ms)", "Model FPS", "E2E FPS"]
    rows: List[str] = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]

    for _, r in df.iterrows():
        row_str = (
            f"| {r.get('model', 'N/A')} "
            f"| {r.get('runtime', 'N/A')} "
            f"| {r.get('provider', 'N/A')} "
            f"| {str(r.get('precision', 'N/A')).upper()} "
            f"| {float(r.get('p50_model_ms', 0.0)):.2f} "
            f"| {float(r.get('p90_model_ms', 0.0)):.2f} "
            f"| {float(r.get('p95_model_ms', 0.0)):.2f} "
            f"| {float(r.get('p99_model_ms', 0.0)):.2f} "
            f"| {float(r.get('p50_e2e_ms', 0.0)):.2f} "
            f"| {float(r.get('model_throughput_fps', 0.0)):.1f} "
            f"| {float(r.get('e2e_throughput_fps', 0.0)):.1f} |"
        )
        rows.append(row_str)

    content = "\n".join(rows)
    output_path.write_text(content, encoding="utf-8")
    return content


def generate_table3_memory_footprint(df: pd.DataFrame, output_path: Path) -> str:
    """Generates Table 3: Memory Allocations & Storage Footprint."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = ["Model", "Runtime", "Provider", "Precision", "Peak VRAM (MB)", "Peak Reserved VRAM (MB)", "Process RSS (MB)", "Model Disk Size (MB)", "Storage Red. %"]
    rows: List[str] = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]

    for _, r in df.iterrows():
        row_str = (
            f"| {r.get('model', 'N/A')} "
            f"| {r.get('runtime', 'N/A')} "
            f"| {r.get('provider', 'N/A')} "
            f"| {str(r.get('precision', 'N/A')).upper()} "
            f"| {float(r.get('peak_vram_mb', 0.0)):.2f} "
            f"| {float(r.get('peak_vram_reserved_mb', 0.0)):.2f} "
            f"| {float(r.get('process_rss_mb', 0.0)):.2f} "
            f"| {float(r.get('model_size_mb', 0.0)):.2f} "
            f"| {float(r.get('storage_compression_pct', 0.0)):.1f}% |"
        )
        rows.append(row_str)

    content = "\n".join(rows)
    output_path.write_text(content, encoding="utf-8")
    return content


def generate_table4_quality_retention(df: pd.DataFrame, output_path: Path) -> str:
    """Generates Table 4: Precision & Task Quality Retention."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = ["Model", "Runtime", "Provider", "Precision", "Metric", "Metric Value", "Quality $\\Delta$", "Verdict"]
    rows: List[str] = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]

    for _, r in df.iterrows():
        passed = bool(r.get("quality_passed", True))
        verdict = "✅ ACCEPTABLE" if passed else "❌ DEGRADED"
        row_str = (
            f"| {r.get('model', 'N/A')} "
            f"| {r.get('runtime', 'N/A')} "
            f"| {r.get('provider', 'N/A')} "
            f"| {str(r.get('precision', 'N/A')).upper()} "
            f"| {r.get('quality_metric', 'N/A')} "
            f"| {float(r.get('quality_value', 0.0)):.4f} "
            f"| {float(r.get('quality_delta', 0.0)):+.4f} "
            f"| {verdict} |"
        )
        rows.append(row_str)

    content = "\n".join(rows)
    output_path.write_text(content, encoding="utf-8")
    return content


def generate_table5_int8_audit(df: pd.DataFrame, output_path: Path) -> str:
    """Generates Table 5: Static INT8 Quantization Audit."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    headers = ["Model", "Runtime", "Provider", "Calibration Method", "Size Reduction %", "Compute Speedup", "Quality Delta", "Deployable"]
    rows: List[str] = [f"| {' | '.join(headers)} |", f"| {' | '.join(['---'] * len(headers))} |"]

    int8_df = df[df["precision"] == "int8"].copy()
    if int8_df.empty:
        rows.append("| N/A | No INT8 models found | N/A | N/A | 0.0% | 1.0x | 0.0 | N/A |")
    else:
        for _, r in int8_df.iterrows():
            speedup = float(r.get("speedup_model", 1.0))
            delta = float(r.get("quality_delta", 0.0))
            deployable = "✅ YES" if abs(delta) <= 0.05 else "⚠️ EVALUATE"
            row_str = (
                f"| {r.get('model', 'N/A')} "
                f"| {r.get('runtime', 'N/A')} "
                f"| {r.get('provider', 'N/A')} "
                f"| MinMax Symmetric "
                f"| {float(r.get('storage_compression_pct', 0.0)):.1f}% "
                f"| {speedup:.1f}x "
                f"| {delta:+.4f} "
                f"| {deployable} |"
            )
            rows.append(row_str)

    content = "\n".join(rows)
    output_path.write_text(content, encoding="utf-8")
    return content


def generate_all_tables(df: pd.DataFrame, output_dir: Path) -> None:
    """Generates all 5 Markdown report tables in results/tables/."""
    output_dir.mkdir(parents=True, exist_ok=True)
    generate_table1_numerical_correctness(df, output_dir / "table1_numerical_correctness.md")
    generate_table2_latency_throughput(df, output_dir / "table2_latency_throughput.md")
    generate_table3_memory_footprint(df, output_dir / "table3_memory_footprint.md")
    generate_table4_quality_retention(df, output_dir / "table4_quality_retention.md")
    generate_table5_int8_audit(df, output_dir / "table5_int8_quantization_audit.md")
