"""
Publication figures (300 DPI) generated from one benchmark session. Every plotted quantity is a measured value
from results/runs.csv or the session's run records.
"""

from pathlib import Path
from typing import Any, Dict, List
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.pareto import identify_pareto_frontier
from src.common.logging import setup_logger

logger = setup_logger("plots")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
})

MODELS = [("yolo_nano", "Detector (YOLO-Nano-style)", "mAP@50"),
          ("industrial_autoencoder", "Conv. autoencoder", "Image AUROC")]
RUNTIME_COLORS = {"PyTorch": "#1f77b4", "ORT_CPU": "#ff7f0e", "ORT_CUDA": "#2ca02c", "TensorRT": "#d62728"}
PRECISION_MARKERS = {"fp32": "o", "fp16": "^", "int8": "s"}


def config_label(r: pd.Series, with_model: bool = False) -> str:
    dev = "GPU" if any(k in str(r["provider"]).upper() for k in ("CUDA", "TENSORRT")) else "CPU"
    rt = {"PyTorch": "PyTorch", "ORT_CPU": "ORT", "ORT_CUDA": "ORT", "TensorRT": "TensorRT"}.get(r["runtime"], r["runtime"])
    lbl = f"{rt} {dev} {str(r['precision']).upper()}"
    return f"{'YOLO' if r['model'] == 'yolo_nano' else 'AE'} | {lbl}" if with_model else lbl


def _empty(fig_path: Path, msg: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.text(0.5, 0.5, msg, ha="center", va="center")
    ax.axis("off")
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return fig_path


def plot_pareto_frontier(df: pd.DataFrame, output_dir: Path) -> Path:
    """Task quality vs model-path p50 latency (95% CI error bars) with the non-dominated frontier."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "pareto_quality_vs_latency.png"
    if df.empty:
        return _empty(fig_path, "No benchmark data available")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    for ax, (model, title, metric) in zip(axes, MODELS):
        sub = df[df["model"] == model]
        if sub.empty:
            ax.set_title(title)
            continue
        for _, r in sub.iterrows():
            x, y = float(r["p50_model_ms"]), float(r["quality_value"])
            xerr = [[x - r["p50_model_ci_low_ms"]], [r["p50_model_ci_high_ms"] - x]]
            ax.errorbar(x, y, xerr=xerr, fmt=PRECISION_MARKERS.get(r["precision"], "o"), ms=8,
                        color=RUNTIME_COLORS.get(r["runtime"], "#333"), mec="black", capsize=3, zorder=4)
            ax.annotate(config_label(r), (x, y), textcoords="offset points", xytext=(5, 6), fontsize=7.5)
        front = identify_pareto_frontier(sub, "p50_model_ms", "quality_value", minimize_x=True, maximize_y=True)
        if len(front) > 1:
            ax.plot(front["p50_model_ms"], front["quality_value"], "--", color="#e74c3c", lw=1.5, label="Pareto frontier", zorder=3)
            ax.legend(loc="lower right")
        ax.set_xscale("log")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Model-path latency p50 (ms, log scale)")
        ax.set_ylabel(f"{metric} (test split)")
        ax.grid(True, ls=":", alpha=0.6)
    fig.suptitle("Task quality vs. latency (batch 1; markers: o FP32, ^ FP16, s INT8)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return fig_path


def plot_latency_breakdown(df: pd.DataFrame, output_dir: Path) -> Path:
    """Measured end-to-end stage medians: decode+preprocess, inference (incl. host<->device copies), postprocess."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "latency_breakdown_stacked.png"
    if df.empty:
        return _empty(fig_path, "No data available")

    labels = [config_label(r, with_model=True) for _, r in df.iterrows()]
    pre, inf, post = df["stage_preprocess_ms"].to_numpy(), df["stage_inference_ms"].to_numpy(), df["stage_postprocess_ms"].to_numpy()
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(labels) + 1.5), dpi=300)
    ax.barh(y, pre, 0.6, label="Decode + preprocess", color="#3498db", edgecolor="black")
    ax.barh(y, inf, 0.6, left=pre, label="Inference (incl. H2D/D2H)", color="#2ecc71", edgecolor="black")
    ax.barh(y, post, 0.6, left=pre + inf, label="Postprocess (NMS / anomaly score)", color="#e67e22", edgecolor="black")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Per-stage median latency (ms)")
    ax.set_title("Measured end-to-end stage breakdown (batch 1)", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, ls=":", alpha=0.5, axis="x")
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return fig_path


def plot_speedup_barchart(df: pd.DataFrame, output_dir: Path) -> Path:
    """Model-path speedup relative to PyTorch FP32 on CPU, one panel per model."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "speedup_comparison.png"
    if df.empty:
        return _empty(fig_path, "No data available")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    for ax, (model, title, _) in zip(axes, MODELS):
        sub = df[df["model"] == model]
        labels = [config_label(r) for _, r in sub.iterrows()]
        vals = sub["speedup_model"].to_numpy(dtype=float)
        x = np.arange(len(labels))
        bars = ax.bar(x, vals, color=["#2ecc71" if v >= 1 else "#e74c3c" for v in vals], edgecolor="black")
        ax.axhline(1.0, color="gray", ls="--", lw=1.2)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}x", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel("Speedup vs PyTorch FP32 CPU (p50 ratio)")
        ax.grid(True, ls=":", alpha=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return fig_path


def plot_tail_latencies(df: pd.DataFrame, output_dir: Path) -> Path:
    """p95/p50 and p99/p50 tail ratios of the pooled model-path latencies."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "tail_latency_p50_p95.png"
    if df.empty:
        return _empty(fig_path, "No data available")

    labels = [config_label(r, with_model=True) for _, r in df.iterrows()]
    r95 = (df["p95_model_ms"] / df["p50_model_ms"]).to_numpy()
    r99 = (df["p99_model_ms"] / df["p50_model_ms"]).to_numpy()
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(labels) + 1.5), dpi=300)
    ax.barh(y - 0.2, r95, 0.4, label="p95 / p50", color="#9b59b6", edgecolor="black")
    ax.barh(y + 0.2, r99, 0.4, label="p99 / p50", color="#34495e", edgecolor="black")
    ax.axvline(1.0, color="gray", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Tail-to-median latency ratio (pooled over sessions)")
    ax.set_title("Model-path tail latency", fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, ls=":", alpha=0.5, axis="x")
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return fig_path


def plot_memory_footprints(df: pd.DataFrame, output_dir: Path) -> Path:
    """Serialized artifact size per precision and peak PyTorch CUDA allocation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "memory_vs_footprint.png"
    if df.empty:
        return _empty(fig_path, "No data available")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8), dpi=300)
    files = df[(df["model_size_source"] == "serialized_file") & (df["runtime"] == "ORT_CPU")]
    labels = [f"{'YOLO' if r['model'] == 'yolo_nano' else 'AE'} {str(r['precision']).upper()}" for _, r in files.iterrows()]
    bars = ax1.bar(np.arange(len(labels)), files["model_size_mb"], color="#34495e", edgecolor="black")
    for b, v in zip(bars, files["model_size_mb"]):
        ax1.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax1.set_xticks(np.arange(len(labels)))
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("ONNX file size (MB)")
    ax1.set_title("Serialized model size", fontweight="bold")
    ax1.grid(True, ls=":", alpha=0.5, axis="y")

    gpu = df[(df["runtime"] == "PyTorch") & df["provider"].str.contains("CUDA")]
    if gpu.empty:
        ax2.text(0.5, 0.5, "No PyTorch CUDA runs in this session", ha="center", va="center")
        ax2.axis("off")
    else:
        labels = [config_label(r, with_model=True) for _, r in gpu.iterrows()]
        bars = ax2.bar(np.arange(len(labels)), gpu["peak_vram_mb"], color="#e74c3c", edgecolor="black")
        for b, v in zip(bars, gpu["peak_vram_mb"]):
            ax2.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax2.set_xticks(np.arange(len(labels)))
        ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
        ax2.set_ylabel("Peak torch.cuda.max_memory_allocated (MB)")
        ax2.set_title("Peak PyTorch CUDA allocation", fontweight="bold")
        ax2.grid(True, ls=":", alpha=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return fig_path


def plot_stability_trends(records: List[Dict[str, Any]], output_dir: Path) -> Path:
    """Per-session median latency normalized by each configuration's median-of-sessions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = output_dir / "stability_variance_trends.png"
    if not records:
        return _empty(fig_path, "No multi-session stability data found")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300, sharey=True)
    cmap = plt.get_cmap("tab10")
    for ax, (model, title, _) in zip(axes, MODELS):
        recs = [r for r in records if r["model_metadata"]["model"] == model]
        max_sessions = 1
        for i, rec in enumerate(recs):
            vals = np.asarray(rec["stability_audit"]["session_p50_values"], dtype=float)
            max_sessions = max(max_sessions, len(vals))
            row = pd.Series({"model": model, "runtime": rec["runtime_metadata"]["runtime"],
                             "provider": rec["runtime_metadata"]["provider"], "precision": rec["model_metadata"]["precision"]})
            cv = rec["stability_audit"]["cv_p50"]
            ax.plot(np.arange(1, len(vals) + 1), vals / np.median(vals), marker="os^Dvx"[i % 6], color=cmap(i % 10),
                    lw=1.5, alpha=0.9, label=f"{config_label(row)} (CV {cv:.3f})")
        ax.axhspan(0.95, 1.05, color="gray", alpha=0.12, label="+/-5% band")
        ax.set_xticks(np.arange(1, max_sessions + 1))
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Measurement session")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(fontsize=7, loc="upper left", ncol=1)
    axes[0].set_ylabel("Session p50 / median of session p50s")
    fig.suptitle("Session-to-session latency stability", fontweight="bold")
    fig.tight_layout()
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return fig_path


def plot_all_figures(df: pd.DataFrame, records: List[Dict[str, Any]], output_dir: Path) -> List[Path]:
    """Generates the six session figures."""
    return [
        plot_pareto_frontier(df, output_dir),
        plot_latency_breakdown(df, output_dir),
        plot_speedup_barchart(df, output_dir),
        plot_tail_latencies(df, output_dir),
        plot_memory_footprints(df, output_dir),
        plot_stability_trends(records, output_dir),
    ]
