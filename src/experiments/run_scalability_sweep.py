"""
Scalability Profiling Sweep across Batch Sizes (1, 2, 4, 8) and Input Resolutions (320, 416, 512, 640).
Measures latency scaling, throughput (FPS), and memory footprint.
"""

from pathlib import Path
import csv
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.memory import MemoryProfiler
from src.benchmarking.throughput import compute_throughput
from src.benchmarking.timer import ModelPathTimer
from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.models.yolo_adapter import YOLONanoDetector
from src.runtimes.pytorch_runtime import PyTorchRuntime

logger = setup_logger("run_scalability_sweep")


def run_scalability_sweeps(
    resolutions: list = [320, 416, 512, 640],
    batch_sizes: list = [1, 2, 4, 8],
    warmup_iters: int = 15,
    timed_iters: int = 40,
    out_csv: Path = None,
    fig_dir: Path = None,
    save_artifacts: bool = True,
) -> pd.DataFrame:
    """
    Executes 2D scalability grid sweep over resolutions and batch dimensions.
    """
    seed_everything(42)
    logger.info(f"Starting scalability sweep: Resolutions={resolutions}, Batch Sizes={batch_sizes}...")

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)

    model = YOLONanoDetector(num_classes=80).to(device)
    model.eval()
    runtime = PyTorchRuntime(model, device=device_str, precision="fp32")
    profiler = MemoryProfiler()
    timer = ModelPathTimer(warmup_iterations=warmup_iters, timed_iterations=timed_iters)

    records = []

    for res in resolutions:
        for bs in batch_sizes:
            logger.info(f"Profiling Batch Size={bs}, Resolution={res}x{res} on {device_str.upper()}...")
            dummy_in = torch.randn(bs, 3, res, res, dtype=torch.float32, device=device)

            profiler.start_tracking(is_cuda=(device_str == "cuda"))
            timing_stats = timer.benchmark_device(runtime, dummy_in)
            mem_stats = profiler.stop_tracking()

            p50 = timing_stats["p50_ms"]
            p90 = timing_stats["p90_ms"]
            p95 = timing_stats["p95_ms"]
            fps = compute_throughput(p50, batch_size=bs)

            records.append({
                "resolution": res,
                "batch_size": bs,
                "input_dim": f"{bs}x3x{res}x{res}",
                "p50_ms": round(p50, 3),
                "p90_ms": round(p90, 3),
                "p95_ms": round(p95, 3),
                "throughput_fps": round(fps, 1),
                "peak_vram_mb": mem_stats["peak_vram_allocated_mb"],
                "process_rss_mb": mem_stats["process_rss_mb"],
            })

    df = pd.DataFrame(records)
    if save_artifacts:
        csv_dest = Path(out_csv) if out_csv else PROJECT_ROOT / "results" / "scalability_sweep.csv"
        csv_dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_dest, index=False)
        logger.info(f"Saved scalability sweep CSV -> {csv_dest}")

        # Render Publication Figure: Scalability Batch & Resolution Sweep
        target_fig_dir = Path(fig_dir) if fig_dir else PROJECT_ROOT / "results" / "figures"
        target_fig_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.0, 5.0), dpi=300)

        # Subplot 1: Latency (p50) vs Resolution per Batch Size
        for bs in batch_sizes:
            sub_df = df[df["batch_size"] == bs]
            ax1.plot(sub_df["resolution"], sub_df["p50_ms"], marker="o", lw=2.0, label=f"Batch Size = {bs}")

        ax1.set_title(r"Inference Latency ($p_{50}$) Scaling vs. Resolution", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Input Resolution (Square Pixels)", fontsize=10, fontweight="bold")
        ax1.set_ylabel(r"Model Latency $p_{50}$ (ms)", fontsize=10, fontweight="bold")
        ax1.set_xticks(resolutions)
        ax1.legend(frameon=True, fontsize=9)

        # Subplot 2: Throughput (FPS) vs Batch Size per Resolution
        for res in resolutions:
            sub_df = df[df["resolution"] == res]
            ax2.plot(sub_df["batch_size"], sub_df["throughput_fps"], marker="s", lw=2.0, label=f"Res {res}x{res}")

        ax2.set_title("System Throughput (FPS) vs. Batch Dimension", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Inference Batch Size", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Throughput (Frames Per Second)", fontsize=10, fontweight="bold")
        ax2.set_xticks(batch_sizes)
        ax2.legend(frameon=True, fontsize=9)

        plt.tight_layout()
        fig_path = target_fig_dir / "scalability_batch_resolution.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        logger.info(f"Generated Scalability Figure -> {fig_path}")

    return df


if __name__ == "__main__":
    run_scalability_sweeps()
