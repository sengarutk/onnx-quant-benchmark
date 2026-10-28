"""
Scalability sweep: batch size {1, 2, 4, 8} x input resolution {320, 416, 512, 640} for the trained detector
in PyTorch FP32 (CUDA when available, otherwise CPU).

Each grid point is measured in ``repeats`` independent sessions; the reported latency percentiles are pooled over
all sessions and throughput is batch / p50. Output: results/scalability_sweep.csv and
results/figures/scalability_batch_resolution.png.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking.memory import MemoryProfiler
from src.benchmarking.timer import ModelPathTimer
from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.models.yolo_adapter import YOLOAdapter
from src.runtimes.pytorch_runtime import PyTorchRuntime
from src.visualization.style import TEXT_WIDTH_IN

logger = setup_logger("run_scalability_sweep")


def run_scalability_sweeps(
    resolutions: Optional[List[int]] = None,
    batch_sizes: Optional[List[int]] = None,
    warmup_iters: int = 30,
    timed_iters: int = 200,
    repeats: int = 3,
    out_csv: Optional[Path] = None,
    fig_dir: Optional[Path] = None,
    save_artifacts: bool = True,
    device: Optional[str] = None,
) -> pd.DataFrame:
    resolutions = resolutions or [320, 416, 512, 640]
    batch_sizes = batch_sizes or [1, 2, 4, 8]
    seed_everything(42)
    device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    adapter = YOLOAdapter()
    runtime = PyTorchRuntime(adapter.get_pytorch_model(), device=device, precision="fp32")
    runtime.load()
    profiler = MemoryProfiler()
    timer = ModelPathTimer(warmup_iterations=warmup_iters, timed_iterations=timed_iters, device=device)
    dev_name = torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU"
    logger.info(f"Scalability sweep on {dev_name} (trained weights: {adapter.weights_loaded})")

    records = []
    for res in resolutions:
        for bs in batch_sizes:
            x = torch.randn(bs, 3, res, res, device=device)
            profiler.start_tracking(is_cuda=device.startswith("cuda"))
            sessions = [timer.benchmark_device(runtime, x) for _ in range(repeats)]
            mem = profiler.stop_tracking()
            pooled = np.concatenate([s["latencies_ms"] for s in sessions])
            p50 = float(np.percentile(pooled, 50))
            session_p50 = [s["p50_ms"] for s in sessions]
            records.append({
                "device": dev_name,
                "resolution": res,
                "batch_size": bs,
                "input_dim": f"{bs}x3x{res}x{res}",
                "p50_ms": round(p50, 4),
                "p90_ms": round(float(np.percentile(pooled, 90)), 4),
                "p95_ms": round(float(np.percentile(pooled, 95)), 4),
                "p50_per_image_ms": round(p50 / bs, 4),
                "throughput_fps": round(bs * 1000.0 / p50, 1),
                "session_cv": round(float(np.std(session_p50) / np.mean(session_p50)), 4),
                "peak_vram_mb": mem["peak_vram_allocated_mb"],
                "samples": int(pooled.size),
            })
            logger.info(f"bs={bs} res={res}: p50 {p50:.3f} ms, {records[-1]['throughput_fps']} FPS")

    df = pd.DataFrame(records)
    if save_artifacts:
        csv_dest = Path(out_csv) if out_csv else PROJECT_ROOT / "results" / "scalability_sweep.csv"
        csv_dest.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_dest, index=False)

        plot_scalability(df, Path(fig_dir) if fig_dir else PROJECT_ROOT / "results" / "figures")
        logger.info(f"Saved {csv_dest} and scalability figure")

    runtime.cleanup()
    return df


def plot_scalability(df: pd.DataFrame, fig_dir: Path) -> Path:
    """Latency vs resolution and throughput vs batch size from a sweep table (results/scalability_sweep.csv)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    resolutions = sorted(df["resolution"].unique())
    batch_sizes = sorted(df["batch_size"].unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.7))
    for bs in batch_sizes:
        sub = df[df["batch_size"] == bs]
        ax1.plot(sub["resolution"], sub["p50_ms"], marker="o", label=f"Batch {bs}")
    ax1.set_title("Batch latency p50 vs. input resolution", fontweight="bold")
    ax1.set_xlabel("Input resolution (square, px)")
    ax1.set_ylabel("Model-path p50 per batch (ms)")
    ax1.set_xticks(resolutions)
    ax1.grid(True, ls=":", alpha=0.6)
    ax1.legend()
    for res in resolutions:
        sub = df[df["resolution"] == res]
        ax2.plot(sub["batch_size"], sub["throughput_fps"], marker="s", label=f"{res}x{res}")
    ax2.set_title("Throughput (batch / p50) vs. batch size", fontweight="bold")
    ax2.set_xlabel("Batch size")
    ax2.set_ylabel("Images per second")
    ax2.set_xticks(batch_sizes)
    ax2.grid(True, ls=":", alpha=0.6)
    ax2.legend()
    fig.suptitle(f"Detector scalability, PyTorch FP32 on {df['device'].iloc[0]}", fontweight="bold")
    fig.tight_layout()
    path = fig_dir / "scalability_batch_resolution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None)
    parser.add_argument("--plot-only", action="store_true", help="Redraw the figure from results/scalability_sweep.csv")
    args = parser.parse_args()
    if args.plot_only:
        plot_scalability(pd.read_csv(PROJECT_ROOT / "results" / "scalability_sweep.csv"), PROJECT_ROOT / "results" / "figures")
    else:
        run_scalability_sweeps(device=args.device)
