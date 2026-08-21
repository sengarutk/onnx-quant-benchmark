"""
Device & Process Memory Profiler (VRAM Peak, Host RSS, NVML GPU tracking, Artifact Footprint).
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import psutil
import torch

try:
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import pynvml
    NVML_AVAILABLE = True
except ImportError:
    try:
        import nvidia_smi as pynvml
        NVML_AVAILABLE = True
    except ImportError:
        NVML_AVAILABLE = False
        pynvml = None


def get_artifact_size_mb(file_path: Union[str, Path]) -> float:
    """Returns file size in megabytes."""
    p = Path(file_path)
    if not p.is_file():
        return 0.0
    return float(p.stat().st_size / (1024.0 * 1024.0))


class MemoryProfiler:
    """
    Monitors process RSS, PyTorch CUDA allocations, and NVML device memory.
    """

    def __init__(self) -> None:
        self.process = psutil.Process(os.getpid())
        self.baseline_rss_mb = 0.0
        self.is_cuda = False

    def start_tracking(self, is_cuda: bool = False, device_id: int = 0) -> None:
        """Resets peak statistics and takes baseline RSS snapshot."""
        self.is_cuda = is_cuda and torch.cuda.is_available()
        self.baseline_rss_mb = float(self.process.memory_info().rss / (1024.0 * 1024.0))
        if self.is_cuda:
            torch.cuda.reset_peak_memory_stats(device_id)

    def stop_tracking(self, device_id: int = 0) -> Dict[str, float]:
        """
        Collects memory utilization metrics.

        Returns:
            Dictionary with peak_vram_allocated_mb, peak_vram_reserved_mb, process_rss_mb, nvml_gpu_memory_used_mb.
        """
        current_rss_mb = float(self.process.memory_info().rss / (1024.0 * 1024.0))

        if self.is_cuda:
            peak_alloc = float(torch.cuda.max_memory_allocated(device_id) / (1024.0 * 1024.0))
            peak_res = float(torch.cuda.max_memory_reserved(device_id) / (1024.0 * 1024.0))
            nvml_used_mb = 0.0
            if NVML_AVAILABLE:
                try:
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    nvml_used_mb = float(mem_info.used / (1024.0 * 1024.0))
                    pynvml.nvmlShutdown()
                except Exception:
                    nvml_used_mb = peak_alloc
        else:
            peak_alloc = 0.0
            peak_res = 0.0
            nvml_used_mb = 0.0

        return {
            "peak_vram_allocated_mb": round(peak_alloc, 2),
            "peak_vram_reserved_mb": round(peak_res, 2),
            "process_rss_mb": round(current_rss_mb, 2),
            "nvml_gpu_memory_used_mb": round(nvml_used_mb, 2),
        }
