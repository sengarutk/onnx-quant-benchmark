"""
Hardware and software environment introspection module.
Captures comprehensive host system metadata, GPU topology, and runtime package versions.
"""

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import psutil
from pydantic import BaseModel, Field


class EnvironmentManifest(BaseModel):
    """Pydantic schema representing the complete host and runtime environment."""

    timestamp: str = Field(description="ISO-8601 UTC timestamp of collection")
    git_commit: str = Field(description="Git commit hash and working tree status")
    python_version: str = Field(description="Active Python interpreter version")
    os_info: str = Field(description="Operating system, platform, and kernel release")
    cpu_info: str = Field(description="CPU model and core/thread topology")
    ram_total_gb: float = Field(description="Total physical RAM in gigabytes")
    gpu_available: bool = Field(description="Flag indicating if a CUDA GPU is accessible")
    gpu_name: Optional[str] = Field(default=None, description="Primary GPU model name")
    gpu_vram_total_gb: Optional[float] = Field(default=None, description="Total GPU VRAM in GB")
    gpu_compute_capability: Optional[str] = Field(default=None, description="CUDA compute capability (e.g. 8.6)")
    nvidia_driver_version: Optional[str] = Field(default=None, description="NVIDIA display driver version")
    cuda_runtime_version: Optional[str] = Field(default=None, description="CUDA runtime version linked to PyTorch")
    cudnn_version: Optional[str] = Field(default=None, description="cuDNN library version")
    tensorrt_version: str = Field(default="NOT_INSTALLED", description="TensorRT library version")
    onnx_version: str = Field(default="NOT_INSTALLED", description="ONNX package version")
    onnxruntime_version: str = Field(default="NOT_INSTALLED", description="ONNX Runtime package version")
    torch_version: str = Field(default="NOT_INSTALLED", description="PyTorch package version")


def _get_git_commit() -> str:
    """Safely retrieves the short git commit hash and dirty status."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return f"{commit}-dirty" if status else commit
    except Exception:
        return "UNKNOWN"


def _get_cpu_info() -> str:
    """Extracts human-readable CPU model and topology."""
    cores_phys = psutil.cpu_count(logical=False) or 1
    cores_log = psutil.cpu_count(logical=True) or 1
    processor = platform.processor() or platform.machine()
    return f"{processor} ({cores_phys} physical cores, {cores_log} logical threads)"


def get_physical_core_count() -> int:
    """
    Detects the number of unique physical CPU cores available to the current process,
    accounting for cgroups/affinity masks and hyperthreading topology.
    """
    try:
        if hasattr(os, "sched_getaffinity"):
            affinity_cpus = os.sched_getaffinity(0)
            if affinity_cpus:
                physical_cores = set()
                for cpu_id in affinity_cpus:
                    core_id_file = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology/core_id")
                    pkg_id_file = Path(f"/sys/devices/system/cpu/cpu{cpu_id}/topology/physical_package_id")
                    if core_id_file.is_file():
                        core_id = core_id_file.read_text().strip()
                        pkg_id = pkg_id_file.read_text().strip() if pkg_id_file.is_file() else "0"
                        physical_cores.add((pkg_id, core_id))
                if physical_cores:
                    return len(physical_cores)
    except Exception:
        pass
    return psutil.cpu_count(logical=False) or os.cpu_count() or 1


def configure_cpu_threads(num_threads: Optional[int] = None) -> int:
    """
    Explicitly binds execution to physical core count using threadpoolctl and PyTorch native APIs,
    and suppresses OpenCV background worker thread thrashing.
    """
    cores = num_threads or get_physical_core_count()

    try:
        import threadpoolctl
        threadpoolctl.threadpool_limits(limits=cores)
    except Exception:
        pass

    try:
        import torch
        torch.set_num_threads(cores)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass

    try:
        import cv2
        cv2.setNumThreads(0)
    except Exception:
        pass

    return cores


def get_unified_ort_session_options(
    intra_op_threads: Optional[int] = None,
    inter_op_threads: int = 1,
    enable_profiling: bool = False,
) -> Any:
    """
    Creates and configures unified ONNX Runtime SessionOptions with deterministic thread pool
    and sequential execution mode across all benchmark backends.
    """
    import onnxruntime as ort
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    cores = intra_op_threads or get_physical_core_count()
    opts.intra_op_num_threads = cores
    opts.inter_op_num_threads = inter_op_threads
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.enable_profiling = enable_profiling
    return opts


def collect_environment_manifest() -> EnvironmentManifest:
    """
    Inspects and aggregates system hardware, OS details, GPU architecture, and runtime package versions.

    Returns:
        Populated EnvironmentManifest instance.
    """
    ts = datetime.now(timezone.utc).isoformat()
    git_hash = _get_git_commit()
    py_ver = sys.version.replace("\n", " ")
    os_str = f"{platform.system()} {platform.release()} ({platform.machine()})"
    cpu_str = _get_cpu_info()
    ram_gb = round(psutil.virtual_memory().total / (1024.0 ** 3), 2)

    # Initialize GPU fields with defaults
    gpu_avail = False
    gpu_model: Optional[str] = None
    gpu_vram: Optional[float] = None
    gpu_cc: Optional[str] = None
    driver_ver: Optional[str] = None
    cuda_ver: Optional[str] = None
    cudnn_ver: Optional[str] = None
    torch_ver = "NOT_INSTALLED"
    trt_ver = "NOT_INSTALLED"
    onnx_ver = "NOT_INSTALLED"
    ort_ver = "NOT_INSTALLED"

    # PyTorch inspection
    try:
        import torch

        torch_ver = torch.__version__
        gpu_avail = torch.cuda.is_available()
        cuda_ver = torch.version.cuda
        if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
            cudnn_ver = str(torch.backends.cudnn.version())

        if gpu_avail:
            gpu_model = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            gpu_vram = round(props.total_memory / (1024.0 ** 3), 2)
            gpu_cc = f"{props.major}.{props.minor}"
    except Exception:
        pass

    # NVML driver version inspection
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            import pynvml

        pynvml.nvmlInit()
        driver_ver = pynvml.nvmlSystemGetDriverVersion()
        if not gpu_model:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_model = pynvml.nvmlDeviceGetName(handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_vram = round(mem_info.total / (1024.0 ** 3), 2)
            gpu_avail = True
        pynvml.nvmlShutdown()
    except Exception:
        pass

    # TensorRT inspection
    try:
        import tensorrt

        trt_ver = str(tensorrt.__version__)
    except Exception:
        trt_ver = "NOT_INSTALLED"

    # ONNX inspection
    try:
        import onnx

        onnx_ver = str(onnx.__version__)
    except Exception:
        onnx_ver = "NOT_INSTALLED"

    # ONNX Runtime inspection
    try:
        import onnxruntime

        ort_ver = str(onnxruntime.__version__)
    except Exception:
        ort_ver = "NOT_INSTALLED"

    return EnvironmentManifest(
        timestamp=ts,
        git_commit=git_hash,
        python_version=py_ver,
        os_info=os_str,
        cpu_info=cpu_str,
        ram_total_gb=ram_gb,
        gpu_available=gpu_avail,
        gpu_name=gpu_model,
        gpu_vram_total_gb=gpu_vram,
        gpu_compute_capability=gpu_cc,
        nvidia_driver_version=driver_ver,
        cuda_runtime_version=cuda_ver,
        cudnn_version=cudnn_ver,
        tensorrt_version=trt_ver,
        onnx_version=onnx_ver,
        onnxruntime_version=ort_ver,
        torch_version=torch_ver,
    )


def save_environment_manifest(output_dir: Union[str, Path] = "results/manifests") -> Path:
    """
    Collects and saves the environment manifest to a timestamped JSON file.

    Args:
        output_dir: Target directory path.

    Returns:
        Path to the saved JSON manifest file.
    """
    manifest = collect_environment_manifest()
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    safe_ts = manifest.timestamp.replace(":", "-").replace("+", "_")
    target_file = out_dir_path / f"environment_{safe_ts}.json"
    target_file.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return target_file


def generate_hardware_doc(output_path: Union[str, Path] = "docs/hardware.md") -> None:
    """
    Generates a clean Markdown hardware and runtime specification table.

    Args:
        output_path: Path where the hardware.md file will be rendered.
    """
    manifest = collect_environment_manifest()
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    md_content = f"""# Hardware & Runtime Environment Specification

*Auto-generated by `src.common.environment` on `{manifest.timestamp}` (Git: `{manifest.git_commit}`)*

---

## 1. System Host Architecture

| Component | Specification |
| :--- | :--- |
| **Operating System** | `{manifest.os_info}` |
| **CPU Model & Topology** | `{manifest.cpu_info}` |
| **System Physical RAM** | `{manifest.ram_total_gb} GB` |
| **Python Interpreter** | `{manifest.python_version}` |

---

## 2. GPU Accelerator & Compute Architecture

| Component | Specification |
| :--- | :--- |
| **GPU Device Name** | `{manifest.gpu_name or "N/A (CPU Only)"}` |
| **GPU Total VRAM** | `{str(manifest.gpu_vram_total_gb) + " GB" if manifest.gpu_vram_total_gb else "N/A"}` |
| **Compute Capability** | `{manifest.gpu_compute_capability or "N/A"}` |
| **NVIDIA Driver Version** | `{manifest.nvidia_driver_version or "N/A"}` |
| **CUDA Runtime (PyTorch)** | `{manifest.cuda_runtime_version or "N/A"}` |
| **cuDNN Version** | `{manifest.cudnn_version or "N/A"}` |

---

## 3. Deep Learning & Inference Engine Toolchain

| Package | Version |
| :--- | :--- |
| **PyTorch** | `{manifest.torch_version}` |
| **ONNX** | `{manifest.onnx_version}` |
| **ONNX Runtime** | `{manifest.onnxruntime_version}` |
| **TensorRT** | `{manifest.tensorrt_version}` |

---
"""
    p.write_text(md_content, encoding="utf-8")
