from pathlib import Path
import os
import stat

TARGET_ROOT = Path("/home/sengar/onnx-quant-benchmark")

files = {}

# ============================================================================
# 1. ROOT BUILD & CONFIGURATION FILES
# ============================================================================

files["requirements.txt"] = """torch>=2.2.0
torchvision>=0.17.0
onnx>=1.16.0
onnxruntime-gpu>=1.18.0
onnxsim>=0.4.36
numpy>=1.24.0,<2.0.0
pandas>=2.0.0
opencv-python-headless>=4.8.0
pyyaml>=6.0.1
pydantic>=2.5.0
psutil>=5.9.0
nvidia-ml-py>=12.535.0
matplotlib>=3.8.0
seaborn>=0.13.0
scipy>=1.11.0
scikit-learn>=1.3.0
tqdm>=4.66.0
pytest>=8.0.0
pytest-cov>=4.1.0
"""

files["environment.yml"] = """name: edge-benchmark
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults
dependencies:
  - python=3.10
  - pip
  - cudatoolkit=12.1
  - pip:
      - -r requirements.txt
"""

files["pyproject.toml"] = """[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "onnx-edge-inference-benchmark"
version = "0.1.0"
description = "A reproducible study of correctness, latency, throughput, memory, and accuracy trade-offs across PyTorch, ONNX Runtime, and TensorRT"
readme = "README.md"
requires-python = ">=3.10"
dependencies = [
    "torch>=2.2.0",
    "onnx>=1.16.0",
    "onnxruntime-gpu>=1.18.0",
    "pydantic>=2.5.0",
    "pyyaml>=6.0.1",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = "-v --strict-markers"

[tool.coverage.run]
source = ["src"]
omit = ["tests/*", "scripts/*"]
"""

files[".gitignore"] = """# Python & Bytecode
__pycache__/
*.py[cod]
*$py.class
*.so
.pytest_cache/
.coverage
htmlcov/

# Virtual Environments
venv/
.venv/
env/

# Large Model Weights, Engines, and Data (preserve .gitkeep)
models/weights/*
!models/weights/.gitkeep
models/exported/*
!models/exported/.gitkeep
models/engines/*
!models/engines/.gitkeep
!models/engines/manifests/
models/engines/manifests/*
!models/engines/manifests/.gitkeep
data/calibration/*
!data/calibration/.gitkeep
!data/calibration/README.md
data/sample_images/*
!data/sample_images/.gitkeep

# Raw Benchmark Results & Profiles (preserve .gitkeep)
results/raw/*
!results/raw/.gitkeep
results/manifests/*
!results/manifests/.gitkeep
results/tables/*
!results/tables/.gitkeep
results/figures/*
!results/figures/.gitkeep
results/profiles/*
!results/profiles/.gitkeep

# IDE & OS
.vscode/
.idea/
.DS_Store
*.swp
*.swo
"""

files["Makefile"] = """.PHONY: manifest test lint clean check-env

manifest:
	python scripts/generate_env_manifest.py

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	python -m py_compile src/**/*.py
	python -m py_compile tests/**/*.py
	python -m py_compile scripts/**/*.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	rm -rf .pytest_cache .coverage htmlcov
"""

# ============================================================================
# 2. SOURCE PACKAGE & COMMON UTILITIES
# ============================================================================

files["src/__init__.py"] = '"""ONNX Edge Inference Benchmark package."""\n__version__ = "0.1.0"\n'
files["src/models/__init__.py"] = '"""Model architectures and ingestion."""\n'
files["src/export/__init__.py"] = '"""Model export to ONNX and engine serialization."""\n'
files["src/runtimes/__init__.py"] = '"""Inference runtime wrappers (PyTorch, ORT CPU/CUDA, TensorRT)."""\n'
files["src/quantization/__init__.py"] = '"""Quantization pipelines (Dynamic, Static PTQ, QAT)."""\n'
files["src/validation/__init__.py"] = '"""Correctness and numerical parity validation."""\n'
files["src/benchmarking/__init__.py"] = '"""Latency, throughput, and system resource benchmark engine."""\n'
files["src/analysis/__init__.py"] = '"""Results parsing and tabular summary generators."""\n'
files["src/visualization/__init__.py"] = '"""Publication-grade visualization generators."""\n'

files["src/common/__init__.py"] = """\"\"\"Common utility subsystem.\"\"\"
from src.common.logging import setup_logger
from src.common.seed import seed_everything
from src.common.hashes import compute_file_sha256, compute_tensor_sha256, compute_dict_sha256
from src.common.environment import EnvironmentManifest, collect_environment_manifest, save_environment_manifest, generate_hardware_doc
from src.common.config import MasterConfig, ModelConfig, RuntimeConfig, BenchmarkConfig, QualityThresholdConfig, PathConfig, load_config, save_config

__all__ = [
    "setup_logger",
    "seed_everything",
    "compute_file_sha256",
    "compute_tensor_sha256",
    "compute_dict_sha256",
    "EnvironmentManifest",
    "collect_environment_manifest",
    "save_environment_manifest",
    "generate_hardware_doc",
    "MasterConfig",
    "ModelConfig",
    "RuntimeConfig",
    "BenchmarkConfig",
    "QualityThresholdConfig",
    "PathConfig",
    "load_config",
    "save_config",
]
"""

files["src/common/logging.py"] = """\"\"\"
Logging utility module providing standardized, structured logging across all benchmark components.
\"\"\"

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "edge_benchmark",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    \"\"\"
    Configures and returns a logger instance with ISO-8601 timestamps and standard formatting.

    Args:
        name: Name of the logger namespace.
        log_file: Optional path to a file where log entries will be appended.
        level: Logging verbosity level (default: logging.INFO).

    Returns:
        Configured logging.Logger instance with duplicate handler protection.
    \"\"\"
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid adding duplicate handlers if logger was already configured
    if logger.handlers:
        return logger

    # ISO-8601 format: [%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d]: %(message)s
    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d]: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    # Standard stream output (stdout)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Optional file output handler
    if log_file is not None:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_file_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
"""

files["src/common/seed.py"] = """\"\"\"
Strict deterministic random state initialization utility for reproducible benchmarking.
\"\"\"

import os
import random
import numpy as np


def seed_everything(seed: int = 42) -> None:
    \"\"\"
    Seeds all random number generators and forces deterministic behavior in PyTorch/cuDNN.

    Args:
        seed: Integer seed value to initialize PRNGs.
    \"\"\"
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
"""

files["src/common/hashes.py"] = """\"\"\"
Memory-safe streaming hash verification utilities for models, datasets, and runtime tensors.
\"\"\"

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Union
import numpy as np


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    \"\"\"
    Computes the SHA-256 hex digest of a file using buffered streaming.

    Args:
        file_path: Path to the target file.
        chunk_size: Byte buffer size for chunked reading (default: 64 KB).

    Returns:
        Hexadecimal SHA-256 digest string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the target path is a directory.
    \"\"\"
    p = Path(file_path)
    if not p.is_file():
        raise FileNotFoundError(f"File not found for hash calculation: {file_path}")

    hasher = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_tensor_sha256(tensor: Any) -> str:
    \"\"\"
    Computes a deterministic SHA-256 hash of a PyTorch Tensor or NumPy ndarray byte buffer.

    Enforces memory contiguity and canonical byte representation.

    Args:
        tensor: PyTorch tensor or NumPy ndarray.

    Returns:
        Hexadecimal SHA-256 digest of the raw contiguous tensor memory.

    Raises:
        TypeError: If tensor is not a recognized tensor type.
    \"\"\"
    if hasattr(tensor, "detach") and hasattr(tensor, "cpu") and hasattr(tensor, "numpy"):
        # PyTorch Tensor: force contiguous host memory then extract bytes
        contiguous_tensor = tensor.detach().contiguous().cpu()
        np_arr = contiguous_tensor.numpy()
        byte_data = np.ascontiguousarray(np_arr).tobytes()
    elif isinstance(tensor, np.ndarray):
        # NumPy Array: ensure contiguous C-order layout
        byte_data = np.ascontiguousarray(tensor).tobytes()
    else:
        raise TypeError(f"Unsupported tensor type for hashing: {type(tensor)}")

    return hashlib.sha256(byte_data).hexdigest()


def compute_dict_sha256(data: Dict[str, Any]) -> str:
    \"\"\"
    Computes a deterministic SHA-256 hash of a dictionary by canonicalizing key order.

    Args:
        data: Arbitrary dictionary to hash.

    Returns:
        Hexadecimal SHA-256 digest string.
    \"\"\"
    canonical_json = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
"""

files["src/common/environment.py"] = """\"\"\"
Hardware and software environment introspection module.
Captures comprehensive host system metadata, GPU topology, and runtime package versions.
\"\"\"

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
    \"\"\"Pydantic schema representing the complete host and runtime environment.\"\"\"

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
    \"\"\"Safely retrieves the short git commit hash and dirty status.\"\"\"
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
    \"\"\"Extracts human-readable CPU model and topology.\"\"\"
    cores_phys = psutil.cpu_count(logical=False) or 1
    cores_log = psutil.cpu_count(logical=True) or 1
    processor = platform.processor() or platform.machine()
    return f"{processor} ({cores_phys} physical cores, {cores_log} logical threads)"


def get_physical_core_count() -> int:
    \"\"\"
    Detects the number of unique physical CPU cores available to the current process,
    accounting for cgroups/affinity masks and hyperthreading topology.
    \"\"\"
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
    \"\"\"
    Explicitly binds execution to physical core count using threadpoolctl and PyTorch native APIs,
    and suppresses OpenCV background worker thread thrashing.
    \"\"\"
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
    \"\"\"
    Creates and configures unified ONNX Runtime SessionOptions with deterministic thread pool
    and sequential execution mode across all benchmark backends.
    \"\"\"
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
    \"\"\"
    Inspects and aggregates system hardware, OS details, GPU architecture, and runtime package versions.

    Returns:
        Populated EnvironmentManifest instance.
    \"\"\"
    ts = datetime.now(timezone.utc).isoformat()
    git_hash = _get_git_commit()
    py_ver = sys.version.replace("\\n", " ")
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
    \"\"\"
    Collects and saves the environment manifest to a timestamped JSON file.

    Args:
        output_dir: Target directory path.

    Returns:
        Path to the saved JSON manifest file.
    \"\"\"
    manifest = collect_environment_manifest()
    out_dir_path = Path(output_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    safe_ts = manifest.timestamp.replace(":", "-").replace("+", "_")
    target_file = out_dir_path / f"environment_{safe_ts}.json"
    target_file.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return target_file


def generate_hardware_doc(output_path: Union[str, Path] = "docs/hardware.md") -> None:
    \"\"\"
    Generates a clean Markdown hardware and runtime specification table.

    Args:
        output_path: Path where the hardware.md file will be rendered.
    \"\"\"
    manifest = collect_environment_manifest()
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    md_content = f\"\"\"# Hardware & Runtime Environment Specification

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
\"\"\"
    p.write_text(md_content, encoding="utf-8")
"""

files["src/common/config.py"] = """\"\"\"
Configuration subsystem built on Pydantic v2.
Provides strict validation, hierarchical composition, and YAML serialization/deserialization.
\"\"\"

from pathlib import Path
from typing import List, Literal, Optional, Union
import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    \"\"\"Model definition, task domain, input geometry, and detection/reconstruction parameters.\"\"\"

    name: str = Field(..., description="Unique model identifier (e.g. yolo_nano, industrial_autoencoder)")
    task: Literal["detection", "reconstruction", "classification"] = Field(
        ..., description="Task domain determining metric evaluations"
    )
    input_shape: List[int] = Field(
        ..., description="Static input tensor shape [Batch, Channels, Height, Width]"
    )
    weights_path: str = Field(..., description="Path to PyTorch checkpoint or base weights")
    opset: int = Field(default=17, ge=11, le=21, description="ONNX opset target version")
    class_names: List[str] = Field(default_factory=list, description="Class label names for object detectors")
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Confidence threshold for detections")
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="NMS IoU threshold for detections")

    @field_validator("input_shape")
    @classmethod
    def validate_input_shape(cls, v: List[int]) -> List[int]:
        if len(v) != 4:
            raise ValueError(f"input_shape must contain exactly 4 dimensions [B, C, H, W], got {len(v)}")
        if any(d <= 0 for d in v):
            raise ValueError(f"All dimensions in input_shape must be positive integers, got {v}")
        return v


class RuntimeConfig(BaseModel):
    \"\"\"Inference execution provider configuration, precision, and thread concurrency settings.\"\"\"

    engine: Literal["pytorch", "ort_cpu", "ort_cuda", "tensorrt"] = Field(
        ..., description="Target runtime execution backend"
    )
    precision: Literal["fp32", "fp16", "int8"] = Field(
        ..., description="Target mathematical precision format"
    )
    provider: str = Field(
        default="CPUExecutionProvider", description="ONNX Runtime Execution Provider identifier"
    )
    device_id: int = Field(default=0, ge=0, description="Target GPU device index")
    intra_op_num_threads: int = Field(
        default=8, ge=1, description="Number of intra-operator parallel threads for CPU"
    )
    inter_op_num_threads: int = Field(
        default=1, ge=1, description="Number of inter-operator parallel threads for CPU"
    )
    io_binding: bool = Field(
        default=True, description="Enable pre-allocated GPU VRAM buffer binding"
    )
    workspace_gb: float = Field(
        default=4.0, gt=0.0, description="TensorRT scratch memory allocation limit in GB"
    )
    dynamic_shapes: bool = Field(
        default=False, description="Enable dynamic batch and sequence axes"
    )


class BenchmarkConfig(BaseModel):
    \"\"\"Benchmark execution protocol and timing hyperparameters.\"\"\"

    batch_size: int = Field(default=1, ge=1, description="Inference batch size")
    warmup_iterations: int = Field(
        default=50, ge=1, description="Unmeasured iterations to prime cache and warm kernels"
    )
    timed_iterations: int = Field(
        default=300, ge=1, description="Measured iterations for latency percentile estimation"
    )
    stability_sessions: int = Field(
        default=3, ge=1, description="Number of independent benchmark sessions for stability assessment"
    )
    cooldown_seconds: float = Field(
        default=2.0, ge=0.0, description="Inter-session thermal cooldown duration in seconds"
    )
    collect_nvml: bool = Field(
        default=True, description="Sample GPU power, temperature, and clock frequencies during runs"
    )
    device_id: int = Field(default=0, ge=0, description="Target execution device index")


class QualityThresholdConfig(BaseModel):
    \"\"\"Numerical parity tolerances and domain metric degradation thresholds.\"\"\"

    max_abs_error: float = Field(
        default=1e-4, gt=0.0, description="Maximum allowable element-wise absolute difference vs FP32"
    )
    mean_abs_error: float = Field(
        default=1e-5, gt=0.0, description="Maximum allowable mean absolute difference vs FP32"
    )
    max_map_drop: float = Field(
        default=0.005, ge=0.0, description="Maximum allowable drop in mAP@0.50:0.95 for detectors"
    )
    max_auroc_drop: float = Field(
        default=0.005, ge=0.0, description="Maximum allowable drop in AUROC for anomaly detectors"
    )
    max_aupro_drop: float = Field(
        default=0.010, ge=0.0, description="Maximum allowable drop in AUPRO for anomaly detectors"
    )


class PathConfig(BaseModel):
    \"\"\"Project filesystem directory layout and resource anchors.\"\"\"

    weights_dir: str = Field(default="models/weights", description="Directory for baseline checkpoints")
    exported_dir: str = Field(default="models/exported", description="Directory for exported ONNX graphs")
    engines_dir: str = Field(default="models/engines", description="Directory for compiled TensorRT engines")
    results_raw: str = Field(default="results/raw", description="Directory for raw latency/metric arrays")
    results_manifests: str = Field(default="results/manifests", description="Directory for environment and engine manifests")
    results_tables: str = Field(default="results/tables", description="Directory for aggregated CSV tables")
    results_figures: str = Field(default="results/figures", description="Directory for publication plots")
    results_profiles: str = Field(default="results/profiles", description="Directory for memory and engine profiles")
    calibration_data: str = Field(default="data/calibration", description="Directory for INT8 calibration data")
    sample_data: str = Field(default="data/sample_images", description="Directory for sample images")


class MasterConfig(BaseModel):
    \"\"\"Comprehensive root benchmark configuration composing all domain sub-schemas.\"\"\"

    model: Optional[ModelConfig] = Field(default=None, description="Model definition block")
    runtime: Optional[RuntimeConfig] = Field(default=None, description="Runtime execution block")
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig, description="Benchmarking protocol block")
    quality_thresholds: QualityThresholdConfig = Field(
        default_factory=QualityThresholdConfig, description="Validation threshold block"
    )
    paths: PathConfig = Field(default_factory=PathConfig, description="Filesystem paths block")


def load_config(yaml_path: Union[str, Path]) -> MasterConfig:
    \"\"\"
    Loads, parses, and validates a MasterConfig instance from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Validated MasterConfig object.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValidationError: If schema constraints or types are violated.
    \"\"\"
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    return MasterConfig.model_validate(raw_dict)


def save_config(config: MasterConfig, output_path: Union[str, Path]) -> None:
    \"\"\"
    Serializes a MasterConfig instance to a structured YAML file.

    Args:
        config: MasterConfig object to serialize.
        output_path: Target path for the output YAML file.
    \"\"\"
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    data_dict = config.model_dump(mode="json", exclude_none=True)
    with open(target_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_dict, f, default_flow_style=False, sort_keys=False)
"""

# ============================================================================
# 3. YAML CONFIGURATION FILES
# ============================================================================

files["configs/benchmark.yaml"] = """benchmark:
  batch_size: 1
  warmup_iterations: 50
  timed_iterations: 300
  stability_sessions: 3
  cooldown_seconds: 2.0
  device_id: 0
  collect_nvml: true

paths:
  weights_dir: "models/weights"
  exported_dir: "models/exported"
  engines_dir: "models/engines"
  results_raw: "results/raw"
  results_manifests: "results/manifests"
  results_tables: "results/tables"
  results_figures: "results/figures"
  results_profiles: "results/profiles"
  calibration_data: "data/calibration"
  sample_data: "data/sample_images"
"""

files["configs/yolo/fp32.yaml"] = """model:
  name: "yolo_nano"
  task: "detection"
  input_shape: [1, 3, 640, 640]
  weights_path: "models/weights/yolov8n.pt"
  opset: 17
  class_names: ["object"]
  conf_threshold: 0.25
  iou_threshold: 0.45

runtime:
  engine: "ort_cuda"
  precision: "fp32"
  provider: "CUDAExecutionProvider"
  device_id: 0
  intra_op_num_threads: 8
  inter_op_num_threads: 1
  io_binding: true
  workspace_gb: 4.0
  dynamic_shapes: false

benchmark:
  batch_size: 1
  warmup_iterations: 50
  timed_iterations: 300
  stability_sessions: 3
  cooldown_seconds: 2.0
  collect_nvml: true

quality_thresholds:
  max_abs_error: 1.0e-4
  mean_abs_error: 1.0e-5
  max_map_drop: 0.001
"""

files["configs/yolo/fp16.yaml"] = """model:
  name: "yolo_nano"
  task: "detection"
  input_shape: [1, 3, 640, 640]
  weights_path: "models/weights/yolov8n.pt"
  opset: 17
  class_names: ["object"]
  conf_threshold: 0.25
  iou_threshold: 0.45

runtime:
  engine: "tensorrt"
  precision: "fp16"
  provider: "TensorrtExecutionProvider"
  device_id: 0
  intra_op_num_threads: 8
  inter_op_num_threads: 1
  io_binding: true
  workspace_gb: 4.0
  dynamic_shapes: false

benchmark:
  batch_size: 1
  warmup_iterations: 50
  timed_iterations: 300
  stability_sessions: 3
  cooldown_seconds: 2.0
  collect_nvml: true

quality_thresholds:
  max_abs_error: 5.0e-3
  mean_abs_error: 5.0e-4
  max_map_drop: 0.005
"""

files["configs/yolo/int8.yaml"] = """model:
  name: "yolo_nano"
  task: "detection"
  input_shape: [1, 3, 640, 640]
  weights_path: "models/weights/yolov8n.pt"
  opset: 17
  class_names: ["object"]
  conf_threshold: 0.25
  iou_threshold: 0.45

runtime:
  engine: "tensorrt"
  precision: "int8"
  provider: "TensorrtExecutionProvider"
  device_id: 0
  intra_op_num_threads: 8
  inter_op_num_threads: 1
  io_binding: true
  workspace_gb: 4.0
  dynamic_shapes: false

benchmark:
  batch_size: 1
  warmup_iterations: 50
  timed_iterations: 300
  stability_sessions: 3
  cooldown_seconds: 2.0
  collect_nvml: true

quality_thresholds:
  max_abs_error: 5.0e-2
  mean_abs_error: 5.0e-3
  max_map_drop: 0.015
"""

files["configs/industrial_model/fp32.yaml"] = """model:
  name: "industrial_autoencoder"
  task: "reconstruction"
  input_shape: [1, 3, 256, 256]
  weights_path: "models/weights/industrial_autoencoder.pt"
  opset: 17
  class_names: []
  conf_threshold: 0.5
  iou_threshold: 0.5

runtime:
  engine: "ort_cuda"
  precision: "fp32"
  provider: "CUDAExecutionProvider"
  device_id: 0
  intra_op_num_threads: 8
  inter_op_num_threads: 1
  io_binding: true
  workspace_gb: 4.0
  dynamic_shapes: false

benchmark:
  batch_size: 1
  warmup_iterations: 50
  timed_iterations: 300
  stability_sessions: 3
  cooldown_seconds: 2.0
  collect_nvml: true

quality_thresholds:
  max_abs_error: 1.0e-4
  mean_abs_error: 1.0e-5
  max_auroc_drop: 0.001
  max_aupro_drop: 0.002
"""

files["configs/industrial_model/fp16.yaml"] = """model:
  name: "industrial_autoencoder"
  task: "reconstruction"
  input_shape: [1, 3, 256, 256]
  weights_path: "models/weights/industrial_autoencoder.pt"
  opset: 17
  class_names: []
  conf_threshold: 0.5
  iou_threshold: 0.5

runtime:
  engine: "tensorrt"
  precision: "fp16"
  provider: "TensorrtExecutionProvider"
  device_id: 0
  intra_op_num_threads: 8
  inter_op_num_threads: 1
  io_binding: true
  workspace_gb: 4.0
  dynamic_shapes: false

benchmark:
  batch_size: 1
  warmup_iterations: 50
  timed_iterations: 300
  stability_sessions: 3
  cooldown_seconds: 2.0
  collect_nvml: true

quality_thresholds:
  max_abs_error: 5.0e-3
  mean_abs_error: 5.0e-4
  max_auroc_drop: 0.005
  max_aupro_drop: 0.010
"""

files["configs/industrial_model/int8.yaml"] = """model:
  name: "industrial_autoencoder"
  task: "reconstruction"
  input_shape: [1, 3, 256, 256]
  weights_path: "models/weights/industrial_autoencoder.pt"
  opset: 17
  class_names: []
  conf_threshold: 0.5
  iou_threshold: 0.5

runtime:
  engine: "tensorrt"
  precision: "int8"
  provider: "TensorrtExecutionProvider"
  device_id: 0
  intra_op_num_threads: 8
  inter_op_num_threads: 1
  io_binding: true
  workspace_gb: 4.0
  dynamic_shapes: false

benchmark:
  batch_size: 1
  warmup_iterations: 50
  timed_iterations: 300
  stability_sessions: 3
  cooldown_seconds: 2.0
  collect_nvml: true

quality_thresholds:
  max_abs_error: 5.0e-2
  mean_abs_error: 5.0e-3
  max_auroc_drop: 0.010
  max_aupro_drop: 0.015
"""

# ============================================================================
# 4. SCRIPTS & AUTOMATION
# ============================================================================

files["scripts/generate_env_manifest.py"] = """#!/usr/bin/env python3
\"\"\"
CLI tool to inspect system hardware, runtime environment, and serialize manifests.
\"\"\"

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.environment import (
    collect_environment_manifest,
    save_environment_manifest,
    generate_hardware_doc,
)
from src.common.logging import setup_logger

logger = setup_logger("generate_env_manifest")


def main() -> None:
    logger.info("Inspecting hardware topology and runtime environment...")
    manifest = collect_environment_manifest()

    manifest_path = save_environment_manifest(PROJECT_ROOT / "results" / "manifests")
    logger.info(f"Environment manifest saved -> {manifest_path}")

    doc_path = PROJECT_ROOT / "docs" / "hardware.md"
    generate_hardware_doc(doc_path)
    logger.info(f"Hardware documentation updated -> {doc_path}")

    print("\\n" + "=" * 65)
    print("        ENVIRONMENT & HARDWARE AUDIT SUMMARY")
    print("=" * 65)
    print(f"Timestamp:          {manifest.timestamp}")
    print(f"Git Commit:         {manifest.git_commit}")
    print(f"OS Info:            {manifest.os_info}")
    print(f"CPU Model:          {manifest.cpu_info}")
    print(f"System RAM:         {manifest.ram_total_gb} GB")
    print(f"GPU Available:      {manifest.gpu_available}")
    print(f"GPU Device:         {manifest.gpu_name or 'N/A'}")
    print(f"GPU VRAM:           {manifest.gpu_vram_total_gb or 'N/A'} GB")
    print(f"CUDA Driver:        {manifest.nvidia_driver_version or 'N/A'}")
    print(f"CUDA Runtime:       {manifest.cuda_runtime_version or 'N/A'}")
    print(f"cuDNN:              {manifest.cudnn_version or 'N/A'}")
    print(f"TensorRT:           {manifest.tensorrt_version}")
    print(f"ONNX Runtime:       {manifest.onnxruntime_version}")
    print(f"PyTorch:            {manifest.torch_version}")
    print("=" * 65 + "\\n")


if __name__ == "__main__":
    main()
"""

files["scripts/setup_env.sh"] = """#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================="
echo " Setting up ONNX Edge Inference Benchmark Environment"
echo "=========================================================="

# Check Python version
PYTHON_CMD="python3"
if ! command -v "$PYTHON_CMD" &> /dev/null; then
    echo "ERROR: python3 could not be found."
    exit 1
fi

PY_VER=$($PYTHON_CMD -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Detected Python version: $PY_VER"

# Create required directories
mkdir -p models/weights models/exported models/engines/manifests \\
         data/sample_images data/calibration \\
         results/raw results/manifests results/tables results/figures results/profiles

echo "Directories verified."

# Check NVIDIA GPU support
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA Driver detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    echo "WARNING: nvidia-smi not found. Running in CPU-only or container mode."
fi

# Run environment manifest generation
$PYTHON_CMD scripts/generate_env_manifest.py

echo "Environment setup complete."
"""

# ============================================================================
# 5. DOCUMENTATION FILES
# ============================================================================

files["docs/hardware.md"] = """# Hardware & Runtime Environment Specification

*Run `python scripts/generate_env_manifest.py` to populate with active system specifications.*
"""

files["docs/environment.md"] = """# Environment & Software Stack Architecture

This document specifies the software dependencies, CUDA runtime compatibility, and execution provider toolchain used for the `onnx-edge-inference-benchmark` project.

---

## 1. Supported Runtime Providers

1. **PyTorch (Native):** Baseline eager execution and TorchScript tracing on CPU and CUDA.
2. **ONNX Runtime (CPU):** Multi-threaded CPU execution with AVX2/AVX-512 VNNI vectorization.
3. **ONNX Runtime (CUDA):** GPU inference leveraging cuDNN and cuBLAS execution backends with IOBinding support.
4. **TensorRT:** NVIDIA's high-performance inference optimizer utilizing FP16 and INT8 Tensor Cores with kernel auto-tuning and layer fusion.

---

## 2. Dependency Matrix

| Component | Target Version | Purpose |
| :--- | :--- | :--- |
| **Python** | `>=3.10` | Core language runtime |
| **PyTorch** | `>=2.2.0` | Ingestion, PyTorch reference runs, and model export |
| **ONNX** | `>=1.16.0` | Graph representation and schema validation |
| **ONNX Runtime GPU** | `>=1.18.0` | Multi-backend inference runtime |
| **TensorRT** | `10.x / 8.6+` | Low-latency edge GPU inference engine |
| **Pydantic** | `>=2.5.0` | Type-safe configuration and manifest schemas |
| **PyYAML** | `>=6.0.1` | YAML configuration parsing |
| **pytest** | `>=8.0.0` | Automated test suite and coverage reporting |
"""

files["docs/methodology.md"] = """# Benchmarking & Validation Methodology

This document outlines the experimental protocol, timing measurement safeguards, numerical parity standards, and statistical metrics.

---

## 1. Latency & Throughput Measurement Protocol

### 1.1. GPU Synchronization & Monotonic Clock
- **CUDA Stream Synchronization:** To prevent measuring asynchronous CPU dispatch time, the CUDA stream is synchronized before and after inference iterations.
- **High-Resolution Clock:** Latencies are measured using `time.perf_counter_ns()` with nanosecond precision.

### 1.2. Memory Management (IOBinding)
- Input and output memory buffers are pre-allocated in device VRAM (`cuda:0`).
- By benchmarking using `session.run_with_iobinding()` or TensorRT enqueue buffers, pure compute kernel latency is isolated from host-device PCIe bus transfer overhead.

### 1.3. Warmup & Sample Sizes
- **Warmup:** $\\\\ge 50$ iterations to compile execution kernels, prime CUDA memory allocators, and stabilize power states.
- **Timed Iterations:** $\\\\ge 300$ iterations per session across 3 independent sessions to ensure mathematically sound percentile estimations ($p_{50}, p_{90}, p_{99}$).
- **Garbage Collection:** Python's automatic garbage collection is temporarily suspended during timed loops to prevent latency spikes.

---

## 2. Numerical Parity & Correctness Standards

Every exported and quantized model must pass rigorous numerical validation against the PyTorch FP32 baseline:

1. **Max Absolute Error (MAE):** $\\\\max |y_{\\\\text{candidate}} - y_{\\\\text{baseline}}| \\\\le \\\\epsilon_{\\\\text{max}}$
2. **Mean Absolute Error:** $\\\\frac{1}{N} \\\\sum |y_{\\\\text{candidate}} - y_{\\\\text{baseline}}| \\\\le \\\\epsilon_{\\\\text{mean}}$
3. **Domain Task Accuracy:**
   - **Object Detection (YOLO):** $\\\\Delta \\\\text{mAP@0.50:0.95} \\\\le 0.005$ (FP16), $\\\\le 0.015$ (INT8).
   - **Industrial Anomaly Reconstruction:** $\\\\Delta \\\\text{AUROC} \\\\le 0.005$ (FP16), $\\\\le 0.010$ (INT8).
"""

files["models/README.md"] = """# Models Directory

This directory stores model checkpoints, exported ONNX graphs, and compiled TensorRT engines.

```
models/
├── weights/           # Base PyTorch checkpoints (.pt / .pth)
├── exported/          # Exported and optimized ONNX models (.onnx)
└── engines/           # Compiled TensorRT plan engines (.engine)
    └── manifests/     # Serialized engine compilation manifests (.json)
```
"""

files["data/README.md"] = """# Data Directory

This directory contains evaluation datasets, sample images, and INT8 calibration batches.

```
data/
├── sample_images/     # Static sample images for quick correctness tests
└── calibration/       # Representative domain dataset for INT8 calibration
```
"""

files["data/calibration/README.md"] = """# INT8 Calibration Data

Store representative unlabelled image batches (100–500 samples) in this directory for TensorRT INT8 entropy / min-max calibration.
"""

# ============================================================================
# 6. COMPREHENSIVE UNIT TEST SUITE (tests/)
# ============================================================================

files["tests/__init__.py"] = '"""Unit test suite for onnx-edge-inference-benchmark."""\n'

files["tests/conftest.py"] = """\"\"\"
Pytest shared fixtures providing configuration samples and test files.
\"\"\"

from pathlib import Path
from typing import Any, Dict
import pytest
import yaml


@pytest.fixture
def valid_config_dict() -> Dict[str, Any]:
    \"\"\"Returns a valid dictionary matching the MasterConfig schema.\"\"\"
    return {
        "model": {
            "name": "yolo_nano",
            "task": "detection",
            "input_shape": [1, 3, 640, 640],
            "weights_path": "models/weights/yolov8n.pt",
            "opset": 17,
            "class_names": ["person", "car"],
            "conf_threshold": 0.25,
            "iou_threshold": 0.45,
        },
        "runtime": {
            "engine": "ort_cuda",
            "precision": "fp32",
            "provider": "CUDAExecutionProvider",
            "device_id": 0,
            "intra_op_num_threads": 8,
            "inter_op_num_threads": 1,
            "io_binding": True,
            "workspace_gb": 4.0,
            "dynamic_shapes": False,
        },
        "benchmark": {
            "batch_size": 1,
            "warmup_iterations": 50,
            "timed_iterations": 300,
            "stability_sessions": 3,
            "cooldown_seconds": 2.0,
            "collect_nvml": True,
            "device_id": 0,
        },
        "quality_thresholds": {
            "max_abs_error": 1e-4,
            "mean_abs_error": 1e-5,
            "max_map_drop": 0.005,
            "max_auroc_drop": 0.005,
            "max_aupro_drop": 0.010,
        },
        "paths": {
            "weights_dir": "models/weights",
            "exported_dir": "models/exported",
            "engines_dir": "models/engines",
            "results_raw": "results/raw",
            "results_manifests": "results/manifests",
            "results_tables": "results/tables",
            "results_figures": "results/figures",
            "results_profiles": "results/profiles",
            "calibration_data": "data/calibration",
            "sample_data": "data/sample_images",
        },
    }


@pytest.fixture
def temp_yaml_file(tmp_path: Path, valid_config_dict: Dict[str, Any]) -> Path:
    \"\"\"Writes a valid configuration to a temporary YAML file.\"\"\"
    yaml_file = tmp_path / "test_config.yaml"
    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(valid_config_dict, f)
    return yaml_file


@pytest.fixture
def temp_dummy_file(tmp_path: Path) -> Path:
    \"\"\"Creates a temporary file with a known byte sequence for hashing tests.\"\"\"
    file_path = tmp_path / "dummy.bin"
    file_path.write_bytes(b"Benchmark test data for SHA-256 verification\\n")
    return file_path
"""

files["tests/test_hashes.py"] = """\"\"\"
Unit tests for memory-safe streaming hashing utilities.
\"\"\"

import hashlib
from pathlib import Path
import numpy as np
import pytest

from src.common.hashes import compute_dict_sha256, compute_file_sha256, compute_tensor_sha256


class TestHashes:
    \"\"\"Test suite verifying SHA-256 integrity functions.\"\"\"

    def test_compute_file_sha256_known_string(self, temp_dummy_file: Path) -> None:
        \"\"\"Tests compute_file_sha256 against known raw bytes digest.\"\"\"
        expected = hashlib.sha256(b"Benchmark test data for SHA-256 verification\\n").hexdigest()
        actual = compute_file_sha256(temp_dummy_file)
        assert actual == expected
        assert len(actual) == 64

    def test_compute_file_sha256_empty_file(self, tmp_path: Path) -> None:
        \"\"\"Tests compute_file_sha256 on an empty file against standard SHA-256 empty digest.\"\"\"
        empty_file = tmp_path / "empty.bin"
        empty_file.write_bytes(b"")
        expected_empty = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        assert compute_file_sha256(empty_file) == expected_empty

    def test_compute_file_sha256_missing_file_raises(self, tmp_path: Path) -> None:
        \"\"\"Ensures compute_file_sha256 raises FileNotFoundError for nonexistent paths.\"\"\"
        with pytest.raises(FileNotFoundError):
            compute_file_sha256(tmp_path / "nonexistent.bin")

    def test_compute_tensor_sha256_numpy(self) -> None:
        \"\"\"Tests hashing of NumPy ndarrays ensuring consistency and value sensitivity.\"\"\"
        arr1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        arr2 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        arr3 = np.array([1.0, 2.0, 3.0, 4.0001], dtype=np.float32)

        hash1 = compute_tensor_sha256(arr1)
        hash2 = compute_tensor_sha256(arr2)
        hash3 = compute_tensor_sha256(arr3)

        assert hash1 == hash2
        assert hash1 != hash3
        assert len(hash1) == 64

    def test_compute_tensor_sha256_torch(self) -> None:
        \"\"\"Tests hashing of PyTorch tensors across memory views and non-contiguous layouts.\"\"\"
        try:
            import torch

            t1 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
            t2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
            t_permuted = t1.t()  # Non-contiguous view

            hash1 = compute_tensor_sha256(t1)
            hash2 = compute_tensor_sha256(t2)
            hash_perm = compute_tensor_sha256(t_permuted)

            assert hash1 == hash2
            assert hash1 != hash_perm
        except ImportError:
            pytest.skip("PyTorch not installed")

    def test_compute_tensor_sha256_invalid_type_raises(self) -> None:
        \"\"\"Tests compute_tensor_sha256 raises TypeError on invalid types.\"\"\"
        with pytest.raises(TypeError):
            compute_tensor_sha256("not a tensor")  # type: ignore

    def test_compute_dict_sha256_key_order_invariance(self) -> None:
        \"\"\"Tests compute_dict_sha256 yields identical digests regardless of key insertion order.\"\"\"
        d1 = {"model": "yolo", "batch_size": 1, "precision": "fp16", "threads": 8}
        d2 = {"threads": 8, "precision": "fp16", "batch_size": 1, "model": "yolo"}
        d3 = {"threads": 8, "precision": "fp32", "batch_size": 1, "model": "yolo"}

        h1 = compute_dict_sha256(d1)
        h2 = compute_dict_sha256(d2)
        h3 = compute_dict_sha256(d3)

        assert h1 == h2
        assert h1 != h3
        assert len(h1) == 64
"""

files["tests/test_config.py"] = """\"\"\"
Unit tests for the Pydantic v2 configuration subsystem.
\"\"\"

from pathlib import Path
import pytest
from pydantic import ValidationError

from src.common.config import (
    BenchmarkConfig,
    MasterConfig,
    ModelConfig,
    PathConfig,
    QualityThresholdConfig,
    RuntimeConfig,
    load_config,
    save_config,
)


class TestConfig:
    \"\"\"Test suite validating configuration schemas and serialization.\"\"\"

    def test_load_all_existing_yaml_configs(self) -> None:
        \"\"\"Verifies all default YAML configuration files in configs/ parse and validate cleanly.\"\"\"
        root_dir = Path(__file__).resolve().parent.parent
        configs_dir = root_dir / "configs"
        yaml_files = list(configs_dir.rglob("*.yaml"))

        assert len(yaml_files) >= 7, f"Expected at least 7 YAML config files, found {len(yaml_files)}"

        for yf in yaml_files:
            cfg = load_config(yf)
            assert isinstance(cfg, MasterConfig), f"Failed to load {yf}"
            assert isinstance(cfg.benchmark, BenchmarkConfig)
            assert isinstance(cfg.paths, PathConfig)

    def test_model_config_validation(self) -> None:
        \"\"\"Validates ModelConfig fields and input_shape constraints.\"\"\"
        # Valid model config
        m = ModelConfig(
            name="yolo_nano",
            task="detection",
            input_shape=[1, 3, 640, 640],
            weights_path="models/weights/yolov8n.pt",
        )
        assert m.opset == 17
        assert m.input_shape == [1, 3, 640, 640]

        # Invalid shape: 3 dimensions instead of 4
        with pytest.raises(ValidationError):
            ModelConfig(
                name="invalid",
                task="detection",
                input_shape=[3, 640, 640],
                weights_path="test.pt",
            )

        # Invalid shape: non-positive dimension
        with pytest.raises(ValidationError):
            ModelConfig(
                name="invalid",
                task="detection",
                input_shape=[1, 3, -640, 640],
                weights_path="test.pt",
            )

        # Invalid task name
        with pytest.raises(ValidationError):
            ModelConfig(
                name="invalid",
                task="segmentation",  # type: ignore
                input_shape=[1, 3, 640, 640],
                weights_path="test.pt",
            )

    def test_runtime_config_validation(self) -> None:
        \"\"\"Validates RuntimeConfig allowable backends and precisions.\"\"\"
        r = RuntimeConfig(engine="ort_cuda", precision="fp16")
        assert r.provider == "CPUExecutionProvider"
        assert r.io_binding is True

        # Invalid engine
        with pytest.raises(ValidationError):
            RuntimeConfig(engine="vllm", precision="fp16")  # type: ignore

        # Invalid precision
        with pytest.raises(ValidationError):
            RuntimeConfig(engine="tensorrt", precision="int4")  # type: ignore

    def test_benchmark_config_validation(self) -> None:
        \"\"\"Validates BenchmarkConfig bounds.\"\"\"
        b = BenchmarkConfig(batch_size=4, warmup_iterations=10, timed_iterations=100)
        assert b.batch_size == 4

        # Zero or negative batch size
        with pytest.raises(ValidationError):
            BenchmarkConfig(batch_size=0)

        # Negative cooldown
        with pytest.raises(ValidationError):
            BenchmarkConfig(cooldown_seconds=-1.0)

    def test_config_serialization_roundtrip(self, tmp_path: Path, temp_yaml_file: Path) -> None:
        \"\"\"Tests load -> save -> load roundtrip fidelity.\"\"\"
        cfg1 = load_config(temp_yaml_file)

        saved_path = tmp_path / "roundtrip.yaml"
        save_config(cfg1, saved_path)

        cfg2 = load_config(saved_path)
        assert cfg1.model_dump() == cfg2.model_dump()

    def test_load_missing_file_raises(self, tmp_path: Path) -> None:
        \"\"\"Ensures load_config raises FileNotFoundError for missing paths.\"\"\"
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "does_not_exist.yaml")
"""

files["tests/test_logging.py"] = """\"\"\"
Unit tests for the structured logging subsystem.
\"\"\"

import logging
from pathlib import Path
from src.common.logging import setup_logger


class TestLogging:
    \"\"\"Test suite validating logger initialization, formatting, and file outputs.\"\"\"

    def test_setup_logger_basic(self) -> None:
        \"\"\"Tests basic stream logger setup.\"\"\"
        logger = setup_logger("test_basic", level=logging.DEBUG)
        assert logger.name == "test_basic"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) >= 1

    def test_setup_logger_file_output(self, tmp_path: Path) -> None:
        \"\"\"Tests logger file handler creation and log persistence.\"\"\"
        log_file = tmp_path / "sub_dir" / "benchmark.log"
        logger = setup_logger("test_file_logger", log_file=log_file, level=logging.INFO)
        logger.info("Testing file logging output.")

        assert log_file.is_file()
        content = log_file.read_text(encoding="utf-8")
        assert "Testing file logging output." in content
        assert "[INFO]" in content
        assert "[test_file_logger:" in content

    def test_setup_logger_duplicate_protection(self) -> None:
        \"\"\"Ensures repeated setup calls do not duplicate handlers.\"\"\"
        logger1 = setup_logger("test_dup", level=logging.INFO)
        num_handlers_initial = len(logger1.handlers)

        logger2 = setup_logger("test_dup", level=logging.INFO)
        assert len(logger2.handlers) == num_handlers_initial
"""

files["tests/test_seed.py"] = """\"\"\"
Unit tests for strict deterministic random state initialization.
\"\"\"

import os
import random
import numpy as np
import pytest
from src.common.seed import seed_everything


class TestSeed:
    \"\"\"Test suite validating deterministic PRNG initialization.\"\"\"

    def test_seed_everything_deterministic_numpy(self) -> None:
        \"\"\"Verifies seed_everything produces deterministic NumPy random sequences.\"\"\"
        seed_everything(1234)
        sample1 = np.random.rand(5)

        seed_everything(1234)
        sample2 = np.random.rand(5)

        assert np.allclose(sample1, sample2)

    def test_seed_everything_deterministic_python(self) -> None:
        \"\"\"Verifies seed_everything produces deterministic Python random choices.\"\"\"
        seed_everything(42)
        v1 = [random.random() for _ in range(5)]

        seed_everything(42)
        v2 = [random.random() for _ in range(5)]

        assert v1 == v2

    def test_seed_everything_env_var(self) -> None:
        \"\"\"Verifies PYTHONHASHSEED is set to the provided seed.\"\"\"
        seed_everything(999)
        assert os.environ.get("PYTHONHASHSEED") == "999"

    def test_seed_everything_torch(self) -> None:
        \"\"\"Verifies seed_everything controls PyTorch determinism and cuDNN flags.\"\"\"
        try:
            import torch

            seed_everything(42)
            t1 = torch.rand(4, 4)

            seed_everything(42)
            t2 = torch.rand(4, 4)

            assert torch.equal(t1, t2)
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False
        except ImportError:
            pytest.skip("PyTorch not installed")
"""

files["tests/test_environment.py"] = """\"\"\"
Unit tests for hardware and environment introspection routines.
\"\"\"

from pathlib import Path
from unittest.mock import patch
import pytest

from src.common.environment import (
    EnvironmentManifest,
    collect_environment_manifest,
    generate_hardware_doc,
    save_environment_manifest,
)


class TestEnvironment:
    \"\"\"Test suite validating environment introspection and markdown generation.\"\"\"

    def test_collect_environment_manifest_returns_valid_instance(self) -> None:
        \"\"\"Tests collect_environment_manifest returns populated EnvironmentManifest.\"\"\"
        manifest = collect_environment_manifest()
        assert isinstance(manifest, EnvironmentManifest)
        assert manifest.timestamp != ""
        assert manifest.python_version != ""
        assert manifest.os_info != ""
        assert manifest.cpu_info != ""
        assert manifest.ram_total_gb > 0.0

    def test_save_environment_manifest(self, tmp_path: Path) -> None:
        \"\"\"Tests serialization of EnvironmentManifest to a JSON file.\"\"\"
        saved_file = save_environment_manifest(tmp_path)
        assert saved_file.is_file()
        assert saved_file.name.startswith("environment_")
        assert saved_file.name.endswith(".json")
        assert saved_file.stat().st_size > 0

    def test_generate_hardware_doc(self, tmp_path: Path) -> None:
        \"\"\"Tests generation of Markdown hardware summary.\"\"\"
        doc_file = tmp_path / "hardware.md"
        generate_hardware_doc(doc_file)
        assert doc_file.is_file()

        content = doc_file.read_text(encoding="utf-8")
        assert "# Hardware & Runtime Environment Specification" in content
        assert "System Host Architecture" in content
        assert "Deep Learning & Inference Engine Toolchain" in content

    def test_graceful_fallback_without_gpu(self) -> None:
        \"\"\"Mocks GPU unavailability to verify graceful fallback.\"\"\"
        with patch("torch.cuda.is_available", return_value=False):
            manifest = collect_environment_manifest()
            assert isinstance(manifest, EnvironmentManifest)
            # Should not crash and should record valid CPU environment
            assert manifest.cpu_info != ""
            assert manifest.ram_total_gb > 0.0

    def test_configure_cpu_threads(self) -> None:
        \"\"\"Tests thread configuration helper.\"\"\"
        from src.common.environment import configure_cpu_threads, get_physical_core_count
        cores = get_physical_core_count()
        assert cores >= 1
        applied = configure_cpu_threads(cores)
        assert applied == cores

    def test_get_unified_ort_session_options(self) -> None:
        \"\"\"Tests unified ORT session options factory.\"\"\"
        from src.common.environment import get_unified_ort_session_options
        import onnxruntime as ort
        opts = get_unified_ort_session_options(intra_op_threads=4, inter_op_threads=1)
        assert opts.intra_op_num_threads == 4
        assert opts.inter_op_num_threads == 1
        assert opts.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
"""

# ============================================================================
# WRITE ALL FILES TO TARGET_ROOT
# ============================================================================

for rel_path, content in files.items():
    dest = TARGET_ROOT / rel_path
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(content, encoding="utf-8")
    if rel_path.endswith(".sh") or rel_path.endswith(".py") and rel_path.startswith("scripts/"):
        dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(f"  [CREATED] {rel_path}")

print(f"\\nAll {len(files)} Phase 0 files generated successfully at {TARGET_ROOT}.")
