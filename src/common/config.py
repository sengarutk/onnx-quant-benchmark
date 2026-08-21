"""
Configuration subsystem built on Pydantic v2.
Provides strict validation, hierarchical composition, and YAML serialization/deserialization.
"""

from pathlib import Path
from typing import List, Literal, Optional, Union
import yaml
from pydantic import BaseModel, Field, field_validator


class ModelConfig(BaseModel):
    """Model definition, task domain, input geometry, and detection/reconstruction parameters."""

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
    """Inference execution provider configuration, precision, and thread concurrency settings."""

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
    """Benchmark execution protocol and timing hyperparameters."""

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
    """Numerical parity tolerances and domain metric degradation thresholds."""

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
    """Project filesystem directory layout and resource anchors."""

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
    """Comprehensive root benchmark configuration composing all domain sub-schemas."""

    model: Optional[ModelConfig] = Field(default=None, description="Model definition block")
    runtime: Optional[RuntimeConfig] = Field(default=None, description="Runtime execution block")
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig, description="Benchmarking protocol block")
    quality_thresholds: QualityThresholdConfig = Field(
        default_factory=QualityThresholdConfig, description="Validation threshold block"
    )
    paths: PathConfig = Field(default_factory=PathConfig, description="Filesystem paths block")


def load_config(yaml_path: Union[str, Path]) -> MasterConfig:
    """
    Loads, parses, and validates a MasterConfig instance from a YAML file.

    Args:
        yaml_path: Path to the YAML configuration file.

    Returns:
        Validated MasterConfig object.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ValidationError: If schema constraints or types are violated.
    """
    path = Path(yaml_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_dict = yaml.safe_load(f) or {}

    return MasterConfig.model_validate(raw_dict)


def save_config(config: MasterConfig, output_path: Union[str, Path]) -> None:
    """
    Serializes a MasterConfig instance to a structured YAML file.

    Args:
        config: MasterConfig object to serialize.
        output_path: Target path for the output YAML file.
    """
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    data_dict = config.model_dump(mode="json", exclude_none=True)
    with open(target_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data_dict, f, default_flow_style=False, sort_keys=False)
