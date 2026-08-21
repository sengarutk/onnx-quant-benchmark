"""
Unit tests for FP16 conversion, Static INT8 PTQ, and quality degradation gating.
"""

import json
from pathlib import Path
import numpy as np
import onnx
import pytest
import torch
import torch.nn as nn

from src.common.config import QualityThresholdConfig
from src.export.export_onnx import export_to_onnx
from src.quantization.calibration_reader import BenchmarkCalibrationDataReader
from src.quantization.convert_fp16 import convert_onnx_to_fp16
from src.quantization.quantize_onnx import quantize_onnx_static
from src.validation.validate_quantization import validate_quantized_model


class SimpleToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class TestQuantization:
    """Test suite validating quantization transformations and decision gating."""

    def test_convert_onnx_to_fp16(self, tmp_path: Path) -> None:
        """Tests FP16 conversion produces valid ONNX model with half-precision weights."""
        model = SimpleToyModel()
        dummy_in = torch.randn(1, 3, 16, 16)
        fp32_path = tmp_path / "toy_fp32.onnx"
        fp16_path = tmp_path / "toy_fp16.onnx"

        export_to_onnx(model, dummy_in, fp32_path, ["input"], ["output"], opset_version=17)
        res_path = convert_onnx_to_fp16(fp32_path, fp16_path, keep_io_types=True)

        assert res_path.is_file()
        assert fp16_path.with_name(fp16_path.name + ".sha256").is_file()

        loaded = onnx.load(str(res_path))
        # Check that weight initializers are float16
        float16_inits = [init for init in loaded.graph.initializer if init.data_type == onnx.TensorProto.FLOAT16]
        assert len(float16_inits) > 0

    def test_quantize_onnx_static(self, tmp_path: Path) -> None:
        """Tests static INT8 PTQ creates quantized graph with QuantizeLinear/DequantizeLinear nodes."""
        model = SimpleToyModel()
        dummy_in = torch.randn(1, 3, 16, 16)
        fp32_path = tmp_path / "toy_fp32.onnx"
        int8_path = tmp_path / "toy_int8.onnx"

        export_to_onnx(model, dummy_in, fp32_path, ["input"], ["output"], opset_version=17)

        # Create dummy calibration data
        calib_images = []
        for i in range(5):
            img_p = tmp_path / f"c_img_{i}.npy"
            np.save(str(img_p), np.random.randn(1, 3, 16, 16).astype(np.float32))
            calib_images.append(img_p)

        def toy_preprocess(p):
            return np.load(str(p))

        reader = BenchmarkCalibrationDataReader(
            image_paths=calib_images,
            input_name="input",
            input_shape=(1, 3, 16, 16),
            preprocess_fn=toy_preprocess,
        )

        res_path = quantize_onnx_static(
            input_onnx_path=fp32_path,
            output_onnx_path=int8_path,
            calibration_data_reader=reader,
            quant_format="QDQ",
        )

        assert res_path.is_file()
        assert int8_path.with_name(int8_path.name + ".sha256").is_file()

        loaded = onnx.load(str(res_path))
        op_types = {node.op_type for node in loaded.graph.node}
        assert "QuantizeLinear" in op_types or "DequantizeLinear" in op_types or "QLinearConv" in op_types

    def test_validate_quantized_model_quality_gate(self, tmp_path: Path) -> None:
        """Verifies degradation gating marks severe drops as REJECTED and acceptable drops as PASS."""
        root = Path(__file__).resolve().parent.parent
        manifest_p = root / "data" / "sample_images" / "manifest.json"
        base_p = root / "results" / "raw" / "pytorch_fp32" / "yolo_nano" / "baseline_metrics.json"
        onnx_p = root / "models" / "exported" / "yolo_nano_fp32_opset17.onnx"

        if not manifest_p.is_file() or not base_p.is_file() or not onnx_p.is_file():
            pytest.skip("Required benchmark artifacts not present")

        # Create mini manifest for fast unit testing
        full_manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        mini_manifest = {
            "detection_samples": full_manifest.get("detection_samples", [])[:2],
            "industrial_samples": full_manifest.get("industrial_samples", [])[:2],
        }
        mini_manifest_p = tmp_path / "mini_manifest.json"
        mini_manifest_p.write_text(json.dumps(mini_manifest), encoding="utf-8")

        # Test with standard threshold -> should PASS
        pass_cfg = QualityThresholdConfig(max_map_drop=0.05)
        rep_pass = validate_quantized_model("yolo_nano", "fp32", onnx_p, mini_manifest_p, base_p, pass_cfg)
        assert rep_pass["status"] == "PASS"

        # Test with strict threshold on artificially high baseline -> should be REJECTED
        mock_base = tmp_path / "mock_base.json"
        mock_base.write_text(json.dumps({"metrics": {"mAP_50": 0.99, "mAP_50_95": 0.95}}), encoding="utf-8")

        strict_cfg = QualityThresholdConfig(max_map_drop=0.01)
        rep_reject = validate_quantized_model("yolo_nano", "fp32", onnx_p, mini_manifest_p, mock_base, strict_cfg)
        assert rep_reject["status"] == "REJECTED"
        assert "exceeds threshold" in rep_reject["rejection_reason"]

    def test_validate_industrial_quantized_model(self, tmp_path: Path) -> None:
        """Tests validate_quantized_model on industrial autoencoder model and rejection gating."""
        root = Path(__file__).resolve().parent.parent
        manifest_p = root / "data" / "sample_images" / "manifest.json"
        base_p = root / "results" / "raw" / "pytorch_fp32" / "industrial_autoencoder" / "baseline_metrics.json"
        onnx_p = root / "models" / "exported" / "industrial_autoencoder_fp32_opset17.onnx"

        if not manifest_p.is_file() or not base_p.is_file() or not onnx_p.is_file():
            pytest.skip("Required benchmark artifacts not present")

        # Create mini manifest for fast unit testing
        full_manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        mini_manifest = {
            "detection_samples": full_manifest.get("detection_samples", [])[:2],
            "industrial_samples": full_manifest.get("industrial_samples", [])[:2],
        }
        mini_manifest_p = tmp_path / "mini_manifest_ind.json"
        mini_manifest_p.write_text(json.dumps(mini_manifest), encoding="utf-8")

        # Test pass case
        pass_cfg = QualityThresholdConfig(max_auroc_drop=0.05)
        rep_pass = validate_quantized_model("industrial_autoencoder", "fp32", onnx_p, mini_manifest_p, base_p, pass_cfg)
        assert rep_pass["status"] == "PASS"

        # Test reject case
        mock_base = tmp_path / "mock_ind_base.json"
        mock_base.write_text(json.dumps({"metrics": {"image_auroc": 2.0, "aupro": 2.0}}), encoding="utf-8")

        strict_cfg = QualityThresholdConfig(max_auroc_drop=0.0)
        rep_reject = validate_quantized_model("industrial_autoencoder", "fp32", onnx_p, mini_manifest_p, mock_base, strict_cfg)
        assert rep_reject["status"] == "REJECTED"
