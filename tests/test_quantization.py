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
from tests.conftest import requires_artifacts

ROOT = Path(__file__).resolve().parent.parent


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

    def test_quantize_includes_conv_transpose(self, tmp_path: Path) -> None:
        """ConvTranspose layers (autoencoder decoder) are quantized, not silently left in FP32."""
        model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU(), nn.ConvTranspose2d(4, 3, 4, stride=2, padding=1))
        fp32, int8 = tmp_path / "ct_fp32.onnx", tmp_path / "ct_int8.onnx"
        export_to_onnx(model, torch.randn(1, 3, 16, 16), fp32, ["input"], ["output"], opset_version=17)
        paths = []
        for i in range(4):
            p = tmp_path / f"c{i}.npy"
            np.save(str(p), np.random.rand(1, 3, 16, 16).astype(np.float32))
            paths.append(p)
        reader = BenchmarkCalibrationDataReader(paths, "input", (1, 3, 16, 16), lambda q: np.load(str(q)))
        quantize_onnx_static(fp32, int8, reader)
        g = onnx.load(str(int8)).graph
        ct = [n for n in g.node if n.op_type == "ConvTranspose"][0]
        producers = {o: n.op_type for n in g.node for o in n.output}
        assert producers[ct.input[1]] == "DequantizeLinear"  # weights pass through Q/DQ

    @requires_artifacts
    def test_validate_quantized_model_gate(self, tmp_path: Path) -> None:
        """PASS against the real baseline; REJECTED against an artificially higher baseline."""
        manifest = json.loads((ROOT / "data" / "sample_images" / "manifest.json").read_text())
        mini = tmp_path / "mini.json"
        mini.write_text(json.dumps({"detection_samples": manifest["detection_samples"][:4],
                                    "industrial_samples": manifest["industrial_samples"][:2] + manifest["industrial_samples"][-2:]}))
        for model, metric in (("yolo_nano", "mAP_50"), ("industrial_autoencoder", "image_auroc")):
            onnx_p = ROOT / "models" / "exported" / f"{model}_fp32_opset17.onnx"
            fake = tmp_path / f"{model}_base.json"
            # baseline equal to the candidate itself -> delta 0 -> PASS
            from src.validation.numerical_equivalence import evaluate_onnx_dataset
            own = evaluate_onnx_dataset(model, onnx_p, mini)["metrics"]
            fake.write_text(json.dumps({"metrics": own}))
            assert validate_quantized_model(model, "fp32", onnx_p, mini, fake)["status"] == "PASS"
            fake.write_text(json.dumps({"metrics": {**own, metric: own[metric] + 0.5}}))
            rep = validate_quantized_model(model, "fp32", onnx_p, mini, fake, QualityThresholdConfig(max_map_drop=0.01, max_auroc_drop=0.01))
            assert rep["status"] == "REJECTED" and rep["rejection_reason"]
