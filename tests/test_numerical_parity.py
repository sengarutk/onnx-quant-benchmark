"""
Unit tests for PyTorch vs ONNX Runtime numerical equivalence and parity validation.
"""

from pathlib import Path
import pytest
import torch

from src.export.export_onnx import export_model_family
from src.models.industrial_model_adapter import IndustrialModelAdapter
from src.models.yolo_adapter import YOLOAdapter
from src.validation.numerical_equivalence import compare_pytorch_vs_onnx, evaluate_onnx_dataset
from tests.conftest import requires_artifacts


class TestNumericalParity:
    """Test suite validating numerical parity between PyTorch and ONNX Runtime."""

    def test_yolo_nano_pytorch_vs_onnx_parity(self, tmp_path: Path) -> None:
        """Verifies YOLO nano PyTorch vs ONNX Runtime raw tensor parity satisfies L_inf < 1e-4."""
        adapter = YOLOAdapter()
        model = adapter.get_pytorch_model()
        dummy_in = torch.randn(1, 3, 640, 640)

        onnx_file = export_model_family("yolo_nano", output_dir=tmp_path, model_instance=model)
        report = compare_pytorch_vs_onnx(model, onnx_file, dummy_in)

        assert report["all_passed"] is True
        assert "output0" in report["outputs"]
        assert report["outputs"]["output0"]["max_rel_error"] < 1e-5
        assert report["outputs"]["output0"]["cosine_similarity"] > 0.99999

    def test_industrial_autoencoder_pytorch_vs_onnx_parity(self, tmp_path: Path) -> None:
        """Verifies Industrial Autoencoder PyTorch vs ONNX parity on both outputs."""
        class Wrapper(torch.nn.Module):
            def __init__(self, core):
                super().__init__()
                self.core = core

            def forward(self, x):
                recon = self.core(x)
                a_map = torch.mean(torch.abs(x - recon), dim=1, keepdim=True)
                return recon, a_map

        adapter = IndustrialModelAdapter()
        model = Wrapper(adapter.get_pytorch_model())
        dummy_in = torch.randn(1, 3, 256, 256)

        onnx_file = export_model_family("industrial_autoencoder", output_dir=tmp_path, model_instance=model)
        report = compare_pytorch_vs_onnx(model, onnx_file, dummy_in)

        assert report["all_passed"] is True
        assert "reconstruction" in report["outputs"]
        assert "anomaly_map" in report["outputs"]
        assert report["outputs"]["reconstruction"]["max_rel_error"] < 1e-5
        assert report["outputs"]["anomaly_map"]["max_rel_error"] < 1e-5

    @requires_artifacts
    def test_evaluate_onnx_dataset_matches_pytorch_baseline(self) -> None:
        """ONNX Runtime FP32 reproduces the stored PyTorch FP32 test-split metrics."""
        import json
        root = Path(__file__).resolve().parent.parent
        for model, key in (("yolo_nano", "mAP_50"), ("industrial_autoencoder", "image_auroc")):
            base = root / "results" / "baselines" / f"{model}.json"
            if not base.is_file():
                pytest.skip("baselines not generated")
            report = evaluate_onnx_dataset(model, root / "models" / "exported" / f"{model}_fp32_opset17.onnx")
            assert abs(report["metrics"][key] - json.loads(base.read_text())["metrics"][key]) <= 1e-3
