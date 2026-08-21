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
        assert report["outputs"]["output0"]["max_abs_error"] < 1e-4
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
        assert report["outputs"]["reconstruction"]["max_abs_error"] < 1e-4
        assert report["outputs"]["anomaly_map"]["max_abs_error"] < 1e-4

    def test_evaluate_onnx_dataset_parity(self) -> None:
        """Verifies full dataset task metrics match PyTorch baseline within 0.0001."""
        root_dir = Path(__file__).resolve().parent.parent
        manifest_file = root_dir / "data" / "sample_images" / "manifest.json"

        if not manifest_file.is_file():
            pytest.skip("Sample dataset manifest not generated")

        # Test industrial autoencoder dataset parity if exported model exists
        exported_dir = root_dir / "models" / "exported"
        ind_onnx = exported_dir / "industrial_autoencoder_fp32_opset17.onnx"

        if not ind_onnx.is_file():
            ind_onnx = export_model_family("industrial_autoencoder", output_dir=exported_dir)

        report = evaluate_onnx_dataset(
            "industrial_autoencoder",
            ind_onnx,
            sample_manifest_path=manifest_file,
        )
        assert report["parity_passed"] is True
