"""
Unit tests for ONNX export, graph verification, simplification, and inspection.
"""

from pathlib import Path
import onnx
import pytest
import torch
import torch.nn as nn

from src.export.export_onnx import export_to_onnx
from src.export.inspect_graph import inspect_onnx_graph
from src.export.simplify_onnx import simplify_onnx_graph
from src.export.validate_onnx_graph import validate_onnx_graph


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


class TestExportValidation:
    """Test suite validating ONNX export pipelines and graph utilities."""

    def test_export_to_onnx_and_metadata(self, tmp_path: Path) -> None:
        """Tests export_to_onnx creates valid model with injected metadata."""
        model = DummyModel()
        dummy_in = torch.randn(1, 3, 32, 32)
        out_file = tmp_path / "dummy_model.onnx"

        meta = {"model_name": "dummy", "custom_key": "custom_value"}
        export_path = export_to_onnx(
            model=model,
            dummy_input=dummy_in,
            output_path=out_file,
            input_names=["input"],
            output_names=["output"],
            opset_version=17,
            metadata=meta,
        )

        assert export_path.is_file()
        assert out_file.with_name(out_file.name + ".sha256").is_file()
        assert out_file.with_name(out_file.stem + ".metadata.json").is_file()

        # Check metadata properties
        loaded = onnx.load(str(export_path))
        prop_dict = {p.key: p.value for p in loaded.metadata_props}
        assert prop_dict["model_name"] == "dummy"
        assert prop_dict["custom_key"] == "custom_value"
        assert "git_commit" in prop_dict

    def test_validate_onnx_graph(self, tmp_path: Path) -> None:
        """Tests validate_onnx_graph extracts correct input/output signatures."""
        model = DummyModel()
        dummy_in = torch.randn(1, 3, 32, 32)
        out_file = tmp_path / "valid_model.onnx"

        export_to_onnx(model, dummy_in, out_file, ["input"], ["output"], opset_version=17)
        report = validate_onnx_graph(out_file)

        assert report["valid"] is True
        assert report["opset_version"] == 17
        assert "input" in report["inputs"]
        assert report["inputs"]["input"]["shape"] == [1, 3, 32, 32]
        assert "output" in report["outputs"]
        assert report["outputs"]["output"]["shape"] == [1, 8, 32, 32]

    def test_simplify_onnx_graph(self, tmp_path: Path) -> None:
        """Tests simplify_onnx_graph runs onnxsim and updates checksum."""
        model = DummyModel()
        dummy_in = torch.randn(1, 3, 32, 32)
        out_file = tmp_path / "orig_model.onnx"
        sim_out_file = tmp_path / "sim_model.onnx"

        export_to_onnx(model, dummy_in, out_file, ["input"], ["output"], opset_version=17)
        sim_path, report = simplify_onnx_graph(out_file, sim_out_file)

        assert sim_path.is_file()
        assert report["check_ok"] is True
        assert report["nodes_after"] <= report["nodes_before"]

    def test_inspect_onnx_graph(self, tmp_path: Path) -> None:
        """Tests inspect_onnx_graph calculates parameter count and operator distributions."""
        model = DummyModel()
        dummy_in = torch.randn(1, 3, 32, 32)
        out_file = tmp_path / "inspect_model.onnx"

        export_to_onnx(model, dummy_in, out_file, ["input"], ["output"], opset_version=17)
        report = inspect_onnx_graph(out_file)

        assert report["total_nodes"] > 0
        assert "Conv" in report["operator_counts"]
        assert report["total_parameters"] > 0
        assert report["file_size_mb"] > 0.0
