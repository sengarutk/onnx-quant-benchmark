"""
Unit tests for hardware and environment introspection routines.
"""

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
    """Test suite validating environment introspection and markdown generation."""

    def test_collect_environment_manifest_returns_valid_instance(self) -> None:
        """Tests collect_environment_manifest returns populated EnvironmentManifest."""
        manifest = collect_environment_manifest()
        assert isinstance(manifest, EnvironmentManifest)
        assert manifest.timestamp != ""
        assert manifest.python_version != ""
        assert manifest.os_info != ""
        assert manifest.cpu_info != ""
        assert manifest.ram_total_gb > 0.0

    def test_save_environment_manifest(self, tmp_path: Path) -> None:
        """Tests serialization of EnvironmentManifest to a JSON file."""
        saved_file = save_environment_manifest(tmp_path)
        assert saved_file.is_file()
        assert saved_file.name.startswith("environment_")
        assert saved_file.name.endswith(".json")
        assert saved_file.stat().st_size > 0

    def test_generate_hardware_doc(self, tmp_path: Path) -> None:
        """Tests generation of Markdown hardware summary."""
        doc_file = tmp_path / "hardware.md"
        generate_hardware_doc(doc_file)
        assert doc_file.is_file()

        content = doc_file.read_text(encoding="utf-8")
        assert "# Hardware & Runtime Environment Specification" in content
        assert "System Host Architecture" in content
        assert "Deep Learning & Inference Engine Toolchain" in content

    def test_graceful_fallback_without_gpu(self) -> None:
        """Mocks GPU unavailability to verify graceful fallback."""
        with patch("torch.cuda.is_available", return_value=False):
            manifest = collect_environment_manifest()
            assert isinstance(manifest, EnvironmentManifest)
            # Should not crash and should record valid CPU environment
            assert manifest.cpu_info != ""
            assert manifest.ram_total_gb > 0.0

    def test_configure_cpu_threads(self) -> None:
        """Tests thread configuration helper."""
        from src.common.environment import configure_cpu_threads, get_physical_core_count
        cores = get_physical_core_count()
        assert cores >= 1
        applied = configure_cpu_threads(cores)
        assert applied == cores

    def test_get_unified_ort_session_options(self) -> None:
        """Tests unified ORT session options factory."""
        from src.common.environment import get_unified_ort_session_options
        import onnxruntime as ort
        opts = get_unified_ort_session_options(intra_op_threads=4, inter_op_threads=1)
        assert opts.intra_op_num_threads == 4
        assert opts.inter_op_num_threads == 1
        assert opts.execution_mode == ort.ExecutionMode.ORT_SEQUENTIAL
