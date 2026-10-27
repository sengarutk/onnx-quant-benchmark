"""Tests for the pipeline orchestrator and repository metadata."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class TestFullPipeline:
    def test_run_full_pipeline_dry_run(self) -> None:
        res = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_full_pipeline.py"), "--dry-run"],
                             cwd=str(ROOT), capture_output=True, text=True)
        assert res.returncode == 0
        assert "Train detector and autoencoder" in res.stdout
        assert "Q-Aware NMS" in res.stdout

    def test_metadata_files_consistent(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
        citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
        pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert len(readme) > 2000
        assert "Apache License" in license_text and "APPENDIX: How to apply the Apache License" in license_text
        assert "9. Accepting Warranty or Additional Liability" in license_text
        assert "Utkarsh" in citation and "sengarutk/onnx-quant-benchmark" in citation
        version = [l for l in pyproject.splitlines() if l.startswith("version")][0].split('"')[1]
        assert f"version: {version}" in citation or f'version: "{version}"' in citation
        import src
        assert src.__version__ == version
