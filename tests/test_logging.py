"""
Unit tests for the structured logging subsystem.
"""

import logging
from pathlib import Path
from src.common.logging import setup_logger


class TestLogging:
    """Test suite validating logger initialization, formatting, and file outputs."""

    def test_setup_logger_basic(self) -> None:
        """Tests basic stream logger setup."""
        logger = setup_logger("test_basic", level=logging.DEBUG)
        assert logger.name == "test_basic"
        assert logger.level == logging.DEBUG
        assert len(logger.handlers) >= 1

    def test_setup_logger_file_output(self, tmp_path: Path) -> None:
        """Tests logger file handler creation and log persistence."""
        log_file = tmp_path / "sub_dir" / "benchmark.log"
        logger = setup_logger("test_file_logger", log_file=log_file, level=logging.INFO)
        logger.info("Testing file logging output.")

        assert log_file.is_file()
        content = log_file.read_text(encoding="utf-8")
        assert "Testing file logging output." in content
        assert "[INFO]" in content
        assert "[test_file_logger:" in content

    def test_setup_logger_duplicate_protection(self) -> None:
        """Ensures repeated setup calls do not duplicate handlers."""
        logger1 = setup_logger("test_dup", level=logging.INFO)
        num_handlers_initial = len(logger1.handlers)

        logger2 = setup_logger("test_dup", level=logging.INFO)
        assert len(logger2.handlers) == num_handlers_initial
