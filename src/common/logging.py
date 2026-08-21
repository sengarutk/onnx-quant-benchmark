"""
Logging utility module providing standardized, structured logging across all benchmark components.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "edge_benchmark",
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configures and returns a logger instance with ISO-8601 timestamps and standard formatting.

    Args:
        name: Name of the logger namespace.
        log_file: Optional path to a file where log entries will be appended.
        level: Logging verbosity level (default: logging.INFO).

    Returns:
        Configured logging.Logger instance with duplicate handler protection.
    """
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
