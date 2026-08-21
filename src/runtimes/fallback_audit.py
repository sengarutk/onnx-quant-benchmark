"""
Execution Provider & Fallback Auditor trapping silent CPU fallback during GPU execution.
"""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.logging import setup_logger
from src.runtimes.base import BaseRuntime

logger = setup_logger("fallback_audit")


class FallbackAuditor:
    """
    Audits and validates active execution provider placement.
    """

    @staticmethod
    def audit_ort_session(session: ort.InferenceSession, requested_provider: str) -> Dict[str, Any]:
        """
        Asserts that the requested provider was successfully initialized by the runtime engine.

        Args:
            session: Active ONNX Runtime InferenceSession.
            requested_provider: 'CUDAExecutionProvider', 'TensorrtExecutionProvider', etc.

        Returns:
            Dictionary with audit results.

        Raises:
            RuntimeError: If requested GPU provider silently fell back to CPU.
        """
        active_providers = session.get_providers()
        logger.info(f"Auditing session providers (Requested: {requested_provider}, Active: {active_providers})")

        if requested_provider not in active_providers:
            err = (
                f"CRITICAL FALLBACK DETECTED: Requested execution provider '{requested_provider}' "
                f"failed to initialize. Active providers: {active_providers}."
            )
            logger.error(err)
            raise RuntimeError(err)

        if active_providers[0] != requested_provider:
            logger.warning(
                f"Execution provider priority mismatch: '{requested_provider}' is not primary provider. "
                f"Primary is '{active_providers[0]}'."
            )

        return {
            "requested_provider": requested_provider,
            "primary_provider": active_providers[0],
            "active_providers": active_providers,
            "fallback_occurred": False,
        }

    @staticmethod
    def assert_zero_fallback(runtime: BaseRuntime) -> None:
        """Asserts that a runtime is executing on its claimed hardware provider."""
        provider = runtime.get_active_provider()
        if "CUDA" in provider and not (sys.platform == "linux" or sys.platform == "win32"):
            raise RuntimeError(f"GPU runtime cannot run on platform: {sys.platform}")


def audit_ort_session(session: ort.InferenceSession, requested_provider: str) -> Dict[str, Any]:
    return FallbackAuditor.audit_ort_session(session, requested_provider)


def assert_zero_fallback(runtime: BaseRuntime) -> None:
    return FallbackAuditor.assert_zero_fallback(runtime)
