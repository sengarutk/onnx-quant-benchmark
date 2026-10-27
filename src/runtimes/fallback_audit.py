"""
Execution Provider & Fallback Auditor trapping silent CPU fallback during GPU execution.

Scope: session-level provider registration only. ONNX Runtime may still assign individual unsupported nodes to
the CPU provider inside a GPU session; per-node placement can be inspected with ORT verbose logging/profiling.
"""

from typing import Any, Dict
import onnxruntime as ort

from src.common.logging import setup_logger
from src.runtimes.base import BaseRuntime

logger = setup_logger("fallback_audit")


class FallbackAuditor:
    """Audits and validates active execution provider placement."""

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
            RuntimeError: If the requested provider is not registered in the session.
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
        """
        Asserts that a runtime executes on the provider it claims.

        * ORT runtimes: the claimed provider must be the primary provider of the live session.
        * PyTorch CUDA: the model parameters must reside on a CUDA device.
        """
        claimed = runtime.get_active_provider()
        session = getattr(runtime, "session", None)
        if session is not None:
            FallbackAuditor.audit_ort_session(session, claimed)
            primary = session.get_providers()[0]
            if primary != claimed:
                raise RuntimeError(f"Runtime claims '{claimed}' but session primary provider is '{primary}'")
            return
        model = getattr(runtime, "model", None)
        if model is not None and "CUDA" in claimed.upper():
            devices = {p.device.type for p in model.parameters()}
            if devices != {"cuda"}:
                raise RuntimeError(f"Runtime claims '{claimed}' but parameters reside on {sorted(devices)}")


def audit_ort_session(session: ort.InferenceSession, requested_provider: str) -> Dict[str, Any]:
    return FallbackAuditor.audit_ort_session(session, requested_provider)


def assert_zero_fallback(runtime: BaseRuntime) -> None:
    return FallbackAuditor.assert_zero_fallback(runtime)
