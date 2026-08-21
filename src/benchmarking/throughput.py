"""
Throughput calculation utilities.
"""


def compute_throughput(latency_ms: float, batch_size: int = 1) -> float:
    """
    Computes throughput in frames per second (FPS).

    Args:
        latency_ms: Latency in milliseconds.
        batch_size: Number of images per inference step.

    Returns:
        Throughput in FPS (frames / sec).
    """
    if latency_ms <= 0.0:
        return 0.0
    return float(batch_size / (latency_ms / 1000.0))
