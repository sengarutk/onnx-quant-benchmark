"""
Unit tests for strict deterministic random state initialization.
"""

import os
import random
import numpy as np
import pytest
from src.common.seed import seed_everything


class TestSeed:
    """Test suite validating deterministic PRNG initialization."""

    def test_seed_everything_deterministic_numpy(self) -> None:
        """Verifies seed_everything produces deterministic NumPy random sequences."""
        seed_everything(1234)
        sample1 = np.random.rand(5)

        seed_everything(1234)
        sample2 = np.random.rand(5)

        assert np.allclose(sample1, sample2)

    def test_seed_everything_deterministic_python(self) -> None:
        """Verifies seed_everything produces deterministic Python random choices."""
        seed_everything(42)
        v1 = [random.random() for _ in range(5)]

        seed_everything(42)
        v2 = [random.random() for _ in range(5)]

        assert v1 == v2

    def test_seed_everything_env_var(self) -> None:
        """Verifies PYTHONHASHSEED is set to the provided seed."""
        seed_everything(999)
        assert os.environ.get("PYTHONHASHSEED") == "999"

    def test_seed_everything_torch(self) -> None:
        """Verifies seed_everything controls PyTorch determinism and cuDNN flags."""
        try:
            import torch

            seed_everything(42)
            t1 = torch.rand(4, 4)

            seed_everything(42)
            t2 = torch.rand(4, 4)

            assert torch.equal(t1, t2)
            assert torch.backends.cudnn.deterministic is True
            assert torch.backends.cudnn.benchmark is False
        except ImportError:
            pytest.skip("PyTorch not installed")
