"""
Strict deterministic random state initialization utility for reproducible benchmarking.
"""

import os
import random
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """
    Seeds all random number generators and forces deterministic behavior in PyTorch/cuDNN.

    Args:
        seed: Integer seed value to initialize PRNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
