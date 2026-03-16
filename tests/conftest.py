"""Pytest configuration and fixtures."""

import random

import numpy as np
import pytest
import torch


@pytest.fixture(autouse=True)
def deterministic_seed() -> None:
    """Set deterministic seed for reproducibility.

    Resets Python, NumPy, and PyTorch random seeds to 42 before each test.
    """
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
