"""Device management utilities for PyTorch models."""

import torch


def get_device(cuda: bool | None = None) -> torch.device:
    """Get the appropriate device for model training.

    Args:
        cuda: If True, use CUDA if available. If False, use CPU.
              If None (default), use CUDA if available, else CPU.

    Returns:
        torch.device: The selected device.
    """
    if cuda is False:
        return torch.device("cpu")
    if cuda is True:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    # cuda is None: auto-select
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
