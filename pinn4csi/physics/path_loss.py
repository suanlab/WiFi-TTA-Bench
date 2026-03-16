"""Log-distance path loss model for WiFi signal propagation.

This module implements the log-distance path loss model, which describes
how signal strength decreases with distance in indoor environments.
The model is anchored to the Friis free-space equation for physical consistency.
"""

import numpy as np
import torch
from torch import Tensor


def compute_path_loss(
    distance: Tensor | float,
    frequency: float,
    n: float = 2.0,
    reference_distance: float = 1.0,
) -> Tensor:
    """Compute log-distance path loss using a 1m reference.

    The log-distance path loss model describes signal attenuation with distance:

        PL(d) = PL(d0) + 10*n*log10(d/d0)

    where:
    - PL(d) is path loss at distance d (in dB)
    - PL(d0) is path loss at reference distance d0 (computed from Friis equation)
    - n is the path loss exponent (2.0 for free space, 1.6-3.3 for indoor)
    - d is the distance in meters
    - d0 is the reference distance in meters (default: 1.0m)

    The reference level PL(d0) is computed from the Friis free-space equation:

        PL(d0) = 20*log10(4*pi*d0*f/c)

    where f is frequency in Hz and c is the speed of light (3e8 m/s).

    Args:
        distance: Tx-Rx distance(s) in meters. Can be a scalar or tensor.
                  Shape: scalar or any shape (e.g., (batch,), (batch, n_samples)).
        frequency: Carrier frequency in Hz (e.g., 2.4e9 for 2.4 GHz).
        n: Path loss exponent (default: 2.0 for free space).
           Typical range: 1.6-3.3 for indoor environments.
        reference_distance: Reference distance d0 in meters (default: 1.0).
                           Must be positive.

    Returns:
        Path loss in dB. Shape matches input distance shape.
        Supports autograd: gradients flow through distance parameter.

    Raises:
        ValueError: If distance contains non-positive values.
        ValueError: If reference_distance is non-positive.

    Example:
        >>> import torch
        >>> from pinn4csi.physics.path_loss import compute_path_loss
        >>> distance = torch.tensor([1.0, 10.0, 100.0])
        >>> pl = compute_path_loss(distance, frequency=2.4e9, n=2.0)
        >>> print(pl)  # Path loss at 1m, 10m, 100m
    """
    # Convert distance to tensor if needed, preserving dtype
    if not isinstance(distance, Tensor):
        distance = torch.tensor(distance, dtype=torch.float32)
    # Keep original dtype (don't force float32)

    # Validate inputs
    if (distance <= 0).any():
        raise ValueError(
            f"distance must be positive, got min={distance.min().item():.6f}"
        )
    if reference_distance <= 0:
        raise ValueError(
            f"reference_distance must be positive, got {reference_distance}"
        )

    # Speed of light
    c = 3e8

    # Compute reference path loss at d0 using Friis equation
    # PL(d0) = 20*log10(4*pi*d0*f/c)
    # Create constant tensor on same device/dtype as distance for device safety
    friis_arg = 4 * np.pi * reference_distance * frequency / c
    pl_ref = 20 * torch.log10(
        torch.tensor(friis_arg, dtype=distance.dtype, device=distance.device)
    )

    # Compute path loss at distance d
    # PL(d) = PL(d0) + 10*n*log10(d/d0)
    pl = pl_ref + 10 * n * torch.log10(distance / reference_distance)

    return pl
