"""OFDM channel response and lightweight frequency-domain constraints.

This module provides a minimal OFDM analytical response model that can be used as
physics supervision for CSI prediction tasks.
"""

# pyright: reportMissingImports=false

import math

import torch
from torch import Tensor


def ofdm_channel_response(h_l: Tensor, tau_l: Tensor, f_k: Tensor) -> Tensor:
    """Compute OFDM frequency response from multipath parameters.

    The model follows:

        H(f_k) = sum_l h_l * exp(-j * 2 * pi * f_k * tau_l)

    Args:
        h_l: Complex path gains. Shape: (..., num_paths).
        tau_l: Path delays in seconds. Shape: (..., num_paths).
        f_k: Subcarrier frequencies in Hz. Shape: (num_subcarriers,) or
            (..., num_subcarriers) with leading dimensions matching `h_l`.

    Returns:
        Complex channel frequency response. Shape: (..., num_subcarriers).

    Raises:
        TypeError: If `h_l` is not complex-valued.
        ValueError: If path-gain and delay shapes are incompatible.
        ValueError: If subcarrier frequency shape is incompatible.
    """
    if not torch.is_complex(h_l):
        raise TypeError("h_l must be a complex-valued tensor")
    if h_l.shape != tau_l.shape:
        raise ValueError(
            f"h_l and tau_l must share shape, got {h_l.shape} and {tau_l.shape}"
        )

    real_dtype = h_l.real.dtype
    tau = tau_l.to(device=h_l.device, dtype=real_dtype)
    frequencies = f_k.to(device=h_l.device, dtype=real_dtype)

    if frequencies.ndim == 1:
        phase = -2.0 * math.pi * tau.unsqueeze(-1) * frequencies
    elif frequencies.shape[:-1] == tau.shape[:-1]:
        phase = -2.0 * math.pi * tau.unsqueeze(-1) * frequencies.unsqueeze(-2)
    else:
        raise ValueError("f_k must be 1D or have leading dimensions matching h_l/tau_l")

    complex_phase = torch.complex(torch.zeros_like(phase), phase)
    steering = torch.exp(complex_phase)
    return torch.sum(h_l.unsqueeze(-1) * steering, dim=-2)


def ofdm_residual(
    predicted_csi: Tensor,
    path_gains: Tensor,
    path_delays: Tensor,
    subcarrier_frequencies: Tensor,
) -> Tensor:
    """Compute OFDM analytical consistency residual for predicted CSI.

    Args:
        predicted_csi: Predicted complex CSI. Shape: (..., num_subcarriers).
        path_gains: Complex path gains. Shape: (..., num_paths).
        path_delays: Path delays in seconds. Shape: (..., num_paths).
        subcarrier_frequencies: Subcarrier frequencies in Hz. Shape:
            (num_subcarriers,) or (..., num_subcarriers).

    Returns:
        Mean-squared complex mismatch to analytical OFDM response. Scalar tensor.

    Raises:
        ValueError: If predicted and analytical CSI shapes differ.
    """
    analytical_csi = ofdm_channel_response(
        h_l=path_gains,
        tau_l=path_delays,
        f_k=subcarrier_frequencies,
    )
    if predicted_csi.shape != analytical_csi.shape:
        raise ValueError(
            f"predicted_csi shape must match analytical OFDM response shape, "
            f"got {predicted_csi.shape} and {analytical_csi.shape}"
        )

    mismatch = predicted_csi - analytical_csi
    return torch.mean(torch.abs(mismatch) ** 2)


def subcarrier_correlation_loss(csi: Tensor) -> Tensor:
    """Encourage neighboring-subcarrier smoothness in CSI frequency response.

    This is a minimal regularizer that penalizes first-order differences across
    adjacent subcarriers.

    Args:
        csi: Complex CSI tensor with subcarrier axis in the last dimension.
            Shape: (..., num_subcarriers).

    Returns:
        Mean squared adjacent-subcarrier difference. Scalar tensor.
    """
    if csi.shape[-1] < 2:
        return torch.zeros((), dtype=csi.real.dtype, device=csi.device)

    adjacent_diff = csi[..., 1:] - csi[..., :-1]
    return torch.mean(torch.abs(adjacent_diff) ** 2)
