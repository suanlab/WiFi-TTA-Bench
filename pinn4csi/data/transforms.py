"""CSI data preprocessing and transformation utilities."""

import torch
from torch import Tensor


def has_meaningful_phase(csi: Tensor, energy_threshold: float = 1e-4) -> bool:
    """Detect whether CSI tensor contains non-trivial phase information.

    Checks the ratio of imaginary energy to total energy. Real-valued datasets
    (e.g. UT-HAR) have zero imaginary component, making phase-dependent physics
    losses meaningless.

    Args:
        csi: Complex or real CSI tensor. Any shape.
        energy_threshold: Minimum ratio of imaginary energy to total energy
            to consider phase meaningful.

    Returns:
        True if the imaginary component carries signal, False if absent
        or noise-level.
    """
    if not csi.is_complex():
        return False
    imag_energy = torch.mean(csi.imag**2).item()
    total_energy = torch.mean(torch.abs(csi) ** 2).item()
    if total_energy < 1e-12:
        return False
    return (imag_energy / total_energy) > energy_threshold


def amplitude_phase_split(csi_complex: Tensor) -> Tensor:
    """Convert complex CSI to amplitude and phase channels.

    Args:
        csi_complex: Complex-valued CSI tensor. Shape:
            (..., num_subcarriers, num_antennas) or (..., num_subcarriers).

    Returns:
        Real-valued tensor with amplitude and phase as separate channels.
        Shape: (..., num_subcarriers, 2*num_antennas) or
        (..., num_subcarriers, 2).
    """
    amplitude = torch.abs(csi_complex)
    phase = torch.angle(csi_complex)
    # Stack amplitude and phase for each antenna/subcarrier
    # Shape: (..., num_subcarriers, num_antennas) ->
    # (..., num_subcarriers, 2*num_antennas)
    if csi_complex.dim() == 2:
        # Shape: (num_subcarriers, num_antennas)
        result = torch.stack(
            [amplitude, phase], dim=-1
        )  # (num_subcarriers, num_antennas, 2)
        result = result.reshape(
            csi_complex.shape[0], -1
        )  # (num_subcarriers, 2*num_antennas)
    else:
        # Shape: (batch, num_subcarriers, num_antennas)
        result = torch.stack(
            [amplitude, phase], dim=-1
        )  # (batch, num_subcarriers, num_antennas, 2)
        result = result.reshape(
            result.shape[0], result.shape[1], -1
        )  # (batch, num_subcarriers, 2*num_antennas)
    return result


def normalize_amplitude(amplitude: Tensor, eps: float = 1e-8) -> Tensor:
    """Normalize amplitude to [0, 1] range per subcarrier.

    Args:
        amplitude: Amplitude tensor. Shape: (..., num_subcarriers, num_antennas).
        eps: Small constant to avoid division by zero.

    Returns:
        Normalized amplitude. Shape: same as input.
    """
    # Normalize per subcarrier across antennas
    max_val = torch.amax(amplitude, dim=-1, keepdim=True)
    min_val = torch.amin(amplitude, dim=-1, keepdim=True)
    normalized = (amplitude - min_val) / (max_val - min_val + eps)
    return normalized


def select_subcarriers(
    csi: Tensor, subcarrier_indices: list[int] | None = None
) -> Tensor:
    """Select a subset of subcarriers from CSI data.

    Args:
        csi: CSI tensor. Shape: (..., num_subcarriers, num_antennas) or
            (..., num_subcarriers, features).
        subcarrier_indices: List of subcarrier indices to select.
            If None, use all.

    Returns:
        CSI with selected subcarriers. Shape:
        (..., len(subcarrier_indices), num_antennas/features).
    """
    if subcarrier_indices is None:
        return csi
    return csi[..., subcarrier_indices, :]


def center_frequency_to_subcarrier_index(
    center_freq_hz: float, num_subcarriers: int = 56, bandwidth_mhz: float = 20.0
) -> int:
    """Convert center frequency to subcarrier index for 802.11n/ac.

    Args:
        center_freq_hz: Center frequency in Hz.
        num_subcarriers: Total number of subcarriers (default 56 for 20MHz).
        bandwidth_mhz: Bandwidth in MHz (default 20.0).

    Returns:
        Subcarrier index (0-based).
    """
    # Subcarrier spacing: bandwidth / num_subcarriers
    subcarrier_spacing = (bandwidth_mhz * 1e6) / num_subcarriers
    # Assume center frequency is at subcarrier index num_subcarriers // 2
    center_index = num_subcarriers // 2
    # Offset from center
    offset = center_freq_hz / subcarrier_spacing
    index = int(center_index + offset)
    return max(0, min(index, num_subcarriers - 1))
