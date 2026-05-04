"""Helmholtz PDE utilities for physics-informed WiFi imaging."""

# pyright: basic, reportMissingImports=false, reportAttributeAccessIssue=false

from __future__ import annotations

import math

import torch
from torch import Tensor

SPEED_OF_LIGHT = 3.0e8


def helmholtz_wavenumber(
    frequency: Tensor | float,
    wave_speed: float = SPEED_OF_LIGHT,
) -> Tensor:
    """Compute wavenumber from frequency using ``k = 2*pi*f/c``.

    Args:
        frequency: Carrier frequency in Hz. Shape: scalar or any tensor shape.
        wave_speed: Wave propagation speed in m/s. Defaults to free-space light
            speed.

    Returns:
        Wavenumber in rad/m. Shape matches input frequency shape.

    Raises:
        ValueError: If frequency contains non-positive values.
        ValueError: If wave_speed is non-positive.
    """
    if wave_speed <= 0:
        raise ValueError(f"wave_speed must be positive, got {wave_speed}")

    frequency_tensor = torch.as_tensor(frequency)

    if (frequency_tensor <= 0).any():
        min_frequency = float(frequency_tensor.min().item())
        raise ValueError(
            f"frequency must be positive, got minimum value {min_frequency}"
        )

    return (2.0 * math.pi * frequency_tensor) / wave_speed


def helmholtz_residual(
    field: Tensor,
    coordinates: Tensor,
    wavenumber: Tensor | float,
) -> Tensor:
    """Compute Helmholtz residual ``∇²u + k²u``.

    Args:
        field: Predicted scalar field values. Shape: (*point_shape,) or
            (*point_shape, 1).
        coordinates: Spatial coordinates where `field` is evaluated.
            Shape: (*point_shape, spatial_dim). Must have `requires_grad=True`.
        wavenumber: Helmholtz wavenumber in rad/m. Can be scalar or
            per-point tensor with shape (*point_shape,) or (num_points_flat,).

    Returns:
        Pointwise residual tensor. Shape: (*point_shape,).

    Raises:
        TypeError: If field is complex-valued.
        ValueError: If shapes are incompatible.
        ValueError: If coordinates do not track gradients.
        ValueError: If field is not differentiable with respect to coordinates.
    """
    if torch.is_complex(field):
        raise TypeError("field must be real-valued for helmholtz_residual")

    if coordinates.dim() < 2:
        raise ValueError(
            "coordinates must have shape (*point_shape, spatial_dim), "
            f"got {tuple(coordinates.shape)}"
        )
    if coordinates.shape[-1] < 1:
        raise ValueError("coordinates must include at least one spatial dimension")
    if not coordinates.requires_grad:
        raise ValueError("coordinates must have requires_grad=True")

    point_shape = tuple(coordinates.shape[:-1])
    field_values = _as_vector_field(field=field, point_shape=point_shape)

    try:
        first_order = torch.autograd.grad(
            field_values,
            coordinates,
            grad_outputs=torch.ones_like(field_values),
            create_graph=True,
            retain_graph=True,
        )[0]
    except RuntimeError as exc:
        raise ValueError(
            "field must be differentiable with respect to coordinates"
        ) from exc

    laplacian = torch.zeros_like(field_values)
    for dim_index in range(coordinates.shape[-1]):
        first_component = first_order[..., dim_index].reshape(-1)
        second_component = torch.autograd.grad(
            first_component,
            coordinates,
            grad_outputs=torch.ones_like(first_component),
            create_graph=True,
            retain_graph=True,
        )[0][..., dim_index].reshape(-1)
        laplacian = laplacian + second_component

    wave_number = _prepare_wavenumber(
        wavenumber=wavenumber,
        reference=field_values,
        point_shape=point_shape,
    )
    residual = laplacian + (wave_number**2) * field_values
    result: Tensor = residual.reshape(point_shape)
    return result


def helmholtz_residual_loss(
    field: Tensor,
    coordinates: Tensor,
    wavenumber: Tensor | float,
) -> Tensor:
    """Return mean-squared Helmholtz residual loss.

    Args:
        field: Predicted scalar field values. Shape: (num_points,) or
            (num_points, 1).
        coordinates: Spatial coordinates. Shape: (num_points, spatial_dim).
        wavenumber: Helmholtz wavenumber in rad/m.

    Returns:
        Mean squared pointwise Helmholtz residual. Scalar tensor.
    """
    residual = helmholtz_residual(
        field=field,
        coordinates=coordinates,
        wavenumber=wavenumber,
    )
    return torch.mean(residual**2)


def _as_vector_field(field: Tensor, point_shape: tuple[int, ...]) -> Tensor:
    if tuple(field.shape) == point_shape:
        return field.reshape(-1)
    if tuple(field.shape) == (*point_shape, 1):
        return field.reshape(-1)
    raise ValueError(
        "field must have shape (*point_shape,) or (*point_shape, 1), "
        f"got {tuple(field.shape)}"
    )


def _prepare_wavenumber(
    wavenumber: Tensor | float,
    reference: Tensor,
    point_shape: tuple[int, ...],
) -> Tensor:
    wave_number = torch.as_tensor(
        wavenumber,
        device=reference.device,
        dtype=reference.dtype,
    )

    if wave_number.dim() == 0:
        return wave_number
    if tuple(wave_number.shape) == point_shape:
        return wave_number.reshape(-1)
    if wave_number.dim() == 1 and wave_number.shape[0] == reference.shape[0]:
        return wave_number

    expected_shapes = [str(point_shape), f"({reference.shape[0]},)"]
    raise ValueError(
        "wavenumber must be scalar or per-point. "
        f"Expected one of {expected_shapes}, got {tuple(wave_number.shape)}"
    )
