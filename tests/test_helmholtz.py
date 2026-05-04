# pyright: basic, reportMissingImports=false

import math

import pytest
import torch

from pinn4csi.physics import (
    helmholtz_residual,
    helmholtz_residual_loss,
    helmholtz_wavenumber,
)


def test_helmholtz_wavenumber_matches_closed_form() -> None:
    frequency = torch.tensor([2.4e9], dtype=torch.float64)
    wavenumber = helmholtz_wavenumber(frequency)
    expected = (2.0 * math.pi * frequency) / 3.0e8
    assert torch.allclose(wavenumber, expected)


def test_helmholtz_residual_is_small_for_analytical_solution() -> None:
    num_points = 128
    coordinates = torch.linspace(
        -1.0,
        1.0,
        num_points,
        dtype=torch.float64,
        requires_grad=True,
    ).unsqueeze(-1)
    wavenumber = 5.0

    field = torch.sin(wavenumber * coordinates[:, 0])
    residual = helmholtz_residual(
        field=field,
        coordinates=coordinates,
        wavenumber=wavenumber,
    )

    assert residual.shape == (num_points,)
    assert torch.isfinite(residual).all()
    assert torch.max(torch.abs(residual)).item() < 1e-5


def test_helmholtz_residual_detects_non_solution() -> None:
    coordinates = torch.linspace(
        -1.0,
        1.0,
        64,
        dtype=torch.float64,
        requires_grad=True,
    ).unsqueeze(-1)
    field = coordinates[:, 0] ** 2

    residual = helmholtz_residual(
        field=field,
        coordinates=coordinates,
        wavenumber=4.0,
    )

    assert torch.mean(torch.abs(residual)).item() > 0.1


def test_helmholtz_residual_loss_has_gradient_flow() -> None:
    coordinates = torch.linspace(-1.0, 1.0, 64, dtype=torch.float64).unsqueeze(-1)
    coordinates = coordinates.requires_grad_(True)
    amplitude = torch.tensor(0.7, dtype=torch.float64, requires_grad=True)
    field = amplitude * (coordinates[:, 0] ** 2)

    loss = helmholtz_residual_loss(
        field=field,
        coordinates=coordinates,
        wavenumber=3.5,
    )
    grad_amplitude = torch.autograd.grad(
        loss,
        amplitude,
        create_graph=True,
    )[0]
    grad_coordinates = torch.autograd.grad(
        loss,
        coordinates,
        create_graph=True,
    )[0]

    assert torch.isfinite(loss)
    assert grad_amplitude is not None
    assert grad_coordinates is not None
    assert torch.isfinite(grad_amplitude)
    assert torch.isfinite(grad_coordinates).all()
    assert torch.abs(grad_amplitude).item() > 0.0


# ============================================================================
# Error handling tests for helmholtz_wavenumber
# ============================================================================


def test_helmholtz_wavenumber_rejects_zero_wave_speed() -> None:
    """Test that wave_speed=0 raises ValueError."""
    frequency = torch.tensor([2.4e9], dtype=torch.float64)
    with pytest.raises(ValueError, match="wave_speed must be positive"):
        helmholtz_wavenumber(frequency, wave_speed=0.0)


def test_helmholtz_wavenumber_rejects_negative_wave_speed() -> None:
    """Test that negative wave_speed raises ValueError."""
    frequency = torch.tensor([2.4e9], dtype=torch.float64)
    with pytest.raises(ValueError, match="wave_speed must be positive"):
        helmholtz_wavenumber(frequency, wave_speed=-1.5e8)


def test_helmholtz_wavenumber_rejects_zero_frequency() -> None:
    """Test that zero frequency raises ValueError."""
    with pytest.raises(ValueError, match="frequency must be positive"):
        helmholtz_wavenumber(0.0)


def test_helmholtz_wavenumber_rejects_negative_frequency() -> None:
    """Test that negative frequency raises ValueError."""
    with pytest.raises(ValueError, match="frequency must be positive"):
        helmholtz_wavenumber(-2.4e9)


def test_helmholtz_wavenumber_rejects_tensor_with_zero_frequency() -> None:
    """Test that tensor with zero frequency raises ValueError."""
    frequency = torch.tensor([2.4e9, 0.0, 5.0e9], dtype=torch.float64)
    with pytest.raises(ValueError, match="frequency must be positive"):
        helmholtz_wavenumber(frequency)


def test_helmholtz_wavenumber_rejects_tensor_with_negative_frequency() -> None:
    """Test that tensor with negative frequency raises ValueError."""
    frequency = torch.tensor([2.4e9, -1.0e9, 5.0e9], dtype=torch.float64)
    with pytest.raises(ValueError, match="frequency must be positive"):
        helmholtz_wavenumber(frequency)


def test_helmholtz_wavenumber_accepts_scalar_float() -> None:
    """Test that scalar float frequency works."""
    wavenumber = helmholtz_wavenumber(2.4e9)
    assert torch.is_tensor(wavenumber)
    assert wavenumber.shape == torch.Size([])


def test_helmholtz_wavenumber_accepts_multidim_tensor() -> None:
    """Test that multi-dimensional frequency tensor works."""
    frequency = torch.tensor([[2.4e9, 5.0e9], [3.0e9, 4.0e9]], dtype=torch.float64)
    wavenumber = helmholtz_wavenumber(frequency)
    assert wavenumber.shape == frequency.shape


# ============================================================================
# Error handling tests for helmholtz_residual
# ============================================================================


def test_helmholtz_residual_rejects_complex_field() -> None:
    """Test that complex-valued field raises TypeError."""
    coordinates = torch.linspace(
        -1.0, 1.0, 32, dtype=torch.complex128, requires_grad=True
    ).unsqueeze(-1)
    field = torch.complex(
        torch.ones(32, dtype=torch.float64),
        torch.ones(32, dtype=torch.float64),
    )
    with pytest.raises(TypeError, match="field must be real-valued"):
        helmholtz_residual(field=field, coordinates=coordinates, wavenumber=5.0)


def test_helmholtz_residual_rejects_1d_coordinates() -> None:
    """Test that 1D coordinates (missing spatial_dim) raises ValueError."""
    coordinates = torch.linspace(-1.0, 1.0, 32, dtype=torch.float64)
    field = torch.sin(coordinates)
    with pytest.raises(ValueError, match="coordinates must have shape"):
        helmholtz_residual(field=field, coordinates=coordinates, wavenumber=5.0)


def test_helmholtz_residual_rejects_zero_spatial_dim() -> None:
    """Test that coordinates with zero spatial dimensions raises ValueError."""
    coordinates = torch.zeros((32, 0), dtype=torch.float64, requires_grad=True)
    field = torch.ones(32, dtype=torch.float64)
    with pytest.raises(ValueError, match="at least one spatial dimension"):
        helmholtz_residual(field=field, coordinates=coordinates, wavenumber=5.0)


def test_helmholtz_residual_rejects_coordinates_without_requires_grad() -> None:
    """Test that coordinates without requires_grad=True raises ValueError."""
    coordinates = torch.linspace(
        -1.0, 1.0, 32, dtype=torch.float64, requires_grad=False
    ).unsqueeze(-1)
    field = torch.sin(coordinates[:, 0])
    with pytest.raises(ValueError, match="requires_grad=True"):
        helmholtz_residual(field=field, coordinates=coordinates, wavenumber=5.0)


def test_helmholtz_residual_rejects_non_differentiable_field() -> None:
    """Test that non-differentiable field raises ValueError."""
    coordinates = torch.linspace(
        -1.0, 1.0, 32, dtype=torch.float64, requires_grad=True
    ).unsqueeze(-1)
    # Create a field that doesn't depend on coordinates
    field = torch.ones(32, dtype=torch.float64)
    with pytest.raises(ValueError, match="must be differentiable"):
        helmholtz_residual(field=field, coordinates=coordinates, wavenumber=5.0)


def test_helmholtz_residual_accepts_field_with_trailing_dimension() -> None:
    """Test that field with shape (*point_shape, 1) is accepted."""
    coordinates = torch.linspace(
        -1.0, 1.0, 32, dtype=torch.float64, requires_grad=True
    ).unsqueeze(-1)
    # Field with shape (32, 1) instead of (32,)
    field = torch.sin(coordinates[:, 0]).unsqueeze(-1)
    residual = helmholtz_residual(field=field, coordinates=coordinates, wavenumber=5.0)
    assert residual.shape == (32,)


def test_helmholtz_residual_accepts_multidim_coordinates() -> None:
    """Test that multi-dimensional coordinates work."""
    # 2D grid: (4, 4, 2) coordinates
    x = torch.linspace(-1.0, 1.0, 4, dtype=torch.float64)
    y = torch.linspace(-1.0, 1.0, 4, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    coordinates = torch.stack([xx, yy], dim=-1).requires_grad_(True)
    # Field must depend on coordinates for gradient computation
    field = torch.sin(3.0 * coordinates[..., 0]) * torch.cos(3.0 * coordinates[..., 1])
    residual = helmholtz_residual(field=field, coordinates=coordinates, wavenumber=3.0)
    assert residual.shape == (4, 4)


# ============================================================================
# Error handling tests for wavenumber handling in helmholtz_residual
# ============================================================================


def test_helmholtz_residual_accepts_scalar_wavenumber() -> None:
    """Test that scalar wavenumber works."""
    coordinates = torch.linspace(
        -1.0, 1.0, 32, dtype=torch.float64, requires_grad=True
    ).unsqueeze(-1)
    field = torch.sin(5.0 * coordinates[:, 0])
    residual = helmholtz_residual(field=field, coordinates=coordinates, wavenumber=5.0)
    assert residual.shape == (32,)


def test_helmholtz_residual_accepts_per_point_wavenumber() -> None:
    """Test that per-point wavenumber with shape (num_points,) works."""
    coordinates = torch.linspace(
        -1.0, 1.0, 32, dtype=torch.float64, requires_grad=True
    ).unsqueeze(-1)
    field = torch.sin(5.0 * coordinates[:, 0])
    wavenumber = torch.full((32,), 5.0, dtype=torch.float64)
    residual = helmholtz_residual(
        field=field, coordinates=coordinates, wavenumber=wavenumber
    )
    assert residual.shape == (32,)


def test_helmholtz_residual_accepts_per_point_wavenumber_multidim() -> None:
    """Test that per-point wavenumber matching point_shape works."""
    # 2D grid: (4, 4, 2) coordinates
    x = torch.linspace(-1.0, 1.0, 4, dtype=torch.float64)
    y = torch.linspace(-1.0, 1.0, 4, dtype=torch.float64)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    coordinates = torch.stack([xx, yy], dim=-1).requires_grad_(True)
    # Field must depend on coordinates for gradient computation
    field = torch.sin(3.0 * coordinates[..., 0]) * torch.cos(3.0 * coordinates[..., 1])
    wavenumber = torch.full((4, 4), 3.0, dtype=torch.float64)
    residual = helmholtz_residual(
        field=field, coordinates=coordinates, wavenumber=wavenumber
    )
    assert residual.shape == (4, 4)


def test_helmholtz_residual_rejects_invalid_wavenumber_shape() -> None:
    """Test that invalid wavenumber shape raises ValueError."""
    coordinates = torch.linspace(
        -1.0, 1.0, 32, dtype=torch.float64, requires_grad=True
    ).unsqueeze(-1)
    field = torch.sin(5.0 * coordinates[:, 0])
    # Wavenumber with wrong shape (16,) instead of (32,)
    wavenumber = torch.full((16,), 5.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="wavenumber must be scalar or per-point"):
        helmholtz_residual(field=field, coordinates=coordinates, wavenumber=wavenumber)


def test_helmholtz_residual_rejects_invalid_field_shape() -> None:
    """Test that field with invalid shape raises ValueError."""
    coordinates = torch.linspace(
        -1.0, 1.0, 32, dtype=torch.float64, requires_grad=True
    ).unsqueeze(-1)
    # Field with wrong shape (16,) instead of (32,)
    field = torch.sin(5.0 * torch.linspace(-1.0, 1.0, 16, dtype=torch.float64))
    with pytest.raises(ValueError, match="field must have shape"):
        helmholtz_residual(field=field, coordinates=coordinates, wavenumber=5.0)
