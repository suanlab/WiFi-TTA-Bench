"""Tests for model architectures."""

import pytest
import torch
from torch import Tensor

from pinn4csi.models import PINN


@pytest.mark.parametrize("batch_size", [1, 16, 256])
def test_pinn_output_shape(batch_size: int) -> None:
    """PINN output shape matches batch and output dimensions."""
    model = PINN(in_features=12, out_features=3, hidden_dim=32, num_layers=3)
    x = torch.randn(batch_size, 12)
    y = model(x)
    assert y.shape == (batch_size, 3)


def test_pinn_backward_has_finite_gradients() -> None:
    """PINN backward pass produces finite gradients."""
    model = PINN(in_features=8, out_features=1, hidden_dim=16, num_layers=2)
    x = torch.randn(32, 8)
    target = torch.randn(32, 1)

    prediction = model(x)
    loss = torch.nn.functional.mse_loss(prediction, target)
    loss.backward()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))


@pytest.mark.parametrize(
    ("activation", "expected_type"),
    [("relu", torch.nn.ReLU), ("tanh", torch.nn.Tanh), ("gelu", torch.nn.GELU)],
)
def test_pinn_activation_selection(
    activation: str, expected_type: type[torch.nn.Module]
) -> None:
    """PINN selects requested activation module."""
    model = PINN(
        in_features=6,
        out_features=2,
        hidden_dim=16,
        num_layers=2,
        activation=activation,
    )
    assert isinstance(model.activation, expected_type)


def test_pinn_invalid_activation_raises() -> None:
    """PINN raises error for unsupported activation names."""
    with pytest.raises(ValueError, match="Unsupported activation"):
        PINN(in_features=4, out_features=1, activation="sigmoid")


def test_pinn_forward_dtype_consistency() -> None:
    """PINN preserves floating dtype through forward pass."""
    model = PINN(in_features=5, out_features=2, hidden_dim=8, num_layers=2)
    x: Tensor = torch.randn(10, 5, dtype=torch.float32)
    y = model(x)
    assert y.dtype == torch.float32
