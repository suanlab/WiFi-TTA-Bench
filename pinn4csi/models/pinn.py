"""Base PINN architecture implemented as a simple MLP."""

from typing import cast

import torch.nn as nn
from torch import Tensor


class PINN(nn.Module):
    """Physics-Informed Neural Network base model for CSI tasks.

    This model only performs data prediction. Physics loss terms are computed
    outside of the model in trainer/physics modules.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        activation: str = "relu",
    ) -> None:
        """Initialize a configurable MLP-based PINN backbone.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            hidden_dim: Hidden layer dimension.
            num_layers: Number of hidden layers.
            activation: Activation function name. One of "relu", "tanh", "gelu".

        Raises:
            ValueError: If num_layers < 1 or activation is unsupported.
        """
        super().__init__()

        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.layers = nn.ModuleList()
        prev_dim = in_features

        for _ in range(num_layers):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim

        self.output_layer = nn.Linear(prev_dim, out_features)
        self.activation = self._build_activation(activation)

    def _build_activation(self, activation: str) -> nn.Module:
        """Build activation module from string name.

        Args:
            activation: Activation function name.

        Returns:
            Activation module.

        Raises:
            ValueError: If activation is unsupported.
        """
        activations: dict[str, nn.Module] = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "gelu": nn.GELU(),
        }
        key = activation.lower()
        if key not in activations:
            valid = ", ".join(sorted(activations))
            msg = f"Unsupported activation: {activation}. Expected one of: {valid}"
            raise ValueError(msg)
        return activations[key]

    def forward(self, x: Tensor) -> Tensor:
        """Run data prediction forward pass.

        Args:
            x: Input tensor. Shape: (batch, in_features).

        Returns:
            Predicted outputs. Shape: (batch, out_features).
        """
        hidden = x
        for layer in self.layers:
            hidden = self.activation(layer(hidden))
        return cast(Tensor, self.output_layer(hidden))
