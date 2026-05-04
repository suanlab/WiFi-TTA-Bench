"""Minimal DeepONet-style neural operator with OFDM physics hooks."""

# pyright: basic, reportMissingImports=false

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from pinn4csi.physics import ofdm_residual, subcarrier_correlation_loss


class PhysicsInformedDeepONet(nn.Module):
    """Lightweight DeepONet for environment-parameter to CSI mapping.

    The branch network encodes per-environment context features, while the trunk
    network encodes query coordinates (for example, subcarrier coordinates). The
    final output is produced with a latent inner-product, yielding CSI-like
    real/imag channels per query point.
    """

    def __init__(
        self,
        environment_dim: int,
        query_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
        output_channels: int = 2,
    ) -> None:
        """Initialize a minimal DeepONet-style operator.

        Args:
            environment_dim: Environment/context feature dimension.
            query_dim: Query-coordinate feature dimension.
            hidden_dim: Hidden layer size for branch/trunk MLPs.
            latent_dim: Latent operator width.
            num_layers: Number of hidden layers per branch/trunk MLP.
            output_channels: Number of output channels per query point.

        Raises:
            ValueError: If dimensions or layer counts are invalid.
        """
        super().__init__()
        if environment_dim < 1:
            raise ValueError(f"environment_dim must be >= 1, got {environment_dim}")
        if query_dim < 1:
            raise ValueError(f"query_dim must be >= 1, got {query_dim}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        if output_channels < 1:
            raise ValueError(f"output_channels must be >= 1, got {output_channels}")

        self.output_channels = output_channels

        self.branch = _build_mlp(
            in_features=environment_dim,
            hidden_dim=hidden_dim,
            out_features=latent_dim,
            num_layers=num_layers,
        )
        self.trunk = _build_mlp(
            in_features=query_dim,
            hidden_dim=hidden_dim,
            out_features=latent_dim * output_channels,
            num_layers=num_layers,
        )

    def forward(
        self, environment_features: Tensor, query_coordinates: Tensor
    ) -> Tensor:
        """Predict CSI-like outputs at query coordinates.

        Args:
            environment_features: Environment/context tensor. Shape:
                (batch, environment_dim).
            query_coordinates: Query tensor. Shape: (num_queries, query_dim) for a
                shared query grid, or (batch, num_queries, query_dim).

        Returns:
            Predicted per-query outputs. Shape: (batch, num_queries, output_channels).
        """
        if environment_features.dim() != 2:
            raise ValueError(
                "environment_features must have shape (batch, environment_dim). "
                f"Got {tuple(environment_features.shape)}."
            )

        branch_features = cast(Tensor, self.branch(environment_features))
        batched_queries = _to_batched_queries(
            query_coordinates, environment_features.shape[0]
        )
        trunk_features = cast(Tensor, self.trunk(batched_queries))

        trunk_reshaped = trunk_features.reshape(
            batched_queries.shape[0],
            batched_queries.shape[1],
            branch_features.shape[-1],
            self.output_channels,
        )
        return torch.einsum("bl,bqlc->bqc", branch_features, trunk_reshaped)

    def compute_losses(
        self,
        predicted_response: Tensor,
        target_response: Tensor,
        physics: dict[str, Tensor] | None = None,
        data_weight: float = 1.0,
        ofdm_weight: float = 1.0,
        correlation_weight: float = 0.0,
    ) -> dict[str, Tensor]:
        """Compute operator data + OFDM-consistency losses.

        Args:
            predicted_response: Predicted response. Shape: (batch, queries, channels).
            target_response: Supervised target with identical shape.
            physics: Optional OFDM metadata with keys `path_gains_real`,
                `path_gains_imag`, `path_delays`, and `subcarrier_frequencies`.
            data_weight: Scalar weight for supervised MSE term.
            ofdm_weight: Scalar weight for OFDM consistency residual.
            correlation_weight: Scalar weight for neighboring-subcarrier smoothness.

        Returns:
            Dictionary containing each component and total loss.
        """
        if predicted_response.shape != target_response.shape:
            raise ValueError(
                "predicted_response and target_response must share shape. "
                f"Got {tuple(predicted_response.shape)} and "
                f"{tuple(target_response.shape)}."
            )

        loss_data = functional.mse_loss(predicted_response, target_response)
        zero = torch.zeros(
            (), device=predicted_response.device, dtype=predicted_response.dtype
        )
        loss_ofdm = zero
        loss_correlation = zero

        if physics is not None and _has_ofdm_metadata(physics):
            predicted_complex = _stacked_to_complex(predicted_response)
            path_gains = torch.complex(
                physics["path_gains_real"].to(predicted_response.device),
                physics["path_gains_imag"].to(predicted_response.device),
            )
            loss_ofdm = ofdm_residual(
                predicted_csi=predicted_complex,
                path_gains=path_gains,
                path_delays=physics["path_delays"].to(predicted_response.device),
                subcarrier_frequencies=physics["subcarrier_frequencies"].to(
                    predicted_response.device
                ),
            )
            loss_correlation = subcarrier_correlation_loss(predicted_complex)

        loss_total = (
            float(data_weight) * loss_data
            + float(ofdm_weight) * loss_ofdm
            + float(correlation_weight) * loss_correlation
        )
        return {
            "loss_total": loss_total,
            "loss_data": loss_data,
            "loss_ofdm": loss_ofdm,
            "loss_correlation": loss_correlation,
        }


def create_physics_informed_deeponet(
    environment_dim: int,
    query_dim: int,
    hidden_dim: int = 64,
    latent_dim: int = 64,
    num_layers: int = 2,
    output_channels: int = 2,
) -> PhysicsInformedDeepONet:
    """Factory helper for the minimal physics-informed DeepONet."""
    return PhysicsInformedDeepONet(
        environment_dim=environment_dim,
        query_dim=query_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
        output_channels=output_channels,
    )


def _build_mlp(
    in_features: int,
    hidden_dim: int,
    out_features: int,
    num_layers: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev_dim = in_features
    for _ in range(num_layers):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, out_features))
    return nn.Sequential(*layers)


def _to_batched_queries(query_coordinates: Tensor, batch_size: int) -> Tensor:
    if query_coordinates.dim() == 2:
        return query_coordinates.unsqueeze(0).expand(batch_size, -1, -1)
    if query_coordinates.dim() == 3:
        if query_coordinates.shape[0] != batch_size:
            raise ValueError(
                "Batched query_coordinates must have the same batch size as "
                "environment_features. "
                f"Got {query_coordinates.shape[0]} and {batch_size}."
            )
        return query_coordinates
    raise ValueError(
        "query_coordinates must have shape (num_queries, query_dim) or "
        f"(batch, num_queries, query_dim), got {tuple(query_coordinates.shape)}"
    )


def _stacked_to_complex(stacked_response: Tensor) -> Tensor:
    if stacked_response.shape[-1] != 2:
        raise ValueError(
            "OFDM consistency expects stacked real/imag channels in the last "
            f"dimension, got {stacked_response.shape[-1]} channels."
        )
    real = stacked_response[..., 0]
    imag = stacked_response[..., 1]
    return torch.complex(real, imag)


def _has_ofdm_metadata(physics: dict[str, Tensor]) -> bool:
    required = {
        "path_gains_real",
        "path_gains_imag",
        "path_delays",
        "subcarrier_frequencies",
    }
    return required.issubset(physics.keys())
