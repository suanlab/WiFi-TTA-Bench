"""Minimal inverse-imaging PINN core for WiFi field reconstruction."""

# pyright: basic, reportMissingImports=false

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from pinn4csi.physics import helmholtz_residual, helmholtz_wavenumber


class WiFiImagingPINN(nn.Module):
    """Predict field and permittivity maps with Helmholtz-consistent loss.

    The model encodes per-sample CSI features into a latent embedding and predicts
    two inverse-imaging targets at queried coordinates:
    - scalar field value ``u(x)``
    - relative permittivity proxy ``eps_r(x)``
    """

    def __init__(
        self,
        csi_feature_dim: int,
        coordinate_dim: int = 2,
        hidden_dim: int = 64,
        latent_dim: int = 64,
        num_layers: int = 2,
    ) -> None:
        """Initialize WiFi imaging PINN core.

        Args:
            csi_feature_dim: Input CSI feature width per sample.
            coordinate_dim: Spatial coordinate dimension (2 for 2D, 3 for 3D).
            hidden_dim: Hidden width for MLP blocks.
            latent_dim: Latent width after CSI encoding.
            num_layers: Number of hidden layers in each MLP block.

        Raises:
            ValueError: If one of the dimensions or layer count is invalid.
        """
        super().__init__()
        if csi_feature_dim < 1:
            raise ValueError(f"csi_feature_dim must be >= 1, got {csi_feature_dim}")
        if coordinate_dim < 1:
            raise ValueError(f"coordinate_dim must be >= 1, got {coordinate_dim}")
        if hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {hidden_dim}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.coordinate_dim = coordinate_dim
        self.encoder = _build_mlp(
            in_features=csi_feature_dim,
            hidden_dim=hidden_dim,
            out_features=latent_dim,
            num_layers=num_layers,
        )
        self.field_decoder = _build_mlp(
            in_features=latent_dim + coordinate_dim,
            hidden_dim=hidden_dim,
            out_features=2,
            num_layers=num_layers,
        )

    def forward(self, csi_features: Tensor, coordinates: Tensor) -> dict[str, Tensor]:
        """Predict field and permittivity on queried coordinates.

        Args:
            csi_features: CSI embedding input. Shape: (batch, csi_feature_dim).
            coordinates: Spatial query coordinates. Shape:
                (batch, num_points, coordinate_dim).

        Returns:
            Dict with keys:
            - `latent`: Shape (batch, latent_dim)
            - `field`: Shape (batch, num_points)
            - `permittivity`: Shape (batch, num_points), positive by construction

        Raises:
            ValueError: If input shapes are incompatible.
        """
        self._validate_inputs(csi_features=csi_features, coordinates=coordinates)

        latent = cast(Tensor, self.encoder(csi_features))
        batch_size, num_points, _ = coordinates.shape
        latent_expanded = latent.unsqueeze(1).expand(batch_size, num_points, -1)
        decoder_input = torch.cat([latent_expanded, coordinates], dim=-1)
        decoder_output = cast(Tensor, self.field_decoder(decoder_input))

        field = decoder_output[..., 0]
        permittivity = functional.softplus(decoder_output[..., 1])
        return {
            "latent": latent,
            "field": field,
            "permittivity": permittivity,
        }

    def compute_losses(
        self,
        csi_features: Tensor,
        coordinates: Tensor,
        frequency: Tensor | float,
        field_target: Tensor | None = None,
        permittivity_target: Tensor | None = None,
        lambda_field: float = 1.0,
        lambda_permittivity: float = 1.0,
        lambda_physics: float = 1.0,
    ) -> dict[str, Tensor]:
        """Compute supervised inverse losses and Helmholtz physics loss.

        Args:
            csi_features: CSI embedding input. Shape: (batch, csi_feature_dim).
            coordinates: Spatial query coordinates. Shape:
                (batch, num_points, coordinate_dim).
            frequency: Carrier frequency in Hz. Supports scalar, per-sample, or
                per-point values.
            field_target: Optional supervised field target. Shape:
                (batch, num_points).
            permittivity_target: Optional supervised permittivity target. Shape:
                (batch, num_points).
            lambda_field: Weight for field reconstruction loss.
            lambda_permittivity: Weight for permittivity reconstruction loss.
            lambda_physics: Weight for Helmholtz residual loss.

        Returns:
            Dictionary with `loss_total`, `loss_field`, `loss_permittivity`, and
            `loss_physics`.
        """
        coordinates_for_physics = _ensure_requires_grad(coordinates)
        outputs = self.forward(
            csi_features=csi_features,
            coordinates=coordinates_for_physics,
        )

        field_prediction = outputs["field"]
        permittivity_prediction = outputs["permittivity"]
        zero = torch.zeros(
            (),
            dtype=field_prediction.dtype,
            device=field_prediction.device,
        )

        loss_field = zero
        if field_target is not None:
            loss_field = functional.mse_loss(field_prediction, field_target)

        loss_permittivity = zero
        if permittivity_target is not None:
            loss_permittivity = functional.mse_loss(
                permittivity_prediction,
                permittivity_target,
            )

        flat_field = field_prediction.reshape(-1)
        wave_number = helmholtz_wavenumber(frequency)
        wave_number_for_points = _expand_wavenumber(
            wavenumber=wave_number,
            batch_size=coordinates.shape[0],
            num_points=coordinates.shape[1],
            reference=flat_field,
        )
        residual = helmholtz_residual(
            field=field_prediction,
            coordinates=coordinates_for_physics,
            wavenumber=wave_number_for_points,
        )
        loss_physics = torch.mean(residual**2)

        loss_total = (
            float(lambda_field) * loss_field
            + float(lambda_permittivity) * loss_permittivity
            + float(lambda_physics) * loss_physics
        )
        return {
            "loss_total": loss_total,
            "loss_field": loss_field,
            "loss_permittivity": loss_permittivity,
            "loss_physics": loss_physics,
        }

    def _validate_inputs(self, csi_features: Tensor, coordinates: Tensor) -> None:
        if csi_features.dim() != 2:
            raise ValueError(
                "csi_features must have shape (batch, csi_feature_dim), "
                f"got {tuple(csi_features.shape)}"
            )
        if coordinates.dim() != 3:
            raise ValueError(
                "coordinates must have shape (batch, num_points, coordinate_dim), "
                f"got {tuple(coordinates.shape)}"
            )
        if coordinates.shape[0] != csi_features.shape[0]:
            raise ValueError(
                "csi_features and coordinates must share batch size. "
                f"Got {csi_features.shape[0]} and {coordinates.shape[0]}"
            )
        if coordinates.shape[2] != self.coordinate_dim:
            raise ValueError(
                "coordinates last dimension must match coordinate_dim. "
                f"Got {coordinates.shape[2]} and {self.coordinate_dim}"
            )


def _build_mlp(
    in_features: int,
    hidden_dim: int,
    out_features: int,
    num_layers: int,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_features = in_features
    for _ in range(num_layers):
        layers.append(nn.Linear(previous_features, hidden_dim))
        layers.append(nn.ReLU())
        previous_features = hidden_dim
    layers.append(nn.Linear(previous_features, out_features))
    return nn.Sequential(*layers)


def _ensure_requires_grad(coordinates: Tensor) -> Tensor:
    if coordinates.requires_grad:
        return coordinates
    return coordinates.detach().clone().requires_grad_(True)


def _expand_wavenumber(
    wavenumber: Tensor,
    batch_size: int,
    num_points: int,
    reference: Tensor,
) -> Tensor:
    wave_number = wavenumber.to(device=reference.device, dtype=reference.dtype)
    total_points = batch_size * num_points

    if wave_number.dim() == 0:
        return wave_number
    if wave_number.dim() == 1:
        if wave_number.shape[0] == total_points:
            return wave_number
        if wave_number.shape[0] == batch_size:
            return wave_number.repeat_interleave(num_points)
    if wave_number.dim() == 2 and wave_number.shape == (batch_size, num_points):
        return wave_number.reshape(-1)

    raise ValueError(
        "frequency must map to scalar, (batch,), (batch, num_points), or "
        f"(batch*num_points,) wavenumber. Got shape {tuple(wave_number.shape)}"
    )
