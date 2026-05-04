"""CSI-specific physics-informed encoder/autoencoder models."""

# pyright: basic, reportMissingImports=false

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, cast

import torch
import torch.nn as nn
from torch import Tensor

from pinn4csi.physics import compute_path_loss, ofdm_residual


@dataclass(frozen=True)
class LossToggles:
    """Ablation toggles for individual loss components."""

    loss_reconstruction: bool = True
    loss_task: bool = True
    loss_ofdm: bool = True
    loss_path: bool = True


class FourierFeatureEmbedding(nn.Module):
    """Optional Fourier feature mapping for CSI inputs.

    Args:
        in_features: Input feature dimension.
        num_frequencies: Number of Fourier frequencies.
        sigma: Scale of random frequency projection.
    """

    def __init__(
        self, in_features: int, num_frequencies: int = 16, sigma: float = 1.0
    ) -> None:
        super().__init__()
        if num_frequencies < 1:
            raise ValueError(f"num_frequencies must be >= 1, got {num_frequencies}")
        projection = torch.randn(in_features, num_frequencies) * sigma
        self.projection = nn.Parameter(projection, requires_grad=False)
        self.in_features = in_features
        self.num_frequencies = num_frequencies

    @property
    def out_features(self) -> int:
        """Return output feature dimension after embedding."""
        return self.in_features + 2 * self.num_frequencies

    def forward(self, x: Tensor) -> Tensor:
        """Apply Fourier embedding.

        Args:
            x: Input tensor. Shape: (batch, in_features).

        Returns:
            Embedded tensor. Shape: (batch, in_features + 2*num_frequencies).
        """
        projected = (2.0 * math.pi) * (x @ self.projection)
        return torch.cat([x, torch.sin(projected), torch.cos(projected)], dim=-1)


class CSIPhysicsAutoencoder(nn.Module):
    """Minimal CSI encoder/autoencoder with optional physics-aware interfaces.

    The model supports:
    - Encoder -> latent representation
    - Decoder -> reconstruction
    - Optional task head
    - Optional residual-prior reconstruction interface
    - Optional Fourier feature embedding
    """

    def __init__(
        self,
        in_features: int,
        latent_dim: int,
        hidden_dim: int = 64,
        num_subcarriers: int = 32,
        reconstruction_dim: int | None = None,
        use_fourier_features: bool = False,
        fourier_num_frequencies: int = 16,
        fourier_sigma: float = 1.0,
        task_output_dim: int | None = None,
        use_residual_prior: bool = True,
        reconstruction_representation: Literal[
            "real_imag", "amplitude_phase"
        ] = "real_imag",
    ) -> None:
        """Initialize CSI physics-informed autoencoder.

        Args:
            in_features: Input feature size.
            latent_dim: Latent representation size.
            hidden_dim: Hidden feature size.
            num_subcarriers: Number of OFDM subcarriers represented in outputs.
            reconstruction_dim: Output reconstruction width. If None, defaults to
                `2 * num_subcarriers` for stacked real/imag OFDM outputs.
            use_fourier_features: Enable Fourier feature mapping before encoder.
            fourier_num_frequencies: Number of Fourier frequencies.
            fourier_sigma: Fourier projection scale.
            task_output_dim: Optional downstream task output dimension.
            use_residual_prior: If True, decoder output is residual added to prior.
            reconstruction_representation: Representation used by reconstruction
                tensors. `real_imag` expects the last dimension to be split into
                real and imaginary halves. `amplitude_phase` expects adjacent
                feature pairs to represent amplitude and phase.
        """
        super().__init__()
        self.num_subcarriers = num_subcarriers
        self.use_fourier_features = use_fourier_features
        self.use_residual_prior = use_residual_prior
        self.reconstruction_dim = reconstruction_dim or (2 * num_subcarriers)
        self.reconstruction_representation = reconstruction_representation

        if use_fourier_features:
            self.fourier_embedding: FourierFeatureEmbedding | None = (
                FourierFeatureEmbedding(
                    in_features=in_features,
                    num_frequencies=fourier_num_frequencies,
                    sigma=fourier_sigma,
                )
            )
            encoder_in_features = self.fourier_embedding.out_features
        else:
            self.fourier_embedding = None
            encoder_in_features = in_features

        self.encoder = nn.Sequential(
            nn.Linear(encoder_in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.reconstruction_dim),
        )

        self.task_head: nn.Module | None
        if task_output_dim is not None:
            self.task_head = nn.Linear(latent_dim, task_output_dim)
        else:
            self.task_head = None

    def encode(self, x: Tensor) -> Tensor:
        """Encode input CSI into latent representation."""
        encoder_input = self.apply_feature_embedding(x)
        return cast(Tensor, self.encoder(encoder_input))

    def decode(self, latent: Tensor) -> Tensor:
        """Decode latent representation into reconstruction/residual."""
        return cast(Tensor, self.decoder(latent))

    def apply_feature_embedding(self, x: Tensor) -> Tensor:
        """Apply optional Fourier feature embedding to input."""
        if self.fourier_embedding is None:
            return x
        return cast(Tensor, self.fourier_embedding(x))

    def forward(
        self, x: Tensor, prior_reconstruction: Tensor | None = None
    ) -> dict[str, Tensor]:
        """Run model forward path.

        Args:
            x: Input features. Shape: (batch, in_features).
            prior_reconstruction: Optional OFDM prior in stacked real/imag space.
                Shape: (batch, 2*num_subcarriers).

        Returns:
            Dictionary with latent, residual reconstruction, final reconstruction,
            and optional task prediction.
        """
        latent = self.encode(x)
        residual = self.decode(latent)

        if self.use_residual_prior and prior_reconstruction is not None:
            reconstruction = prior_reconstruction + residual
        else:
            reconstruction = residual

        outputs: dict[str, Tensor] = {
            "latent": latent,
            "reconstruction": reconstruction,
            "reconstruction_residual": residual,
        }
        if prior_reconstruction is not None:
            outputs["reconstruction_prior"] = prior_reconstruction
        if self.task_head is not None:
            outputs["task_prediction"] = self.task_head(latent)
        return outputs

    def compute_losses(
        self,
        outputs: dict[str, Tensor],
        target_reconstruction: Tensor,
        task_target: Tensor | None = None,
        physics: dict[str, Tensor] | None = None,
        toggles: LossToggles | None = None,
        task_loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Tensor]:
        """Compute per-component and aggregated losses.

        Physics dictionary keys for OFDM loss:
        - `path_gains_real`, `path_gains_imag`, `path_delays`,
          `subcarrier_frequencies`

        Physics dictionary keys for path loss:
        - `distance`, `frequency`, `tx_power_dbm` (optional `path_loss_exponent`)
        """
        if toggles is None:
            toggles = LossToggles()
        if task_loss_fn is None:
            task_loss_func: Callable[[Tensor, Tensor], Tensor] = nn.functional.mse_loss
        else:
            task_loss_func = task_loss_fn
        if weights is None:
            weights = {}

        reconstruction = outputs["reconstruction"]
        loss_reconstruction = nn.functional.mse_loss(
            reconstruction, target_reconstruction
        )

        task_prediction = outputs.get("task_prediction")
        if task_prediction is not None and task_target is not None:
            loss_task = task_loss_func(task_prediction, task_target)
        else:
            loss_task = torch.zeros(
                (), device=reconstruction.device, dtype=reconstruction.dtype
            )

        loss_ofdm = torch.zeros(
            (), device=reconstruction.device, dtype=reconstruction.dtype
        )
        if physics is not None and self._has_ofdm_metadata(physics):
            pred_complex = self._stacked_to_complex(reconstruction)
            path_gains = torch.complex(
                physics["path_gains_real"].to(reconstruction.device),
                physics["path_gains_imag"].to(reconstruction.device),
            )
            loss_ofdm = ofdm_residual(
                predicted_csi=pred_complex,
                path_gains=path_gains,
                path_delays=physics["path_delays"].to(reconstruction.device),
                subcarrier_frequencies=physics["subcarrier_frequencies"].to(
                    reconstruction.device
                ),
            )

        loss_path = torch.zeros(
            (), device=reconstruction.device, dtype=reconstruction.dtype
        )
        if physics is not None and self._has_path_metadata(physics):
            predicted_complex = self._stacked_to_complex(reconstruction)
            predicted_rx_dbm = 20.0 * torch.log10(
                torch.clamp(torch.mean(torch.abs(predicted_complex), dim=-1), min=1e-6)
            )
            distance = physics["distance"].to(reconstruction.device)
            frequency = physics["frequency"].to(reconstruction.device)
            tx_power_dbm = physics["tx_power_dbm"].to(reconstruction.device)
            path_loss_exponent = physics.get("path_loss_exponent")
            if path_loss_exponent is None:
                path_loss_exponent = torch.full_like(distance, 2.0)
            else:
                path_loss_exponent = path_loss_exponent.to(reconstruction.device)

            expected_path_loss = compute_path_loss(
                distance=distance,
                frequency=float(torch.mean(frequency).item()),
                n=float(torch.mean(path_loss_exponent).item()),
            )
            expected_rx_dbm = tx_power_dbm - expected_path_loss
            loss_path = torch.mean((predicted_rx_dbm - expected_rx_dbm) ** 2)

        component_losses = {
            "loss_reconstruction": loss_reconstruction,
            "loss_task": loss_task,
            "loss_ofdm": loss_ofdm,
            "loss_path": loss_path,
        }

        active = {
            "loss_reconstruction": toggles.loss_reconstruction,
            "loss_task": toggles.loss_task,
            "loss_ofdm": toggles.loss_ofdm,
            "loss_path": toggles.loss_path,
        }
        total_loss = torch.zeros(
            (), device=reconstruction.device, dtype=reconstruction.dtype
        )
        for key, value in component_losses.items():
            if active[key]:
                total_loss = total_loss + float(weights.get(key, 1.0)) * value

        return {
            **component_losses,
            "loss_total": total_loss,
        }

    def _stacked_to_complex(self, stacked: Tensor) -> Tensor:
        """Convert stacked real/imag tensor to complex CSI."""
        if self.reconstruction_representation == "real_imag":
            if stacked.shape[-1] != 2 * self.num_subcarriers:
                raise ValueError(
                    "Expected stacked reconstruction width "
                    f"{2 * self.num_subcarriers}, got {stacked.shape[-1]}"
                )
            real = stacked[:, : self.num_subcarriers]
            imag = stacked[:, self.num_subcarriers :]
            return torch.complex(real, imag)

        if stacked.shape[-1] % 2 != 0:
            raise ValueError(
                "Amplitude/phase reconstruction width must be even, got "
                f"{stacked.shape[-1]}"
            )
        amplitude = torch.clamp(stacked[:, 0::2], min=1e-6)
        phase = stacked[:, 1::2]
        return torch.polar(amplitude, phase)

    @staticmethod
    def _has_ofdm_metadata(physics: dict[str, Tensor]) -> bool:
        required = {
            "path_gains_real",
            "path_gains_imag",
            "path_delays",
            "subcarrier_frequencies",
        }
        return required.issubset(physics.keys())

    @staticmethod
    def _has_path_metadata(physics: dict[str, Tensor]) -> bool:
        required = {"distance", "frequency", "tx_power_dbm"}
        return required.issubset(physics.keys())
