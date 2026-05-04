# pyright: basic, reportMissingImports=false

"""Reusable Paper 1 baseline models and factory helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from pinn4csi.models.csi_pinn import CSIPhysicsAutoencoder, LossToggles
from pinn4csi.models.pinn import PINN
from pinn4csi.physics import compute_path_loss, ofdm_residual

_SURROGATE_WARNING_EMITTED = False


class Paper1Model(nn.Module):
    """Common interface for Paper 1 experiment models."""

    requires_prior: bool = False
    uses_reconstruction_loss: bool = False
    uses_ofdm_loss: bool = False
    uses_path_loss: bool = False
    uses_fourier_features: bool = False
    uses_adaptive_physics_weighting: bool = False
    physics_weighting_mode: str = "none"

    def forward(self, x: Tensor, prior: Tensor | None = None) -> dict[str, Tensor]:
        del x, prior
        raise NotImplementedError

    def compute_batch_losses(
        self,
        x: Tensor,
        labels: Tensor,
        prior: Tensor | None = None,
        has_prior: Tensor | None = None,
    ) -> dict[str, Tensor]:
        del x, labels, prior, has_prior
        raise NotImplementedError


class CSIClassifierMLP(Paper1Model):
    def __init__(
        self,
        input_shape: tuple[int, int],
        num_classes: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ) -> None:
        super().__init__()
        input_dim = input_shape[0] * input_shape[1]
        self.backbone = PINN(
            in_features=input_dim,
            out_features=num_classes,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            activation="relu",
        )

    def forward(self, x: Tensor, prior: Tensor | None = None) -> dict[str, Tensor]:
        del prior
        logits = self.backbone(x.flatten(start_dim=1))
        return {"logits": logits}

    def compute_batch_losses(
        self,
        x: Tensor,
        labels: Tensor,
        prior: Tensor | None = None,
        has_prior: Tensor | None = None,
    ) -> dict[str, Tensor]:
        del prior, has_prior
        outputs = self.forward(x)
        loss_task = functional.cross_entropy(outputs["logits"], labels)
        zero = torch.zeros((), device=x.device, dtype=loss_task.dtype)
        return {
            **outputs,
            "loss_task": loss_task,
            "loss_reconstruction": zero,
            "loss_ofdm": zero,
            "loss_path": zero,
            "loss_total": loss_task,
        }


class CSICNNClassifier(Paper1Model):
    def __init__(
        self,
        input_shape: tuple[int, int],
        num_classes: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        _, num_features = input_shape
        self.encoder = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: Tensor, prior: Tensor | None = None) -> dict[str, Tensor]:
        del prior
        encoded = self.encoder(x.transpose(1, 2)).squeeze(-1)
        logits = self.classifier(encoded)
        return {"logits": logits}

    def compute_batch_losses(
        self,
        x: Tensor,
        labels: Tensor,
        prior: Tensor | None = None,
        has_prior: Tensor | None = None,
    ) -> dict[str, Tensor]:
        del prior, has_prior
        outputs = self.forward(x)
        loss_task = functional.cross_entropy(outputs["logits"], labels)
        zero = torch.zeros((), device=x.device, dtype=loss_task.dtype)
        return {
            **outputs,
            "loss_task": loss_task,
            "loss_reconstruction": zero,
            "loss_ofdm": zero,
            "loss_path": zero,
            "loss_total": loss_task,
        }


class CSIDGSenseLiteClassifier(Paper1Model):
    """Lightweight WiFi-specific CSI baseline for Paper 1.

    This model is an in-repo approximation baseline inspired by CSI-aware designs
    (for internal comparison only), not a full paper-faithful DGSense
    reproduction.
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        num_classes: int,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        _, num_features = input_shape
        self.subcarrier_encoder = nn.Sequential(
            nn.Conv1d(
                num_features,
                num_features,
                kernel_size=5,
                padding=2,
                groups=num_features,
            ),
            nn.ReLU(),
            nn.Conv1d(num_features, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor, prior: Tensor | None = None) -> dict[str, Tensor]:
        del prior
        encoded = self.subcarrier_encoder(x.transpose(1, 2))
        pooled_mean = torch.mean(encoded, dim=-1)
        pooled_std = torch.std(encoded, dim=-1, unbiased=False)
        pooled = torch.cat([pooled_mean, pooled_std], dim=1)
        logits = self.classifier(pooled)
        return {"logits": logits}

    def compute_batch_losses(
        self,
        x: Tensor,
        labels: Tensor,
        prior: Tensor | None = None,
        has_prior: Tensor | None = None,
    ) -> dict[str, Tensor]:
        del prior, has_prior
        outputs = self.forward(x)
        loss_task = functional.cross_entropy(outputs["logits"], labels)
        zero = torch.zeros((), device=x.device, dtype=loss_task.dtype)
        return {
            **outputs,
            "loss_task": loss_task,
            "loss_reconstruction": zero,
            "loss_ofdm": zero,
            "loss_path": zero,
            "loss_total": loss_task,
        }


class CSIDGSenseLitePhysicsClassifier(Paper1Model):
    """CSI-aware baseline with lightweight physics auxiliary losses.

    This model keeps the strong DGSense-Lite-style subcarrier encoder while adding
    a small reconstruction head that predicts per-subcarrier amplitude/phase so the
    existing OFDM and path-loss constraints can supervise the representation.
    """

    uses_reconstruction_loss: bool = True
    uses_ofdm_loss: bool = True
    uses_path_loss: bool = True
    uses_adaptive_physics_weighting: bool = True
    physics_weighting_mode: str = "adaptive"

    def __init__(
        self,
        input_shape: tuple[int, int],
        num_classes: int,
        hidden_dim: int = 64,
        reconstruction_weight: float = 0.1,
        ofdm_weight: float = 0.05,
        path_weight: float = 0.05,
        adaptive_weight_eps: float = 1e-8,
        adaptive_weight_min: float = 1e-6,
        adaptive_weight_max: float = 1e6,
        use_fourier_features: bool = False,
        allow_surrogate_physics_metadata: bool = False,
        reconstruction_mode: str = "per_antenna",
    ) -> None:
        super().__init__()
        _, num_features = input_shape
        self.num_subcarriers = input_shape[0]
        self.feature_width = input_shape[1]
        self.reconstruction_weight = reconstruction_weight
        self.ofdm_weight = ofdm_weight
        self.path_weight = path_weight
        self.adaptive_weight_eps = adaptive_weight_eps
        self.adaptive_weight_min = adaptive_weight_min
        self.adaptive_weight_max = adaptive_weight_max
        self.current_ofdm_weight = ofdm_weight
        self.current_path_weight = path_weight
        self.current_reconstruction_weight = reconstruction_weight
        self.uses_fourier_features = use_fourier_features
        self.allow_surrogate_physics_metadata = allow_surrogate_physics_metadata
        self.reconstruction_mode = reconstruction_mode

        if reconstruction_mode == "per_antenna":
            recon_channels = num_features
        elif reconstruction_mode == "first_antenna":
            recon_channels = 2
        else:
            recon_channels = 2

        self.fourier_scale = nn.Parameter(torch.ones(1), requires_grad=False)
        self.subcarrier_encoder = nn.Sequential(
            nn.Conv1d(
                num_features,
                num_features,
                kernel_size=5,
                padding=2,
                groups=num_features,
            ),
            nn.ReLU(),
            nn.Conv1d(num_features, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        self.reconstruction_head = nn.Conv1d(hidden_dim, recon_channels, kernel_size=1)

    def forward(self, x: Tensor, prior: Tensor | None = None) -> dict[str, Tensor]:
        del prior
        model_input = x.transpose(1, 2)
        if self.uses_fourier_features:
            model_input = model_input * self.fourier_scale
        encoded = self.subcarrier_encoder(model_input)
        pooled_mean = torch.mean(encoded, dim=-1)
        pooled_std = torch.std(encoded, dim=-1, unbiased=False)
        pooled = torch.cat([pooled_mean, pooled_std], dim=1)
        logits = self.classifier(pooled)
        recon_input = encoded.detach()
        reconstruction = self.reconstruction_head(recon_input).transpose(1, 2)
        return {"logits": logits, "reconstruction": reconstruction}

    def compute_batch_losses(
        self,
        x: Tensor,
        labels: Tensor,
        prior: Tensor | None = None,
        has_prior: Tensor | None = None,
    ) -> dict[str, Tensor]:
        del has_prior
        outputs = self.forward(x, prior=prior)
        reconstruction = outputs["reconstruction"]
        logits = outputs["logits"]
        loss_task = functional.cross_entropy(logits, labels)

        target_reconstruction = self._build_reconstruction_target(x)
        loss_reconstruction = functional.mse_loss(reconstruction, target_reconstruction)

        zero = torch.zeros((), device=x.device, dtype=loss_task.dtype)
        loss_ofdm = zero
        loss_path = zero
        physics = self._build_physics_metadata(prior=prior)
        if physics is not None:
            pred_complex = torch.polar(
                torch.clamp(reconstruction[..., 0], min=1e-6),
                reconstruction[..., 1],
            )
            path_gains = torch.complex(
                physics["path_gains_real"],
                physics["path_gains_imag"],
            )
            loss_ofdm = ofdm_residual(
                predicted_csi=pred_complex,
                path_gains=path_gains,
                path_delays=physics["path_delays"],
                subcarrier_frequencies=physics["subcarrier_frequencies"],
            )
            predicted_rx_dbm = 20.0 * torch.log10(
                torch.clamp(torch.mean(torch.abs(pred_complex), dim=-1), min=1e-6)
            )
            expected_path_loss = compute_path_loss(
                distance=physics["distance"],
                frequency=float(torch.mean(physics["frequency"]).item()),
                n=float(torch.mean(physics["path_loss_exponent"]).item()),
            )
            expected_rx_dbm = physics["tx_power_dbm"] - expected_path_loss
            loss_path = torch.mean((predicted_rx_dbm - expected_rx_dbm) ** 2)

        component_weights = self._build_component_weights(
            loss_task=loss_task,
            loss_reconstruction=loss_reconstruction,
            loss_ofdm=loss_ofdm,
            loss_path=loss_path,
        )
        loss_total = (
            component_weights["loss_task"] * loss_task
            + component_weights["loss_reconstruction"] * loss_reconstruction
            + component_weights["loss_ofdm"] * loss_ofdm
            + component_weights["loss_path"] * loss_path
        )
        return {
            **outputs,
            "loss_task": loss_task,
            "loss_reconstruction": loss_reconstruction,
            "loss_ofdm": loss_ofdm,
            "loss_path": loss_path,
            "loss_total": loss_total,
        }

    def _build_reconstruction_target(self, x: Tensor) -> Tensor:
        if x.dim() != 3 or x.shape[-1] % 2 != 0:
            raise ValueError(
                "Expected amplitude/phase-expanded CSI with shape (N,S,F_even), "
                f"got {tuple(x.shape)}"
            )
        if self.reconstruction_mode == "per_antenna":
            return x
        if self.reconstruction_mode == "first_antenna":
            return x[..., :2]
        # antenna_mean
        amplitude = torch.mean(x[..., 0::2], dim=-1)
        phase = torch.mean(x[..., 1::2], dim=-1)
        return torch.stack([amplitude, phase], dim=-1)

    def _build_physics_metadata(self, prior: Tensor | None) -> dict[str, Tensor] | None:
        if not self.allow_surrogate_physics_metadata:
            return None
        if prior is None or prior.dim() != 3 or prior.shape[-1] < 2:
            return None
        global _SURROGATE_WARNING_EMITTED  # noqa: PLW0603
        if not _SURROGATE_WARNING_EMITTED:
            warnings.warn(
                "Using surrogate physics metadata fabricated from CSI amplitude. "
                "Distances and path gains are heuristic, not measured. "
                "Results may be circular.",
                UserWarning,
                stacklevel=2,
            )
            _SURROGATE_WARNING_EMITTED = True
        amplitude = torch.clamp(torch.mean(prior[..., 0::2], dim=-1), min=1e-6)
        phase = torch.mean(prior[..., 1::2], dim=-1)
        complex_csi = torch.polar(amplitude, phase)
        average_gain = complex_csi.mean(dim=1, keepdim=True)
        batch_size = int(prior.shape[0])
        device = prior.device
        dtype = prior.dtype
        subcarrier_indices = torch.arange(
            self.num_subcarriers,
            device=device,
            dtype=dtype,
        ) - float(self.num_subcarriers // 2)
        subcarrier_frequencies = 5.32e9 + (312.5e3 * subcarrier_indices)
        subcarrier_frequencies = subcarrier_frequencies.unsqueeze(0).expand(
            batch_size, -1
        )
        distance = 1.0 + 8.0 / (torch.mean(amplitude, dim=1) + 1.0)
        return {
            "path_gains_real": average_gain.real,
            "path_gains_imag": average_gain.imag,
            "path_delays": torch.zeros(batch_size, 1, device=device, dtype=dtype),
            "subcarrier_frequencies": subcarrier_frequencies,
            "distance": distance,
            "frequency": torch.full((batch_size,), 2.4e9, device=device, dtype=dtype),
            "tx_power_dbm": torch.full((batch_size,), 20.0, device=device, dtype=dtype),
            "path_loss_exponent": torch.full(
                (batch_size,), 2.0, device=device, dtype=dtype
            ),
        }

    def _grad_norm(self, loss: Tensor) -> Tensor:
        params = [
            parameter for parameter in self.parameters() if parameter.requires_grad
        ]
        gradients = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        squared_norm = torch.zeros((), device=loss.device, dtype=loss.dtype)
        for grad in gradients:
            if grad is not None:
                squared_norm = squared_norm + torch.sum(grad * grad)
        return torch.sqrt(squared_norm + self.adaptive_weight_eps)

    def _safe_clamp_ratio(self, numerator: Tensor, denominator: Tensor) -> float:
        ratio = numerator / (denominator + self.adaptive_weight_eps)
        clamped = torch.clamp(
            ratio,
            min=self.adaptive_weight_min,
            max=self.adaptive_weight_max,
        )
        return float(clamped.detach().item())

    def _build_component_weights(
        self,
        loss_task: Tensor,
        loss_reconstruction: Tensor,
        loss_ofdm: Tensor,
        loss_path: Tensor,
    ) -> dict[str, float]:
        weights = {
            "loss_task": 1.0,
            "loss_reconstruction": self.reconstruction_weight,
            "loss_ofdm": self.ofdm_weight,
            "loss_path": self.path_weight,
        }
        if not torch.is_grad_enabled():
            return weights
        reference_loss = loss_task + (self.reconstruction_weight * loss_reconstruction)
        reference_norm = self._grad_norm(reference_loss)
        if float(loss_ofdm.item()) > 0.0:
            weights["loss_ofdm"] = self.ofdm_weight * self._safe_clamp_ratio(
                reference_norm,
                self._grad_norm(loss_ofdm),
            )
        if float(loss_path.item()) > 0.0:
            weights["loss_path"] = self.path_weight * self._safe_clamp_ratio(
                reference_norm,
                self._grad_norm(loss_path),
            )
        self.current_reconstruction_weight = weights["loss_reconstruction"]
        self.current_ofdm_weight = weights["loss_ofdm"]
        self.current_path_weight = weights["loss_path"]
        return weights


class CSIAutoencoderClassifier(Paper1Model):
    def __init__(
        self,
        input_shape: tuple[int, int],
        num_classes: int,
        latent_dim: int = 32,
        hidden_dim: int = 64,
        reconstruction_weight: float = 0.2,
        ofdm_weight: float = 0.05,
        path_weight: float = 0.05,
        use_residual_prior: bool = False,
        use_reconstruction_loss: bool = True,
        use_ofdm_loss: bool = True,
        use_path_loss: bool = True,
        use_fourier_features: bool = False,
        use_adaptive_physics_weighting: bool = True,
        adaptive_weight_eps: float = 1e-8,
        adaptive_weight_min: float = 1e-6,
        adaptive_weight_max: float = 1e6,
        allow_surrogate_physics_metadata: bool = False,
        reconstruction_mode: str = "per_antenna",
    ) -> None:
        super().__init__()
        self.requires_prior = use_residual_prior
        self.uses_reconstruction_loss = use_reconstruction_loss
        self.uses_ofdm_loss = use_ofdm_loss
        self.uses_path_loss = use_path_loss
        self.uses_fourier_features = use_fourier_features
        self.uses_adaptive_physics_weighting = use_adaptive_physics_weighting
        self.physics_weighting_mode = (
            "adaptive" if use_adaptive_physics_weighting else "fixed"
        )
        self.reconstruction_weight = reconstruction_weight
        self.ofdm_weight = ofdm_weight
        self.path_weight = path_weight
        self.adaptive_weight_eps = adaptive_weight_eps
        self.adaptive_weight_min = adaptive_weight_min
        self.adaptive_weight_max = adaptive_weight_max
        self.current_ofdm_weight = ofdm_weight
        self.current_path_weight = path_weight
        self.allow_surrogate_physics_metadata = allow_surrogate_physics_metadata
        self.reconstruction_mode = reconstruction_mode
        input_dim = input_shape[0] * input_shape[1]
        self.num_subcarriers = input_shape[0]
        self.feature_width = input_shape[1]

        if reconstruction_mode == "per_antenna":
            recon_dim = input_dim
        elif reconstruction_mode == "first_antenna":
            recon_dim = 2 * input_shape[0]
        else:
            recon_dim = 2 * input_shape[0]

        self.model = CSIPhysicsAutoencoder(
            in_features=input_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_subcarriers=input_shape[0],
            task_output_dim=num_classes,
            use_residual_prior=use_residual_prior,
            use_fourier_features=use_fourier_features,
            reconstruction_dim=recon_dim,
            reconstruction_representation="amplitude_phase",
        )

    def forward(self, x: Tensor, prior: Tensor | None = None) -> dict[str, Tensor]:
        flat_x = x.flatten(start_dim=1)
        flat_prior = (
            self._build_reconstruction_target(prior) if prior is not None else None
        )
        outputs = self.model(flat_x, prior_reconstruction=flat_prior)
        logits = outputs.get("task_prediction")
        if logits is None:
            raise RuntimeError("Autoencoder classifier did not produce task logits.")
        return {**outputs, "logits": logits}

    def compute_batch_losses(
        self,
        x: Tensor,
        labels: Tensor,
        prior: Tensor | None = None,
        has_prior: Tensor | None = None,
    ) -> dict[str, Tensor]:
        if self.requires_prior and (
            prior is None or has_prior is None or not bool(torch.all(has_prior).item())
        ):
            raise ValueError(
                "Residual-prior model requires prepared priors for every sample."
            )

        target_reconstruction = self._build_reconstruction_target(x)
        outputs = self.forward(x, prior=prior)
        physics = self._build_physics_metadata(prior=prior)
        losses = self.model.compute_losses(
            outputs=outputs,
            target_reconstruction=target_reconstruction,
            task_target=labels,
            physics=physics,
            toggles=LossToggles(
                loss_reconstruction=self.uses_reconstruction_loss,
                loss_task=True,
                loss_ofdm=self.uses_ofdm_loss,
                loss_path=self.uses_path_loss,
            ),
            task_loss_fn=functional.cross_entropy,
            weights=self._base_component_weights(),
        )
        component_weights = self._build_component_weights(losses=losses)
        losses["loss_total"] = self._compute_weighted_total_loss(
            losses=losses,
            component_weights=component_weights,
        )
        return {**outputs, **losses, "logits": outputs["logits"]}

    def _build_reconstruction_target(self, x: Tensor) -> Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected 3D CSI features, got shape {tuple(x.shape)}")
        if x.shape[-1] % 2 != 0:
            raise ValueError(
                "Paper1 autoencoder expects amplitude/phase-expanded features with "
                f"even width, got {x.shape[-1]}"
            )
        if self.reconstruction_mode == "per_antenna":
            return x.reshape(x.shape[0], -1)
        if self.reconstruction_mode == "first_antenna":
            reduced = x[..., :2]
            return reduced.reshape(x.shape[0], -1)
        # antenna_mean
        amplitude = torch.mean(x[..., 0::2], dim=-1)
        phase = torch.mean(x[..., 1::2], dim=-1)
        reduced = torch.stack([amplitude, phase], dim=-1)
        return reduced.reshape(x.shape[0], -1)

    def _base_component_weights(self) -> dict[str, float]:
        return {
            "loss_reconstruction": self.reconstruction_weight,
            "loss_task": 1.0,
            "loss_ofdm": self.ofdm_weight,
            "loss_path": self.path_weight,
        }

    def _build_component_weights(self, losses: dict[str, Tensor]) -> dict[str, float]:
        weights = self._base_component_weights()
        if not self.uses_adaptive_physics_weighting:
            self.current_ofdm_weight = self.ofdm_weight
            self.current_path_weight = self.path_weight
            return weights
        if not torch.is_grad_enabled():
            weights["loss_ofdm"] = self.current_ofdm_weight
            weights["loss_path"] = self.current_path_weight
            return weights

        params = [p for p in self.parameters() if p.requires_grad]
        reference_loss = losses["loss_task"]
        if self.uses_reconstruction_loss:
            reference_loss = reference_loss + (
                self.reconstruction_weight * losses["loss_reconstruction"]
            )
        reference_norm = self._grad_norm(reference_loss, params)

        if self.uses_ofdm_loss and float(losses["loss_ofdm"].item()) > 0.0:
            ofdm_norm = self._grad_norm(losses["loss_ofdm"], params)
            adaptive_ofdm = self._safe_clamp_ratio(reference_norm, ofdm_norm)
            self.current_ofdm_weight = self.ofdm_weight * adaptive_ofdm
            weights["loss_ofdm"] = self.current_ofdm_weight
        else:
            self.current_ofdm_weight = self.ofdm_weight

        if self.uses_path_loss and float(losses["loss_path"].item()) > 0.0:
            path_norm = self._grad_norm(losses["loss_path"], params)
            adaptive_path = self._safe_clamp_ratio(reference_norm, path_norm)
            self.current_path_weight = self.path_weight * adaptive_path
            weights["loss_path"] = self.current_path_weight
        else:
            self.current_path_weight = self.path_weight

        return weights

    def _compute_weighted_total_loss(
        self,
        losses: dict[str, Tensor],
        component_weights: dict[str, float],
    ) -> Tensor:
        total_loss = torch.zeros_like(losses["loss_total"])
        if self.uses_reconstruction_loss:
            total_loss = (
                total_loss
                + component_weights["loss_reconstruction"]
                * losses["loss_reconstruction"]
            )
        total_loss = total_loss + component_weights["loss_task"] * losses["loss_task"]
        if self.uses_ofdm_loss:
            total_loss = (
                total_loss + component_weights["loss_ofdm"] * losses["loss_ofdm"]
            )
        if self.uses_path_loss:
            total_loss = (
                total_loss + component_weights["loss_path"] * losses["loss_path"]
            )
        return total_loss

    def _grad_norm(self, loss: Tensor, params: list[nn.Parameter]) -> Tensor:
        gradients = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        squared_norm = torch.zeros((), device=loss.device, dtype=loss.dtype)
        for grad in gradients:
            if grad is not None:
                squared_norm = squared_norm + torch.sum(grad * grad)
        return torch.sqrt(squared_norm + self.adaptive_weight_eps)

    def _safe_clamp_ratio(self, numerator: Tensor, denominator: Tensor) -> float:
        ratio = numerator / (denominator + self.adaptive_weight_eps)
        clamped = torch.clamp(
            ratio,
            min=self.adaptive_weight_min,
            max=self.adaptive_weight_max,
        )
        return float(clamped.detach().item())

    def _build_physics_metadata(
        self,
        prior: Tensor | None,
    ) -> dict[str, Tensor] | None:
        if not self.allow_surrogate_physics_metadata:
            return None
        if prior is None:
            return None
        if prior.dim() != 3 or prior.shape[-1] < 2:
            return None
        global _SURROGATE_WARNING_EMITTED  # noqa: PLW0603
        if not _SURROGATE_WARNING_EMITTED:
            warnings.warn(
                "Using surrogate physics metadata fabricated from CSI amplitude. "
                "Distances and path gains are heuristic, not measured. "
                "Results may be circular.",
                UserWarning,
                stacklevel=2,
            )
            _SURROGATE_WARNING_EMITTED = True

        amplitude = torch.clamp(torch.mean(prior[..., 0::2], dim=-1), min=1e-6)
        phase = torch.mean(prior[..., 1::2], dim=-1)
        complex_csi = torch.polar(amplitude, phase)
        flat_complex_csi = complex_csi.reshape(complex_csi.shape[0], -1)
        average_gain = flat_complex_csi.mean(dim=1, keepdim=True)
        batch_size = int(prior.shape[0])
        device = prior.device
        dtype = prior.dtype
        complex_bins = int(flat_complex_csi.shape[1])

        center_frequency_hz = 5.32e9
        subcarrier_spacing_hz = 312.5e3
        subcarrier_indices = torch.arange(
            complex_bins,
            device=device,
            dtype=dtype,
        ) - float(complex_bins // 2)
        subcarrier_frequencies = center_frequency_hz + (
            subcarrier_spacing_hz * subcarrier_indices
        )
        subcarrier_frequencies = subcarrier_frequencies.unsqueeze(0).expand(
            batch_size, -1
        )

        mean_amplitude = torch.mean(amplitude, dim=1)
        distance = 1.0 + 8.0 / (mean_amplitude + 1.0)

        return {
            "path_gains_real": average_gain.real,
            "path_gains_imag": average_gain.imag,
            "path_delays": torch.zeros(batch_size, 1, device=device, dtype=dtype),
            "subcarrier_frequencies": subcarrier_frequencies,
            "distance": distance,
            "frequency": torch.full(
                (batch_size,),
                2.4e9,
                device=device,
                dtype=dtype,
            ),
            "tx_power_dbm": torch.full(
                (batch_size,),
                20.0,
                device=device,
                dtype=dtype,
            ),
            "path_loss_exponent": torch.full(
                (batch_size,),
                2.0,
                device=device,
                dtype=dtype,
            ),
        }


class CSICNNGRUClassifier(Paper1Model):
    """CNN+GRU temporal baseline matching published SOTA architectures.

    Applies 1D convolutions for local feature extraction followed by a
    bidirectional GRU for temporal modeling and mean-pooling classification.
    """

    def __init__(
        self,
        input_shape: tuple[int, int],
        num_classes: int,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        num_timesteps, num_features = input_shape
        self._num_timesteps = num_timesteps
        self.conv_block = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=gru_layers,
            dropout=dropout if gru_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        # Bidirectional GRU output: hidden_dim (hidden_dim//2 * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: Tensor, prior: Tensor | None = None) -> dict[str, Tensor]:
        del prior
        # x: (batch, timesteps, features)
        conv_input = x.transpose(1, 2)  # (batch, features, timesteps)
        conv_out = self.conv_block(conv_input)  # (batch, hidden_dim, timesteps)
        gru_input = conv_out.transpose(1, 2)  # (batch, timesteps, hidden_dim)
        gru_out, _ = self.gru(gru_input)  # (batch, timesteps, hidden_dim)
        pooled = gru_out.mean(dim=1)  # (batch, hidden_dim)
        logits = self.classifier(pooled)
        return {"logits": logits}

    def compute_batch_losses(
        self,
        x: Tensor,
        labels: Tensor,
        prior: Tensor | None = None,
        has_prior: Tensor | None = None,
    ) -> dict[str, Tensor]:
        del prior, has_prior
        outputs = self.forward(x)
        loss_task = functional.cross_entropy(outputs["logits"], labels)
        zero = torch.zeros((), device=x.device, dtype=loss_task.dtype)
        return {
            **outputs,
            "loss_task": loss_task,
            "loss_reconstruction": zero,
            "loss_ofdm": zero,
            "loss_path": zero,
            "loss_total": loss_task,
        }


@dataclass(frozen=True)
class Paper1ModelFactoryConfig:
    hidden_dim: int = 64
    num_layers: int = 3
    latent_dim: int = 32
    reconstruction_weight: float = 0.2
    ofdm_weight: float = 0.05
    path_weight: float = 0.05
    reconstruction_mode: str = "per_antenna"


@dataclass(frozen=True)
class Paper1ModelSpec:
    model_name: str
    variant_name: str = "default"
    use_reconstruction_loss: bool = True
    use_ofdm_loss: bool = True
    use_path_loss: bool = True
    use_fourier_features: bool = False
    use_adaptive_physics_weighting: bool = True

    @property
    def comparison_name(self) -> str:
        return f"{self.model_name}:{self.variant_name}"


def expand_paper1_model_specs(
    model_names: tuple[str, ...],
    include_loss_ablation_variants: bool = True,
    include_fixed_weight_variant: bool = True,
    include_fourier_variant: bool = True,
) -> tuple[Paper1ModelSpec, ...]:
    specs: list[Paper1ModelSpec] = []
    for model_name in model_names:
        normalized_name = model_name.lower()
        specs.append(Paper1ModelSpec(model_name=normalized_name))
        if normalized_name in {
            "autoencoder",
            "residual_prior",
        }:
            if include_fixed_weight_variant:
                specs.append(
                    Paper1ModelSpec(
                        model_name=normalized_name,
                        variant_name="fixed_weight",
                        use_adaptive_physics_weighting=False,
                    )
                )
            if include_fourier_variant:
                specs.append(
                    Paper1ModelSpec(
                        model_name=normalized_name,
                        variant_name="fourier_on",
                        use_fourier_features=True,
                    )
                )
        if include_loss_ablation_variants and normalized_name in {
            "autoencoder",
            "residual_prior",
        }:
            specs.append(
                Paper1ModelSpec(
                    model_name=normalized_name,
                    variant_name="reconstruction_off",
                    use_reconstruction_loss=False,
                )
            )
            specs.append(
                Paper1ModelSpec(
                    model_name=normalized_name,
                    variant_name="ofdm_off",
                    use_ofdm_loss=False,
                )
            )
            specs.append(
                Paper1ModelSpec(
                    model_name=normalized_name,
                    variant_name="path_off",
                    use_path_loss=False,
                )
            )
    return tuple(specs)


def create_paper1_model(
    model_name: str | Paper1ModelSpec,
    input_shape: tuple[int, int],
    num_classes: int,
    config: Paper1ModelFactoryConfig | None = None,
) -> Paper1Model:
    """Create a named Paper 1 baseline model."""
    if config is None:
        config = Paper1ModelFactoryConfig()

    spec = (
        model_name
        if isinstance(model_name, Paper1ModelSpec)
        else Paper1ModelSpec(model_name=model_name)
    )
    key = spec.model_name.lower()
    if key == "mlp":
        return CSIClassifierMLP(
            input_shape=input_shape,
            num_classes=num_classes,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
        )
    if key == "cnn":
        return CSICNNClassifier(
            input_shape=input_shape,
            num_classes=num_classes,
            hidden_dim=config.hidden_dim,
        )
    if key == "cnn_gru":
        return CSICNNGRUClassifier(
            input_shape=input_shape,
            num_classes=num_classes,
            hidden_dim=config.hidden_dim,
        )
    if key == "dgsense_lite":
        return CSIDGSenseLiteClassifier(
            input_shape=input_shape,
            num_classes=num_classes,
            hidden_dim=config.hidden_dim,
        )
    if key == "dgsense_physics":
        return CSIDGSenseLitePhysicsClassifier(
            input_shape=input_shape,
            num_classes=num_classes,
            hidden_dim=config.hidden_dim,
            reconstruction_weight=config.reconstruction_weight,
            ofdm_weight=config.ofdm_weight,
            path_weight=config.path_weight,
            use_fourier_features=spec.use_fourier_features,
            reconstruction_mode=config.reconstruction_mode,
        )
    if key == "autoencoder":
        return CSIAutoencoderClassifier(
            input_shape=input_shape,
            num_classes=num_classes,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            reconstruction_weight=config.reconstruction_weight,
            ofdm_weight=config.ofdm_weight,
            path_weight=config.path_weight,
            use_residual_prior=False,
            use_reconstruction_loss=spec.use_reconstruction_loss,
            use_ofdm_loss=spec.use_ofdm_loss,
            use_path_loss=spec.use_path_loss,
            use_fourier_features=spec.use_fourier_features,
            use_adaptive_physics_weighting=spec.use_adaptive_physics_weighting,
            reconstruction_mode=config.reconstruction_mode,
        )
    if key == "residual_prior":
        return CSIAutoencoderClassifier(
            input_shape=input_shape,
            num_classes=num_classes,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            reconstruction_weight=config.reconstruction_weight,
            ofdm_weight=config.ofdm_weight,
            path_weight=config.path_weight,
            use_residual_prior=True,
            use_reconstruction_loss=spec.use_reconstruction_loss,
            use_ofdm_loss=spec.use_ofdm_loss,
            use_path_loss=spec.use_path_loss,
            use_fourier_features=spec.use_fourier_features,
            use_adaptive_physics_weighting=spec.use_adaptive_physics_weighting,
            reconstruction_mode=config.reconstruction_mode,
        )

    valid = ", ".join(sorted(list_paper1_model_names()))
    raise ValueError(f"Unknown Paper 1 model: {model_name}. Expected one of: {valid}")


def list_paper1_model_names() -> tuple[str, ...]:
    return (
        "autoencoder",
        "cnn",
        "cnn_gru",
        "dgsense_lite",
        "dgsense_physics",
        "mlp",
        "residual_prior",
    )
