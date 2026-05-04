"""Physics-informed domain-invariant feature module for CSI sensing."""

# pyright: basic, reportMissingImports=false

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor


def coral_loss(source_features: Tensor, target_features: Tensor) -> Tensor:
    """Compute CORAL covariance-alignment loss between two feature sets.

    Args:
        source_features: Source-domain features. Shape: (batch_s, feature_dim).
        target_features: Target-domain features. Shape: (batch_t, feature_dim).

    Returns:
        Scalar CORAL loss.
    """
    if source_features.dim() != 2 or target_features.dim() != 2:
        raise ValueError("CORAL expects 2D feature tensors (batch, feature_dim).")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError(
            "Source/target feature dimensions must match for CORAL. "
            f"Got {source_features.shape[1]} and {target_features.shape[1]}."
        )

    source_centered = source_features - source_features.mean(dim=0, keepdim=True)
    target_centered = target_features - target_features.mean(dim=0, keepdim=True)

    source_denominator = max(int(source_features.shape[0]) - 1, 1)
    target_denominator = max(int(target_features.shape[0]) - 1, 1)
    source_cov = (source_centered.T @ source_centered) / source_denominator
    target_cov = (target_centered.T @ target_centered) / target_denominator

    feature_dim = float(source_features.shape[1])
    return torch.mean((source_cov - target_cov) ** 2) / (
        4.0 * feature_dim * feature_dim
    )


def residual_moment_alignment_loss(
    source_residual: Tensor,
    target_residual: Tensor,
) -> Tensor:
    """Align first and second moments of residual-prior mismatch.

    Args:
        source_residual: Source residual (`x - prior`). Shape: (batch_s, features).
        target_residual: Target residual (`x - prior`). Shape: (batch_t, features).

    Returns:
        Scalar moment-alignment loss.
    """
    if source_residual.dim() != 2 or target_residual.dim() != 2:
        raise ValueError("Residual alignment expects 2D tensors (batch, features).")
    if source_residual.shape[1] != target_residual.shape[1]:
        raise ValueError(
            "Source/target residual dimensions must match. "
            f"Got {source_residual.shape[1]} and {target_residual.shape[1]}."
        )

    source_mean = source_residual.mean(dim=0)
    target_mean = target_residual.mean(dim=0)
    source_var = source_residual.var(dim=0, unbiased=False)
    target_var = target_residual.var(dim=0, unbiased=False)
    return torch.mean((source_mean - target_mean) ** 2) + torch.mean(
        (source_var - target_var) ** 2
    )


@dataclass(frozen=True)
class DomainInvariantLossToggles:
    """Ablation toggles for domain-generalization losses."""

    loss_task: bool = True
    loss_invariance: bool = True
    loss_physics_residual: bool = True


class PhysicsDomainInvariantModule(nn.Module):
    """Minimal physics-informed domain-invariant representation learner.

    The module follows the residual-prior direction by explicitly building
    representation inputs from both raw features and residualized features
    (`x - prior`).
    """

    def __init__(
        self,
        in_features: int,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 3,
        task_output_dim: int | None = None,
        use_residual_prior: bool = True,
    ) -> None:
        """Initialize module.

        Args:
            in_features: Input feature size.
            latent_dim: Latent feature size.
            hidden_dim: Hidden-layer width.
            num_layers: Number of hidden layers.
            task_output_dim: Optional class-count for source-domain supervision.
            use_residual_prior: Use `x - prior` branch in feature construction.
        """
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")

        self.in_features = in_features
        self.use_residual_prior = use_residual_prior

        encoder_in_features = in_features * 3 if use_residual_prior else in_features
        layers: list[nn.Module] = []
        current_dim = encoder_in_features
        for _ in range(num_layers):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, latent_dim))
        self.feature_extractor = nn.Sequential(*layers)
        self.invariance_projection = nn.Linear(latent_dim, latent_dim)

        self.task_head: nn.Module | None
        if task_output_dim is None:
            self.task_head = None
        else:
            self.task_head = nn.Linear(latent_dim, task_output_dim)

    def forward(self, x: Tensor, prior: Tensor | None = None) -> dict[str, Tensor]:
        """Compute latent and invariant features for one domain batch.

        Args:
            x: Input features. Shape: (batch, in_features).
            prior: Optional physics prior with same shape as `x`.

        Returns:
            Dict containing latent, invariant features, and residual branch.
        """
        if x.dim() != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"Expected x shape (batch, {self.in_features}), got {tuple(x.shape)}"
            )

        if prior is None:
            prior_tensor = torch.zeros_like(x)
        else:
            if prior.shape != x.shape:
                raise ValueError(
                    "Prior shape must match x shape. "
                    f"Got {tuple(prior.shape)} and {tuple(x.shape)}."
                )
            prior_tensor = prior

        residual = x - prior_tensor
        if self.use_residual_prior:
            encoder_input = torch.cat([x, prior_tensor, residual], dim=-1)
        else:
            encoder_input = x

        features = cast(Tensor, self.feature_extractor(encoder_input))
        invariant_features = cast(Tensor, self.invariance_projection(features))

        outputs: dict[str, Tensor] = {
            "features": features,
            "invariant_features": invariant_features,
            "physics_residual": residual,
        }
        if self.task_head is not None:
            outputs["task_logits"] = self.task_head(invariant_features)
        return outputs

    def compute_domain_losses(
        self,
        source_outputs: dict[str, Tensor],
        target_outputs: dict[str, Tensor],
        source_labels: Tensor | None = None,
        toggles: DomainInvariantLossToggles | None = None,
        weights: dict[str, float] | None = None,
    ) -> dict[str, Tensor]:
        """Compute task + invariance + physics residual alignment losses."""
        if toggles is None:
            toggles = DomainInvariantLossToggles()
        if weights is None:
            weights = {}

        reference = source_outputs["invariant_features"]
        zero = torch.zeros((), device=reference.device, dtype=reference.dtype)

        if "task_logits" in source_outputs and source_labels is not None:
            loss_task = nn.functional.cross_entropy(
                source_outputs["task_logits"], source_labels
            )
        else:
            loss_task = zero

        loss_invariance = coral_loss(
            source_outputs["invariant_features"],
            target_outputs["invariant_features"],
        )
        loss_physics_residual = residual_moment_alignment_loss(
            source_outputs["physics_residual"],
            target_outputs["physics_residual"],
        )

        component_losses = {
            "loss_task": loss_task,
            "loss_invariance": loss_invariance,
            "loss_physics_residual": loss_physics_residual,
        }
        active = {
            "loss_task": toggles.loss_task,
            "loss_invariance": toggles.loss_invariance,
            "loss_physics_residual": toggles.loss_physics_residual,
        }

        total = torch.zeros((), device=reference.device, dtype=reference.dtype)
        for key, value in component_losses.items():
            if active[key]:
                total = total + float(weights.get(key, 1.0)) * value

        return {
            **component_losses,
            "loss_total": total,
        }
