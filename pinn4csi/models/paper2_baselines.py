# pyright: basic, reportMissingImports=false

"""Shared-backbone domain-adaptation baselines for Paper 2."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from pinn4csi.models.domain_invariant import (
    PhysicsDomainInvariantModule,
    coral_loss,
    residual_moment_alignment_loss,
)

CANONICAL_DOMAIN_ADAPTATION_BASELINES = (
    "residual_source_only",
    "coral",
    "dann",
    "maml",
)

_DOMAIN_ADAPTATION_BASELINE_ALIASES = {
    "source_only": "residual_source_only",
    "maml_lite": "maml",
}


@dataclass(frozen=True)
class DomainAdaptationBaselineConfig:
    """Hyperparameters shared across the Paper 2 baseline family."""

    hidden_dim: int = 64
    num_layers: int = 3
    latent_dim: int = 32
    use_residual_prior: bool = True
    invariance_weight: float = 1.0
    physics_weight: float = 0.5
    domain_weight: float = 0.5
    meta_weight: float = 0.5
    meta_inner_lr: float = 0.1
    domain_hidden_dim: int = 32
    backbone: str = "mlp"


class _CNN1DInvariantEncoder(nn.Module):
    """Conv1D encoder matching PhysicsDomainInvariantModule interface."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        latent_dim: int,
        use_residual_prior: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual_prior = use_residual_prior
        in_ch = 3 if use_residual_prior else 1

        # Progressive downsampling to avoid AdaptiveAvgPool(1) information loss
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_ch, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
        )
        self.flatten_proj = nn.Linear(hidden_dim * 8, hidden_dim)
        self.feature_projection = nn.Linear(hidden_dim, latent_dim)
        self.invariance_projection = nn.Linear(latent_dim, latent_dim)
        self._in_features = in_features

    def forward(self, x: Tensor, prior: Tensor | None = None) -> dict[str, Tensor]:
        if self.use_residual_prior and prior is not None:
            residual = x - prior
            stacked = torch.stack([x, prior, residual], dim=1)
        else:
            stacked = x.unsqueeze(1)

        conv_out = self.conv_block(stacked)
        flat = conv_out.reshape(conv_out.shape[0], -1)
        projected = torch.relu(self.flatten_proj(flat))
        features = self.feature_projection(projected)
        invariant_features = self.invariance_projection(features)
        return {
            "features": features,
            "invariant_features": invariant_features,
            "physics_residual": x
            - (prior if prior is not None else torch.zeros_like(x)),
        }


class _MLPBNInvariantEncoder(nn.Module):
    """MLP encoder with BatchNorm — isolates BN's contribution."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 3,
        use_residual_prior: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual_prior = use_residual_prior
        actual_in = in_features * 3 if use_residual_prior else in_features
        layers: list[nn.Module] = []
        prev_dim = actual_in
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                ]
            )
            prev_dim = hidden_dim
        self.feature_extractor = nn.Sequential(*layers)
        self.feature_projection = nn.Linear(hidden_dim, latent_dim)
        self.invariance_projection = nn.Linear(latent_dim, latent_dim)

    def forward(self, x: Tensor, prior: Tensor | None = None) -> dict[str, Tensor]:
        if self.use_residual_prior and prior is not None:
            residual = x - prior
            encoder_input = torch.cat([x, prior, residual], dim=-1)
        else:
            encoder_input = x
        hidden = self.feature_extractor(encoder_input)
        features = self.feature_projection(hidden)
        invariant_features = self.invariance_projection(features)
        return {
            "features": features,
            "invariant_features": invariant_features,
            "physics_residual": x
            - (prior if prior is not None else torch.zeros_like(x)),
        }


class Paper2DomainAdaptationBaseline(nn.Module):
    """Shared-backbone Paper 2 baseline model.

    All baselines reuse the same residual-prior encoder and source classifier so the
    comparison focuses on adaptation losses rather than backbone capacity. The
    `maml` option is intentionally a lightweight first-order hook, not a full
    paper-faithful MAML implementation.

    Set config.backbone to 'cnn1d' for Conv1D variant with BatchNorm
    (required for faithful TENT evaluation).
    """

    def __init__(
        self,
        baseline_name: str,
        input_shape: tuple[int, ...],
        num_classes: int,
        config: DomainAdaptationBaselineConfig | None = None,
    ) -> None:
        super().__init__()
        self.baseline_name = _canonicalize_domain_adaptation_baseline_name(
            baseline_name
        )
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.config = config or DomainAdaptationBaselineConfig()
        in_features = int(prod(input_shape))

        encoder: (
            _CNN1DInvariantEncoder
            | _MLPBNInvariantEncoder
            | PhysicsDomainInvariantModule
        )
        if self.config.backbone == "cnn1d":
            encoder = _CNN1DInvariantEncoder(
                in_features=in_features,
                hidden_dim=self.config.hidden_dim,
                latent_dim=self.config.latent_dim,
                use_residual_prior=self.config.use_residual_prior,
            )
        elif self.config.backbone == "mlp_bn":
            encoder = _MLPBNInvariantEncoder(
                in_features=in_features,
                hidden_dim=self.config.hidden_dim,
                latent_dim=self.config.latent_dim,
                num_layers=self.config.num_layers,
                use_residual_prior=self.config.use_residual_prior,
            )
        else:
            encoder = PhysicsDomainInvariantModule(
                in_features=in_features,
                latent_dim=self.config.latent_dim,
                hidden_dim=self.config.hidden_dim,
                num_layers=self.config.num_layers,
                task_output_dim=None,
                use_residual_prior=self.config.use_residual_prior,
            )
        self.encoder = encoder
        self.task_head = nn.Linear(self.config.latent_dim, num_classes)
        self.domain_head = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.domain_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.config.domain_hidden_dim, 2),
        )

    def forward(
        self,
        x: Tensor,
        prior: Tensor | None = None,
        has_prior: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Encode one batch and return logits plus intermediate features."""
        flat_x = _flatten_batch(x)
        flat_prior = _resolve_prior(flat_x, prior, has_prior)
        encoded = self.encoder(flat_x, flat_prior)
        logits = self.task_head(encoded["invariant_features"])
        return {
            **encoded,
            "task_logits": logits,
        }

    def compute_batch_losses(
        self,
        source_x: Tensor,
        source_labels: Tensor,
        target_x: Tensor,
        source_prior: Tensor | None = None,
        target_prior: Tensor | None = None,
        source_has_prior: Tensor | None = None,
        target_has_prior: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Compute losses for one paired source/target batch."""
        source_outputs = self(source_x, source_prior, source_has_prior)
        target_outputs = self(target_x, target_prior, target_has_prior)

        loss_task = functional.cross_entropy(
            source_outputs["task_logits"], source_labels
        )
        loss_invariance = coral_loss(
            source_outputs["invariant_features"],
            target_outputs["invariant_features"],
        )
        loss_physics_residual = residual_moment_alignment_loss(
            source_outputs["physics_residual"],
            target_outputs["physics_residual"],
        )
        zero = torch.zeros((), device=loss_task.device, dtype=loss_task.dtype)
        loss_domain = zero
        loss_meta = zero

        if self.baseline_name == "residual_source_only":
            loss_total = loss_task
        elif self.baseline_name == "coral":
            loss_total = (
                loss_task
                + self.config.invariance_weight * loss_invariance
                + self.config.physics_weight * loss_physics_residual
            )
        elif self.baseline_name == "dann":
            loss_domain = self._compute_domain_loss(
                source_outputs["invariant_features"],
                target_outputs["invariant_features"],
            )
            loss_total = loss_task + self.config.domain_weight * loss_domain
        elif self.baseline_name == "maml":
            loss_meta = self._compute_maml_hook_loss(
                source_outputs["invariant_features"], source_labels
            )
            loss_total = (
                loss_task
                + self.config.meta_weight * loss_meta
                + self.config.invariance_weight * loss_invariance
                + self.config.physics_weight * loss_physics_residual
            )
        else:
            raise RuntimeError(f"Unsupported baseline: {self.baseline_name}")

        return {
            "loss_total": loss_total,
            "loss_task": loss_task,
            "loss_invariance": loss_invariance,
            "loss_physics_residual": loss_physics_residual,
            "loss_domain": loss_domain,
            "loss_meta": loss_meta,
            "source_logits": source_outputs["task_logits"],
            "target_logits": target_outputs["task_logits"],
        }

    def _compute_domain_loss(
        self, source_features: Tensor, target_features: Tensor
    ) -> Tensor:
        domain_features = torch.cat([source_features, target_features], dim=0)
        reversed_features = _reverse_gradient(domain_features, scale=1.0)
        domain_logits = self.domain_head(reversed_features)
        source_labels = torch.zeros(
            source_features.shape[0],
            device=source_features.device,
            dtype=torch.long,
        )
        target_labels = torch.ones(
            target_features.shape[0],
            device=target_features.device,
            dtype=torch.long,
        )
        domain_labels = torch.cat([source_labels, target_labels], dim=0)
        return functional.cross_entropy(domain_logits, domain_labels)

    def _compute_maml_hook_loss(
        self, source_features: Tensor, source_labels: Tensor
    ) -> Tensor:
        support_features, query_features = _split_support_query(source_features)
        support_labels, query_labels = _split_support_query(source_labels)
        if support_features.shape[0] == 0 or query_features.shape[0] == 0:
            return torch.zeros(
                (), device=source_features.device, dtype=source_features.dtype
            )

        support_logits = self.task_head(support_features)
        support_loss = functional.cross_entropy(support_logits, support_labels)
        gradients = torch.autograd.grad(
            support_loss,
            [self.task_head.weight, self.task_head.bias],
            retain_graph=True,
            create_graph=False,
        )
        adapted_weight = (
            self.task_head.weight - self.config.meta_inner_lr * gradients[0]
        )
        adapted_bias = self.task_head.bias - self.config.meta_inner_lr * gradients[1]
        query_logits = functional.linear(query_features, adapted_weight, adapted_bias)
        return functional.cross_entropy(query_logits, query_labels)


def create_domain_adaptation_baseline(
    baseline_name: str,
    input_shape: tuple[int, ...],
    num_classes: int,
    config: DomainAdaptationBaselineConfig | None = None,
) -> Paper2DomainAdaptationBaseline:
    """Create one Paper 2 domain-adaptation baseline."""
    return Paper2DomainAdaptationBaseline(
        baseline_name=baseline_name,
        input_shape=input_shape,
        num_classes=num_classes,
        config=config,
    )


def list_domain_adaptation_baselines() -> tuple[str, ...]:
    """List canonical Paper 2 baseline names."""
    return CANONICAL_DOMAIN_ADAPTATION_BASELINES


def _canonicalize_domain_adaptation_baseline_name(name: str) -> str:
    key = name.lower()
    key = _DOMAIN_ADAPTATION_BASELINE_ALIASES.get(key, key)
    if key not in CANONICAL_DOMAIN_ADAPTATION_BASELINES:
        valid = ", ".join(CANONICAL_DOMAIN_ADAPTATION_BASELINES)
        raise ValueError(
            f"Unknown domain-adaptation baseline: {name}. Expected one of: {valid}"
        )
    return key


def _flatten_batch(x: Tensor) -> Tensor:
    if x.dim() < 2:
        raise ValueError(f"Expected batched tensor with dim >= 2, got {tuple(x.shape)}")
    if x.dim() == 2:
        return x
    return x.reshape(x.shape[0], -1)


def _resolve_prior(
    flat_x: Tensor,
    prior: Tensor | None,
    has_prior: Tensor | None,
) -> Tensor:
    if prior is None:
        return torch.zeros_like(flat_x)
    flat_prior = _flatten_batch(prior)
    if flat_prior.shape != flat_x.shape:
        raise ValueError(
            "Prior shape must match flattened input shape. "
            f"Got {tuple(flat_prior.shape)} and {tuple(flat_x.shape)}."
        )
    if has_prior is None:
        return flat_prior
    if has_prior.dim() != 1 or has_prior.shape[0] != flat_x.shape[0]:
        raise ValueError(
            "has_prior must have shape (batch,). "
            f"Got {tuple(has_prior.shape)} for batch {flat_x.shape[0]}."
        )
    mask = has_prior.to(device=flat_x.device, dtype=flat_x.dtype).unsqueeze(-1)
    return flat_prior * mask


def _split_support_query(batch: Tensor) -> tuple[Tensor, Tensor]:
    midpoint = batch.shape[0] // 2
    return batch[:midpoint], batch[midpoint:]


def _reverse_gradient(x: Tensor, scale: float) -> Tensor:
    return -scale * x + (1.0 + scale) * x.detach()
