"""Tests for the grid backbones (ResNet2D with BN/GN, Transformer with LN)."""

from __future__ import annotations

import copy

import pytest
import torch

from pinn4csi.models.paper2_baselines import (
    DomainAdaptationBaselineConfig,
    create_domain_adaptation_baseline,
)
from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    adapt_model_for_tta_with_summary,
    build_synthetic_tta_loaders,
    compute_tta_reference_stats,
)

GRID_BACKBONES = ("resnet2d", "resnet2d_gn", "transformer")
SHAPES = ((22, 400), (114, 500), (200, 90))


@pytest.mark.parametrize("backbone", GRID_BACKBONES)
@pytest.mark.parametrize("shape", SHAPES)
def test_forward_shapes_on_every_benchmark_input(
    backbone: str, shape: tuple[int, int]
) -> None:
    model = create_domain_adaptation_baseline(
        "residual_source_only",
        shape,
        6,
        DomainAdaptationBaselineConfig(backbone=backbone, hidden_dim=32, latent_dim=16),
    )
    out = model(torch.randn(3, *shape))
    assert out["task_logits"].shape == (3, 6)
    assert out["features"].shape == (3, 16)


@pytest.mark.parametrize("backbone", GRID_BACKBONES)
def test_backward_gives_finite_gradients(backbone: str) -> None:
    model = create_domain_adaptation_baseline(
        "residual_source_only",
        (22, 400),
        6,
        DomainAdaptationBaselineConfig(backbone=backbone, hidden_dim=32, latent_dim=16),
    )
    loss = model(torch.randn(4, 22, 400))["task_logits"].sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_group_and_batch_variants_differ_only_in_normaliser() -> None:
    def norm_kinds(backbone: str) -> tuple[set[str], int]:
        model = create_domain_adaptation_baseline(
            "residual_source_only",
            (22, 400),
            6,
            DomainAdaptationBaselineConfig(backbone=backbone),
        )
        kinds = {
            type(m).__name__ for m in model.modules() if "Norm" in type(m).__name__
        }
        return kinds, sum(p.numel() for p in model.parameters())

    bn_kinds, bn_params = norm_kinds("resnet2d")
    gn_kinds, gn_params = norm_kinds("resnet2d_gn")
    assert bn_kinds == {"BatchNorm2d"} and gn_kinds == {"GroupNorm"}
    assert bn_params == gn_params


@pytest.mark.parametrize(
    ("backbone", "norm_type"),
    [("resnet2d_gn", torch.nn.GroupNorm), ("transformer", torch.nn.LayerNorm)],
)
def test_tent_updates_only_sample_norm_affines_without_batchnorm(
    backbone: str, norm_type: type
) -> None:
    config = TTAExperimentConfig(
        methods=("tent",),
        source_epochs=1,
        adaptation_steps=2,
        batch_size=12,
        hidden_dim=32,
        latent_dim=16,
        backbone=backbone,
    )
    loaders = build_synthetic_tta_loaders(
        shift_name="moderate",
        seed=0,
        batch_size=config.batch_size,
        source_val_ratio=config.source_val_ratio,
        target_adapt_ratio=config.target_adapt_ratio,
    )
    model = create_domain_adaptation_baseline(
        "residual_source_only",
        loaders[4],
        loaders[5],
        DomainAdaptationBaselineConfig(backbone=backbone, hidden_dim=32, latent_dim=16),
    )
    before = copy.deepcopy(model.state_dict())
    stats = compute_tta_reference_stats(model, loaders[0], torch.device("cpu"))
    adapted = adapt_model_for_tta_with_summary(
        model, loaders[2], stats, config, "tent", torch.device("cpu")
    ).model
    norm_names = {
        f"{name}.{p}"
        for name, m in adapted.named_modules()
        if isinstance(m, norm_type)
        for p in ("weight", "bias")
    }
    after = adapted.state_dict()
    changed = {k for k in before if not torch.equal(before[k], after[k])}
    assert changed, "TENT updated nothing"
    assert changed <= norm_names, (
        f"TENT touched non-norm params: {changed - norm_names}"
    )
