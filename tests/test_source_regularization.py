"""Tests for source-side regularisers and the train/val gap measurement."""

from __future__ import annotations

import math

import pytest
import torch

from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    build_synthetic_tta_loaders,
    run_tta_suite,
    train_source_only_tta_model,
)


def _run(config: TTAExperimentConfig, seed: int = 3) -> list:  # type: ignore[type-arg]
    loaders = build_synthetic_tta_loaders(
        shift_name="moderate",
        seed=seed,
        batch_size=config.batch_size,
        source_val_ratio=config.source_val_ratio,
        target_adapt_ratio=config.target_adapt_ratio,
    )
    return run_tta_suite(
        source_train_loader=loaders[0],
        source_val_loader=loaders[1],
        target_adapt_loader=loaders[2],
        target_test_loader=loaders[3],
        input_shape=loaders[4],
        num_classes=loaders[5],
        metadata=loaders[6],
        seed=seed,
        config=config,
    )


def _config(**overrides: float) -> TTAExperimentConfig:
    return TTAExperimentConfig(
        methods=("no_adapt", "entropy_tta"),
        source_epochs=2,
        adaptation_steps=2,
        batch_size=12,
        hidden_dim=24,
        num_layers=2,
        latent_dim=12,
        **overrides,  # type: ignore[arg-type]
    )


def test_rows_record_source_train_accuracy() -> None:
    rows = _run(_config())
    for row in rows:
        assert not math.isnan(row.source_train_accuracy)
        assert 0.0 <= row.source_train_accuracy <= 1.0


def test_measuring_train_accuracy_does_not_perturb_adaptation() -> None:
    """The train-loader pass must restore the RNG so later methods are unchanged."""
    torch.manual_seed(0)
    first = _run(_config())
    torch.manual_seed(0)
    second = _run(_config())
    assert [r.post_accuracy for r in first] == [r.post_accuracy for r in second]


@pytest.mark.parametrize(
    "overrides",
    [
        {"source_input_noise_std": 0.1},
        {"source_mixup_alpha": 0.4},
    ],
)
def test_each_regulariser_trains_and_changes_the_source_model(
    overrides: dict[str, float],
) -> None:
    baseline = _run(_config())
    regularised = _run(_config(**overrides))
    assert all(0.0 <= r.pre_accuracy <= 1.0 for r in regularised)
    changed = any(
        abs(a.source_pre_accuracy - b.source_pre_accuracy) > 0
        or abs(a.pre_accuracy - b.pre_accuracy) > 0
        or abs(a.source_train_accuracy - b.source_train_accuracy) > 0
        for a, b in zip(baseline, regularised, strict=True)
    )
    assert changed, f"{overrides} had no effect on the source model"


def test_regularisers_default_to_off() -> None:
    config = TTAExperimentConfig()
    assert config.source_weight_decay == 0.0
    assert config.source_input_noise_std == 0.0
    assert config.source_mixup_alpha == 0.0


def _param_norm(weight_decay: float) -> float:
    config = _config(source_weight_decay=weight_decay)
    loaders = build_synthetic_tta_loaders(
        shift_name="moderate",
        seed=4,
        batch_size=config.batch_size,
        source_val_ratio=config.source_val_ratio,
        target_adapt_ratio=config.target_adapt_ratio,
    )
    torch.manual_seed(4)
    model = train_source_only_tta_model(
        loaders[0], loaders[1], loaders[4], loaders[5], config
    )
    return float(sum(p.detach().norm() ** 2 for p in model.parameters()) ** 0.5)


def test_weight_decay_shrinks_source_weights() -> None:
    assert _param_norm(0.5) < _param_norm(0.0)
