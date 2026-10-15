from __future__ import annotations

import pytest
import torch

from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    adapt_model_for_tta_with_summary,
    build_synthetic_tta_loaders,
    compute_tta_reference_stats,
    evaluate_t3a_online,
    run_tta_suite,
    train_source_only_tta_model,
)


def _setup(backbone: str = "mlp_bn", **overrides: object):
    settings: dict[str, object] = {
        "source_epochs": 1,
        "adaptation_steps": 2,
        "batch_size": 8,
        "backbone": backbone,
        "cotta_num_augmentations": 2,
    }
    settings.update(overrides)
    config = TTAExperimentConfig(**settings)  # type: ignore[arg-type]
    (
        source_train,
        source_val,
        target_adapt,
        target_test,
        input_shape,
        num_classes,
        metadata,
    ) = build_synthetic_tta_loaders(
        shift_name="moderate",
        seed=0,
        batch_size=config.batch_size,
        source_val_ratio=0.2,
        target_adapt_ratio=0.5,
    )
    model = train_source_only_tta_model(
        source_train, source_val, input_shape, num_classes, config
    )
    reference = compute_tta_reference_stats(model, source_train, torch.device("cpu"))
    loaders = (source_train, source_val, target_adapt, target_test)
    return config, model, reference, loaders, input_shape, num_classes, metadata


def _changed(a: torch.nn.Module, b: torch.nn.Module) -> set[str]:
    sb = dict(b.named_parameters())
    return {n for n, p in a.named_parameters() if not torch.equal(p, sb[n])}


@pytest.mark.slow
def test_sar_full_updates_only_normalisation_affines() -> None:
    # A permissive margin makes every sample "reliable" so the update runs.
    config, model, reference, loaders, *_ = _setup(sar_margin_factor=10.0)
    summary = adapt_model_for_tta_with_summary(
        model, loaders[2], reference, config, "sar_full", torch.device("cpu")
    )
    changed = _changed(summary.model, model)
    assert summary.accepted_steps > 0
    assert changed
    norm_names = {
        f"{name}.{p}"
        for name, m in model.named_modules()
        if isinstance(m, torch.nn.BatchNorm1d | torch.nn.BatchNorm2d)
        for p in ("weight", "bias")
    }
    assert changed <= norm_names


def test_sar_full_model_recovery_restores_source_weights() -> None:
    # A reset threshold above any entropy forces recovery after every step.
    config, model, reference, loaders, *_ = _setup(
        sar_margin_factor=10.0, sar_reset_threshold=1e9
    )
    summary = adapt_model_for_tta_with_summary(
        model, loaders[2], reference, config, "sar_full", torch.device("cpu")
    )
    assert summary.stop_reason == f"sar_full_resets_{summary.accepted_steps}"
    assert not _changed(summary.model, model)


@pytest.mark.slow
def test_cotta_full_teacher_is_ema_of_student() -> None:
    # Momentum 1.0 freezes the teacher, which is the model that predicts.
    config, model, reference, loaders, *_ = _setup(cotta_ema_momentum=1.0)
    summary = adapt_model_for_tta_with_summary(
        model, loaders[2], reference, config, "cotta_full", torch.device("cpu")
    )
    assert not _changed(summary.model, model)

    config, model, reference, loaders, *_ = _setup(cotta_ema_momentum=0.5)
    summary = adapt_model_for_tta_with_summary(
        model, loaders[2], reference, config, "cotta_full", torch.device("cpu")
    )
    assert _changed(summary.model, model)
    assert not any("task_head" in n for n in _changed(summary.model, model))


def test_t3a_full_is_gradient_free_and_scores_online() -> None:
    config, model, reference, loaders, _, num_classes, _ = _setup(backbone="mlp")
    summary = adapt_model_for_tta_with_summary(
        model, loaders[2], reference, config, "t3a_full", torch.device("cpu")
    )
    assert summary.attempted_steps == 0
    assert not _changed(summary.model, model)
    metrics = evaluate_t3a_online(
        summary.model, loaders[3], torch.device("cpu"), num_classes, filter_k=5
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0


@pytest.mark.slow
def test_faithful_ports_run_in_suite() -> None:
    config, _, _, loaders, input_shape, num_classes, metadata = _setup()
    suite_config = TTAExperimentConfig(
        methods=("no_adapt", "sar_full", "cotta_full", "t3a_full"),
        source_epochs=1,
        adaptation_steps=1,
        batch_size=8,
        backbone="mlp_bn",
        cotta_num_augmentations=2,
    )
    rows = run_tta_suite(
        *loaders,
        input_shape=input_shape,
        num_classes=num_classes,
        metadata=metadata,
        seed=0,
        config=suite_config,
    )
    assert [r.method for r in rows] == list(suite_config.methods)
    assert all(0.0 <= r.post_accuracy <= 1.0 for r in rows)
