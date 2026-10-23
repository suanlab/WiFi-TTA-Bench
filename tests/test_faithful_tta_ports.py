from __future__ import annotations

import copy

import pytest
import torch

from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    _batch_logits,
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


def test_batch_stat_evaluation_leaves_running_statistics_untouched() -> None:
    from pinn4csi.training.paper2_tta import evaluate_tta_classifier

    _, model, _, loaders, *_ = _setup()
    bn = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    before = [m.running_mean.clone() for m in bn]
    running = evaluate_tta_classifier(model, loaders[3], torch.device("cpu"))
    batch = evaluate_tta_classifier(
        model, loaders[3], torch.device("cpu"), bn_batch_stats=True
    )
    assert all(torch.equal(b, m.running_mean) for b, m in zip(before, bn, strict=True))
    assert all(not m.training and m.momentum == 0.1 for m in bn)
    assert 0.0 <= running["accuracy"] <= 1.0 and 0.0 <= batch["accuracy"] <= 1.0


def test_bn_reset_cumulative_replaces_source_statistics() -> None:
    config, model, reference, loaders, *_ = _setup(bn_reset_cumulative=True)
    summary = adapt_model_for_tta_with_summary(
        model, loaders[2], reference, config, "bn_reset", torch.device("cpu")
    )
    adapted = [
        m for m in summary.model.modules() if isinstance(m, torch.nn.BatchNorm1d)
    ]
    source = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert all(m.momentum == 0.1 for m in adapted)
    assert any(
        not torch.allclose(a.running_mean, s.running_mean)
        for a, s in zip(adapted, source, strict=True)
    )


@pytest.mark.slow
def test_guarded_published_methods_roll_back_on_any_diversity_loss() -> None:
    # A floor of 1.0 forbids any diversity loss, so guarded runs stop early or
    # keep every step that does not reduce diversity.
    config, model, reference, loaders, *_ = _setup(
        sar_margin_factor=10.0, collapse_guard_diversity_ratio=1.0
    )
    for method in ("collapse_guarded_sar_full", "collapse_guarded_cotta_full"):
        summary = adapt_model_for_tta_with_summary(
            model, loaders[2], reference, config, method, torch.device("cpu")
        )
        assert summary.accepted_steps <= config.adaptation_steps
        assert summary.stop_reason


def test_frozen_scoring_pass_updates_bn_statistics_once_per_step() -> None:
    def _tracked(freeze: bool) -> int:
        config, model, reference, loaders, *_ = _setup(
            adaptation_learning_rate=0.0, freeze_bn_in_scoring_pass=freeze
        )
        summary = adapt_model_for_tta_with_summary(
            model, loaders[2], reference, config, "tent", torch.device("cpu")
        )
        before = {n: int(b) for n, b in model.named_buffers() if "num_batches" in n}
        after = {
            n: int(b) for n, b in summary.model.named_buffers() if "num_batches" in n
        }
        assert before
        return max(after[n] - before[n] for n in before)

    assert _tracked(freeze=False) == 4
    # num_batches_tracked still counts the frozen pass; the running mean must
    # match a single update per step instead.
    config, model, reference, loaders, *_ = _setup(
        adaptation_learning_rate=0.0, freeze_bn_in_scoring_pass=True
    )
    summary = adapt_model_for_tta_with_summary(
        model, loaders[2], reference, config, "tent", torch.device("cpu")
    )
    expected = copy.deepcopy(model)
    expected.train()
    batches = iter(loaders[2])
    with torch.no_grad():
        for _ in range(config.adaptation_steps):
            _batch_logits(expected, next(batches))
    for (name, got), (_, want) in zip(
        summary.model.named_buffers(), expected.named_buffers(), strict=True
    ):
        if "running" in name:
            assert torch.allclose(got, want, atol=1e-6), name


def test_guard_that_never_fires_leaves_sar_unchanged() -> None:
    # A floor of 0 can never fire: guarded SAR must end exactly where SAR does,
    # including BatchNorm running statistics.
    config, model, reference, loaders, *_ = _setup(
        sar_margin_factor=10.0, collapse_guard_diversity_ratio=0.0
    )
    states = []
    for method in ("sar_full", "collapse_guarded_sar_full"):
        torch.manual_seed(0)
        summary = adapt_model_for_tta_with_summary(
            model, loaders[2], reference, config, method, torch.device("cpu")
        )
        states.append(summary.model.state_dict())
    for name, value in states[0].items():
        # The batch counter only matters for cumulative (momentum=None) BN.
        if name.endswith("num_batches_tracked"):
            continue
        assert torch.equal(value, states[1][name]), name


def test_batch_stat_evaluation_handles_a_one_sample_batch() -> None:
    from torch.utils.data import DataLoader, Subset

    from pinn4csi.training.paper2_tta import evaluate_tta_classifier

    config, model, reference, loaders, *_ = _setup()
    test_set = loaders[3].dataset
    # 33 samples with batch size 32 leave a one-sample remainder batch.
    loader = DataLoader(Subset(test_set, range(33)), batch_size=32)
    metrics = evaluate_tta_classifier(
        model, loader, torch.device("cpu"), bn_batch_stats=True
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0
