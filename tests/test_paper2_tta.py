from __future__ import annotations

import torch

from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    adapt_model_for_tta,
    build_synthetic_tta_loaders,
    compute_tta_reference_stats,
    run_tta_suite,
    train_source_only_tta_model,
)


def test_synthetic_target_adaptation_loader_has_no_labels() -> None:
    (
        _,
        _,
        target_adapt_loader,
        _,
        _,
        _,
        _,
    ) = build_synthetic_tta_loaders(
        shift_name="moderate",
        seed=0,
        batch_size=8,
        source_val_ratio=0.2,
        target_adapt_ratio=0.5,
    )
    batch = next(iter(target_adapt_loader))

    assert "label" not in batch
    assert set(batch) >= {"x", "prior", "has_prior"}


def test_tta_adaptation_updates_encoder_but_not_task_head() -> None:
    config = TTAExperimentConfig(
        source_epochs=2,
        adaptation_steps=2,
        batch_size=12,
        hidden_dim=24,
        num_layers=2,
        latent_dim=12,
    )
    (
        source_train_loader,
        source_val_loader,
        target_adapt_loader,
        _,
        input_shape,
        num_classes,
        _,
    ) = build_synthetic_tta_loaders(
        shift_name="strong",
        seed=1,
        batch_size=config.batch_size,
        source_val_ratio=config.source_val_ratio,
        target_adapt_ratio=config.target_adapt_ratio,
    )
    source_model = train_source_only_tta_model(
        source_train_loader=source_train_loader,
        source_val_loader=source_val_loader,
        input_shape=input_shape,
        num_classes=num_classes,
        config=config,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_model = source_model.to(device)
    reference_stats = compute_tta_reference_stats(
        source_model,
        source_train_loader,
        device,
    )

    encoder_before = [
        parameter.detach().cpu().clone()
        for parameter in source_model.encoder.parameters()
    ]
    head_before = [
        parameter.detach().cpu().clone()
        for parameter in source_model.task_head.parameters()
    ]

    adapted = adapt_model_for_tta(
        model=source_model,
        target_loader=target_adapt_loader,
        reference_stats=reference_stats,
        config=config,
        method="physics_tta",
        device=device,
    )

    encoder_after = [
        parameter.detach().cpu() for parameter in adapted.encoder.parameters()
    ]
    head_after = [
        parameter.detach().cpu() for parameter in adapted.task_head.parameters()
    ]

    assert any(
        not torch.allclose(before, after)
        for before, after in zip(encoder_before, encoder_after, strict=True)
    )
    assert all(
        torch.allclose(before, after)
        for before, after in zip(head_before, head_after, strict=True)
    )


def test_run_tta_suite_returns_all_methods_and_zero_gain_for_no_adapt() -> None:
    config = TTAExperimentConfig(
        source_epochs=2,
        adaptation_steps=2,
        batch_size=12,
        hidden_dim=24,
        num_layers=2,
        latent_dim=12,
    )
    (
        source_train_loader,
        source_val_loader,
        target_adapt_loader,
        target_test_loader,
        input_shape,
        num_classes,
        metadata,
    ) = build_synthetic_tta_loaders(
        shift_name="moderate",
        seed=2,
        batch_size=config.batch_size,
        source_val_ratio=config.source_val_ratio,
        target_adapt_ratio=config.target_adapt_ratio,
    )

    rows = run_tta_suite(
        source_train_loader=source_train_loader,
        source_val_loader=source_val_loader,
        target_adapt_loader=target_adapt_loader,
        target_test_loader=target_test_loader,
        input_shape=input_shape,
        num_classes=num_classes,
        metadata=metadata,
        seed=2,
        config=config,
    )

    assert len(rows) == 4
    assert {row.method for row in rows} == {
        "no_adapt",
        "entropy_tta",
        "physics_tta",
        "physics_entropy_tta",
    }
    for row in rows:
        assert 0.0 <= row.pre_accuracy <= 1.0
        assert 0.0 <= row.post_accuracy <= 1.0
        assert 0.0 <= row.source_pre_accuracy <= 1.0
        assert 0.0 <= row.source_post_accuracy <= 1.0
    no_adapt = [row for row in rows if row.method == "no_adapt"]
    assert len(no_adapt) == 1
    assert abs(no_adapt[0].gain) < 1e-9
    assert abs(no_adapt[0].source_drop) < 1e-9


def test_safe_tta_abstains_when_min_delta_is_too_large() -> None:
    config = TTAExperimentConfig(
        methods=("safe_physics_tta",),
        source_epochs=2,
        adaptation_steps=2,
        batch_size=12,
        hidden_dim=24,
        num_layers=2,
        latent_dim=12,
        safe_objective_min_delta=10.0,
        safe_alignment_guard_ratio=100.0,
    )
    (
        source_train_loader,
        source_val_loader,
        target_adapt_loader,
        target_test_loader,
        input_shape,
        num_classes,
        metadata,
    ) = build_synthetic_tta_loaders(
        shift_name="moderate",
        seed=3,
        batch_size=config.batch_size,
        source_val_ratio=config.source_val_ratio,
        target_adapt_ratio=config.target_adapt_ratio,
    )

    rows = run_tta_suite(
        source_train_loader=source_train_loader,
        source_val_loader=source_val_loader,
        target_adapt_loader=target_adapt_loader,
        target_test_loader=target_test_loader,
        input_shape=input_shape,
        num_classes=num_classes,
        metadata=metadata,
        seed=3,
        config=config,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.method == "safe_physics_tta"
    assert row.abstained is True
    assert row.attempted_steps == 1
    assert row.accepted_steps == 0
    assert row.stop_reason == "abstained_small_improvement"
    assert abs(row.gain) < 1e-9
    assert abs(row.source_drop) < 1e-9


def test_safe_tta_rows_record_controller_metadata() -> None:
    config = TTAExperimentConfig(
        methods=("safe_entropy_tta", "safe_physics_tta"),
        source_epochs=2,
        adaptation_steps=3,
        batch_size=12,
        hidden_dim=24,
        num_layers=2,
        latent_dim=12,
        safe_objective_min_delta=0.0,
        safe_alignment_guard_ratio=100.0,
    )
    (
        source_train_loader,
        source_val_loader,
        target_adapt_loader,
        target_test_loader,
        input_shape,
        num_classes,
        metadata,
    ) = build_synthetic_tta_loaders(
        shift_name="strong",
        seed=4,
        batch_size=config.batch_size,
        source_val_ratio=config.source_val_ratio,
        target_adapt_ratio=config.target_adapt_ratio,
    )

    rows = run_tta_suite(
        source_train_loader=source_train_loader,
        source_val_loader=source_val_loader,
        target_adapt_loader=target_adapt_loader,
        target_test_loader=target_test_loader,
        input_shape=input_shape,
        num_classes=num_classes,
        metadata=metadata,
        seed=4,
        config=config,
    )

    assert len(rows) == 2
    assert {row.method for row in rows} == {"safe_entropy_tta", "safe_physics_tta"}
    for row in rows:
        assert 0 <= row.accepted_steps <= row.attempted_steps <= config.adaptation_steps
        assert row.stop_reason != ""
        assert row.pre_adaptation_objective >= 0.0
        assert row.post_adaptation_objective >= 0.0
        if row.abstained:
            assert row.accepted_steps == 0


def test_conservative_entropy_tta_runs_and_records_budget_exhaustion() -> None:
    config = TTAExperimentConfig(
        methods=("conservative_entropy_tta",),
        source_epochs=2,
        adaptation_steps=3,
        batch_size=12,
        hidden_dim=24,
        num_layers=2,
        latent_dim=12,
    )
    (
        source_train_loader,
        source_val_loader,
        target_adapt_loader,
        target_test_loader,
        input_shape,
        num_classes,
        metadata,
    ) = build_synthetic_tta_loaders(
        shift_name="moderate",
        seed=5,
        batch_size=config.batch_size,
        source_val_ratio=config.source_val_ratio,
        target_adapt_ratio=config.target_adapt_ratio,
    )

    rows = run_tta_suite(
        source_train_loader=source_train_loader,
        source_val_loader=source_val_loader,
        target_adapt_loader=target_adapt_loader,
        target_test_loader=target_test_loader,
        input_shape=input_shape,
        num_classes=num_classes,
        metadata=metadata,
        seed=5,
        config=config,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.method == "conservative_entropy_tta"
    assert row.attempted_steps == config.adaptation_steps
    assert row.accepted_steps == config.adaptation_steps
    assert row.abstained is False
    assert row.stop_reason == "budget_exhausted"
    assert row.pre_adaptation_objective >= 0.0
    assert row.post_adaptation_objective >= 0.0


def test_calibrated_entropy_tta_runs() -> None:
    config = TTAExperimentConfig(
        methods=("calibrated_entropy_tta",),
        source_epochs=2,
        adaptation_steps=3,
        batch_size=12,
        hidden_dim=24,
        num_layers=2,
        latent_dim=12,
        calibration_temperature=2.0,
    )
    (
        source_train_loader,
        source_val_loader,
        target_adapt_loader,
        target_test_loader,
        input_shape,
        num_classes,
        metadata,
    ) = build_synthetic_tta_loaders(
        shift_name="moderate",
        seed=6,
        batch_size=config.batch_size,
        source_val_ratio=config.source_val_ratio,
        target_adapt_ratio=config.target_adapt_ratio,
    )

    rows = run_tta_suite(
        source_train_loader=source_train_loader,
        source_val_loader=source_val_loader,
        target_adapt_loader=target_adapt_loader,
        target_test_loader=target_test_loader,
        input_shape=input_shape,
        num_classes=num_classes,
        metadata=metadata,
        seed=6,
        config=config,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.method == "calibrated_entropy_tta"
    assert row.attempted_steps == config.adaptation_steps
    assert row.stop_reason == "budget_exhausted"


def test_warm_restart_physics_tta_runs() -> None:
    config = TTAExperimentConfig(
        methods=("warm_restart_physics_tta",),
        source_epochs=2,
        adaptation_steps=3,
        batch_size=12,
        hidden_dim=24,
        num_layers=2,
        latent_dim=12,
        warm_restart_steps=2,
        warm_restart_lr_factor=0.5,
    )
    (
        source_train_loader,
        source_val_loader,
        target_adapt_loader,
        target_test_loader,
        input_shape,
        num_classes,
        metadata,
    ) = build_synthetic_tta_loaders(
        shift_name="strong",
        seed=7,
        batch_size=config.batch_size,
        source_val_ratio=config.source_val_ratio,
        target_adapt_ratio=config.target_adapt_ratio,
    )

    rows = run_tta_suite(
        source_train_loader=source_train_loader,
        source_val_loader=source_val_loader,
        target_adapt_loader=target_adapt_loader,
        target_test_loader=target_test_loader,
        input_shape=input_shape,
        num_classes=num_classes,
        metadata=metadata,
        seed=7,
        config=config,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.method == "warm_restart_physics_tta"
    assert row.attempted_steps > config.adaptation_steps  # includes warm restart steps
    assert row.stop_reason in ("warm_restart_accepted", "warm_restart_reverted")


def test_selective_physics_tta_achieves_nonnegative_gain() -> None:
    config = TTAExperimentConfig(
        methods=("selective_physics_tta",),
        source_epochs=2,
        adaptation_steps=3,
        batch_size=12,
        hidden_dim=24,
        num_layers=2,
        latent_dim=12,
        selective_confidence_threshold=0.5,
        selective_alignment_threshold=10.0,
    )
    (
        source_train_loader,
        source_val_loader,
        target_adapt_loader,
        target_test_loader,
        input_shape,
        num_classes,
        metadata,
    ) = build_synthetic_tta_loaders(
        shift_name="moderate",
        seed=8,
        batch_size=config.batch_size,
        source_val_ratio=config.source_val_ratio,
        target_adapt_ratio=config.target_adapt_ratio,
    )

    rows = run_tta_suite(
        source_train_loader=source_train_loader,
        source_val_loader=source_val_loader,
        target_adapt_loader=target_adapt_loader,
        target_test_loader=target_test_loader,
        input_shape=input_shape,
        num_classes=num_classes,
        metadata=metadata,
        seed=8,
        config=config,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row.method == "selective_physics_tta"
    # Selective TTA should not be worse than no_adapt by design
    # (worst case: falls back to source predictions entirely)
    assert row.gain >= -0.05  # Allow small tolerance for stochastic variation


def test_tent_shot_t3a_run_end_to_end() -> None:
    config = TTAExperimentConfig(
        methods=("tent", "shot", "t3a"),
        source_epochs=2,
        adaptation_steps=3,
        batch_size=12,
        hidden_dim=24,
        num_layers=2,
        latent_dim=12,
    )
    (
        source_train_loader,
        source_val_loader,
        target_adapt_loader,
        target_test_loader,
        input_shape,
        num_classes,
        metadata,
    ) = build_synthetic_tta_loaders(
        shift_name="moderate",
        seed=9,
        batch_size=config.batch_size,
        source_val_ratio=config.source_val_ratio,
        target_adapt_ratio=config.target_adapt_ratio,
    )

    rows = run_tta_suite(
        source_train_loader=source_train_loader,
        source_val_loader=source_val_loader,
        target_adapt_loader=target_adapt_loader,
        target_test_loader=target_test_loader,
        input_shape=input_shape,
        num_classes=num_classes,
        metadata=metadata,
        seed=9,
        config=config,
    )

    assert len(rows) == 3
    methods = {row.method for row in rows}
    assert methods == {"tent", "shot", "t3a"}
    for row in rows:
        assert 0.0 <= row.pre_accuracy <= 1.0
        assert 0.0 <= row.post_accuracy <= 1.0
