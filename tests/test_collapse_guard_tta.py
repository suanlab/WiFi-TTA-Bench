"""Tests for the collapse-guarded TTA controller."""

from pinn4csi.training.paper2_tta import (
    CANONICAL_TTA_METHODS,
    TTAExperimentConfig,
    build_synthetic_tta_loaders,
    run_tta_suite,
)


def _run(config: TTAExperimentConfig, seed: int = 5) -> list:
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
        seed=seed,
        batch_size=config.batch_size,
        source_val_ratio=config.source_val_ratio,
        target_adapt_ratio=config.target_adapt_ratio,
    )
    return run_tta_suite(
        source_train_loader=source_train_loader,
        source_val_loader=source_val_loader,
        target_adapt_loader=target_adapt_loader,
        target_test_loader=target_test_loader,
        input_shape=input_shape,
        num_classes=num_classes,
        metadata=metadata,
        seed=seed,
        config=config,
    )


def _base_config(**overrides: object) -> TTAExperimentConfig:
    defaults: dict[str, object] = {
        "methods": ("collapse_guarded_entropy_tta",),
        "source_epochs": 2,
        "adaptation_steps": 3,
        "batch_size": 12,
        "hidden_dim": 24,
        "num_layers": 2,
        "latent_dim": 12,
    }
    defaults.update(overrides)
    return TTAExperimentConfig(**defaults)  # type: ignore[arg-type]


def test_collapse_guarded_methods_are_registered() -> None:
    for name in (
        "collapse_guarded_entropy_tta",
        "collapse_guarded_tent",
        "collapse_guarded_sar",
        "collapse_guarded_cotta",
        "collapse_guarded_t3a",
        "collapse_guarded_shot",
        "reset_entropy_tta",
    ):
        assert name in CANONICAL_TTA_METHODS


def test_weight_reset_is_a_noop_at_probability_zero() -> None:
    """`reset_entropy_tta` must reduce to plain entropy TTA when nothing is restored.

    Pins both the alias resolution and the restore call site: a wrong base method
    or a restore applied unconditionally would break this equality.
    """
    config = _base_config(
        methods=("entropy_tta", "reset_entropy_tta"),
        weight_reset_probability=0.0,
    )
    rows = _run(config, seed=13)
    by_method = {r.method: r for r in rows}
    assert set(by_method) == {"entropy_tta", "reset_entropy_tta"}
    assert by_method["reset_entropy_tta"].gain == by_method["entropy_tta"].gain


def test_weight_reset_changes_the_outcome_when_active() -> None:
    """A non-zero restore probability must actually alter adaptation.

    Asserted on the post-adaptation objective rather than accuracy gain: the
    synthetic target split is small enough that accuracy is coarse and both
    variants can land on the same value even when the weights differ.
    """
    rows = _run(
        _base_config(
            methods=("entropy_tta", "reset_entropy_tta"),
            weight_reset_probability=1.0,
            adaptation_steps=3,
        ),
        seed=13,
    )
    by_method = {r.method: r for r in rows}
    assert by_method["reset_entropy_tta"].post_adaptation_objective != (
        by_method["entropy_tta"].post_adaptation_objective
    )


def test_guarded_t3a_matches_t3a_when_guard_is_disabled() -> None:
    """T3A is scored by its own prototype classifier, not the model head.

    With the guard off, the alias must reproduce T3A exactly; if it fell back to
    the standard evaluation path it would be a different method and the
    guarded-vs-unguarded comparison would be meaningless.
    """
    config = _base_config(
        methods=("t3a", "collapse_guarded_t3a"),
        collapse_guard_diversity_ratio=0.0,
    )
    rows = _run(config, seed=11)
    by_method = {r.method: r for r in rows}
    assert set(by_method) == {"t3a", "collapse_guarded_t3a"}
    assert by_method["collapse_guarded_t3a"].post_accuracy == (
        by_method["t3a"].post_accuracy
    )
    assert by_method["collapse_guarded_t3a"].gain == by_method["t3a"].gain


def test_guarded_tent_matches_tent_when_guard_is_disabled() -> None:
    """With the guard off, the TENT alias must reproduce TENT exactly.

    This pins the alias resolution *and* the BN-only update scope: if the
    guarded variant fell back to the default encoder scope it would be a
    different method and the comparison would be meaningless.
    """
    config = _base_config(
        methods=("tent", "collapse_guarded_tent"),
        collapse_guard_diversity_ratio=0.0,
    )
    rows = _run(config, seed=7)
    by_method = {r.method: r for r in rows}
    assert set(by_method) == {"tent", "collapse_guarded_tent"}
    assert by_method["collapse_guarded_tent"].post_accuracy == (
        by_method["tent"].post_accuracy
    )
    assert by_method["collapse_guarded_tent"].gain == by_method["tent"].gain


def test_collapse_guard_abstains_when_no_collapse_is_tolerated() -> None:
    """A ratio above 1.0 is unsatisfiable, so the first step must roll back.

    Diversity is bounded by 1.0, so requiring post-diversity to exceed
    1.5x the reference forces the guard to reject immediately.
    """
    rows = _run(_base_config(collapse_guard_diversity_ratio=1.5))
    assert len(rows) == 1
    row = rows[0]
    assert row.method == "collapse_guarded_entropy_tta"
    assert row.abstained is True
    assert row.accepted_steps == 0
    assert row.attempted_steps == 1
    assert row.stop_reason == "abstained_collapse_guard"
    assert abs(row.gain) < 1e-9
    assert abs(row.source_drop) < 1e-9


def test_collapse_guard_disabled_accepts_every_step() -> None:
    """Ratio 0.0 disables the guard, so the full step budget is spent."""
    config = _base_config(collapse_guard_diversity_ratio=0.0)
    rows = _run(config)
    assert len(rows) == 1
    row = rows[0]
    assert row.abstained is False
    assert row.accepted_steps == config.adaptation_steps
    assert row.stop_reason == "budget_exhausted"


def test_collapse_guard_runs_alongside_unguarded_entropy() -> None:
    """Both variants complete and are reported separately."""
    rows = _run(
        _base_config(
            methods=("entropy_tta", "collapse_guarded_entropy_tta"),
            collapse_guard_diversity_ratio=0.9,
        )
    )
    assert {r.method for r in rows} == {
        "entropy_tta",
        "collapse_guarded_entropy_tta",
    }
    for row in rows:
        assert row.attempted_steps >= 1
