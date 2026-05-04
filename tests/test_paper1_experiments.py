# pyright: basic, reportMissingImports=false

import csv
import json
from pathlib import Path

import pytest
import torch

from pinn4csi.data import (
    create_mock_paper1_prepared_data,
    create_paper1_splits,
    get_paper1_dataset_config,
    load_prepared_paper1_dataset,
)
from pinn4csi.models import (
    Paper1ModelSpec,
    create_paper1_model,
    expand_paper1_model_specs,
)
from pinn4csi.training import (
    Paper1ExperimentConfig,
    Paper1ResultRow,
    analyze_paper1_results,
    build_paper1_summary_rows,
    render_paper1_summary_latex,
    run_paper1_experiments,
    save_paper1_analysis_json,
    save_paper1_results_csv,
    save_paper1_summary_json,
    save_paper1_summary_latex,
    summarize_paper1_results,
)


@pytest.fixture
def prepared_root(tmp_path: Path) -> Path:
    root = tmp_path / "prepared"
    create_mock_paper1_prepared_data(
        prepared_root=root,
        samples_per_class=12,
        num_subcarriers=12,
        num_classes=3,
    )
    return root


def test_dataset_registry_supports_signfi_and_ut_har(prepared_root: Path) -> None:
    signfi = load_prepared_paper1_dataset("signfi", prepared_root)
    ut_har = load_prepared_paper1_dataset("ut_har", prepared_root)

    assert signfi.config.name == "SignFi"
    assert ut_har.config.name == "UT_HAR"
    assert signfi.num_classes == 3
    assert ut_har.environments is not None
    assert set(torch.unique(ut_har.environments).tolist()) == {0, 1}
    assert get_paper1_dataset_config("ut_har").evaluation_mode == "cross_environment"


def test_create_paper1_splits_respects_cross_environment(prepared_root: Path) -> None:
    bundle = load_prepared_paper1_dataset("ut_har", prepared_root)
    splits = create_paper1_splits(bundle, seed=3)

    train_envs = torch.unique(splits.train.bundle.environments)
    test_envs = torch.unique(splits.test.bundle.environments)
    assert set(train_envs.tolist()) == {0}
    assert set(test_envs.tolist()) == {1}
    assert len(splits.train) > 0
    assert len(splits.val) > 0
    assert len(splits.test) > 0
    assert splits.train_environment_ids == (0,)
    assert splits.test_environment_ids == (1,)


def test_model_factory_exposes_multiple_baselines(prepared_root: Path) -> None:
    bundle = load_prepared_paper1_dataset("signfi", prepared_root)
    batch_x = bundle.features[:4]
    batch_labels = bundle.labels[:4]
    batch_prior = bundle.priors[:4] if bundle.priors is not None else None
    has_prior = torch.ones(4, dtype=torch.bool)

    for model_name in (
        "mlp",
        "cnn",
        "dgsense_lite",
        "dgsense_physics",
        "autoencoder",
        "residual_prior",
    ):
        model = create_paper1_model(
            model_name=model_name,
            input_shape=bundle.input_shape,
            num_classes=bundle.num_classes,
        )
        losses = model.compute_batch_losses(
            x=batch_x,
            labels=batch_labels,
            prior=batch_prior,
            has_prior=has_prior,
        )
        assert losses["logits"].shape == (4, bundle.num_classes)
        assert torch.isfinite(losses["loss_total"])
        assert torch.isfinite(losses["loss_task"])
        assert torch.isfinite(losses["loss_reconstruction"])
        assert torch.isfinite(losses["loss_ofdm"])
        assert torch.isfinite(losses["loss_path"])


def test_autoencoder_ablation_toggles_control_physics_losses(
    prepared_root: Path,
) -> None:
    torch.manual_seed(0)
    bundle = load_prepared_paper1_dataset("signfi", prepared_root)
    batch_x = bundle.features[:8]
    batch_labels = bundle.labels[:8]
    batch_prior = bundle.priors[:8] if bundle.priors is not None else None
    has_prior = torch.ones(8, dtype=torch.bool)

    default_model = create_paper1_model(
        model_name="autoencoder",
        input_shape=bundle.input_shape,
        num_classes=bundle.num_classes,
    )
    default_losses = default_model.compute_batch_losses(
        x=batch_x,
        labels=batch_labels,
        prior=batch_prior,
        has_prior=has_prior,
    )
    assert float(default_losses["loss_ofdm"].detach().item()) == 0.0
    assert float(default_losses["loss_path"].detach().item()) == 0.0

    ofdm_off_model = create_paper1_model(
        model_name=Paper1ModelSpec(model_name="autoencoder", use_ofdm_loss=False),
        input_shape=bundle.input_shape,
        num_classes=bundle.num_classes,
    )
    ofdm_off_losses = ofdm_off_model.compute_batch_losses(
        x=batch_x,
        labels=batch_labels,
        prior=batch_prior,
        has_prior=has_prior,
    )
    assert float(ofdm_off_losses["loss_ofdm"].detach().item()) == 0.0

    path_off_model = create_paper1_model(
        model_name=Paper1ModelSpec(model_name="autoencoder", use_path_loss=False),
        input_shape=bundle.input_shape,
        num_classes=bundle.num_classes,
    )
    path_off_losses = path_off_model.compute_batch_losses(
        x=batch_x,
        labels=batch_labels,
        prior=batch_prior,
        has_prior=has_prior,
    )
    assert float(path_off_losses["loss_path"].detach().item()) == 0.0

    no_physics_losses = default_model.compute_batch_losses(
        x=batch_x,
        labels=batch_labels,
        prior=None,
        has_prior=torch.zeros_like(has_prior),
    )
    assert float(no_physics_losses["loss_ofdm"].detach().item()) == 0.0
    assert float(no_physics_losses["loss_path"].detach().item()) == 0.0


def test_autoencoder_default_uses_adaptive_physics_weights(prepared_root: Path) -> None:
    torch.manual_seed(0)
    bundle = load_prepared_paper1_dataset("signfi", prepared_root)
    batch_x = bundle.features[:8]
    batch_labels = bundle.labels[:8]
    batch_prior = bundle.priors[:8] if bundle.priors is not None else None
    has_prior = torch.ones(8, dtype=torch.bool)

    default_model = create_paper1_model(
        model_name="autoencoder",
        input_shape=bundle.input_shape,
        num_classes=bundle.num_classes,
    )
    fixed_weight_model = create_paper1_model(
        model_name=Paper1ModelSpec(
            model_name="autoencoder",
            variant_name="fixed_weight",
            use_adaptive_physics_weighting=False,
        ),
        input_shape=bundle.input_shape,
        num_classes=bundle.num_classes,
    )

    default_losses = default_model.compute_batch_losses(
        x=batch_x,
        labels=batch_labels,
        prior=batch_prior,
        has_prior=has_prior,
    )
    fixed_losses = fixed_weight_model.compute_batch_losses(
        x=batch_x,
        labels=batch_labels,
        prior=batch_prior,
        has_prior=has_prior,
    )

    assert default_model.physics_weighting_mode == "adaptive"
    assert default_model.uses_adaptive_physics_weighting is True
    assert fixed_weight_model.physics_weighting_mode == "fixed"
    assert fixed_weight_model.uses_adaptive_physics_weighting is False
    assert default_model.current_ofdm_weight == pytest.approx(default_model.ofdm_weight)
    assert default_model.current_path_weight == pytest.approx(default_model.path_weight)
    assert fixed_weight_model.current_ofdm_weight == pytest.approx(
        fixed_weight_model.ofdm_weight
    )
    assert fixed_weight_model.current_path_weight == pytest.approx(
        fixed_weight_model.path_weight
    )
    assert torch.isfinite(default_losses["loss_total"])
    assert torch.isfinite(fixed_losses["loss_total"])


def test_paper1_harness_runs_end_to_end_and_writes_csv(
    prepared_root: Path, tmp_path: Path
) -> None:
    output_csv = tmp_path / "paper1_results.csv"
    analysis_json = tmp_path / "paper1_analysis.json"
    summary_json = tmp_path / "paper1_summary.json"
    summary_latex = tmp_path / "paper1_summary.tex"
    embeddings_dir = tmp_path / "embeddings"
    config = Paper1ExperimentConfig(
        prepared_root=prepared_root,
        dataset_names=("signfi", "ut_har"),
        model_names=(
            "mlp",
            "cnn",
            "dgsense_lite",
            "dgsense_physics",
            "autoencoder",
            "residual_prior",
        ),
        seeds=(0,),
        epochs=4,
        batch_size=16,
        hidden_dim=24,
        num_layers=2,
        latent_dim=12,
        output_csv=output_csv,
        analysis_json=analysis_json,
        summary_json=summary_json,
        summary_latex=summary_latex,
        embeddings_dir=embeddings_dir,
    )

    results = run_paper1_experiments(config)
    save_paper1_results_csv(results, output_csv)
    analysis = analyze_paper1_results(results, split="test")
    save_paper1_analysis_json(analysis, analysis_json)
    summary_rows = build_paper1_summary_rows(results, split="test")
    save_paper1_summary_json(summary_rows, summary_json)
    save_paper1_summary_latex(summary_rows, summary_latex)
    summary = summarize_paper1_results(results, split="test")

    model_variant_count = len(
        expand_paper1_model_specs(
            config.model_names,
            include_loss_ablation_variants=config.include_loss_ablation_variants,
            include_fixed_weight_variant=config.include_fixed_weight_variant,
            include_fourier_variant=config.include_fourier_variant,
        )
    )
    expected_rows = model_variant_count * 2 + model_variant_count * 2 * 2
    assert len(results) == expected_rows
    assert output_csv.exists()
    assert analysis_json.exists()
    assert summary_json.exists()
    assert summary_latex.exists()
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == expected_rows
    assert set(rows[0]) == {
        "dataset_name",
        "model_name",
        "variant_name",
        "comparison_name",
        "seed",
        "eval_mode",
        "split",
        "train_environment_ids",
        "test_environment_ids",
        "uses_reconstruction_loss",
        "uses_ofdm_loss",
        "uses_path_loss",
        "uses_fourier_features",
        "uses_adaptive_physics_weighting",
        "physics_weighting_mode",
        "requires_prior",
        "accuracy",
        "macro_f1",
        "loss_total",
        "loss_task",
        "loss_reconstruction",
        "loss_ofdm",
        "loss_path",
        "best_epoch",
        "num_examples",
    }
    assert "SignFi:mlp:default:train_val_test" in summary
    assert "SignFi:dgsense_lite:default:train_val_test" in summary
    assert "SignFi:dgsense_physics:default:train_val_test" in summary
    assert "UT_HAR:residual_prior:default:cross_environment" in summary
    assert "UT_HAR:autoencoder:fixed_weight:train_val_test" in summary
    assert "UT_HAR:autoencoder:fourier_on:train_val_test" in summary
    assert "UT_HAR:autoencoder:reconstruction_off:train_val_test" in summary
    assert "UT_HAR:autoencoder:ofdm_off:train_val_test" in summary
    assert "UT_HAR:autoencoder:path_off:train_val_test" in summary
    assert analysis["per_model_summary"]
    assert analysis["per_dataset_summary"]
    assert analysis["summary_rows"]
    assert analysis["paired_variant_comparisons"]
    assert analysis["ut_har_cross_environment"]
    saved_summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert saved_summary
    assert "tabular" in summary_latex.read_text(encoding="utf-8")
    assert any(
        item["comparison_name"] == "residual_prior:reconstruction_off"
        for item in analysis["ut_har_cross_environment"]
    )
    assert any(path.suffix == ".pt" for path in embeddings_dir.iterdir())
    for row in results:
        assert row.split in {"val", "test"}
        assert 0.0 <= row.accuracy <= 1.0
        assert 0.0 <= row.macro_f1 <= 1.0
        assert row.best_epoch >= 0
        assert row.num_examples > 0
        if row.variant_name == "reconstruction_off":
            assert row.uses_reconstruction_loss is False
        if row.variant_name == "fixed_weight":
            assert row.physics_weighting_mode == "fixed"
            assert row.uses_adaptive_physics_weighting is False
        if row.variant_name == "fourier_on":
            assert row.uses_fourier_features is True
        if row.variant_name == "ofdm_off":
            assert row.uses_ofdm_loss is False
        if row.variant_name == "path_off":
            assert row.uses_path_loss is False


def test_analysis_surfaces_ut_har_gap_and_dataset_best(prepared_root: Path) -> None:
    config = Paper1ExperimentConfig(
        prepared_root=prepared_root,
        dataset_names=("ut_har",),
        model_names=("mlp", "autoencoder", "residual_prior"),
        seeds=(0, 1),
        epochs=3,
        batch_size=16,
        hidden_dim=20,
        num_layers=2,
        latent_dim=10,
    )

    analysis = analyze_paper1_results(run_paper1_experiments(config), split="test")

    assert {item["eval_mode"] for item in analysis["per_dataset_summary"]} == {
        "train_val_test",
        "cross_environment",
    }
    assert all("accuracy_gap" in item for item in analysis["ut_har_cross_environment"])
    assert all(
        item["num_model_variants"] >= 4 for item in analysis["per_dataset_summary"]
    )


def test_summary_rows_capture_seed_stats_and_latex_export(tmp_path: Path) -> None:
    results = [
        Paper1ResultRow(
            dataset_name="SignFi",
            model_name="autoencoder",
            variant_name="default",
            comparison_name="autoencoder:default",
            seed=0,
            eval_mode="train_val_test",
            split="test",
            train_environment_ids="all",
            test_environment_ids="all",
            uses_reconstruction_loss=True,
            uses_ofdm_loss=True,
            uses_path_loss=True,
            uses_fourier_features=False,
            uses_adaptive_physics_weighting=True,
            physics_weighting_mode="adaptive",
            requires_prior=False,
            accuracy=0.80,
            macro_f1=0.78,
            loss_total=0.90,
            loss_task=0.60,
            loss_reconstruction=0.20,
            loss_ofdm=0.06,
            loss_path=0.04,
            best_epoch=1,
            num_examples=12,
        ),
        Paper1ResultRow(
            dataset_name="SignFi",
            model_name="autoencoder",
            variant_name="default",
            comparison_name="autoencoder:default",
            seed=1,
            eval_mode="train_val_test",
            split="test",
            train_environment_ids="all",
            test_environment_ids="all",
            uses_reconstruction_loss=True,
            uses_ofdm_loss=True,
            uses_path_loss=True,
            uses_fourier_features=False,
            uses_adaptive_physics_weighting=True,
            physics_weighting_mode="adaptive",
            requires_prior=False,
            accuracy=0.82,
            macro_f1=0.80,
            loss_total=0.86,
            loss_task=0.58,
            loss_reconstruction=0.18,
            loss_ofdm=0.06,
            loss_path=0.04,
            best_epoch=1,
            num_examples=12,
        ),
        Paper1ResultRow(
            dataset_name="SignFi",
            model_name="autoencoder",
            variant_name="fixed_weight",
            comparison_name="autoencoder:fixed_weight",
            seed=0,
            eval_mode="train_val_test",
            split="test",
            train_environment_ids="all",
            test_environment_ids="all",
            uses_reconstruction_loss=True,
            uses_ofdm_loss=True,
            uses_path_loss=True,
            uses_fourier_features=False,
            uses_adaptive_physics_weighting=False,
            physics_weighting_mode="fixed",
            requires_prior=False,
            accuracy=0.77,
            macro_f1=0.74,
            loss_total=0.95,
            loss_task=0.64,
            loss_reconstruction=0.21,
            loss_ofdm=0.06,
            loss_path=0.04,
            best_epoch=1,
            num_examples=12,
        ),
        Paper1ResultRow(
            dataset_name="SignFi",
            model_name="autoencoder",
            variant_name="fixed_weight",
            comparison_name="autoencoder:fixed_weight",
            seed=1,
            eval_mode="train_val_test",
            split="test",
            train_environment_ids="all",
            test_environment_ids="all",
            uses_reconstruction_loss=True,
            uses_ofdm_loss=True,
            uses_path_loss=True,
            uses_fourier_features=False,
            uses_adaptive_physics_weighting=False,
            physics_weighting_mode="fixed",
            requires_prior=False,
            accuracy=0.78,
            macro_f1=0.76,
            loss_total=0.92,
            loss_task=0.62,
            loss_reconstruction=0.20,
            loss_ofdm=0.06,
            loss_path=0.04,
            best_epoch=1,
            num_examples=12,
        ),
    ]

    summary_rows = build_paper1_summary_rows(results, split="test")

    assert len(summary_rows) == 2
    fixed_row = next(row for row in summary_rows if row.variant_name == "fixed_weight")
    assert fixed_row.reference_comparison_name == "autoencoder:default"
    assert fixed_row.paired_seed_count == 2
    assert fixed_row.paired_accuracy_mean_delta == pytest.approx(-0.035)
    assert fixed_row.paired_accuracy_wins == 0
    assert fixed_row.paired_accuracy_losses == 2
    assert fixed_row.paired_accuracy_sign_test_pvalue == pytest.approx(0.5)

    summary_json = tmp_path / "summary.json"
    summary_latex = tmp_path / "summary.tex"
    save_paper1_summary_json(summary_rows, summary_json)
    save_paper1_summary_latex(summary_rows, summary_latex)

    saved_summary = json.loads(summary_json.read_text(encoding="utf-8"))
    latex = render_paper1_summary_latex(summary_rows)
    assert saved_summary[1]["paired_accuracy_mean_delta"] == pytest.approx(-0.035)
    assert "SignFi" in latex
    assert "autoencoder" in latex
    assert "0.5000" in latex
    assert summary_latex.read_text(encoding="utf-8") == latex
