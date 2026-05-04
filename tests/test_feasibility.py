# pyright: basic, reportMissingImports=false

import csv
import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def feasibility_module() -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "feasibility.py"
    spec = importlib.util.spec_from_file_location("feasibility", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load feasibility module spec.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_feasibility_mini_experiment_runs_and_logs(
    tmp_path: Path,
    feasibility_module: ModuleType,
) -> None:
    output_csv = tmp_path / "feasibility_results.csv"
    config = feasibility_module.FeasibilityConfig(
        seeds=1,
        lambdas=(0.1,),
        epochs=6,
        batch_size=32,
        train_samples=96,
        val_samples=32,
        test_samples=96,
        num_subcarriers=16,
        num_paths=3,
        hidden_dim=32,
        num_layers=2,
        output_csv=output_csv,
    )

    results = feasibility_module.run_feasibility(config)
    feasibility_module.save_results_csv(results, output_csv)

    assert len(results) == 3
    baseline = [row for row in results if row.model_type == "baseline"][0]
    pinn = [row for row in results if row.model_type == "ofdm_pinn"][0]
    residual = [row for row in results if row.model_type == "ofdm_residual"][0]

    assert output_csv.exists()
    with output_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    assert len(rows) == 3
    assert set(rows[0]) >= {
        "seed",
        "model_type",
        "initial_lambda",
        "best_epoch",
        "val_rmse",
        "test_rmse",
        "val_nmse",
        "test_nmse",
        "final_lambda",
    }

    assert pinn.final_lambda > 0.0
    assert math.isfinite(pinn.final_lambda)
    assert residual.final_lambda == 0.0
    assert 0 <= residual.best_epoch < config.epochs
    assert 0 <= pinn.best_epoch < config.epochs
    assert not math.isclose(
        baseline.val_rmse,
        pinn.val_rmse,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def test_summarize_results_uses_validation_selection(
    feasibility_module: ModuleType,
) -> None:
    results = [
        feasibility_module.ExperimentResult(
            seed=0,
            model_type="baseline",
            initial_lambda=0.0,
            best_epoch=2,
            val_rmse=1.0,
            test_rmse=4.0,
            val_nmse=0.1,
            test_nmse=0.2,
            final_lambda=0.0,
        ),
        feasibility_module.ExperimentResult(
            seed=0,
            model_type="ofdm_residual",
            initial_lambda=0.0,
            best_epoch=0,
            val_rmse=0.85,
            test_rmse=3.0,
            val_nmse=0.085,
            test_nmse=0.3,
            final_lambda=0.0,
        ),
        feasibility_module.ExperimentResult(
            seed=0,
            model_type="ofdm_pinn",
            initial_lambda=0.1,
            best_epoch=1,
            val_rmse=0.8,
            test_rmse=9.0,
            val_nmse=0.08,
            test_nmse=0.9,
            final_lambda=0.2,
        ),
        feasibility_module.ExperimentResult(
            seed=0,
            model_type="ofdm_pinn",
            initial_lambda=1.0,
            best_epoch=3,
            val_rmse=0.9,
            test_rmse=1.0,
            val_nmse=0.09,
            test_nmse=0.1,
            final_lambda=1.3,
        ),
    ]

    summary = feasibility_module.summarize_results(results)
    assert summary["baseline_mean_test_nmse"] == 0.2
    assert summary["best_ofdm_pinn_mean_test_nmse"] == 0.9
    assert summary["best_ofdm_residual_mean_test_nmse"] == 0.3
    assert summary["delta_ofdm_pinn_test_nmse"] == 0.7
    assert math.isclose(summary["delta_ofdm_residual_test_nmse"], 0.1)
    assert summary["baseline_mean_test_rmse"] == 4.0
    assert summary["best_ofdm_pinn_mean_test_rmse"] == 9.0
    assert summary["best_ofdm_residual_mean_test_rmse"] == 3.0
    assert summary["delta_ofdm_pinn_test_rmse"] == 5.0
    assert math.isclose(summary["delta_ofdm_residual_test_rmse"], -1.0)


def test_parse_lambdas_reuses_shared_csv_parser(
    feasibility_module: ModuleType,
) -> None:
    assert feasibility_module.parse_lambdas("0.1, 1.0,2.5") == (0.1, 1.0, 2.5)
    with pytest.raises(ValueError, match="comma-separated"):
        feasibility_module.parse_lambdas("  ")
