# pyright: basic, reportMissingImports=false

"""Smoke tests for Paper 3 neural operator experiments."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def no_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_neural_operator_experiments.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_neural_operator_experiments", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_neural_operator_experiments module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_neural_operator_smoke(tmp_path: Path, no_module: ModuleType) -> None:
    """Mini config runs end-to-end and produces expected results."""
    config = no_module.NeuralOperatorExperimentConfig(
        seeds=(0,),
        train_samples=32,
        test_samples=16,
        epochs=2,
        batch_size=16,
        hidden_dim=16,
        latent_dim=16,
        num_subcarriers=8,
        num_paths=2,
        output_csv=tmp_path / "results.csv",
    )
    results = no_module.run_neural_operator_experiments(config)
    assert len(results) == 2  # baseline + physics_informed
    for r in results:
        assert r.test_rmse > 0
        assert r.model_type in ("baseline", "physics_informed")
