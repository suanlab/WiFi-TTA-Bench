# pyright: basic, reportMissingImports=false

"""Smoke tests for Paper 4 WiFi imaging experiments."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def wi_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_wifi_imaging_experiments.py"
    )
    spec = importlib.util.spec_from_file_location(
        "run_wifi_imaging_experiments", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_wifi_imaging_experiments module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_wifi_imaging_smoke(tmp_path: Path, wi_module: ModuleType) -> None:
    """Mini config runs end-to-end and produces expected results."""
    config = wi_module.WiFiImagingExperimentConfig(
        seeds=(0,),
        csi_feature_dim=16,
        grid_resolution=4,
        train_samples=32,
        test_samples=16,
        epochs=2,
        batch_size=16,
        hidden_dim=16,
        latent_dim=16,
        output_csv=tmp_path / "results.csv",
    )
    results = wi_module.run_wifi_imaging_experiments(config)
    assert len(results) == 2  # baseline + helmholtz_pinn
    for r in results:
        assert r.field_rmse > 0
        assert r.permittivity_rmse > 0
        assert r.model_type in ("baseline", "helmholtz_pinn")
