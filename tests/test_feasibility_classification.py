# pyright: basic, reportMissingImports=false

"""Tests for classification feasibility gate."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def feasibility_cls_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "feasibility_classification.py"
    )
    spec = importlib.util.spec_from_file_location(
        "feasibility_classification", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load feasibility_classification module spec.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smoke_run_produces_results(
    tmp_path: Path,
    feasibility_cls_module: ModuleType,
) -> None:
    """Mini config runs end-to-end and produces results."""
    config = feasibility_cls_module.ClassificationFeasibilityConfig(
        seeds=1,
        num_classes=3,
        samples_per_class=24,
        epochs=2,
        batch_size=16,
        hidden_dim=16,
        num_subcarriers=8,
        num_paths=2,
        output_csv=tmp_path / "results.csv",
    )
    results = feasibility_cls_module.run_classification_feasibility(config)
    assert len(results) == 3  # 1 seed × 3 model types
    for r in results:
        assert 0.0 <= r.test_accuracy <= 1.0
        assert 0.0 <= r.test_macro_f1 <= 1.0


def test_summary_includes_effect_size(
    tmp_path: Path,
    feasibility_cls_module: ModuleType,
) -> None:
    """Summary computes Cohen's d for physics methods vs baseline."""
    config = feasibility_cls_module.ClassificationFeasibilityConfig(
        seeds=3,
        num_classes=3,
        samples_per_class=24,
        epochs=2,
        batch_size=16,
        hidden_dim=16,
        num_subcarriers=8,
        num_paths=2,
        output_csv=tmp_path / "results.csv",
    )
    results = feasibility_cls_module.run_classification_feasibility(config)
    summary = feasibility_cls_module.summarize_classification_feasibility(results)
    assert "baseline" in summary
    assert "physics_features" in summary
    assert "cohens_d_vs_baseline" in summary["physics_features"]
