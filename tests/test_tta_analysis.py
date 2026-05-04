# pyright: basic, reportMissingImports=false

"""Smoke tests for TTA analysis script."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest


@pytest.fixture
def analysis_module() -> ModuleType:
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "analyze_tta_adaptation.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_tta_adaptation", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load analyze_tta_adaptation module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_full_analysis_runs(tmp_path: Path, analysis_module: ModuleType) -> None:
    """Full analysis pipeline runs and produces output files."""
    summary = analysis_module.run_full_analysis(tmp_path)
    assert "harm_analysis" in summary
    assert "shift_gain_curve" in summary
    assert (tmp_path / "analysis_summary.json").exists()
    assert (tmp_path / "harm_analysis.json").exists()
    assert (tmp_path / "shift_gain_curve.json").exists()
    assert (tmp_path / "tsne_embeddings.json").exists()

    harm = summary["harm_analysis"]
    assert harm["total_samples"] > 0
    assert harm["negative_adaptation_rate"] >= 0.0
    assert harm["help_rate"] >= 0.0
