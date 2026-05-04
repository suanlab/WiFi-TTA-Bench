from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path


def _load_analysis_module():
    script_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "analyze_tta_results.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_tta_results", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load analyze_tta_results module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_analyze_comparison_tracks_harm_and_shift_pairing() -> None:
    module = _load_analysis_module()
    rows = [
        {
            "dataset_name": "synthetic_tta",
            "shift_name": "mild",
            "held_out_environment_id": "1",
            "seed": "0",
            "method": "safe_physics_tta",
            "gain": "0.10",
            "source_drop": "-0.01",
        },
        {
            "dataset_name": "synthetic_tta",
            "shift_name": "mild",
            "held_out_environment_id": "1",
            "seed": "0",
            "method": "physics_tta",
            "gain": "-0.20",
            "source_drop": "-0.30",
        },
        {
            "dataset_name": "synthetic_tta",
            "shift_name": "strong",
            "held_out_environment_id": "1",
            "seed": "0",
            "method": "safe_physics_tta",
            "gain": "-0.05",
            "source_drop": "-0.02",
        },
        {
            "dataset_name": "synthetic_tta",
            "shift_name": "strong",
            "held_out_environment_id": "1",
            "seed": "0",
            "method": "physics_tta",
            "gain": "-0.40",
            "source_drop": "-0.50",
        },
    ]

    comparison = module.analyze_comparison(
        rows=rows,
        scope="pooled",
        compare_method="safe_physics_tta",
        reference_method="physics_tta",
    )

    assert comparison.total == 2
    assert comparison.wins == 2
    assert comparison.compare_harm_rate == 0.5
    assert comparison.reference_harm_rate == 1.0
    assert math.isclose(comparison.compare_negative_tail_severity, 0.05)
    assert math.isclose(comparison.reference_negative_tail_severity, 0.3)
    assert math.isclose(comparison.compare_source_drop_mean, -0.015)
    assert math.isclose(comparison.reference_source_drop_mean, -0.4)


def test_scope_key_includes_shift_for_synthetic_rows() -> None:
    module = _load_analysis_module()

    synthetic_scope = module.scope_key(
        {
            "dataset_name": "synthetic_tta",
            "shift_name": "moderate",
            "held_out_environment_id": "1",
        }
    )
    prepared_scope = module.scope_key(
        {
            "dataset_name": "widar_bvp",
            "shift_name": "held_out_split",
            "held_out_environment_id": "2",
        }
    )

    assert synthetic_scope == "synthetic_tta:moderate:room1"
    assert prepared_scope == "widar_bvp:room2"
