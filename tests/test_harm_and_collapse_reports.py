from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch


def _load_script(name: str):
    script_path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {name} module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_harm_tolerance_discounts_one_sample_flips_and_counts_abstentions() -> None:
    module = _load_script("harm_tolerance_report")
    sizes = {0: 10, 1: 10}  # one evaluation sample = 0.1
    rows = [
        # environment 0: a one-sample loss, and an abstaining run
        {"split": "env0", "gain": -0.1, "accepted_steps": 2},
        {"split": "env0", "gain": 0.0, "accepted_steps": 0},
        # environment 1: a two-sample loss and a gain
        {"split": "env1", "gain": -0.2, "accepted_steps": 5},
        {"split": "env1", "gain": 0.3, "accepted_steps": 5},
    ]
    out = module._summarise(rows, sizes)
    assert out["harm_rate"] == 0.5
    assert out["harm_rate_gt1_sample"] == 0.25
    assert out["abstain_rate"] == 0.25
    assert math.isclose(out["env_mean_pp"]["0"], -5.0)
    assert math.isclose(out["env_mean_pp"]["1"], 5.0)
    assert out["envs_negative"] == 1


def test_harm_tolerance_reads_widar_room_names() -> None:
    module = _load_script("harm_tolerance_report")
    assert module._env_of({"room": "room1_hall"}) == 1
    assert module._env_of({"split": "env2"}) == 2
    assert module._env_of({"test_env": 0}) == 0


def test_collapse_summary_measures_class_usage() -> None:
    module = _load_script("analyze_collapse_trajectories")
    labels = torch.tensor([0, 1, 2, 0, 1, 2])
    uniform = module._summarise(labels.clone(), labels, num_classes=3)
    assert uniform["accuracy"] == 1.0
    assert math.isclose(uniform["diversity"], 1.0)
    assert uniform["classes_used"] == 3

    collapsed = module._summarise(torch.zeros(6, dtype=torch.long), labels, 3)
    assert math.isclose(collapsed["diversity"], 0.0, abs_tol=1e-12)
    assert collapsed["classes_used"] == 1
    assert collapsed["predicted_histogram"] == [6, 0, 0]
    assert collapsed["confusion"][1] == [2, 0, 0]
