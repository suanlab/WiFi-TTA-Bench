from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_guard_floor_lodo.py"
    spec = importlib.util.spec_from_file_location("run_guard_floor_lodo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load run_guard_floor_lodo.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _cell(m, unguarded: float, by_floor: dict[float, float]) -> dict[str, float]:
    cell = {"unguarded": unguarded}
    for f in m.FLOORS:
        cell[f"{f}"] = by_floor.get(f, unguarded)
        cell[f"{f}_abstained"] = 0.0
    return cell


def test_regret_penalises_lost_gains_and_harm() -> None:
    m = _load()
    harmful = _cell(m, -10.0, {1.0: 0.0})
    helpful = _cell(m, +8.0, {1.0: 0.0})
    assert m.regret(harmful, 1.0) == 0.0  # abstaining removes all harm
    assert m.regret(helpful, 1.0) == -8.0  # abstaining loses the whole gain
    assert m.regret(helpful, 0.8) == 0.0  # keeping the gain costs nothing


def test_floor_is_chosen_on_the_other_two_datasets() -> None:
    m = _load()
    # On NTU-Fi and SignFi a floor of 0.95 keeps gains and removes most harm,
    # while 1.0 abstains and loses the gains; Widar alone would favour 1.0.
    good = {0.95: -1.0, 1.0: 0.0}
    grid = {
        "widar_bvp": {"a": _cell(m, -5.0, {0.95: -4.0, 1.0: 0.0})},
        "ntufi_har": {
            "a": _cell(m, -6.0, good),
            "b": _cell(m, +4.0, {0.95: +4.0, 1.0: 0.0}),
        },
        "signfi_top10": {
            "a": _cell(m, -20.0, good),
            "b": _cell(m, +7.0, {0.95: +7.0, 1.0: 0.0}),
        },
    }
    out = m.lodo(grid)
    assert out["widar_bvp"]["chosen_floor"] == 0.95
    assert out["widar_bvp"]["cells"]["a"]["guarded_pp"] == -4.0
