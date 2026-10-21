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


def test_floor_is_chosen_on_the_other_two_datasets() -> None:
    m = _load()
    floors = {f"{f}": 0.0 for f in m.FLOORS}
    grid = {
        "widar_bvp": {**floors, "unguarded": -10.0, "0.9": -2.0, "1.0": -5.0},
        "ntufi_har": {**floors, "unguarded": -8.0, "0.9": -4.0, "1.0": -1.0},
        "signfi_top10": {**floors, "unguarded": -6.0, "0.9": -3.0, "1.0": -2.0},
    }
    for cell in grid.values():
        for f in ("0.8", "0.95", "0.98"):
            cell[f] = -9.0
    out = m.lodo(grid)
    # Held-out Widar: NTU-Fi and SignFi favour 1.0 (mean -1.5 vs -3.5).
    assert out["widar_bvp"]["chosen_floor"] == 1.0
    assert out["widar_bvp"]["in_sample_best_floor"] == 0.9
    assert out["widar_bvp"]["guarded_pp"] == -5.0
    assert out["widar_bvp"]["harm_removed_pct"] == 50.0
