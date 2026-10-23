"""Collapse-guard floor chosen leave-one-dataset-out, on harmful and helpful cells.

For every dataset the script runs three cells over one grid of floors (tau):
entropy minimisation and IM on the primary MLP, and BN-affine TENT on MLP+BN. Each
cell has its unguarded method and the guarded variant (``collapse_guarded_*``).

Selecting tau by the mean guarded gain of a method that never helps makes
abstention win by construction. Instead tau is chosen by *regret*: for every cell,
guarded gain minus the better of "adapt" (the unguarded gain) and "do not adapt"
(0). Losing a gain costs as much as causing harm. For each held-out dataset the
floor with the least mean regret over the other two datasets' cells is chosen and
the held-out cells are reported at that floor, with abstention counts.

Output: ``outputs/guard_floor_lodo/guard_floor_lodo.json``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATASETS = ("widar_bvp", "ntufi_har", "signfi_top10")
FLOORS = (0.8, 0.9, 0.95, 0.98, 1.0)
SEEDS = (0, 1, 2, 3, 4)
# (backbone, base method): cells evaluated on every dataset.
CELLS = (("mlp", "entropy_tta"), ("mlp", "shot"), ("mlp_bn", "tent"))
OUTPUT = Path("outputs/guard_floor_lodo/guard_floor_lodo.json")


def _runner():  # type: ignore[no-untyped-def]
    path = Path(__file__).with_name("run_extended_tta_per_dataset.py")
    spec = importlib.util.spec_from_file_location("run_extended_tta", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load run_extended_tta_per_dataset.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _mean_gain_pp(rows: list[dict[str, object]], method: str) -> float:
    gains = [float(r["gain"]) for r in rows if r["method"] == method]  # type: ignore[arg-type]
    return 100.0 * sum(gains) / len(gains)


def _abstained(rows: list[dict[str, object]], method: str) -> int:
    return sum(
        int(r["accepted_steps"]) == 0  # type: ignore[call-overload]
        for r in rows
        if r["method"] == method
    )


def cell_key(backbone: str, method: str) -> str:
    return f"{backbone}/{method}"


def run_grid() -> dict[str, dict[str, dict[str, float]]]:
    """grid[dataset][cell] = {"unguarded", "<tau>", "<tau>_abstained", ...}."""
    runner = _runner()
    grid: dict[str, dict[str, dict[str, float]]] = {}
    for dataset in DATASETS:
        grid[dataset] = {}
        for backbone in sorted({b for b, _ in CELLS}):
            bases = [m for b, m in CELLS if b == backbone]
            for floor in FLOORS:
                methods = ["no_adapt"]
                for base in bases:
                    if floor == FLOORS[0]:
                        methods.append(base)
                    methods.append(f"collapse_guarded_{base}")
                result = runner.run_experiment(
                    dataset=dataset,
                    methods=tuple(methods),
                    seeds=SEEDS,
                    source_epochs=10,
                    adaptation_steps=5,
                    adaptation_learning_rate=5e-4,
                    batch_size=32,
                    backbone=backbone,
                    collapse_guard_diversity_ratio=floor,
                )
                rows = result["rows"]
                for base in bases:
                    cell = grid[dataset].setdefault(cell_key(backbone, base), {})
                    guarded = f"collapse_guarded_{base}"
                    if floor == FLOORS[0]:
                        cell["unguarded"] = _mean_gain_pp(rows, base)
                    cell[f"{floor}"] = _mean_gain_pp(rows, guarded)
                    cell[f"{floor}_abstained"] = float(_abstained(rows, guarded))
                    logger.info(
                        "%s %s tau=%.2f guarded %+.2f pp",
                        dataset,
                        cell_key(backbone, base),
                        floor,
                        cell[f"{floor}"],
                    )
    return grid


def regret(cell: dict[str, float], floor: float) -> float:
    """Guarded gain minus the better of adapting and not adapting (pp, 0 is ideal)."""
    return cell[f"{floor}"] - max(cell["unguarded"], 0.0)


def lodo(
    grid: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for held_out in DATASETS:
        others = [d for d in DATASETS if d != held_out]
        cells = [grid[d][c] for d in others for c in grid[d]]
        chosen = max(
            FLOORS, key=lambda f: sum(regret(c, f) for c in cells) / len(cells)
        )
        out[held_out] = {
            "chosen_floor": chosen,
            "cells": {
                name: {
                    "unguarded_pp": cell["unguarded"],
                    "guarded_pp": cell[f"{chosen}"],
                    "abstained_runs": cell[f"{chosen}_abstained"],
                    "regret_pp": regret(cell, chosen),
                }
                for name, cell in grid[held_out].items()
            },
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    grid = run_grid()
    result = {
        "protocol": "3 folds x 5 seeds per cell; cells: MLP entropy_tta, MLP IM "
        "(shot), MLP+BN TENT; tau chosen on the other two datasets by least mean "
        "regret (guarded gain minus max(unguarded gain, 0))",
        "floors": list(FLOORS),
        "runs_per_cell": len(SEEDS) * 3,
        "grid": grid,
        "leave_one_dataset_out": lodo(grid),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
