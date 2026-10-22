"""Collapse-guard floor chosen leave-one-dataset-out.

The paper's recommended floor (tau) was chosen by comparing target-labelled
outcomes on the same three datasets it is reported on. This script removes that
in-sample choice: it runs unguarded and guarded entropy minimisation on the
primary MLP for every dataset over one common grid of floors, then, for each
dataset, picks tau on the *other two* datasets (highest mean guarded gain) and
reports the held-out dataset's result at that tau.

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


def run_grid() -> dict[str, dict[str, float]]:
    runner = _runner()
    grid: dict[str, dict[str, float]] = {}
    for dataset in DATASETS:
        cell: dict[str, float] = {}
        for floor in FLOORS:
            methods = ("no_adapt", "collapse_guarded_entropy_tta")
            if floor == FLOORS[0]:
                methods = ("no_adapt", "entropy_tta", "collapse_guarded_entropy_tta")
            result = runner.run_experiment(
                dataset=dataset,
                methods=methods,
                seeds=SEEDS,
                source_epochs=10,
                adaptation_steps=5,
                adaptation_learning_rate=5e-4,
                batch_size=32,
                backbone="mlp",
                collapse_guard_diversity_ratio=floor,
            )
            rows = result["rows"]
            if floor == FLOORS[0]:
                cell["unguarded"] = _mean_gain_pp(rows, "entropy_tta")
            cell[f"{floor}"] = _mean_gain_pp(rows, "collapse_guarded_entropy_tta")
            guarded = [r for r in rows if r["method"] == "collapse_guarded_entropy_tta"]
            # Runs where the guard accepted no step, i.e. declined to adapt.
            cell[f"{floor}_abstained"] = float(
                sum(int(r["accepted_steps"]) == 0 for r in guarded)  # type: ignore[call-overload]
            )
            cell[f"{floor}_runs"] = float(len(guarded))
            logger.info(
                "%s tau=%.2f guarded %+.2f pp", dataset, floor, cell[f"{floor}"]
            )
        grid[dataset] = cell
    return grid


def lodo(grid: dict[str, dict[str, float]]) -> dict[str, dict[str, float | str]]:
    out: dict[str, dict[str, float | str]] = {}
    for held_out in DATASETS:
        others = [d for d in DATASETS if d != held_out]
        chosen = max(
            FLOORS, key=lambda f: sum(grid[d][f"{f}"] for d in others) / len(others)
        )
        unguarded = grid[held_out]["unguarded"]
        guarded = grid[held_out][f"{chosen}"]
        abstained = grid[held_out].get(f"{chosen}_abstained")
        out[held_out] = {
            "chosen_floor": chosen,
            "unguarded_pp": unguarded,
            "guarded_pp": guarded,
            "harm_removed_pct": (
                100.0 * (guarded - unguarded) / -unguarded if unguarded < 0 else "n/a"
            ),
            "in_sample_best_floor": max(FLOORS, key=lambda f: grid[held_out][f"{f}"]),
            "abstained_runs": abstained if abstained is not None else "n/a",
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    grid = run_grid()
    result = {
        "protocol": "primary MLP, entropy_tta vs collapse_guarded_entropy_tta, "
        "3 folds x 5 seeds per dataset; tau chosen on the other two datasets",
        "floors": list(FLOORS),
        "grid_mean_gain_pp": grid,
        "leave_one_dataset_out": lodo(grid),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
