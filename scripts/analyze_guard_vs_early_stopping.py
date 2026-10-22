"""Is the collapse guard more than early stopping?

The guard rolls back the step that drops marginal diversity below the floor and stops,
so its final model is exactly the unguarded model after its accepted steps (checked
per run below). The question is therefore whether the guard's *choice of when to stop*
beats simpler stopping rules applied to the same runs:

* a fixed budget of k = 1, 2 or 3 steps for every run;
* a random stop with the same distribution of stopping points as the guard
  (the guard's accepted-step counts shuffled across runs of a cell), which matches
  how often and how early it stops but ignores the diversity signal.

Everything is read from ``outputs/collapse_trajectories/collapse_trajectories.json``
(unguarded entropy minimisation re-run for k = 0..5 steps); no model is trained.
Output: ``outputs/statistics/guard_vs_early_stopping.json``.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

INPUT = Path("outputs/collapse_trajectories/collapse_trajectories.json")
OUTPUT = Path("outputs/statistics/guard_vs_early_stopping.json")
PERMUTATIONS = 10_000
SEED = 0


def _rule_stats(gains: np.ndarray) -> dict[str, float]:
    return {
        "mean_gain_pp": 100.0 * float(gains.mean()),
        "harm_rate": float((gains < 0).mean()),
        "harm_rate_gt1pp": float((gains < -0.01).mean()),
    }


def analyse_cell(runs: list[dict[str, object]], rng: np.random.Generator) -> dict:
    # gain[i, k]: run i's accuracy change after k unguarded steps.
    gain = np.array(
        [
            [
                t["accuracy"] - r["entropy_trajectory"][0]["accuracy"]
                for t in r["entropy_trajectory"]
            ]  # type: ignore[index]
            for r in runs
        ]
    )
    accepted = np.array([int(r["guarded"]["accepted_steps"]) for r in runs])  # type: ignore[index]
    guard_gain = gain[np.arange(len(runs)), accepted]
    guarded_acc = np.array([r["guarded"]["accuracy"] for r in runs])  # type: ignore[index]
    source_acc = gain[:, 0] * 0 + np.array(
        [r["entropy_trajectory"][0]["accuracy"] for r in runs]  # type: ignore[index]
    )
    if not np.allclose(guarded_acc - source_acc, guard_gain):
        raise ValueError("Guarded model differs from the unguarded k-step model.")

    out: dict[str, object] = {
        "n": len(runs),
        "guard_accepted_steps": sorted(int(a) for a in accepted),
        "guard": _rule_stats(guard_gain),
        "unguarded_5_steps": _rule_stats(gain[:, -1]),
        # k = 0 is "do not adapt" (gain 0), so the budget rule may also abstain.
        "fixed_budget": {
            str(k): _rule_stats(gain[:, k]) for k in BUDGETS if k < gain.shape[1]
        },
    }
    rows = np.arange(len(runs))
    perm_means = np.array(
        [gain[rows, rng.permutation(accepted)].mean() for _ in range(PERMUTATIONS)]
    )
    out["random_stop_same_distribution"] = {
        "mean_gain_pp": 100.0 * float(perm_means.mean()),
        "p_ge_guard": float((perm_means >= guard_gain.mean() - 1e-12).mean()),
        "permutations": PERMUTATIONS,
    }
    return out


BUDGETS = (0, 1, 2, 3, 5)


def budget_lodo(report: dict[str, dict]) -> dict[str, dict[str, object]]:
    """Choose the step budget k leave-one-dataset-out (best mean gain elsewhere).

    Only the primary-MLP cell of each dataset enters, matching the guard-floor
    selection in ``run_guard_floor_lodo.py``.
    """
    mlp = {c.split("/")[0]: r for c, r in report.items() if c.endswith("/mlp")}
    out: dict[str, dict[str, object]] = {}
    for held_out, cell in mlp.items():
        others = [r for d, r in mlp.items() if d != held_out]
        options = [
            k for k in BUDGETS if all(str(k) in r["fixed_budget"] for r in mlp.values())
        ]
        chosen = max(
            options,
            key=lambda k: float(
                np.mean([r["fixed_budget"][str(k)]["mean_gain_pp"] for r in others])
            ),
        )
        out[held_out] = {
            "chosen_k": chosen,
            "mean_gain_pp": cell["fixed_budget"][str(chosen)]["mean_gain_pp"],
        }
    return out


def main() -> None:
    payload = json.loads(INPUT.read_text())
    cells: dict[str, list[dict[str, object]]] = defaultdict(list)
    for run in payload["runs"]:
        cells[f"{run['dataset']}/{run['backbone']}"].append(run)
    rng = np.random.default_rng(SEED)
    report = {cell: analyse_cell(runs, rng) for cell, runs in sorted(cells.items())}
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "source": str(INPUT),
                "note": "p_ge_guard: share of random stops (same stopping-point "
                "distribution) whose mean gain is at least the guard's",
                "cells": report,
                "fixed_budget_leave_one_dataset_out": budget_lodo(report),
            },
            indent=2,
        )
    )
    for cell, r in report.items():
        logger.info(
            "%s guard %+.2f | k=1 %+.2f k=2 %+.2f | random-stop %+.2f (p=%.3f)",
            cell,
            r["guard"]["mean_gain_pp"],
            r["fixed_budget"]["1"]["mean_gain_pp"],
            r["fixed_budget"]["2"]["mean_gain_pp"],
            r["random_stop_same_distribution"]["mean_gain_pp"],
            r["random_stop_same_distribution"]["p_ge_guard"],
        )


if __name__ == "__main__":
    main()
