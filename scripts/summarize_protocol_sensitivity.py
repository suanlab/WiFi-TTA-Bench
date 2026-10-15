"""Summarise the protocol-sensitivity runs into the numbers the paper reports.

Reads ``outputs/protocol_sensitivity/{sweep,stream}.json`` (from
``scripts/run_protocol_sensitivity.py``) and writes
``outputs/protocol_sensitivity/summary.json``:

* sweep, per (dataset, backbone, method): the benchmark default (lr 5e-4, 5 steps);
  the grid point a deployer could choose without target labels (highest mean
  post-adaptation accuracy on the *source* validation split); the oracle grid point
  (highest mean target gain, which needs target labels); and the worst grid point.
* stream, per (dataset, backbone, method, stream): mean gain, harm rates and the
  guard's accepted-step counts.

Intervals are seeded percentile bootstraps over the 15 (fold, seed) runs.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DIR = Path("outputs/protocol_sensitivity")
DEFAULT = (5e-4, 5)
RESAMPLES = 10_000


def _ci(gains: list[float], rng: np.random.Generator) -> dict[str, float]:
    x = 100.0 * np.asarray(gains)
    boot = x[rng.integers(0, len(x), size=(RESAMPLES, len(x)))].mean(axis=1)
    return {
        "mean_pp": float(x.mean()),
        "ci_lower_pp": float(np.percentile(boot, 2.5)),
        "ci_upper_pp": float(np.percentile(boot, 97.5)),
        "harm_rate": float((x < 0).mean()),
        "harm_rate_gt1pp": float((x < -1.0).mean()),
    }


def summarise_sweep(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    grid: dict[tuple, dict[tuple[float, int], list[dict]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        cell = (r["dataset"], r["backbone"], r["method"])
        grid[cell][(r["learning_rate"], r["steps"])].append(r)
    out = []
    for cell, configs in sorted(grid.items()):

        def mean_gain(k: tuple[float, int], cfgs: dict = configs) -> float:
            return float(np.mean([r["gain"] for r in cfgs[k]]))

        def mean_source(k: tuple[float, int], cfgs: dict = configs) -> float:
            return float(np.mean([r["source_val_after"] for r in cfgs[k]]))

        # Ties on source accuracy go to the smaller step budget, then learning rate.
        source_pick = max(configs, key=lambda k: (mean_source(k), -k[1], -k[0]))
        oracle_pick = max(configs, key=mean_gain)
        worst_pick = min(configs, key=mean_gain)
        entry: dict[str, object] = {
            "dataset": cell[0],
            "backbone": cell[1],
            "method": cell[2],
        }
        for name, key in (
            ("default", DEFAULT),
            ("source_selected", source_pick),
            ("oracle", oracle_pick),
            ("worst", worst_pick),
        ):
            entry[name] = {
                "learning_rate": key[0],
                "steps": key[1],
                **_ci([r["gain"] for r in configs[key]], rng),
            }
        out.append(entry)
    return out


def summarise_stream(rows: list[dict], rng: np.random.Generator) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["backbone"], r["method"], r["stream"])].append(r)
    return [
        {
            "dataset": k[0],
            "backbone": k[1],
            "method": k[2],
            "stream": k[3],
            **_ci([r["gain"] for r in v], rng),
            "accepted_steps": sorted(int(r["accepted_steps"]) for r in v),
        }
        for k, v in sorted(groups.items())
    ]


def main() -> None:
    rng = np.random.default_rng(0)
    sweep = json.loads((DIR / "sweep.json").read_text())["rows"]
    stream = json.loads((DIR / "stream.json").read_text())["rows"]
    summary = {
        "default_setting": {"learning_rate": DEFAULT[0], "steps": DEFAULT[1]},
        "sweep": summarise_sweep(sweep, rng),
        "stream": summarise_stream(stream, rng),
    }
    (DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    for e in summary["sweep"]:
        logger.info(
            "%s/%s %s default %+.2f source-picked %+.2f (lr %g, %d) oracle %+.2f",
            e["dataset"],
            e["backbone"],
            e["method"],
            e["default"]["mean_pp"],
            e["source_selected"]["mean_pp"],
            e["source_selected"]["learning_rate"],
            e["source_selected"]["steps"],
            e["oracle"]["mean_pp"],
        )


if __name__ == "__main__":
    main()
