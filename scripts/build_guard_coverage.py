# pyright: basic, reportMissingImports=false

"""Single source of truth for every collapse-guard number in the paper.

Each guarded measurement lives in a different artifact, written by a different
runner, at a different diversity floor. That spread is how two tables came to
report different values for the same cell. This script declares every artifact
explicitly -- dataset, backbone, floor, and file -- pairs each
``collapse_guarded_*`` entry with its base method inside the same run, and emits
one table. Any number quoted in the paper should be traceable to a row here.

A pairing is only reported when both arms come from the *same* artifact, so an
unguarded baseline is never joined across runs.

Outputs:
    outputs/guard_coverage/guard_coverage.json
    outputs/guard_coverage/guard_coverage.csv

Usage:
    python scripts/build_guard_coverage.py
    python scripts/build_guard_coverage.py --floor 0.95
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from pinn4csi.utils import save_dataclass_rows_csv, save_json_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/guard_coverage")

# Harm below this magnitude (pp) is treated as "little to remove", the regime
# where Table tab:guard_backbone claims neither benefit nor damage.
HARM_FLOOR_PP = 3.0
# Harm above this magnitude is the regime where the guard is expected to act.
HARM_CEILING_PP = 6.0


@dataclass(frozen=True)
class Artifact:
    """One guard measurement file.

    Attributes:
        dataset: Prepared dataset name.
        backbone: Encoder backbone, or None when the summary is keyed by backbone.
        floor: Diversity floor tau the run used.
        path: Summary artifact path.
        note: Why this artifact exists, for the emitted table.
    """

    dataset: str
    backbone: str | None
    floor: float
    path: str
    note: str


ARTIFACTS: tuple[Artifact, ...] = (
    # --- tau = 0.9: the generality evidence (four families) ---
    Artifact(
        "widar_bvp",
        "mlp",
        0.9,
        "outputs/final_ablations/collapse_guard_widar.json",
        "guard-signal ablation",
    ),
    Artifact(
        "widar_bvp",
        "mlp",
        0.9,
        "outputs/collapse_guard_generality/widar_sar_cotta.json",
        "generality across families",
    ),
    Artifact(
        "ntufi_har",
        "mlp",
        0.9,
        "outputs/collapse_guard_ntufi/ntufi_har_lame_sar_cotta.json",
        "generality across families",
    ),
    Artifact(
        "signfi_top10",
        "mlp_bn",
        0.9,
        "outputs/collapse_guard_signfi/signfi_top10_lame_sar_cotta.json",
        "gain preservation",
    ),
    Artifact(
        "signfi_top10",
        "mlp_bn",
        0.9,
        "outputs/collapse_guard_signfi_tent/signfi_top10_lame_sar_cotta.json",
        "gain preservation, faithful TENT",
    ),
    # --- tau = 0.95: the recommended default, 3x3 dataset x backbone grid ---
    Artifact(
        "widar_bvp",
        None,
        0.95,
        "outputs/guard_backbone/widar_3backbone.json",
        "backbone grid",
    ),
    Artifact(
        "ntufi_har",
        "mlp",
        0.95,
        "outputs/collapse_guard_ntufi_sweep/r0.95/ntufi_har_lame_sar_cotta.json",
        "backbone grid",
    ),
    Artifact(
        "ntufi_har",
        "mlp_bn",
        0.95,
        "outputs/guard_backbone/ntufi_mlp_bn/ntufi_har_lame_sar_cotta.json",
        "backbone grid",
    ),
    Artifact(
        "ntufi_har",
        "cnn1d",
        0.95,
        "outputs/guard_backbone/ntufi_cnn1d/ntufi_har_lame_sar_cotta.json",
        "backbone grid",
    ),
    Artifact(
        "signfi_top10",
        "mlp",
        0.95,
        "outputs/guard_backbone/signfi_mlp/signfi_top10_lame_sar_cotta.json",
        "backbone grid",
    ),
    Artifact(
        "signfi_top10",
        "mlp_bn",
        0.95,
        "outputs/guard_backbone/signfi_mlp_bn/signfi_top10_lame_sar_cotta.json",
        "backbone grid",
    ),
    Artifact(
        "signfi_top10",
        "cnn1d",
        0.95,
        "outputs/guard_backbone/signfi_cnn1d/signfi_top10_lame_sar_cotta.json",
        "backbone grid",
    ),
    Artifact(
        "signfi_top10",
        "mlp_bn",
        0.95,
        "outputs/collapse_guard_signfi_tight/r0.95/signfi_top10_lame_sar_cotta.json",
        "gain preservation at the default floor",
    ),
)


@dataclass(frozen=True)
class CoverageRow:
    """One (dataset, backbone, floor, method) guarded-vs-unguarded pairing."""

    dataset: str
    backbone: str
    floor: float
    method: str
    unguarded_pp: float
    guarded_pp: float
    harm_removed_pct: float
    unguarded_harm_rate: float
    guarded_harm_rate: float
    regime: str
    verdict: str
    note: str
    source: str


def _summary_blocks(
    payload: dict[str, object], backbone: str | None
) -> list[tuple[str, dict[str, dict[str, float]]]]:
    """Return (backbone, method-summary) blocks, handling backbone-keyed files."""
    summary = payload.get("summary", payload)
    if not isinstance(summary, dict):
        return []
    keyed_by_backbone = any(
        isinstance(value, dict) and "mean_gain" not in value
        for value in summary.values()
    )
    if keyed_by_backbone:
        return [
            (name, block) for name, block in summary.items() if isinstance(block, dict)
        ]
    if backbone is None:
        raise ValueError("Flat summary needs an explicit backbone in the registry.")
    return [(backbone, summary)]  # type: ignore[list-item]


def collect_rows(artifacts: tuple[Artifact, ...]) -> list[CoverageRow]:
    """Pair every guarded method with its base arm from the same artifact."""
    rows: list[CoverageRow] = []
    for artifact in artifacts:
        path = Path(artifact.path)
        if not path.exists():
            logger.warning("Missing artifact: %s", path)
            continue
        payload = json.loads(path.read_text())
        for backbone, methods in _summary_blocks(payload, artifact.backbone):
            for name, entry in methods.items():
                if not name.startswith("collapse_guarded_"):
                    continue
                base = name[len("collapse_guarded_") :]
                baseline = methods.get(base)
                if baseline is None:
                    logger.warning(
                        "%s has %s without its base arm %s", path, name, base
                    )
                    continue
                unguarded = float(baseline["mean_gain"]) * 100.0
                guarded = float(entry["mean_gain"]) * 100.0
                if unguarded < 0.0:
                    removed = (1.0 - guarded / unguarded) * 100.0
                else:
                    removed = float("nan")
                if unguarded <= -HARM_CEILING_PP:
                    regime = "harmful"
                elif unguarded >= -HARM_FLOOR_PP:
                    regime = "little-harm"
                else:
                    regime = "intermediate"
                if abs(guarded - unguarded) < 1e-9:
                    verdict = "inert"
                elif guarded > unguarded:
                    verdict = "improved"
                else:
                    verdict = "worse"
                rows.append(
                    CoverageRow(
                        dataset=artifact.dataset,
                        backbone=backbone,
                        floor=artifact.floor,
                        method=base,
                        unguarded_pp=unguarded,
                        guarded_pp=guarded,
                        harm_removed_pct=removed,
                        unguarded_harm_rate=float(
                            baseline.get("negative_adaptation_rate", float("nan"))
                        ),
                        guarded_harm_rate=float(
                            entry.get("negative_adaptation_rate", float("nan"))
                        ),
                        regime=regime,
                        verdict=verdict,
                        note=artifact.note,
                        source=artifact.path,
                    )
                )
    return rows


def summarise(rows: list[CoverageRow]) -> dict[str, object]:
    """Count the claims the paper makes, per floor and per harm regime."""
    floors = sorted({row.floor for row in rows})
    per_floor: dict[str, object] = {}
    for floor in floors:
        subset = [row for row in rows if row.floor == floor]
        harmful = [row for row in subset if row.regime == "harmful"]
        little = [row for row in subset if row.regime == "little-harm"]
        per_floor[f"{floor}"] = {
            "n_pairings": len(subset),
            "harmful_cells": len(harmful),
            "harmful_improved": sum(row.verdict == "improved" for row in harmful),
            "harmful_removed_pct_min": (
                min((row.harm_removed_pct for row in harmful), default=float("nan"))
            ),
            "harmful_removed_pct_max": (
                max((row.harm_removed_pct for row in harmful), default=float("nan"))
            ),
            "little_harm_cells": len(little),
            "little_harm_inert": sum(row.verdict == "inert" for row in little),
            "methods": sorted({row.method for row in subset}),
            "datasets": sorted({row.dataset for row in subset}),
        }
    return {
        "harm_regime_thresholds_pp": {
            "harmful_at_or_below": -HARM_CEILING_PP,
            "little_harm_at_or_above": -HARM_FLOOR_PP,
        },
        "per_floor": per_floor,
        "n_rows": len(rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--floor",
        type=float,
        default=None,
        help="Restrict the table to one diversity floor (default: all).",
    )
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    artifacts = ARTIFACTS
    if args.floor is not None:
        artifacts = tuple(a for a in ARTIFACTS if a.floor == args.floor)
    rows = collect_rows(artifacts)
    if not rows:
        raise SystemExit("No guarded/unguarded pairing found.")
    rows.sort(key=lambda r: (r.floor, r.dataset, r.backbone, r.method))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_dataclass_rows_csv(rows, output_dir / "guard_coverage.csv")
    summary = summarise(rows)
    save_json_file(
        {"rows": [row.__dict__ for row in rows], "summary": summary},
        output_dir / "guard_coverage.json",
    )

    for row in rows:
        removed = (
            f"{row.harm_removed_pct:5.0f}%"
            if row.harm_removed_pct == row.harm_removed_pct
            else "    --"
        )
        logger.info(
            "tau=%.2f %-13s %-7s %-14s %+7.2f -> %+7.2f  %s  %-11s %s",
            row.floor,
            row.dataset,
            row.backbone,
            row.method,
            row.unguarded_pp,
            row.guarded_pp,
            removed,
            row.regime,
            row.verdict,
        )
    for floor, stats in summary["per_floor"].items():  # type: ignore[union-attr]
        logger.info(
            "tau=%s: %d/%d harmful cells improved (%.0f-%.0f%% removed); "
            "%d/%d little-harm cells inert",
            floor,
            stats["harmful_improved"],
            stats["harmful_cells"],
            stats["harmful_removed_pct_min"],
            stats["harmful_removed_pct_max"],
            stats["little_harm_inert"],
            stats["little_harm_cells"],
        )


if __name__ == "__main__":
    main()
