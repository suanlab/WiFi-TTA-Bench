# pyright: basic, reportMissingImports=false

"""Does TTA still fail on Widar once the source model is not overfit?

The rejection hinged on one confound: the Widar source model has a large
in-domain train/val gap, so a failure of unlabelled TTA could reflect an overfit
source model rather than the cross-room shift. This script evaluates TTA on a
source model that is regularised to close that gap, in two stages.

Stage ``search`` trains source-only models under a grid of source-side
regularisers and records, per configuration, in-domain train accuracy, in-domain
validation accuracy, and their gap. It also records target accuracy, but
**selection never reads it**: the chosen configuration is the one with the
smallest train/val gap among those whose validation accuracy is not worse than
the unregularised baseline, both measured on source rooms only. Choosing by
target accuracy would leak the held-out room into model selection.

Stage ``tta`` reruns the canonical Widar method set on the unregularised and the
selected regularised source model under the same leave-one-room-out folds and
seeds (n=15 each), so the two arms differ only in source training.

Outputs:
    outputs/regularized_source/search.json
    outputs/regularized_source/tta.json

Usage:
    python scripts/run_regularized_source.py --stage search
    python scripts/run_regularized_source.py --stage tta
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from pathlib import Path

from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    build_prepared_tta_loaders,
    run_tta_suite,
)
from pinn4csi.utils import bootstrap_ci, save_json_file
from pinn4csi.utils.metrics import paired_cohens_d

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PREPARED_ROOT = Path("data/prepared")
OUTPUT_DIR = Path("outputs/regularized_source")
# Maximum drop in source validation accuracy a regularised arm may pay.
VAL_TOLERANCE = 0.03
DATASET = "widar_bvp"
SEEDS = (0, 1, 2, 3, 4)
SPLITS = (
    ((0, 1), (2,), "room2"),
    ((0, 2), (1,), "room1"),
    ((1, 2), (0,), "room0"),
)

# The canonical Widar method set of run_widar_full_tta.py plus the three
# published continual-TTA ports and the collapse guard, so the regularised arm is
# compared on every family the paper makes claims about.
TTA_METHODS = (
    "no_adapt",
    "entropy_tta",
    "conservative_entropy_tta",
    "physics_tta",
    "tent",
    "shot",
    "t3a",
    "sar",
    "cotta",
    "collapse_guarded_entropy_tta",
)

BASE_CONFIG = TTAExperimentConfig(
    methods=("no_adapt",),
    source_epochs=10,
    adaptation_steps=5,
    adaptation_learning_rate=5e-4,
    batch_size=32,
    hidden_dim=64,
    num_layers=3,
    latent_dim=32,
    early_stop_patience=3,
    collapse_guard_diversity_ratio=0.95,
)

# name -> overrides. Grid covers each regulariser alone at two strengths, a
# smaller encoder, longer training for the regularised case, and a combination.
SEARCH_GRID: dict[str, dict[str, float | int]] = {
    "baseline": {},
    "label_smoothing_0.1": {"source_label_smoothing": 0.1},
    "weight_decay_1e-2": {"source_weight_decay": 1e-2},
    "weight_decay_1e-1": {"source_weight_decay": 1e-1},
    "input_noise_0.1": {"source_input_noise_std": 0.1},
    "input_noise_0.3": {"source_input_noise_std": 0.3},
    "mixup_0.2": {"source_mixup_alpha": 0.2},
    "mixup_0.4": {"source_mixup_alpha": 0.4},
    "small_encoder": {"hidden_dim": 32, "latent_dim": 16},
    "combined": {
        "source_weight_decay": 1e-2,
        "source_input_noise_std": 0.1,
        "source_mixup_alpha": 0.2,
        "source_label_smoothing": 0.1,
    },
    # Capacity by training length: the best-validation checkpoint within a short
    # budget cannot reach the high train accuracy that drives the gap.
    "epochs_1": {"source_epochs": 1},
    "epochs_2": {"source_epochs": 2},
    "epochs_3": {"source_epochs": 3},
    "epochs_2_small": {"source_epochs": 2, "hidden_dim": 32, "latent_dim": 16},
    "combined_strong": {
        "source_weight_decay": 1e-1,
        "source_input_noise_std": 0.3,
        "source_mixup_alpha": 0.4,
        "source_label_smoothing": 0.1,
        "source_epochs": 20,
    },
}


def _run_cells(config: TTAExperimentConfig) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for train_envs, test_envs, name in SPLITS:
        for seed in SEEDS:
            loaders = build_prepared_tta_loaders(
                dataset_name=DATASET,
                prepared_root=PREPARED_ROOT,
                seed=seed,
                batch_size=config.batch_size,
                source_val_ratio=config.source_val_ratio,
                target_adapt_ratio=config.target_adapt_ratio,
                train_env_ids=train_envs,
                test_env_ids=test_envs,
            )
            result = run_tta_suite(
                source_train_loader=loaders[0],
                source_val_loader=loaders[1],
                target_adapt_loader=loaders[2],
                target_test_loader=loaders[3],
                input_shape=loaders[4],
                num_classes=loaders[5],
                metadata=loaders[6],
                seed=seed,
                config=config,
            )
            for row in result:
                rows.append({"room": name, **dataclasses.asdict(row)})
    return rows


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def stage_search(
    output_dir: Path, only: tuple[str, ...] = (), force: bool = False
) -> dict[str, object]:
    path = output_dir / "search.json"
    summaries: dict[str, dict[str, float]] = (
        json.loads(path.read_text())["summaries"] if path.exists() and not force else {}
    )
    for name, overrides in SEARCH_GRID.items():
        if name in summaries and name not in only:
            continue
        config = dataclasses.replace(BASE_CONFIG, methods=("no_adapt",), **overrides)
        started = time.perf_counter()
        rows = _run_cells(config)
        train = [float(r["source_train_accuracy"]) for r in rows]
        val = [float(r["source_pre_accuracy"]) for r in rows]
        target = [float(r["pre_accuracy"]) for r in rows]
        summaries[name] = {
            "source_train_accuracy": _mean(train),
            "source_val_accuracy": _mean(val),
            "train_val_gap_pp": (_mean(train) - _mean(val)) * 100.0,
            "target_accuracy": _mean(target),
            "n": len(rows),
            "seconds": time.perf_counter() - started,
        }
        logger.info(
            "%-20s train %.3f val %.3f gap %5.1f pp target %.3f",
            name,
            summaries[name]["source_train_accuracy"],
            summaries[name]["source_val_accuracy"],
            summaries[name]["train_val_gap_pp"],
            summaries[name]["target_accuracy"],
        )
    baseline_val = summaries["baseline"]["source_val_accuracy"]
    baseline_gap = summaries["baseline"]["train_val_gap_pp"]
    # A regularised arm must actually shrink the gap; otherwise it is not an
    # answer to the overfitting confound. Among those, prefer the smallest gap
    # whose validation accuracy is within VAL_TOLERANCE of the baseline.
    shrinks = {
        name: s
        for name, s in summaries.items()
        if name != "baseline" and s["train_val_gap_pp"] < baseline_gap
    }
    eligible = {
        name: s
        for name, s in shrinks.items()
        if s["source_val_accuracy"] >= baseline_val - VAL_TOLERANCE
    }
    pool = eligible or shrinks
    selected = min(pool, key=lambda n: pool[n]["train_val_gap_pp"]) if pool else None
    payload = {
        "selection_rule": (
            "smallest source train/val gap among configurations whose gap is below "
            "the unregularised baseline and whose source validation accuracy is "
            f"within {VAL_TOLERANCE:.2f} of it; target accuracy is recorded but "
            "never read by selection"
        ),
        "selected": selected,
        "selected_meets_val_constraint": selected in eligible,
        "grid": {name: SEARCH_GRID[name] for name in summaries},
        "summaries": summaries,
    }
    save_json_file(payload, path)
    logger.info("Selected %s (val constraint met: %s)", selected, selected in eligible)
    return payload


def _summarise(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    import torch

    out: dict[str, dict[str, float]] = {}
    for method in TTA_METHODS:
        gains = [float(r["gain"]) for r in rows if r["method"] == method]
        if not gains:
            continue
        tensor = torch.tensor(gains)
        _, low, high = bootstrap_ci(tensor)
        out[method] = {
            "mean_gain": float(tensor.mean()),
            "ci_lower": float(low),
            "ci_upper": float(high),
            "negative_adaptation_rate": float((tensor < 0).float().mean()),
            "paired_d": float(paired_cohens_d(tensor)) if method != "no_adapt" else 0.0,
            "n": len(gains),
        }
    return out


def stage_tta(output_dir: Path, extra: tuple[str, ...] = ()) -> None:
    search = json.loads((output_dir / "search.json").read_text())
    selected = search["selected"]
    if selected is None:
        raise SystemExit("No configuration shrinks the gap; nothing to compare.")
    # The rule-selected arm is primary; ``extra`` adds sensitivity arms (e.g. the
    # most gap-closed setting even though it pays more validation accuracy).
    arms: dict[str, dict[str, float | int]] = {"baseline": {}}
    for name in (selected, *extra):
        arms[name] = SEARCH_GRID[name]
    result: dict[str, object] = {"selected": selected, "arms": {}}
    for name, overrides in arms.items():
        config = dataclasses.replace(BASE_CONFIG, methods=TTA_METHODS, **overrides)
        rows = _run_cells(config)
        no_adapt = [r for r in rows if r["method"] == "no_adapt"]
        result["arms"][name] = {  # type: ignore[index]
            "overrides": overrides,
            "source_train_accuracy": _mean(
                [float(r["source_train_accuracy"]) for r in no_adapt]
            ),
            "source_val_accuracy": _mean(
                [float(r["source_pre_accuracy"]) for r in no_adapt]
            ),
            "target_accuracy": _mean([float(r["pre_accuracy"]) for r in no_adapt]),
            "summary": _summarise(rows),
            "rows": rows,
        }
        logger.info("arm %s done", name)
    save_json_file(result, output_dir / "tta.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("search", "tta"), required=True)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated grid names to (re)run; others are reused.",
    )
    parser.add_argument("--force", action="store_true", help="Re-run every search arm.")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "search":
        stage_search(
            output_dir,
            tuple(n for n in args.only.split(",") if n.strip()),
            force=args.force,
        )
    else:
        stage_tta(output_dir, tuple(n for n in args.only.split(",") if n.strip()))


if __name__ == "__main__":
    main()
