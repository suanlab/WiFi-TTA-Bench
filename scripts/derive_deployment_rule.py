# pyright: basic, reportMissingImports=false

"""Derive and validate a pre-deployment decision rule for TTA safety.

Task A6 of the forward plan: turn the shift-severity x architecture x
calibration observation into a rule a practitioner can apply *before* deploying
test-time adaptation, when no target label exists yet.

Every predictor here is computable at that moment:

    norm_backbone      design-time constant -- does the encoder normalise
                       activations per batch (mlp_bn, cnn1d) or not (mlp)
    source_val_ece     calibration error on held-out SOURCE validation, the
                       one labelled split a practitioner still has
    feature_shift      standardized source->target encoder-feature moment gap,
                       computed on UNLABELLED target data
    target_confidence  mean max-softmax on the unlabelled target split
    target_diversity   marginal class-usage diversity on the same split

The outcome is the already-measured unguarded ``entropy_tta`` mean gain per
(dataset, backbone) cell, joined from released artifacts. This script trains
source-only models to read the predictors off them; it never re-runs
adaptation, so the outcome side is untouched measurement.

Sample size is the number of cells, not the number of runs. With a 3x3 grid
that is nine points, so the reported rule is a descriptive threshold validated
leave-one-dataset-out, not a fitted model. Treat the rank correlations as the
headline and the threshold as an operating point.

Stages:
    --stage predictors   train source-only models, write predictors.json
    --stage fit          derive + validate the rule from predictors.json
    --stage all          both (default)

Usage:
    python scripts/derive_deployment_rule.py --seeds 0,1,2
    python scripts/derive_deployment_rule.py --stage fit

Outputs:
    outputs/deployment_rule/predictors.json
    outputs/deployment_rule/rule.json
    outputs/deployment_rule/rule.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    build_prepared_tta_loaders,
    compute_tta_reference_stats,
    train_source_only_tta_model,
)
from pinn4csi.utils import (
    expected_calibration_error,
    prediction_diversity,
    save_dataclass_rows_csv,
    save_json_file,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PREPARED_ROOT = Path("data/prepared")
OUTPUT_DIR = Path("outputs/deployment_rule")

# Leave-one-environment-out: 3 environments => 3 folds, matching every other
# WiFi-TTA-Bench runner.
ENV_SPLITS = (
    {"train": (0, 1), "test": (2,), "name": "env2"},
    {"train": (0, 2), "test": (1,), "name": "env1"},
    {"train": (1, 2), "test": (0,), "name": "env0"},
)

BATCHNORM_BACKBONES = frozenset({"mlp_bn", "cnn1d"})


@dataclass(frozen=True)
class HarmSource:
    """Where a cell's measured unguarded harm is read from.

    Attributes:
        dataset: Prepared dataset name.
        backbone: Encoder backbone the harm was measured on.
        path: Released summary artifact.
        nested_key: Sub-key when the artifact is keyed by backbone first.
        method: Method whose mean gain is the outcome.
    """

    dataset: str
    backbone: str
    path: str
    nested_key: str | None
    method: str = "entropy_tta"


HARM_SOURCES: tuple[HarmSource, ...] = (
    HarmSource(
        "widar_bvp", "mlp", "outputs/guard_backbone/widar_3backbone.json", "mlp"
    ),
    HarmSource(
        "widar_bvp", "mlp_bn", "outputs/guard_backbone/widar_3backbone.json", "mlp_bn"
    ),
    HarmSource(
        "widar_bvp", "cnn1d", "outputs/guard_backbone/widar_3backbone.json", "cnn1d"
    ),
    HarmSource(
        "ntufi_har",
        "mlp",
        "outputs/collapse_guard_ntufi/ntufi_har_lame_sar_cotta.json",
        None,
    ),
    HarmSource(
        "ntufi_har",
        "mlp_bn",
        "outputs/guard_backbone/ntufi_mlp_bn/ntufi_har_lame_sar_cotta.json",
        None,
    ),
    HarmSource(
        "ntufi_har",
        "cnn1d",
        "outputs/guard_backbone/ntufi_cnn1d/ntufi_har_lame_sar_cotta.json",
        None,
    ),
    HarmSource(
        "signfi_top10",
        "mlp_bn",
        "outputs/collapse_guard_signfi/signfi_top10_lame_sar_cotta.json",
        None,
    ),
    HarmSource(
        "signfi_top10",
        "mlp",
        "outputs/guard_backbone/signfi_mlp/signfi_top10_lame_sar_cotta.json",
        None,
    ),
    HarmSource(
        "signfi_top10",
        "cnn1d",
        "outputs/guard_backbone/signfi_cnn1d/signfi_top10_lame_sar_cotta.json",
        None,
    ),
)

PREDICTOR_NAMES = (
    "source_val_ece",
    "feature_shift",
    "target_confidence",
    "target_diversity",
)


@dataclass(frozen=True)
class PredictorRow:
    """Deployable predictors for one (dataset, backbone, fold, seed) run."""

    dataset: str
    backbone: str
    split: str
    seed: int
    norm_backbone: int
    source_val_accuracy: float
    source_val_ece: float
    feature_shift: float
    target_confidence: float
    target_diversity: float
    train_seconds: float


@dataclass(frozen=True)
class CellRow:
    """Cell-level predictors joined to the measured unguarded harm."""

    dataset: str
    backbone: str
    norm_backbone: int
    n_runs: int
    source_val_accuracy: float
    source_val_ece: float
    feature_shift: float
    target_confidence: float
    target_diversity: float
    measured_gain_pp: float
    harmful: int


def _gather_logits(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> tuple[Tensor, Tensor | None]:
    """Run the model over a loader and collect logits (and labels if present).

    Args:
        model: Source-only classifier.
        loader: Labelled source or unlabelled target loader.
        device: Device to run on.

    Returns:
        ``(logits, labels)``. Shape: ``(num_samples, num_classes)`` and
        ``(num_samples,)``; labels is None for an unlabelled loader.
    """
    model.eval()
    logit_batches: list[Tensor] = []
    label_batches: list[Tensor] = []
    with torch.no_grad():
        for batch in loader:
            moved = {key: value.to(device) for key, value in batch.items()}
            prior = (
                moved["prior"] if bool(torch.any(moved["has_prior"]).item()) else None
            )
            logits = model(moved["x"], prior=prior, has_prior=moved["has_prior"])[
                "task_logits"
            ]
            logit_batches.append(logits.cpu())
            if "label" in moved:
                label_batches.append(moved["label"].cpu())
    logits_tensor = torch.cat(logit_batches, dim=0)
    labels_tensor = torch.cat(label_batches, dim=0) if label_batches else None
    return logits_tensor, labels_tensor


def _standardized_moment_gap(
    source_mean: Tensor,
    source_var: Tensor,
    target_mean: Tensor,
) -> float:
    """Feature-space shift, scaled by source spread so cells are comparable.

    Args:
        source_mean: Per-dimension source feature mean. Shape: (latent,).
        source_var: Per-dimension source feature variance. Shape: (latent,).
        target_mean: Per-dimension target feature mean. Shape: (latent,).

    Returns:
        ``||mu_t - mu_s|| / sqrt(mean(sigma_s^2))``, a unitless shift measure.
    """
    scale = float(source_var.mean().clamp_min(1e-12).sqrt().item())
    gap = float((target_mean - source_mean).norm(p=2).item())
    return gap / max(scale, 1e-12)


def compute_predictors(
    dataset: str,
    backbone: str,
    seeds: tuple[int, ...],
    source_epochs: int,
    batch_size: int,
    prepared_root: Path,
) -> list[PredictorRow]:
    """Train source-only models for one cell and read the predictors off them.

    Args:
        dataset: Prepared dataset name.
        backbone: Encoder backbone.
        seeds: Seeds to average over.
        source_epochs: Source training epochs (match the harm runs).
        batch_size: Loader batch size.
        prepared_root: Prepared data root.

    Returns:
        One PredictorRow per (fold, seed).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = TTAExperimentConfig(
        methods=("no_adapt",),
        source_epochs=source_epochs,
        adaptation_steps=0,
        batch_size=batch_size,
        backbone=backbone,
    )
    rows: list[PredictorRow] = []
    for split in ENV_SPLITS:
        for seed in seeds:
            started = time.perf_counter()
            (
                source_train_loader,
                source_val_loader,
                target_adapt_loader,
                _target_test_loader,
                input_shape,
                num_classes,
                _metadata,
            ) = build_prepared_tta_loaders(
                dataset_name=dataset,
                prepared_root=prepared_root,
                seed=seed,
                batch_size=batch_size,
                source_val_ratio=config.source_val_ratio,
                target_adapt_ratio=config.target_adapt_ratio,
                train_env_ids=split["train"],
                test_env_ids=split["test"],
            )
            model = train_source_only_tta_model(
                source_train_loader,
                source_val_loader,
                input_shape,
                num_classes,
                config,
            )

            source_logits, source_labels = _gather_logits(
                model, source_val_loader, device
            )
            if source_labels is None:
                raise RuntimeError("Source validation loader carried no labels.")
            source_accuracy = float(
                (source_logits.argmax(dim=1) == source_labels).float().mean().item()
            )
            source_ece = expected_calibration_error(source_logits, source_labels)

            target_logits, _ = _gather_logits(model, target_adapt_loader, device)
            target_confidence = float(
                torch.softmax(target_logits, dim=1).max(dim=1).values.mean().item()
            )
            target_diversity = prediction_diversity(target_logits)

            source_stats = compute_tta_reference_stats(model, source_val_loader, device)
            target_stats = compute_tta_reference_stats(
                model, target_adapt_loader, device
            )
            feature_shift = _standardized_moment_gap(
                source_stats.feature_mean.cpu(),
                source_stats.feature_var.cpu(),
                target_stats.feature_mean.cpu(),
            )

            rows.append(
                PredictorRow(
                    dataset=dataset,
                    backbone=backbone,
                    split=str(split["name"]),
                    seed=seed,
                    norm_backbone=int(backbone in BATCHNORM_BACKBONES),
                    source_val_accuracy=source_accuracy,
                    source_val_ece=source_ece,
                    feature_shift=feature_shift,
                    target_confidence=target_confidence,
                    target_diversity=target_diversity,
                    train_seconds=time.perf_counter() - started,
                )
            )
            logger.info(
                "  %s/%s %s seed=%d ece=%.3f shift=%.2f conf=%.3f div=%.3f",
                dataset,
                backbone,
                split["name"],
                seed,
                source_ece,
                feature_shift,
                target_confidence,
                target_diversity,
            )
    return rows


def load_measured_gain(source: HarmSource) -> float | None:
    """Read a cell's measured mean gain in percentage points, or None if absent."""
    path = Path(source.path)
    if not path.exists():
        logger.warning(
            "Missing harm artifact for %s/%s: %s", source.dataset, source.backbone, path
        )
        return None
    payload = json.loads(path.read_text())
    summary = payload.get("summary", payload)
    if source.nested_key is not None:
        summary = summary.get(source.nested_key, {})
    entry = summary.get(source.method)
    if entry is None:
        logger.warning(
            "Artifact %s has no method %s for %s/%s",
            path,
            source.method,
            source.dataset,
            source.backbone,
        )
        return None
    return float(entry["mean_gain"]) * 100.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def build_cells(
    predictor_rows: list[PredictorRow],
    harm_threshold_pp: float,
) -> list[CellRow]:
    """Average predictors per cell and join the measured harm.

    Args:
        predictor_rows: Per-run predictors.
        harm_threshold_pp: Mean gain at or below which a cell counts as harmful.

    Returns:
        One CellRow per cell that has both predictors and a measured outcome.
    """
    by_cell: dict[tuple[str, str], list[PredictorRow]] = {}
    for row in predictor_rows:
        by_cell.setdefault((row.dataset, row.backbone), []).append(row)

    cells: list[CellRow] = []
    for source in HARM_SOURCES:
        key = (source.dataset, source.backbone)
        runs = by_cell.get(key)
        if not runs:
            continue
        gain = load_measured_gain(source)
        if gain is None:
            continue
        cells.append(
            CellRow(
                dataset=source.dataset,
                backbone=source.backbone,
                norm_backbone=runs[0].norm_backbone,
                n_runs=len(runs),
                source_val_accuracy=_mean([r.source_val_accuracy for r in runs]),
                source_val_ece=_mean([r.source_val_ece for r in runs]),
                feature_shift=_mean([r.feature_shift for r in runs]),
                target_confidence=_mean([r.target_confidence for r in runs]),
                target_diversity=_mean([r.target_diversity for r in runs]),
                measured_gain_pp=gain,
                harmful=int(gain <= harm_threshold_pp),
            )
        )
    return cells


def spearman(values_a: list[float], values_b: list[float]) -> float:
    """Spearman rank correlation; ties broken by argsort, as in the sibling script."""
    if len(values_a) != len(values_b) or len(values_a) < 2:
        return float("nan")

    def _ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        for rank, index in enumerate(order):
            ranks[index] = float(rank)
        return ranks

    rank_a = _ranks(values_a)
    rank_b = _ranks(values_b)
    mean_a = _mean(rank_a)
    mean_b = _mean(rank_b)
    centred_a = [value - mean_a for value in rank_a]
    centred_b = [value - mean_b for value in rank_b]
    denominator = math.sqrt(
        sum(value * value for value in centred_a)
        * sum(value * value for value in centred_b)
    )
    if denominator == 0.0:
        return float("nan")
    numerator = sum(a * b for a, b in zip(centred_a, centred_b, strict=True))
    return numerator / denominator


def _predictor_value(cell: CellRow, name: str) -> float:
    return float(getattr(cell, name))


def best_threshold(
    cells: list[CellRow],
    name: str,
) -> tuple[float, int, bool]:
    """Find the threshold on one predictor that best separates harmful cells.

    Args:
        cells: Cells to fit on.
        name: Predictor field name.

    Returns:
        ``(threshold, correct, predict_harmful_above)``. Ties resolve to the
        first candidate, and the direction is chosen by accuracy.
    """
    values = sorted({_predictor_value(cell, name) for cell in cells})
    candidates = [
        (low + high) / 2.0 for low, high in zip(values, values[1:], strict=False)
    ]
    if not candidates:
        candidates = list(values)
    best = (float("nan"), -1, True)
    for threshold in candidates:
        for above in (True, False):
            correct = 0
            for cell in cells:
                value = _predictor_value(cell, name)
                predicted = value > threshold if above else value <= threshold
                correct += int(predicted == bool(cell.harmful))
            if correct > best[1]:
                best = (threshold, correct, above)
    return best


def leave_one_dataset_out(cells: list[CellRow], name: str) -> dict[str, object]:
    """Fit the threshold on all but one dataset and score the held-out one."""
    datasets = sorted({cell.dataset for cell in cells})
    folds: list[dict[str, object]] = []
    correct = 0
    total = 0
    for held_out in datasets:
        train = [cell for cell in cells if cell.dataset != held_out]
        test = [cell for cell in cells if cell.dataset == held_out]
        if not train or not test:
            continue
        threshold, _, above = best_threshold(train, name)
        fold_correct = 0
        for cell in test:
            value = _predictor_value(cell, name)
            predicted = value > threshold if above else value <= threshold
            fold_correct += int(predicted == bool(cell.harmful))
        correct += fold_correct
        total += len(test)
        folds.append(
            {
                "held_out_dataset": held_out,
                "threshold": threshold,
                "predict_harmful_above": above,
                "n_test_cells": len(test),
                "correct": fold_correct,
            }
        )
    return {
        "folds": folds,
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else float("nan"),
    }


SENSITIVITY_THRESHOLDS_PP = (-1.0, -1.5, -2.0, -3.0)


def fit_rule(cells: list[CellRow], harm_threshold_pp: float) -> dict[str, object]:
    """Rank-correlate every predictor with harm and validate a threshold rule."""
    gains = [cell.measured_gain_pp for cell in cells]
    predictors: dict[str, object] = {}
    for name in PREDICTOR_NAMES + ("norm_backbone",):
        values = [_predictor_value(cell, name) for cell in cells]
        threshold, correct, above = best_threshold(cells, name)
        predictors[name] = {
            "spearman_rho_vs_gain": spearman(values, gains),
            "in_sample_threshold": threshold,
            "predict_harmful_above": above,
            "in_sample_correct": correct,
            "n_cells": len(cells),
            "leave_one_dataset_out": leave_one_dataset_out(cells, name),
        }
    return {
        "harm_threshold_pp": harm_threshold_pp,
        "n_cells": len(cells),
        "n_harmful_cells": sum(cell.harmful for cell in cells),
        "datasets": sorted({cell.dataset for cell in cells}),
        "predictors": predictors,
        "caveat": (
            "Sample size is the number of (dataset, backbone) cells, not runs. "
            "These are descriptive thresholds validated leave-one-dataset-out; "
            "with fewer than ten cells no predictor ordering should be treated "
            "as established."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        default="all",
        choices=("predictors", "fit", "all"),
        help="Which stage to run (default: all).",
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2",
        help="Comma-separated seeds for the predictor pass (default: 0,1,2).",
    )
    parser.add_argument("--source-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--harm-threshold-pp",
        type=float,
        default=-2.0,
        help="Mean gain (pp) at or below which a cell counts as harmful.",
    )
    parser.add_argument("--prepared-root", default=str(PREPARED_ROOT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictors_path = output_dir / "predictors.json"
    seeds = tuple(int(part) for part in args.seeds.split(",") if part.strip())

    if args.stage in ("predictors", "all"):
        cells_to_run = sorted({(s.dataset, s.backbone) for s in HARM_SOURCES})
        rows: list[PredictorRow] = []
        started = time.perf_counter()
        for dataset, backbone in cells_to_run:
            logger.info("Cell %s/%s", dataset, backbone)
            rows.extend(
                compute_predictors(
                    dataset=dataset,
                    backbone=backbone,
                    seeds=seeds,
                    source_epochs=args.source_epochs,
                    batch_size=args.batch_size,
                    prepared_root=Path(args.prepared_root),
                )
            )
        save_json_file(
            {
                "rows": [row.__dict__ for row in rows],
                "seeds": list(seeds),
                "source_epochs": args.source_epochs,
                "elapsed_seconds": time.perf_counter() - started,
            },
            predictors_path,
        )
        logger.info("Wrote %s (%d rows)", predictors_path, len(rows))

    if args.stage in ("fit", "all"):
        if not predictors_path.exists():
            raise SystemExit(
                f"Missing {predictors_path}; run --stage predictors first."
            )
        payload = json.loads(predictors_path.read_text())
        rows = [PredictorRow(**row) for row in payload["rows"]]
        cells = build_cells(rows, args.harm_threshold_pp)
        if not cells:
            raise SystemExit("No cell had both predictors and a measured outcome.")
        rule = fit_rule(cells, args.harm_threshold_pp)
        # Sensitivity of the screen to the harm threshold that defines labels.
        sensitivity: dict[str, object] = {}
        for threshold in SENSITIVITY_THRESHOLDS_PP:
            alt_cells = build_cells(rows, threshold)
            alt = fit_rule(alt_cells, threshold)
            n_harm = int(alt["n_harmful_cells"])  # type: ignore[call-overload]
            sensitivity[f"{threshold}"] = {
                "n_harmful_cells": n_harm,
                "majority_baseline_correct": max(n_harm, len(alt_cells) - n_harm),
                "lodo_correct": {
                    name: stats["leave_one_dataset_out"]["correct"]  # type: ignore[index]
                    for name, stats in alt["predictors"].items()  # type: ignore[union-attr]
                },
                "n_cells": len(alt_cells),
            }
        rule["threshold_sensitivity"] = sensitivity
        save_json_file(rule, output_dir / "rule.json")
        save_dataclass_rows_csv(cells, output_dir / "rule.csv")
        logger.info(
            "Wrote %s over %d cells (%d harmful)",
            output_dir / "rule.json",
            rule["n_cells"],
            rule["n_harmful_cells"],
        )
        for name, stats in rule["predictors"].items():
            loo = stats["leave_one_dataset_out"]
            logger.info(
                "  %-18s rho=%+.2f  in-sample %d/%d  LODO %d/%d",
                name,
                stats["spearman_rho_vs_gain"],
                stats["in_sample_correct"],
                stats["n_cells"],
                loo["correct"],
                loo["total"],
            )


if __name__ == "__main__":
    main()
