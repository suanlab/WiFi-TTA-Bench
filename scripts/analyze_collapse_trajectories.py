"""Measure prediction collapse directly: confusion matrices and class usage.

For every (dataset, backbone, fold, seed) cell this trains the benchmark source
model, then re-adapts it with ``entropy_tta`` for k = 0..K steps (each k starts
from the same source weights; adaptation is deterministic given the seed, so the
k-step model equals the first k steps of the K-step run). After each k it records,
on the held-out target test half:

* accuracy and the full confusion matrix (true x predicted counts);
* the predicted-class histogram and its normalised entropy
  ``D = H(p_hat) / log C`` (1 = every class used equally, 0 = one class).

It also records the same quantities for ``collapse_guarded_entropy_tta`` at the
paper's floor, so a figure can show what the guard leaves behind.

Usage::

    python scripts/analyze_collapse_trajectories.py            # all default cells
    python scripts/analyze_collapse_trajectories.py --cells widar_bvp:mlp

Output: ``outputs/collapse_trajectories/collapse_trajectories.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from dataclasses import replace
from pathlib import Path

import torch

from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    _move_batch_to_device,
    _optional_prior,
    _set_seed,
    adapt_model_for_tta_with_summary,
    build_prepared_tta_loaders,
    compute_tta_reference_stats,
    train_source_only_tta_model,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PREPARED_ROOT = Path("data/prepared")
OUTPUT = Path("outputs/collapse_trajectories/collapse_trajectories.json")
SEEDS = (0, 1, 2, 3, 4)
ENV_SPLITS = (
    {"train": (0, 1), "test": (2,), "name": "env2"},
    {"train": (0, 2), "test": (1,), "name": "env1"},
    {"train": (1, 2), "test": (0,), "name": "env0"},
)
# Primary-MLP cells on all three datasets, plus the two SignFi cells where the
# guard fails at tau=0.95 ("harm without collapse").
DEFAULT_CELLS = (
    "widar_bvp:mlp",
    "ntufi_har:mlp",
    "signfi_top10:mlp",
    "signfi_top10:resnet2d",
    "signfi_top10:transformer",
)
MAX_STEPS = 5
GUARD_FLOOR = 0.95


@torch.no_grad()
def _predictions(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,  # type: ignore[type-arg]
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    for batch in loader:
        moved = _move_batch_to_device(batch, device)
        logits = model(
            moved["x"], prior=_optional_prior(moved), has_prior=moved["has_prior"]
        )["task_logits"]
        preds.append(logits.argmax(dim=-1).cpu())
        labels.append(moved["label"].cpu())
    return torch.cat(preds), torch.cat(labels)


def _summarise(
    preds: torch.Tensor, labels: torch.Tensor, num_classes: int
) -> dict[str, object]:
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for t, p in zip(labels.tolist(), preds.tolist(), strict=True):
        confusion[t, p] += 1
    hist = torch.bincount(preds, minlength=num_classes).double()
    freq = hist / hist.sum().clamp_min(1.0)
    nz = freq[freq > 0]
    diversity = float(-(nz * nz.log()).sum().item() / math.log(num_classes))
    return {
        "accuracy": float((preds == labels).double().mean().item()),
        "diversity": diversity,
        "classes_used": int((hist > 0).sum().item()),
        "predicted_histogram": [int(v) for v in hist.tolist()],
        "confusion": confusion.tolist(),
    }


def run_cell(dataset: str, backbone: str) -> list[dict[str, object]]:
    base = TTAExperimentConfig(
        methods=("entropy_tta",),
        source_epochs=10,
        adaptation_steps=MAX_STEPS,
        adaptation_learning_rate=5e-4,
        batch_size=32,
        backbone=backbone,
        collapse_guard_diversity_ratio=GUARD_FLOOR,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs: list[dict[str, object]] = []
    for split in ENV_SPLITS:
        for seed in SEEDS:
            (
                source_train,
                source_val,
                target_adapt,
                target_test,
                input_shape,
                num_classes,
                _meta,
            ) = build_prepared_tta_loaders(
                dataset_name=dataset,
                prepared_root=PREPARED_ROOT,
                seed=seed,
                batch_size=base.batch_size,
                source_val_ratio=base.source_val_ratio,
                target_adapt_ratio=base.target_adapt_ratio,
                train_env_ids=split["train"],
                test_env_ids=split["test"],
            )
            # Same seeding order as run_tta_suite, so the source model matches
            # the benchmark's source model for this (fold, seed).
            _set_seed(seed)
            source = train_source_only_tta_model(
                source_train, source_val, input_shape, num_classes, base
            ).to(device)
            reference = compute_tta_reference_stats(source, source_train, device)

            trajectory: list[dict[str, object]] = []
            for k in range(MAX_STEPS + 1):
                if k == 0:
                    model = source
                else:
                    model = adapt_model_for_tta_with_summary(
                        model=source,
                        target_loader=target_adapt,
                        reference_stats=reference,
                        config=replace(base, adaptation_steps=k),
                        method="entropy_tta",
                        device=device,
                    ).model
                preds, labels = _predictions(model, target_test, device)
                trajectory.append(
                    {"steps": k, **_summarise(preds, labels, num_classes)}
                )

            guarded = adapt_model_for_tta_with_summary(
                model=source,
                target_loader=target_adapt,
                reference_stats=reference,
                config=base,
                method="collapse_guarded_entropy_tta",
                device=device,
            )
            g_preds, g_labels = _predictions(guarded.model, target_test, device)
            runs.append(
                {
                    "dataset": dataset,
                    "backbone": backbone,
                    "fold": split["name"],
                    "seed": seed,
                    "num_classes": num_classes,
                    "entropy_trajectory": trajectory,
                    "guarded": {
                        "accepted_steps": guarded.accepted_steps,
                        "stop_reason": guarded.stop_reason,
                        **_summarise(g_preds, g_labels, num_classes),
                    },
                }
            )
            last = trajectory[-1]
            logger.info(
                "%s/%s %s seed%d: acc %.3f->%.3f  D %.2f->%.2f  guard acc %.3f",
                dataset,
                backbone,
                split["name"],
                seed,
                trajectory[0]["accuracy"],
                last["accuracy"],
                trajectory[0]["diversity"],
                last["diversity"],
                runs[-1]["guarded"]["accuracy"],  # type: ignore[index]
            )
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--cells", default=",".join(DEFAULT_CELLS))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    start = time.monotonic()
    runs: list[dict[str, object]] = []
    for cell in args.cells.split(","):
        dataset, backbone = cell.split(":")
        runs.extend(run_cell(dataset, backbone))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": (
            "leave-one-environment-out x 5 seeds; entropy_tta re-run from the "
            f"source weights for k=0..{MAX_STEPS} steps; collapse_guarded_entropy_tta "
            f"at tau={GUARD_FLOOR}; metrics on the target test half"
        ),
        "diversity_definition": "H(predicted-class frequencies) / log(num_classes)",
        "runs": runs,
        "elapsed_seconds": time.monotonic() - start,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s (%d runs)", args.output, len(runs))


if __name__ == "__main__":
    main()
