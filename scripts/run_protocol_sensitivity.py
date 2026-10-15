"""Protocol sensitivity: learning rate x step budget, and non-i.i.d. target streams.

Two questions a reader can fairly ask of a benchmark that runs every method at one
shared, untuned setting on an i.i.d. draw:

1. **Does tuning rescue adaptation?** For each method, the adaptation learning rate
   and step budget are swept over a small grid. We report the benchmark default, the
   setting a deployer could pick without target labels (highest post-adaptation
   accuracy on the *source* validation split), and the oracle setting picked with
   target labels -- an upper bound no deployer has.
2. **Does the collapse guard misfire on correlated streams?** The adaptation half is
   re-ordered into class blocks (labels are used only to order the stream), so each
   batch is mostly one class, as when one activity is repeated. A diversity guard
   expects mixed batches; this measures what happens when they are not.

The source model is trained once per (fold, seed) and every adaptation starts from
it. Evaluation is always on the untouched evaluation half.

Usage::

    python scripts/run_protocol_sensitivity.py --part sweep
    python scripts/run_protocol_sensitivity.py --part stream

Output: ``outputs/protocol_sensitivity/{sweep,stream}.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    UnlabeledPreparedCSIDataset,
    _set_seed,
    adapt_model_for_tta_with_summary,
    build_prepared_tta_loaders,
    compute_tta_reference_stats,
    evaluate_tta_classifier,
    train_source_only_tta_model,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PREPARED_ROOT = Path("data/prepared")
OUTPUT_DIR = Path("outputs/protocol_sensitivity")
SEEDS = (0, 1, 2, 3, 4)
ENV_SPLITS = (
    {"train": (0, 1), "test": (2,), "name": "env2"},
    {"train": (0, 2), "test": (1,), "name": "env1"},
    {"train": (1, 2), "test": (0,), "name": "env0"},
)
SWEEP_CELLS = (
    ("widar_bvp", "mlp"),
    ("widar_bvp", "mlp_bn"),
    ("ntufi_har", "mlp"),
    ("ntufi_har", "mlp_bn"),
    ("signfi_top10", "mlp_bn"),
)
SWEEP_METHODS = ("entropy_tta", "tent")
LEARNING_RATES = (1e-4, 5e-4, 1e-3, 5e-3)
STEP_BUDGETS = (1, 5, 20)
STREAM_CELLS = (
    ("widar_bvp", "mlp", ("entropy_tta", "collapse_guarded_entropy_tta")),
    ("ntufi_har", "mlp", ("entropy_tta", "collapse_guarded_entropy_tta")),
    ("signfi_top10", "mlp_bn", ("tent", "collapse_guarded_tent")),
)
GUARD_FLOOR = 0.95


def _class_blocked(loader: DataLoader, seed: int, batch_size: int) -> DataLoader:  # type: ignore[type-arg]
    """Same samples, re-ordered into contiguous class blocks in random class order."""
    dataset: UnlabeledPreparedCSIDataset = loader.dataset  # type: ignore[assignment]
    labels = dataset.bundle.labels
    generator = torch.Generator().manual_seed(seed + 101)
    classes = torch.unique(labels)
    order = classes[torch.randperm(classes.numel(), generator=generator)]
    indices = torch.cat(
        [torch.nonzero(labels == c, as_tuple=False).squeeze(-1) for c in order]
    )
    blocked = UnlabeledPreparedCSIDataset(dataset.bundle.subset(indices))
    return DataLoader(blocked, batch_size=batch_size, shuffle=False)


def _source(dataset: str, backbone: str, split: dict[str, object], seed: int):
    config = TTAExperimentConfig(
        source_epochs=10,
        adaptation_steps=5,
        adaptation_learning_rate=5e-4,
        batch_size=32,
        backbone=backbone,
        collapse_guard_diversity_ratio=GUARD_FLOOR,
    )
    loaders = build_prepared_tta_loaders(
        dataset_name=dataset,
        prepared_root=PREPARED_ROOT,
        seed=seed,
        batch_size=config.batch_size,
        source_val_ratio=config.source_val_ratio,
        target_adapt_ratio=config.target_adapt_ratio,
        train_env_ids=split["train"],  # type: ignore[arg-type]
        test_env_ids=split["test"],  # type: ignore[arg-type]
    )
    source_train, source_val, target_adapt, target_test, shape, n_cls, _ = loaders
    # Same seeding order as run_tta_suite, so the source model is the benchmark's.
    _set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = train_source_only_tta_model(source_train, source_val, shape, n_cls, config)
    model = model.to(device)
    reference = compute_tta_reference_stats(model, source_train, device)
    return config, model, reference, source_val, target_adapt, target_test, device


def _evaluate(model, source_val, target_test, device) -> tuple[float, float]:  # type: ignore[no-untyped-def]
    return (
        evaluate_tta_classifier(model, target_test, device)["accuracy"],
        evaluate_tta_classifier(model, source_val, device)["accuracy"],
    )


def run_sweep() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, backbone in SWEEP_CELLS:
        for split in ENV_SPLITS:
            for seed in SEEDS:
                config, model, reference, s_val, t_adapt, t_test, device = _source(
                    dataset, backbone, split, seed
                )
                pre_target, pre_source = _evaluate(model, s_val, t_test, device)
                for method in SWEEP_METHODS:
                    for lr in LEARNING_RATES:
                        for steps in STEP_BUDGETS:
                            adapted = adapt_model_for_tta_with_summary(
                                model=model,
                                target_loader=t_adapt,
                                reference_stats=reference,
                                config=replace(
                                    config,
                                    adaptation_learning_rate=lr,
                                    adaptation_steps=steps,
                                ),
                                method=method,
                                device=device,
                            ).model
                            post_target, post_source = _evaluate(
                                adapted, s_val, t_test, device
                            )
                            rows.append(
                                {
                                    "dataset": dataset,
                                    "backbone": backbone,
                                    "fold": split["name"],
                                    "seed": seed,
                                    "method": method,
                                    "learning_rate": lr,
                                    "steps": steps,
                                    "gain": post_target - pre_target,
                                    "source_val_after": post_source,
                                    "source_val_before": pre_source,
                                }
                            )
                logger.info(
                    "sweep %s/%s %s seed%d done", dataset, backbone, split["name"], seed
                )
    return rows


def run_stream() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, backbone, methods in STREAM_CELLS:
        for split in ENV_SPLITS:
            for seed in SEEDS:
                config, model, reference, s_val, t_adapt, t_test, device = _source(
                    dataset, backbone, split, seed
                )
                pre_target, _ = _evaluate(model, s_val, t_test, device)
                streams = {
                    "iid": t_adapt,
                    "class_blocked": _class_blocked(t_adapt, seed, config.batch_size),
                }
                for stream, loader in streams.items():
                    for method in methods:
                        summary = adapt_model_for_tta_with_summary(
                            model=model,
                            target_loader=loader,
                            reference_stats=reference,
                            config=config,
                            method=method,
                            device=device,
                        )
                        post_target, _ = _evaluate(summary.model, s_val, t_test, device)
                        rows.append(
                            {
                                "dataset": dataset,
                                "backbone": backbone,
                                "fold": split["name"],
                                "seed": seed,
                                "stream": stream,
                                "method": method,
                                "gain": post_target - pre_target,
                                "accepted_steps": summary.accepted_steps,
                                "stop_reason": summary.stop_reason,
                            }
                        )
                logger.info(
                    "stream %s/%s %s seed%d done",
                    dataset,
                    backbone,
                    split["name"],
                    seed,
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--part", choices=("sweep", "stream"), required=True)
    args = parser.parse_args()
    start = time.monotonic()
    rows = run_sweep() if args.part == "sweep" else run_stream()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config: dict[str, object] = {"seeds": list(SEEDS), "guard_floor": GUARD_FLOOR}
    if args.part == "sweep":
        config.update(
            cells=[list(c) for c in SWEEP_CELLS],
            methods=list(SWEEP_METHODS),
            learning_rates=list(LEARNING_RATES),
            step_budgets=list(STEP_BUDGETS),
        )
    else:
        config.update(cells=[[d, b, list(m)] for d, b, m in STREAM_CELLS])
    out = OUTPUT_DIR / f"{args.part}.json"
    out.write_text(
        json.dumps(
            {
                "config": config,
                "rows": rows,
                "elapsed_seconds": time.monotonic() - start,
            },
            indent=2,
        )
    )
    logger.info("Wrote %s (%d rows)", out, len(rows))


if __name__ == "__main__":
    main()
