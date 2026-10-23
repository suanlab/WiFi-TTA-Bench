# pyright: basic, reportMissingImports=false

"""Source and target accuracy per dataset, measured one consistent way.

The dataset table and the shift-severity ladder previously quoted accuracies
from a single fixed-split run. This script measures them under the benchmark's
own protocol so every dataset is described the same way: source-only models,
leave-one-environment-out, 5 seeds (n=15), reporting in-domain train accuracy,
in-domain validation accuracy (held-out source data), and target accuracy on
the held-out environment. The severity of a shift is the drop from in-domain
validation accuracy to target accuracy, so it is not inflated by overfitting
to the training split.

Output: outputs/severity/source_target_gap.json

Usage:
    python scripts/compute_source_target_gap.py
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
from pathlib import Path

from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    build_prepared_tta_loaders,
    run_tta_suite,
)
from pinn4csi.utils import save_json_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PREPARED_ROOT = Path("data/prepared")
OUTPUT = Path("outputs/severity/source_target_gap.json")
SEEDS = (0, 1, 2, 3, 4)
SPLITS = (((0, 1), (2,)), ((0, 2), (1,)), ((1, 2), (0,)))
DATASETS = ("widar_bvp", "ntufi_har", "signfi_top10")
BACKBONES = ("mlp", "mlp_bn")
BASE = TTAExperimentConfig(
    methods=("no_adapt",),
    source_epochs=10,
    adaptation_steps=5,
    batch_size=32,
    hidden_dim=64,
    num_layers=3,
    latent_dim=32,
    early_stop_patience=0,
)


def measure(dataset: str, backbone: str) -> dict[str, float]:
    config = dataclasses.replace(BASE, backbone=backbone)
    train: list[float] = []
    val: list[float] = []
    target: list[float] = []
    for train_envs, test_envs in SPLITS:
        for seed in SEEDS:
            loaders = build_prepared_tta_loaders(
                dataset_name=dataset,
                prepared_root=PREPARED_ROOT,
                seed=seed,
                batch_size=config.batch_size,
                source_val_ratio=config.source_val_ratio,
                target_adapt_ratio=config.target_adapt_ratio,
                train_env_ids=train_envs,
                test_env_ids=test_envs,
            )
            row = run_tta_suite(
                source_train_loader=loaders[0],
                source_val_loader=loaders[1],
                target_adapt_loader=loaders[2],
                target_test_loader=loaders[3],
                input_shape=loaders[4],
                num_classes=loaders[5],
                metadata=loaders[6],
                seed=seed,
                config=config,
            )[0]
            train.append(row.source_train_accuracy)
            val.append(row.source_pre_accuracy)
            target.append(row.pre_accuracy)

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    return {
        "source_train_accuracy": mean(train),
        "source_val_accuracy": mean(val),
        "target_accuracy": mean(target),
        "val_to_target_drop_pp": (mean(val) - mean(target)) * 100.0,
        "train_to_target_drop_pp": (mean(train) - mean(target)) * 100.0,
        "n": len(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--backbones", default=",".join(BACKBONES))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    datasets = tuple(d for d in args.datasets.split(",") if d)
    backbones = tuple(b for b in args.backbones.split(",") if b)
    result: dict[str, dict[str, dict[str, float]]] = {}
    for dataset in datasets:
        for backbone in backbones:
            stats = measure(dataset, backbone)
            result.setdefault(dataset, {})[backbone] = stats
            logger.info(
                "%s/%s train %.3f val %.3f target %.3f drop(val->target) %.1f pp",
                dataset,
                backbone,
                stats["source_train_accuracy"],
                stats["source_val_accuracy"],
                stats["target_accuracy"],
                stats["val_to_target_drop_pp"],
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json_file(
        {
            "protocol": "source-only, leave-one-environment-out x 5 seeds (n=15), "
            "primary configuration (10 source epochs, early-stop patience 3)",
            "definition": "severity = in-domain validation accuracy - target accuracy",
            "results": result,
        },
        args.output,
    )


if __name__ == "__main__":
    main()
