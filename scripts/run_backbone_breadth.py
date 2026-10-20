# pyright: basic, reportMissingImports=false

"""TTA across stronger backbones and other normalisers, on every method.

Review objected that conclusions rest on shallow backbones (MLP, MLP+BN,
CNN1D+BN) and that the cross-backbone evidence covered mainly TENT. This script
runs the full method set on three further backbones:

    resnet2d     residual 2D CNN over the CSI grid, BatchNorm (the SenseFi-style
                 treatment of CSI as an image)
    resnet2d_gn  the same network with GroupNorm, isolating the normaliser
    transformer  self-attention over grid rows, LayerNorm (no batch statistics)

on all three datasets, leave-one-environment-out, 5 seeds (n=15 per cell), with
the source configuration of the primary benchmark. TENT adapts BatchNorm affines
when present and GroupNorm/LayerNorm affines otherwise.

Outputs:
    outputs/backbone_breadth/<dataset>_<backbone>.json   (rows + summary)

Usage:
    python scripts/run_backbone_breadth.py
    python scripts/run_backbone_breadth.py --datasets widar_bvp --backbones transformer
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from pathlib import Path

import torch

from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    build_prepared_tta_loaders,
    run_tta_suite,
)
from pinn4csi.utils import bootstrap_ci, save_json_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PREPARED_ROOT = Path("data/prepared")
OUTPUT_DIR = Path("outputs/backbone_breadth")
SEEDS = (0, 1, 2, 3, 4)
SPLITS = (((0, 1), (2,), "env2"), ((0, 2), (1,), "env1"), ((1, 2), (0,), "env0"))
BACKBONES = ("resnet2d", "resnet2d_gn", "transformer")
DATASETS = ("widar_bvp", "ntufi_har", "signfi_top10")
METHODS = (
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
    methods=METHODS,
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


def summarise(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for method in METHODS:
        gains = torch.tensor([float(r["gain"]) for r in rows if r["method"] == method])
        if gains.numel() == 0:
            continue
        _, low, high = bootstrap_ci(gains)
        out[method] = {
            "mean_gain": float(gains.mean()),
            "ci_lower": low,
            "ci_upper": high,
            "negative_adaptation_rate": float((gains < 0).float().mean()),
            "n": int(gains.numel()),
        }
    return out


def run_cell(dataset: str, backbone: str) -> dict[str, object]:
    config = dataclasses.replace(BASE_CONFIG, backbone=backbone)
    rows: list[dict[str, object]] = []
    started = time.perf_counter()
    for train_envs, test_envs, name in SPLITS:
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
            for row in run_tta_suite(
                source_train_loader=loaders[0],
                source_val_loader=loaders[1],
                target_adapt_loader=loaders[2],
                target_test_loader=loaders[3],
                input_shape=loaders[4],
                num_classes=loaders[5],
                metadata=loaders[6],
                seed=seed,
                config=config,
            ):
                rows.append({"split": name, **dataclasses.asdict(row)})
    no_adapt = [r for r in rows if r["method"] == "no_adapt"]

    def mean(key: str) -> float:
        return sum(float(r[key]) for r in no_adapt) / len(no_adapt)

    return {
        "dataset": dataset,
        "backbone": backbone,
        "config": dataclasses.asdict(config),
        "source_train_accuracy": mean("source_train_accuracy"),
        "source_val_accuracy": mean("source_pre_accuracy"),
        "target_accuracy": mean("pre_accuracy"),
        "summary": summarise(rows),
        "rows": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--backbones", default=",".join(BACKBONES))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument(
        "--force", action="store_true", help="Re-run cells whose output exists."
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in [d for d in args.datasets.split(",") if d]:
        for backbone in [b for b in args.backbones.split(",") if b]:
            path = output_dir / f"{dataset}_{backbone}.json"
            if path.exists() and not args.force:
                logger.info("skip %s (exists)", path)
                continue
            result = run_cell(dataset, backbone)
            save_json_file(result, path)
            positive = [
                m
                for m, s in result["summary"].items()  # type: ignore[union-attr]
                if s["ci_lower"] > 0
            ]
            logger.info(
                "CELL %s/%s target=%.3f positive-CI=%s (%.0fs)",
                dataset,
                backbone,
                result["target_accuracy"],
                positive or "none",
                result["elapsed_seconds"],
            )


if __name__ == "__main__":
    main()
