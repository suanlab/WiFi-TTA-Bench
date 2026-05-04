# pyright: basic, reportMissingImports=false

"""Run extended TTA method coverage (SAR, CoTTA, LAME + core set) on NTU-Fi and SignFi.

This script closes the method-coverage asymmetry flagged by reviewers: Widar_BVP
has SAR/CoTTA/LAME in outputs/new_methods/widar_lame_sar_cotta.json, but NTU-Fi
and SignFi-10 were previously missing these methods. Running this script produces
outputs/new_methods/<dataset>_lame_sar_cotta.json for the specified dataset with
the same leave-one-environment-out protocol as run_widar_full_tta.py.

Usage:
    python scripts/run_extended_tta_per_dataset.py \\
        --dataset ntufi_har --methods sar,cotta,lame,no_adapt
    python scripts/run_extended_tta_per_dataset.py \\
        --dataset signfi_top10 --methods sar,cotta,lame,no_adapt
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import torch

from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    build_prepared_tta_loaders,
    run_tta_suite,
)
from pinn4csi.utils.metrics import bootstrap_ci, cohens_d, paired_cohens_d

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PREPARED_ROOT = Path("data/prepared")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

# Leave-one-environment-out splits: 3 envs => 3 folds (train on 2, test on 1)
DEFAULT_ENV_SPLITS = [
    {"train": (0, 1), "test": (2,), "name": "env2"},
    {"train": (0, 2), "test": (1,), "name": "env1"},
    {"train": (1, 2), "test": (0,), "name": "env0"},
]


def run_experiment(
    dataset: str,
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    source_epochs: int,
    adaptation_steps: int,
    adaptation_learning_rate: float,
    batch_size: int,
) -> dict[str, object]:
    all_rows: list[dict[str, object]] = []
    config = TTAExperimentConfig(
        methods=methods,
        source_epochs=source_epochs,
        adaptation_steps=adaptation_steps,
        adaptation_learning_rate=adaptation_learning_rate,
        batch_size=batch_size,
    )

    for split in DEFAULT_ENV_SPLITS:
        train_envs = split["train"]
        test_envs = split["test"]
        split_name = split["name"]
        logger.info("Split %s: train=%s test=%s", split_name, train_envs, test_envs)
        for seed in seeds:
            logger.info("  seed=%d", seed)
            t0 = time.monotonic()
            try:
                (
                    src_train,
                    src_val,
                    tgt_adapt,
                    tgt_test,
                    input_shape,
                    num_classes,
                    metadata,
                ) = build_prepared_tta_loaders(
                    dataset_name=dataset,
                    prepared_root=PREPARED_ROOT,
                    seed=seed,
                    batch_size=config.batch_size,
                    source_val_ratio=config.source_val_ratio,
                    target_adapt_ratio=config.target_adapt_ratio,
                    train_env_ids=train_envs,
                    test_env_ids=test_envs,
                )
                rows = run_tta_suite(
                    source_train_loader=src_train,
                    source_val_loader=src_val,
                    target_adapt_loader=tgt_adapt,
                    target_test_loader=tgt_test,
                    input_shape=input_shape,
                    num_classes=num_classes,
                    metadata=metadata,
                    seed=seed,
                    config=config,
                )
                elapsed = time.monotonic() - t0
                for row in rows:
                    all_rows.append(
                        {
                            "split": split_name,
                            "train_envs": str(train_envs),
                            "test_env": int(test_envs[0]),
                            "seed": seed,
                            "method": row.method,
                            "pre_accuracy": row.pre_accuracy,
                            "post_accuracy": row.post_accuracy,
                            "gain": row.gain,
                            "source_drop": row.source_drop,
                            "time_seconds": elapsed / max(len(methods), 1),
                        }
                    )
            except Exception as e:  # noqa: BLE001
                logger.error("  FAILED: %s", e)
                continue

    return {"rows": all_rows, "config": asdict(config), "dataset": dataset}


def compute_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    by_method: dict[str, list[float]] = {}
    for r in rows:
        method = str(r["method"])
        by_method.setdefault(method, []).append(float(r["gain"]))

    noadapt = torch.tensor(by_method.get("no_adapt", [0.0]))
    summary: dict[str, dict[str, float]] = {}
    for method, gains in by_method.items():
        g = torch.tensor(gains)
        point, ci_lo, ci_hi = bootstrap_ci(g)
        d = float(cohens_d(g, noadapt)) if method != "no_adapt" else 0.0
        nr = float((g < 0).float().mean().item())
        if method != "no_adapt" and len(gains) == len(by_method.get("no_adapt", [])):
            diffs = g - noadapt
            pd = paired_cohens_d(diffs)
        else:
            pd = 0.0
        summary[method] = {
            "mean_gain": point,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "cohens_d_vs_noadapt": d,
            "paired_d_vs_noadapt": pd,
            "negative_adaptation_rate": nr,
            "n_observations": len(gains),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--dataset",
        required=True,
        choices=("ntufi_har", "signfi_top10"),
    )
    parser.add_argument(
        "--methods",
        default="no_adapt,sar,cotta,lame",
        help="Comma-separated method list (default: no_adapt,sar,cotta,lame).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/new_methods"),
    )
    parser.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated seeds (default 5 seeds x 3 folds = n=15).",
    )
    parser.add_argument("--source-epochs", type=int, default=10)
    parser.add_argument("--adaptation-steps", type=int, default=5)
    parser.add_argument("--adaptation-learning-rate", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    results = run_experiment(
        dataset=args.dataset,
        methods=methods,
        seeds=seeds,
        source_epochs=args.source_epochs,
        adaptation_steps=args.adaptation_steps,
        adaptation_learning_rate=args.adaptation_learning_rate,
        batch_size=args.batch_size,
    )
    rows = results["rows"]

    tag = args.dataset.replace("_", "-")
    full_path = args.output_dir / f"{args.dataset}_lame_sar_cotta_full.json"
    summary_path = args.output_dir / f"{args.dataset}_lame_sar_cotta.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    summary = compute_summary(rows)  # type: ignore[arg-type]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{tag.upper()} extended TTA (SAR/CoTTA/LAME) — {len(rows)} obs")
    print("=" * 70)
    print(f"{'Method':24s} {'Gain':>8s} {'95% CI':>22s} {'d':>6s} {'NegRate':>8s}")
    print("-" * 72)
    for method in methods:
        if method in summary:
            s = summary[method]
            ci = f"[{s['ci_lower']:+.4f}, {s['ci_upper']:+.4f}]"
            print(
                f"  {method:22s} {s['mean_gain']:+.4f} {ci:>22s} "
                f"{s['cohens_d_vs_noadapt']:+.3f} {s['negative_adaptation_rate']:.2f}"
            )
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
