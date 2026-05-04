# pyright: basic, reportMissingImports=false

"""Faithful SAR + CoTTA on Widar_BVP with MLP+BN backbone (n=15).

Closes the reviewer-identified baseline-fidelity gap: the primary MLP port of
SAR/CoTTA does not have BatchNorm and therefore cannot apply the original
methods faithfully. This script runs the same implementations on an MLP+BN
backbone using the leave-one-room-out protocol (3 folds x 5 seeds = 15 paired
observations per method).

Output:
    outputs/new_methods/widar_mlpbn_faithful_sar_cotta.json
    outputs/new_methods/widar_mlpbn_faithful_sar_cotta_full.json

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_widar_faithful_sar_cotta.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
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
OUTPUT_DIR = Path("outputs/new_methods")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)

ROOM_SPLITS = [
    {"train": (0, 1), "test": (2,), "name": "room2"},
    {"train": (0, 2), "test": (1,), "name": "room1"},
    {"train": (1, 2), "test": (0,), "name": "room0"},
]


def run_experiment(
    methods: tuple[str, ...],
    seeds: tuple[int, ...],
    source_epochs: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    config = TTAExperimentConfig(
        methods=methods,
        backbone="mlp_bn",
        source_epochs=source_epochs,
        adaptation_steps=5,
        adaptation_learning_rate=5e-4,
        batch_size=32,
    )

    for split in ROOM_SPLITS:
        train_envs, test_envs, name = split["train"], split["test"], split["name"]
        logger.info("split %s: train=%s test=%s", name, train_envs, test_envs)
        for seed in seeds:
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
                    dataset_name="widar_bvp",
                    prepared_root=PREPARED_ROOT,
                    seed=seed,
                    batch_size=config.batch_size,
                    source_val_ratio=config.source_val_ratio,
                    target_adapt_ratio=config.target_adapt_ratio,
                    train_env_ids=train_envs,
                    test_env_ids=test_envs,
                )
                run_rows = run_tta_suite(
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
                for r in run_rows:
                    rows.append(
                        {
                            "backbone": "mlp_bn",
                            "split": name,
                            "train_envs": str(train_envs),
                            "test_env": int(test_envs[0]),
                            "seed": seed,
                            "method": r.method,
                            "pre_accuracy": r.pre_accuracy,
                            "post_accuracy": r.post_accuracy,
                            "gain": r.gain,
                            "source_drop": r.source_drop,
                            "time_seconds": elapsed / max(len(methods), 1),
                        }
                    )
            except Exception as e:  # noqa: BLE001
                logger.error("  FAILED: %s", e)
                continue

    return {"rows": rows, "config": {"backbone": "mlp_bn", "methods": list(methods)}}


def compute_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    by_method: dict[str, list[float]] = {}
    for r in rows:
        m = str(r["method"])
        by_method.setdefault(m, []).append(float(r["gain"]))
    noadapt = torch.tensor(by_method.get("no_adapt", [0.0]))
    out: dict[str, dict[str, float]] = {}
    for m, gains in by_method.items():
        g = torch.tensor(gains)
        point, ci_lo, ci_hi = bootstrap_ci(g)
        d = float(cohens_d(g, noadapt)) if m != "no_adapt" else 0.0
        nr = float((g < 0).float().mean().item())
        pd = 0.0
        if m != "no_adapt" and len(gains) == len(by_method.get("no_adapt", [])):
            pd = paired_cohens_d(g - noadapt)
        out[m] = {
            "mean_gain": point,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "cohens_d_vs_noadapt": d,
            "paired_d_vs_noadapt": pd,
            "negative_adaptation_rate": nr,
            "n_observations": len(gains),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--methods",
        default="no_adapt,sar,cotta,tent",
        help="Comma-separated methods (default: no_adapt,sar,cotta,tent).",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--source-epochs", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    t0 = time.monotonic()
    results = run_experiment(methods, seeds, args.source_epochs)
    rows = results["rows"]

    full = args.output_dir / "widar_mlpbn_faithful_sar_cotta_full.json"
    summary_path = args.output_dir / "widar_mlpbn_faithful_sar_cotta.json"
    with open(full, "w") as f:
        json.dump(results, f, indent=2)
    summary = compute_summary(rows)  # type: ignore[arg-type]
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 72)
    print("WIDAR_BVP faithful SAR/CoTTA on MLP+BN (n=15 per method)")
    print("=" * 72)
    print(f"{'Method':<12s} {'Gain':>8s} {'95% CI':>22s} {'d':>6s} {'NegRate':>8s}")
    print("-" * 72)
    for m in methods:
        if m in summary:
            s = summary[m]
            ci = f"[{s['ci_lower']:+.4f}, {s['ci_upper']:+.4f}]"
            print(
                f"{m:<12s} {s['mean_gain']:+.4f} {ci:>22s} "
                f"{s['cohens_d_vs_noadapt']:+.3f} {s['negative_adaptation_rate']:.2f}"
            )
    print(f"\nSaved: {summary_path}")
    print(f"Elapsed: {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    main()
