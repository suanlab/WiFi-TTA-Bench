# pyright: basic, reportMissingImports=false

"""Run ALL TTA methods on real Widar_BVP data with leave-one-room-out protocol.
This is the critical experiment for NeurIPS: demonstrating real-data performance."""

from __future__ import annotations

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

OUTPUT_DIR = Path("outputs/widar_full_tta")

ALL_METHODS = (
    "no_adapt",
    "entropy_tta",
    "conservative_entropy_tta",
    "physics_tta",
    "physics_entropy_tta",
    "safe_physics_tta",
    "selective_physics_tta",
    "tent",
    "shot",
    "t3a",
)

PREPARED_ROOT = Path("data/prepared")
SEEDS = (0, 1, 2, 3, 4)
# Leave-one-room-out: train on 2 rooms, test on 1
ROOM_SPLITS = [
    {"train": (0, 1), "test": (2,), "name": "room2_office"},
    {"train": (0, 2), "test": (1,), "name": "room1_hall"},
    {"train": (1, 2), "test": (0,), "name": "room0_classroom"},
]


def run_widar_full_experiment() -> dict[str, object]:
    """Run all methods × all room splits × all seeds on real Widar_BVP."""
    all_rows: list[dict[str, object]] = []

    config = TTAExperimentConfig(
        methods=ALL_METHODS,
        source_epochs=10,
        adaptation_steps=5,
        adaptation_learning_rate=5e-4,
        batch_size=32,
        hidden_dim=64,
        num_layers=3,
        latent_dim=32,
        invariance_weight=1.0,
        residual_weight=1.0,
        entropy_weight=0.1,
        selective_confidence_threshold=0.2,
        selective_alignment_threshold=2.0,
        early_stop_patience=3,
    )

    for split in ROOM_SPLITS:
        train_envs = split["train"]
        test_envs = split["test"]
        room_name = split["name"]
        logger.info(
            "=== Room split: train=%s test=%s (%s) ===",
            train_envs,
            test_envs,
            room_name,
        )

        for seed in SEEDS:
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
                    dataset_name="widar_bvp",
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
                            "room": room_name,
                            "train_envs": str(train_envs),
                            "test_env": int(test_envs[0]),
                            "seed": seed,
                            "method": row.method,
                            "pre_accuracy": row.pre_accuracy,
                            "post_accuracy": row.post_accuracy,
                            "gain": row.gain,
                            "source_drop": row.source_drop,
                            "time_seconds": elapsed / len(ALL_METHODS),
                        }
                    )
            except Exception as e:
                logger.error("  FAILED: %s", e)
                continue

    return {"rows": all_rows, "config": asdict(config)}


def compute_summary(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Compute per-method summary with effect sizes."""
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
        # Paired Cohen's d (differences from no_adapt)
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = run_widar_full_experiment()
    rows = results["rows"]

    with open(OUTPUT_DIR / "full_results.json", "w") as f:
        json.dump(results, f, indent=2)

    summary = compute_summary(rows)  # type: ignore[arg-type]
    with open(OUTPUT_DIR / "method_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 70)
    print("WIDAR_BVP REAL DATA: FULL TTA METHOD COMPARISON")
    print(f"  Leave-one-room-out × {len(SEEDS)} seeds = {len(rows)} total observations")
    print("=" * 70)
    print(f"{'Method':32s} {'Gain':>8s} {'95% CI':>20s} {'d':>6s} {'NegRate':>8s}")
    print("-" * 76)
    for method in ALL_METHODS:
        if method in summary:
            s = summary[method]
            ci = f"[{s['ci_lower']:+.4f}, {s['ci_upper']:+.4f}]"
            print(
                f"  {method:30s} {s['mean_gain']:+.4f} {ci:>20s} "
                f"{s['cohens_d_vs_noadapt']:+.3f} {s['negative_adaptation_rate']:.2f}"
            )

    # Save evidence
    evidence_dir = Path(".sisyphus/evidence/widar_full_tta")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    with open(evidence_dir / "method_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/ and .sisyphus/evidence/widar_full_tta/")


if __name__ == "__main__":
    main()
