#!/usr/bin/env python3
# pyright: basic, reportMissingImports=false

"""Sweep selective-physics thresholds on Widar_BVP.

The sweep reuses the main real-data protocol: 3 leave-one-room-out folds,
5 seeds, 5 adaptation steps, and lr=5e-4. For each fold/seed we train the source
model once, adapt it once with physics_tta, and then vary only the selective
evaluation gate. This isolates threshold sensitivity from retraining noise.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from statistics import mean, stdev

import torch

from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    adapt_model_for_tta_with_summary,
    build_prepared_tta_loaders,
    compute_tta_reference_stats,
    evaluate_selective_tta_classifier,
    evaluate_tta_classifier,
    train_source_only_tta_model,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT = Path("outputs/final_ablations/selective_threshold_widar_main_protocol.json")
PREPARED_ROOT = Path("data/prepared")
ROOM_SPLITS = (
    ((0, 1), (2,), "office"),
    ((0, 2), (1,), "hall"),
    ((1, 2), (0,), "classroom"),
)
SEEDS = tuple(range(5))
CONFIDENCE_THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.5)
ALIGNMENT_THRESHOLDS = (1.0, 2.0, 5.0, 10.0)


def main() -> None:
    """Run the sweep and write JSON rows plus summary statistics."""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    config = TTAExperimentConfig(
        methods=("physics_tta",),
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[dict[str, float | int | str]] = []

    for train_envs, test_envs, room_name in ROOM_SPLITS:
        for seed in SEEDS:
            logger.info("fold=%s seed=%d", room_name, seed)
            (
                source_train,
                source_val,
                target_adapt,
                target_test,
                input_shape,
                num_classes,
                _metadata,
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
            source_model = train_source_only_tta_model(
                source_train_loader=source_train,
                source_val_loader=source_val,
                input_shape=input_shape,
                num_classes=num_classes,
                config=config,
            ).to(device)
            reference_stats = compute_tta_reference_stats(
                source_model, source_train, device
            )
            pre_accuracy = evaluate_tta_classifier(source_model, target_test, device)[
                "accuracy"
            ]
            source_pre_accuracy = evaluate_tta_classifier(
                source_model, source_val, device
            )["accuracy"]
            adapted_model = adapt_model_for_tta_with_summary(
                model=source_model,
                target_loader=target_adapt,
                reference_stats=reference_stats,
                config=config,
                method="physics_tta",
                device=device,
            ).model

            for confidence_threshold in CONFIDENCE_THRESHOLDS:
                for alignment_threshold in ALIGNMENT_THRESHOLDS:
                    post_accuracy = evaluate_selective_tta_classifier(
                        source_model=source_model,
                        adapted_model=adapted_model,
                        loader=target_test,
                        reference_stats=reference_stats,
                        device=device,
                        confidence_threshold=confidence_threshold,
                        alignment_threshold=alignment_threshold,
                    )["accuracy"]
                    source_post_accuracy = evaluate_selective_tta_classifier(
                        source_model=source_model,
                        adapted_model=adapted_model,
                        loader=source_val,
                        reference_stats=reference_stats,
                        device=device,
                        confidence_threshold=confidence_threshold,
                        alignment_threshold=alignment_threshold,
                    )["accuracy"]
                    rows.append(
                        {
                            "room": room_name,
                            "seed": seed,
                            "confidence_threshold": confidence_threshold,
                            "alignment_threshold": alignment_threshold,
                            "pre_accuracy": pre_accuracy,
                            "post_accuracy": post_accuracy,
                            "gain": post_accuracy - pre_accuracy,
                            "source_drop": source_post_accuracy - source_pre_accuracy,
                        }
                    )

    summary = _summarize(rows)
    OUTPUT.write_text(
        json.dumps(
            {
                "rows": rows,
                "summary": summary,
                "config": {
                    "adaptation_steps": config.adaptation_steps,
                    "adaptation_learning_rate": config.adaptation_learning_rate,
                    "seeds": list(SEEDS),
                    "confidence_thresholds": list(CONFIDENCE_THRESHOLDS),
                    "alignment_thresholds": list(ALIGNMENT_THRESHOLDS),
                    "elapsed_seconds": time.monotonic() - start,
                },
            },
            indent=2,
        )
    )
    logger.info("saved %s", OUTPUT)


def _summarize(
    rows: list[dict[str, float | int | str]],
) -> list[dict[str, float | int]]:
    summary: list[dict[str, float | int]] = []
    for confidence_threshold in CONFIDENCE_THRESHOLDS:
        for alignment_threshold in ALIGNMENT_THRESHOLDS:
            selected = [
                row
                for row in rows
                if row["confidence_threshold"] == confidence_threshold
                and row["alignment_threshold"] == alignment_threshold
            ]
            gains = [float(row["gain"]) for row in selected]
            drops = [float(row["source_drop"]) for row in selected]
            summary.append(
                {
                    "confidence_threshold": confidence_threshold,
                    "alignment_threshold": alignment_threshold,
                    "mean_gain": mean(gains),
                    "std_gain": stdev(gains) if len(gains) > 1 else 0.0,
                    "neg_rate": sum(gain < 0.0 for gain in gains) / max(len(gains), 1),
                    "mean_source_drop": mean(drops),
                    "n": len(gains),
                }
            )
    return summary


if __name__ == "__main__":
    main()
