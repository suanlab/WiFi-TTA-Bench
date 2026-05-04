# pyright: basic, reportMissingImports=false

"""Per-class confidence and accuracy on Widar_BVP target to anchor the paper's
86-94% overconfidence claim.

The main paper, abstract, and root-cause section state that Widar source-model
confidence on target samples remains high (86-94%) while per-class accuracy can
be as low as 10%. Previously no inline table backed this claim. This script
computes per-class (mean max-softmax confidence, accuracy, sample count) on the
held-out target rooms, averaged over 3 folds x 5 seeds = 15 runs.

Output:
    outputs/overconfidence/widar_per_class.json
    outputs/overconfidence/widar_per_class.csv

Usage:
    CUDA_VISIBLE_DEVICES=2 python scripts/analyze_widar_overconfidence.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    build_prepared_tta_loaders,
    train_source_only_tta_model,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PREPARED_ROOT = Path("data/prepared")
OUTPUT_DIR = Path("outputs/overconfidence")
SEEDS = (0, 1, 2, 3, 4)
ROOM_SPLITS = [
    {"train": (0, 1), "test": (2,), "name": "room2"},
    {"train": (0, 2), "test": (1,), "name": "room1"},
    {"train": (1, 2), "test": (0,), "name": "room0"},
]

_WIDAR_CLASSES = {
    0: "slide",
    1: "draw-zigzag",
    2: "push&pull",
    3: "sweep",
    4: "clap",
    5: "draw-circle",
}


def _collect_per_class(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return per-class (sum_correct, sum_confidence, count)."""
    sums = torch.zeros(num_classes, device=device)
    confs = torch.zeros(num_classes, device=device)
    counts = torch.zeros(num_classes, device=device)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["label"].to(device)
            prior = batch.get("prior")
            if prior is not None:
                prior = prior.to(device)
            has_prior = batch.get("has_prior")
            if has_prior is not None:
                has_prior = has_prior.to(device)
            out = model(x, prior=prior, has_prior=has_prior)
            logits = out["task_logits"]
            probs = F.softmax(logits, dim=-1)
            max_conf, pred = probs.max(dim=-1)
            correct = (pred == y).float()
            for c in range(num_classes):
                mask = y == c
                if mask.any():
                    sums[c] += correct[mask].sum()
                    confs[c] += max_conf[mask].sum()
                    counts[c] += mask.sum()
    return sums.cpu(), confs.cpu(), counts.cpu()


def run() -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # accumulate per-class stats over (room, seed) pairs
    num_classes = 6
    sum_correct = torch.zeros(num_classes)
    sum_conf = torch.zeros(num_classes)
    sum_count = torch.zeros(num_classes)
    per_run: list[dict[str, object]] = []
    config = TTAExperimentConfig(
        backbone="mlp", source_epochs=10, batch_size=32, adaptation_steps=0
    )
    for split in ROOM_SPLITS:
        train_envs, test_envs, name = split["train"], split["test"], split["name"]
        logger.info("split %s train=%s test=%s", name, train_envs, test_envs)
        for seed in SEEDS:
            (
                src_train,
                src_val,
                _tgt_adapt,
                tgt_test,
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
            model = train_source_only_tta_model(
                source_train_loader=src_train,
                source_val_loader=src_val,
                input_shape=input_shape,
                num_classes=num_classes,
                config=config,
            )
            sc, cf, ct = _collect_per_class(model, tgt_test, device, num_classes)
            sum_correct += sc
            sum_conf += cf
            sum_count += ct
            per_run.append(
                {
                    "split": name,
                    "seed": seed,
                    "class_accuracy": (sc / ct.clamp(min=1)).tolist(),
                    "class_confidence": (cf / ct.clamp(min=1)).tolist(),
                    "class_count": ct.int().tolist(),
                }
            )

    acc = (sum_correct / sum_count.clamp(min=1)).tolist()
    conf = (sum_conf / sum_count.clamp(min=1)).tolist()
    counts = sum_count.int().tolist()

    per_class = []
    for c in range(num_classes):
        per_class.append(
            {
                "class_id": c,
                "class_name": _WIDAR_CLASSES.get(c, f"class_{c}"),
                "accuracy_mean": acc[c],
                "confidence_mean": conf[c],
                "n_target_samples_aggregated": counts[c],
            }
        )

    payload = {
        "protocol": (
            "leave-one-room-out x 5 seeds = 15 runs; source-only (no adaptation)"
        ),
        "dataset": "widar_bvp",
        "backbone": "mlp",
        "n_runs": 15,
        "per_class": per_class,
        "per_run": per_run,
    }
    return payload


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = run()
    with open(OUTPUT_DIR / "widar_per_class.json", "w") as f:
        json.dump(payload, f, indent=2)
    with open(OUTPUT_DIR / "widar_per_class.csv", "w") as f:
        f.write("class_id,class_name,accuracy,confidence,n_samples\n")
        for row in payload["per_class"]:
            f.write(
                f"{row['class_id']},{row['class_name']},"
                f"{row['accuracy_mean']:.4f},{row['confidence_mean']:.4f},"
                f"{row['n_target_samples_aggregated']}\n"
            )
    print("\n" + "=" * 72)
    print("WIDAR_BVP per-class overconfidence (source-only, n=15 aggregated)")
    print("=" * 72)
    print(f"{'class':<14s} {'acc':>8s} {'conf':>8s} {'n':>6s}")
    print("-" * 72)
    for row in payload["per_class"]:
        print(
            f"{row['class_name']:<14s} "
            f"{row['accuracy_mean']:>8.3f} "
            f"{row['confidence_mean']:>8.3f} "
            f"{row['n_target_samples_aggregated']:>6d}"
        )
    print(f"\nSaved: {OUTPUT_DIR / 'widar_per_class.json'}")


if __name__ == "__main__":
    main()
