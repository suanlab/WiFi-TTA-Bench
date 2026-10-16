# pyright: basic, reportMissingImports=false

"""Per-class confidence and accuracy on Widar/NTU-Fi/SignFi target.

Anchors the paper's overconfidence claim with measured per-class statistics.
Supports the three real datasets via --dataset and produces JSON/CSV per
dataset with the same n=15 (3 folds x 5 seeds) source-only protocol.

Output:
    outputs/overconfidence/<dataset>_per_class.json
    outputs/overconfidence/<dataset>_per_class.csv

Usage:
    python scripts/analyze_widar_overconfidence.py --dataset widar_bvp
    python scripts/analyze_widar_overconfidence.py --dataset ntufi_har
    python scripts/analyze_widar_overconfidence.py --dataset signfi_top10
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch.utils.data import DataLoader

from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    _set_seed,
    build_prepared_tta_loaders,
    train_source_only_tta_model,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PREPARED_ROOT = Path("data/prepared")
OUTPUT_DIR = Path("outputs/overconfidence")
SEEDS = (0, 1, 2, 3, 4)
ENV_SPLITS = [
    {"train": (0, 1), "test": (2,), "name": "env2"},
    {"train": (0, 2), "test": (1,), "name": "env1"},
    {"train": (1, 2), "test": (0,), "name": "env0"},
]


def _class_names(dataset: str) -> dict[int, str]:
    """Class names from the prepared bundle's metadata (the labels' source of truth)."""
    meta = json.loads((PREPARED_ROOT / dataset / "metadata.json").read_text())
    return {int(k): str(v) for k, v in meta.get("label_names", {}).items()}


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


def run(dataset: str) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes_default = 10 if dataset == "signfi_top10" else 6
    sum_correct = torch.zeros(num_classes_default)
    sum_conf = torch.zeros(num_classes_default)
    sum_count = torch.zeros(num_classes_default)
    per_run: list[dict[str, object]] = []
    num_classes = num_classes_default
    config = TTAExperimentConfig(
        backbone="mlp", source_epochs=10, batch_size=32, adaptation_steps=0
    )
    for split in ENV_SPLITS:
        train_envs, test_envs, name = split["train"], split["test"], split["name"]
        logger.info(
            "[%s] split %s train=%s test=%s", dataset, name, train_envs, test_envs
        )
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
                dataset_name=dataset,
                prepared_root=PREPARED_ROOT,
                seed=seed,
                batch_size=config.batch_size,
                source_val_ratio=config.source_val_ratio,
                target_adapt_ratio=config.target_adapt_ratio,
                train_env_ids=train_envs,
                test_env_ids=test_envs,
            )
            if num_classes != sum_correct.numel():
                # First seed - resize accumulators to actual num_classes
                sum_correct = torch.zeros(num_classes)
                sum_conf = torch.zeros(num_classes)
                sum_count = torch.zeros(num_classes)
            # Same seeding as run_tta_suite, so these are the benchmark's models.
            _set_seed(seed)
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
    class_names = _class_names(dataset)

    per_class = []
    for c in range(num_classes):
        per_class.append(
            {
                "class_id": c,
                "class_name": class_names.get(c, f"class_{c}"),
                "accuracy_mean": acc[c],
                "confidence_mean": conf[c],
                "n_target_samples_aggregated": counts[c],
            }
        )

    payload = {
        "protocol": (
            "leave-one-environment-out x 5 seeds = 15 runs; source-only (no adaptation)"
        ),
        "dataset": dataset,
        "backbone": "mlp",
        "n_runs": 15,
        "per_class": per_class,
        "per_run": per_run,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--dataset",
        default="widar_bvp",
        choices=("widar_bvp", "ntufi_har", "signfi_top10"),
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = run(args.dataset)
    out_json = OUTPUT_DIR / f"{args.dataset}_per_class.json"
    out_csv = OUTPUT_DIR / f"{args.dataset}_per_class.csv"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    with open(out_csv, "w") as f:
        f.write("class_id,class_name,accuracy,confidence,n_samples\n")
        for row in payload["per_class"]:
            f.write(
                f"{row['class_id']},{row['class_name']},"
                f"{row['accuracy_mean']:.4f},{row['confidence_mean']:.4f},"
                f"{row['n_target_samples_aggregated']}\n"
            )
    print("\n" + "=" * 72)
    print(f"{args.dataset} per-class overconfidence (source-only, n=15 aggregated)")
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
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
