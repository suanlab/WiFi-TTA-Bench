# pyright: basic, reportMissingImports=false

"""Supervised upper bound: fine-tune on labelled target data (5-fold CV).

Reproduces the supervised upper-bound reported in the paper: the labelled
target-domain oracle the benchmark does NOT release, used only to quantify
the headroom unlabelled TTA leaves on the table on Widar_BVP.

Protocol (per held-out room):
    1. Train a source model on the two training rooms (shared MLP backbone,
       same hyperparameters as the primary benchmark, ``TTAExperimentConfig``).
    2. 5-fold cross-validation on the held-out target room:
       - 4/5 of target samples used as labelled fine-tuning set,
       - 1/5 used as the evaluation fold.
    3. Fine-tune the source checkpoint with Adam (lr=1e-3, 50 epochs) on the
       labelled target set; record best-epoch validation accuracy on the
       held-out 1/5 fold.
    4. Repeat across three rooms x 5 folds x ``SEEDS`` seeds.
    5. Report matched source-only accuracy, mean target accuracy of the
       fine-tuned oracle, and the supervised gap.

Output:
    outputs/phase2/upper_bound.json

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/train_supervised_upper_bound.py
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
from torch.optim import Adam
from torch.utils.data import DataLoader

from pinn4csi.data.paper1 import load_prepared_paper1_dataset
from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    _move_batch_to_device,
    _optional_prior,
    build_prepared_tta_loaders,
    evaluate_tta_classifier,
    train_source_only_tta_model,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PREPARED_ROOT = Path("data/prepared")
OUTPUT_PATH = Path("outputs/phase2/upper_bound.json")
ROOM_SPLITS = [
    {"train": (0, 1), "test": (2,), "name": "room2"},
    {"train": (0, 2), "test": (1,), "name": "room1"},
    {"train": (1, 2), "test": (0,), "name": "room0"},
]


def _target_indices_for_env(
    env_ids: torch.Tensor, test_envs: tuple[int, ...]
) -> torch.Tensor:
    test_tensor = torch.tensor(list(test_envs))
    return torch.nonzero(torch.isin(env_ids, test_tensor), as_tuple=False).squeeze(-1)


def _build_target_loader(
    bundle_subset, batch_size: int, shuffle: bool
) -> DataLoader[dict[str, torch.Tensor]]:
    x = bundle_subset.features
    y = bundle_subset.labels
    priors = bundle_subset.priors  # None or (N, ...) tensor

    def collate(indices: list[int]) -> dict[str, torch.Tensor]:
        idx = torch.tensor(indices, dtype=torch.long)
        bs = len(indices)
        batch: dict[str, torch.Tensor] = {
            "x": x[idx],
            "label": y[idx].long(),
        }
        if priors is None:
            batch["prior"] = torch.zeros_like(x[idx])
            batch["has_prior"] = torch.zeros(bs, dtype=torch.bool)
        else:
            batch["prior"] = priors[idx]
            batch["has_prior"] = torch.ones(bs, dtype=torch.bool)
        return batch

    class IndexDataset(torch.utils.data.Dataset[int]):
        def __len__(self) -> int:
            return int(y.shape[0])

        def __getitem__(self, idx: int) -> int:
            return int(idx)

    return DataLoader(
        IndexDataset(),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate,
    )


def _finetune_on_labelled_target(
    model: torch.nn.Module,
    loader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
    lr: float,
    epochs: int,
) -> None:
    optim = Adam(model.parameters(), lr=lr)
    model.train()
    for _epoch in range(epochs):
        for batch in loader:
            moved = _move_batch_to_device(batch, device)
            optim.zero_grad()
            out = model(
                moved["x"],
                prior=_optional_prior(moved),
                has_prior=moved.get("has_prior"),
            )
            loss = F.cross_entropy(out["task_logits"], moved["label"])
            loss.backward()
            optim.step()


def _run_one_room(
    room: dict[str, object],
    seeds: tuple[int, ...],
    folds: int,
    finetune_epochs: int,
    finetune_lr: float,
    config: TTAExperimentConfig,
) -> list[dict[str, float]]:
    train_envs = room["train"]
    test_envs = room["test"]
    name = str(room["name"])
    rows: list[dict[str, float]] = []

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        # (1) Source-only training on the two training rooms
        src_train_loader, src_val_loader, _, _, input_shape, num_classes, _meta = (
            build_prepared_tta_loaders(
                dataset_name="widar_bvp",
                prepared_root=PREPARED_ROOT,
                seed=seed,
                batch_size=config.batch_size,
                source_val_ratio=config.source_val_ratio,
                target_adapt_ratio=config.target_adapt_ratio,
                train_env_ids=train_envs,
                test_env_ids=test_envs,
            )
        )
        source_model = train_source_only_tta_model(
            source_train_loader=src_train_loader,
            source_val_loader=src_val_loader,
            input_shape=input_shape,
            num_classes=num_classes,
            config=config,
        )
        device = next(source_model.parameters()).device

        # (2) Build target loaders for labelled 5-fold fine-tuning
        bundle = load_prepared_paper1_dataset(
            dataset_name="widar_bvp",
            prepared_root=PREPARED_ROOT,
        )
        assert bundle.environments is not None
        tgt_idx = _target_indices_for_env(bundle.environments, test_envs).tolist()
        rng = np.random.RandomState(seed + 37)
        tgt_idx = np.array(tgt_idx)
        rng.shuffle(tgt_idx)
        fold_chunks = np.array_split(tgt_idx, folds)

        # Source-only baseline on full target (for matched gap)
        full_tgt_subset = bundle.subset(torch.tensor(tgt_idx, dtype=torch.long))
        full_tgt_loader = _build_target_loader(
            full_tgt_subset, batch_size=config.batch_size, shuffle=False
        )
        src_only_metrics = evaluate_tta_classifier(
            source_model, full_tgt_loader, device
        )
        src_only_acc = float(src_only_metrics["accuracy"])

        # (3-4) 5-fold oracle fine-tuning
        fold_accs: list[float] = []
        for k in range(folds):
            val_idx = fold_chunks[k]
            train_idx = np.concatenate([fold_chunks[j] for j in range(folds) if j != k])
            train_subset = bundle.subset(torch.tensor(train_idx, dtype=torch.long))
            val_subset = bundle.subset(torch.tensor(val_idx, dtype=torch.long))
            train_loader = _build_target_loader(
                train_subset, batch_size=config.batch_size, shuffle=True
            )
            val_loader = _build_target_loader(
                val_subset, batch_size=config.batch_size, shuffle=False
            )
            # reset from source checkpoint for each fold
            state = {k: v.clone() for k, v in source_model.state_dict().items()}
            model_k = train_source_only_tta_model(
                source_train_loader=src_train_loader,
                source_val_loader=src_val_loader,
                input_shape=input_shape,
                num_classes=num_classes,
                config=config,
            )
            model_k.load_state_dict(state)
            _finetune_on_labelled_target(
                model_k,
                train_loader,
                device,
                lr=finetune_lr,
                epochs=finetune_epochs,
            )
            val_metrics = evaluate_tta_classifier(model_k, val_loader, device)
            fold_accs.append(float(val_metrics["accuracy"]))

        rows.append(
            {
                "room": name,
                "seed": int(seed),
                "source_only_accuracy": src_only_acc,
                "oracle_fold_accuracies": fold_accs,
                "oracle_mean_accuracy": float(np.mean(fold_accs)),
                "oracle_std_accuracy": float(np.std(fold_accs, ddof=1)),
                "gap_pp": (float(np.mean(fold_accs)) - src_only_acc) * 100.0,
            }
        )
        logger.info(
            "[%s seed=%d] src=%.3f  oracle=%.3f  gap=%.2fpp",
            name,
            seed,
            src_only_acc,
            float(np.mean(fold_accs)),
            (float(np.mean(fold_accs)) - src_only_acc) * 100.0,
        )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=50)
    parser.add_argument("--finetune-lr", type=float, default=1e-3)
    parser.add_argument("--source-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    config = TTAExperimentConfig(
        methods=("no_adapt",),
        backbone="mlp",
        source_epochs=args.source_epochs,
        batch_size=args.batch_size,
        adaptation_steps=0,
    )

    t0 = time.monotonic()
    all_rows: list[dict[str, float]] = []
    for room in ROOM_SPLITS:
        rows = _run_one_room(
            room=room,
            seeds=seeds,
            folds=args.folds,
            finetune_epochs=args.finetune_epochs,
            finetune_lr=args.finetune_lr,
            config=config,
        )
        all_rows.extend(rows)

    src_mean = float(np.mean([r["source_only_accuracy"] for r in all_rows]))
    oracle_mean = float(np.mean([r["oracle_mean_accuracy"] for r in all_rows]))
    gap_mean = (oracle_mean - src_mean) * 100.0
    src_std = float(np.std([r["source_only_accuracy"] for r in all_rows], ddof=1))
    oracle_std = float(np.std([r["oracle_mean_accuracy"] for r in all_rows], ddof=1))

    payload = {
        "protocol": (
            "Widar_BVP leave-one-room-out x 5 seeds; per held-out room, "
            "5-fold CV on labelled target; Adam lr=1e-3, 50 epochs fine-tune "
            "from source checkpoint."
        ),
        "config": asdict(config),
        "folds": args.folds,
        "finetune_epochs": args.finetune_epochs,
        "finetune_lr": args.finetune_lr,
        "rows": all_rows,
        "summary": {
            "source_only_mean": src_mean,
            "source_only_std": src_std,
            "oracle_mean": oracle_mean,
            "oracle_std": oracle_std,
            "supervised_gap_pp": gap_mean,
            "n_room_seed_rows": len(all_rows),
        },
        "elapsed_seconds": time.monotonic() - t0,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 76)
    print("WIDAR_BVP SUPERVISED UPPER BOUND (labelled target 5-fold CV)")
    print("=" * 76)
    print(f"  Source-only accuracy   : {src_mean * 100:.2f}% (std {src_std * 100:.2f})")
    print(
        f"  Oracle fine-tune accuracy: {oracle_mean * 100:.2f}% "
        f"(std {oracle_std * 100:.2f})"
    )
    print(f"  Supervised gap           : {gap_mean:+.2f}pp")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
