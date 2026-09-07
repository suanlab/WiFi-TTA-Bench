# pyright: basic, reportMissingImports=false

"""Audit benchmark splits for duplicate and environment leakage.

Contrasts the benchmark's leave-one-environment-out (LOEO) protocol against a
naive random split -- the partitioning practice that has been shown to leak in
prior WiFi CSI activity-recognition benchmarks. A clean protocol should report
near-zero duplicate leakage and exactly zero environment leakage, while a random
split over overlapping windows inflates both.

Output: outputs/leakage_audit/split_leakage.json (+ .csv)
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor

from pinn4csi.data.paper1 import load_prepared_paper1_dataset
from pinn4csi.utils.experiment import save_dataclass_rows_csv
from pinn4csi.utils.metrics import split_leakage_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PREPARED_ROOT = Path("data/prepared")
OUTPUT_PATH = Path("outputs/leakage_audit/split_leakage.json")

DEFAULT_DATASETS = ("widar_bvp", "ntufi_har", "signfi_top10")
ROOM_SPLITS = [
    {"train": (0, 1), "test": (2,)},
    {"train": (0, 2), "test": (1,)},
    {"train": (1, 2), "test": (0,)},
]


@dataclass
class LeakageRow:
    """One audited split."""

    dataset: str
    protocol: str
    split_name: str
    duplicate_leakage_rate: float
    environment_leakage_rate: float
    min_nn_distance: float
    median_nn_distance: float
    feature_scale: float
    num_train: int
    num_test: int


def _subsample(indices: Tensor, limit: int, seed: int) -> Tensor:
    """Cap an index tensor at `limit` entries with a fixed permutation."""
    if indices.numel() <= limit:
        return indices
    generator = torch.Generator().manual_seed(seed)
    choice = torch.randperm(indices.numel(), generator=generator)[:limit]
    return indices[choice]


def _audit(
    features: Tensor,
    environments: Tensor,
    train_idx: Tensor,
    test_idx: Tensor,
    dataset: str,
    protocol: str,
    split_name: str,
    relative_threshold: float,
) -> LeakageRow:
    report = split_leakage_report(
        train_features=features[train_idx],
        test_features=features[test_idx],
        train_environments=environments[train_idx],
        test_environments=environments[test_idx],
        relative_threshold=relative_threshold,
    )
    return LeakageRow(
        dataset=dataset,
        protocol=protocol,
        split_name=split_name,
        duplicate_leakage_rate=report["duplicate_leakage_rate"],
        environment_leakage_rate=report["environment_leakage_rate"],
        min_nn_distance=report["min_nn_distance"],
        median_nn_distance=report["median_nn_distance"],
        feature_scale=report["feature_scale"],
        num_train=int(train_idx.numel()),
        num_test=int(test_idx.numel()),
    )


def audit_dataset(
    dataset: str,
    prepared_root: Path,
    max_samples: int,
    relative_threshold: float,
    seed: int,
) -> list[LeakageRow]:
    """Audit LOEO and random splits for one dataset."""
    bundle = load_prepared_paper1_dataset(
        dataset_name=dataset, prepared_root=prepared_root
    )
    if bundle.environments is None:
        raise ValueError(f"Dataset {dataset} has no environment IDs to audit.")
    features = bundle.features
    environments = bundle.environments
    rows: list[LeakageRow] = []

    for split in ROOM_SPLITS:
        train_envs = torch.tensor(split["train"])
        test_envs = torch.tensor(split["test"])
        train_idx = torch.nonzero(
            torch.isin(environments, train_envs), as_tuple=False
        ).squeeze(-1)
        test_idx = torch.nonzero(
            torch.isin(environments, test_envs), as_tuple=False
        ).squeeze(-1)
        if train_idx.numel() == 0 or test_idx.numel() == 0:
            logger.warning("[%s] skipping empty split %s", dataset, split)
            continue
        held_out = int(split["test"][0])
        rows.append(
            _audit(
                features=features,
                environments=environments,
                train_idx=_subsample(train_idx, max_samples, seed),
                test_idx=_subsample(test_idx, max_samples, seed + 1),
                dataset=dataset,
                protocol="leave_one_environment_out",
                split_name=f"held_out_env{held_out}",
                relative_threshold=relative_threshold,
            )
        )

        # Naive random split at the same train/test ratio, ignoring environments.
        total = features.shape[0]
        ratio = train_idx.numel() / (train_idx.numel() + test_idx.numel())
        generator = torch.Generator().manual_seed(seed + held_out)
        shuffled = torch.randperm(total, generator=generator)
        cut = int(total * ratio)
        rows.append(
            _audit(
                features=features,
                environments=environments,
                train_idx=_subsample(shuffled[:cut], max_samples, seed),
                test_idx=_subsample(shuffled[cut:], max_samples, seed + 1),
                dataset=dataset,
                protocol="random_split",
                split_name=f"random_match_env{held_out}",
                relative_threshold=relative_threshold,
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated dataset names.",
    )
    parser.add_argument("--prepared-root", type=Path, default=PREPARED_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=1500,
        help="Cap per split; nearest-neighbour search is quadratic.",
    )
    parser.add_argument(
        "--relative-threshold",
        type=float,
        default=0.01,
        help="Duplicate cutoff as a fraction of feature scale.",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    datasets = tuple(d.strip() for d in args.datasets.split(",") if d.strip())
    rows: list[LeakageRow] = []
    for dataset in datasets:
        logger.info("auditing %s", dataset)
        rows.extend(
            audit_dataset(
                dataset=dataset,
                prepared_root=args.prepared_root,
                max_samples=args.max_samples,
                relative_threshold=args.relative_threshold,
                seed=args.seed,
            )
        )

    summary: dict[str, dict[str, float]] = {}
    for dataset in datasets:
        for protocol in ("leave_one_environment_out", "random_split"):
            matching = [
                r for r in rows if r.dataset == dataset and r.protocol == protocol
            ]
            if not matching:
                continue
            n = len(matching)
            summary[f"{dataset}/{protocol}"] = {
                "mean_duplicate_leakage_rate": sum(
                    r.duplicate_leakage_rate for r in matching
                )
                / n,
                "mean_environment_leakage_rate": sum(
                    r.environment_leakage_rate for r in matching
                )
                / n,
                "n_splits": float(n),
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "datasets": list(datasets),
            "max_samples": args.max_samples,
            "relative_threshold": args.relative_threshold,
            "seed": args.seed,
        },
        "rows": [asdict(r) for r in rows],
        "summary": summary,
    }
    args.output.write_text(json.dumps(payload, indent=2))
    save_dataclass_rows_csv(rows, args.output.with_suffix(".csv"))
    logger.info("wrote %s", args.output)

    for key, value in summary.items():
        logger.info(
            "%s: duplicate=%.4f environment=%.4f",
            key,
            value["mean_duplicate_leakage_rate"],
            value["mean_environment_leakage_rate"],
        )


if __name__ == "__main__":
    main()
