"""Export sample-level split manifests for every benchmark fold and seed.

For each prepared dataset this writes ``outputs/splits/<dataset>.json`` with, per
sample index into the prepared arrays, its environment and label, and for each
(held-out environment, seed) the exact index lists the benchmark uses: source
train / source validation, and target adaptation half / target evaluation half.
The indices are computed with the same splitting functions as
``build_prepared_tta_loaders``, so they are the splits behind every reported
number. A SHA-256 of the prepared ``csi.npy`` ties a manifest to its bundle.

The prepared arrays themselves are not redistributed; these manifests let anyone
who rebuilds a bundle check that it matches ours sample for sample.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import torch

from pinn4csi.data.paper1 import PAPER1_DATASET_REGISTRY
from pinn4csi.training.paper2_tta import TTAExperimentConfig, _split_index_tensor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PREPARED_ROOT = Path("data/prepared")
OUTPUT_DIR = Path("outputs/splits")
DATASETS = ("widar_bvp", "ntufi_har", "signfi_top10")
SEEDS = (0, 1, 2, 3, 4)
FOLDS = (((0, 1), (2,)), ((0, 2), (1,)), ((1, 2), (0,)))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest(dataset: str) -> dict[str, object]:
    config = TTAExperimentConfig()
    registry = PAPER1_DATASET_REGISTRY[dataset]
    root = PREPARED_ROOT / registry.directory_name
    environments = torch.from_numpy(np.load(root / registry.environments_filename))
    labels = np.load(root / registry.labels_filename)
    folds: list[dict[str, object]] = []
    for train_envs, test_envs in FOLDS:
        source = torch.nonzero(
            torch.isin(environments, torch.tensor(train_envs)), as_tuple=False
        ).squeeze(-1)
        target = torch.nonzero(
            torch.isin(environments, torch.tensor(test_envs)), as_tuple=False
        ).squeeze(-1)
        for seed in SEEDS:
            train, val = _split_index_tensor(
                source, 1.0 - config.source_val_ratio, seed
            )
            adapt, evaluation = _split_index_tensor(
                target, config.target_adapt_ratio, seed + 17
            )
            folds.append(
                {
                    "train_envs": list(train_envs),
                    "test_env": test_envs[0],
                    "seed": seed,
                    "source_train": train.tolist(),
                    "source_val": val.tolist(),
                    "target_adapt": adapt.tolist(),
                    "target_eval": evaluation.tolist(),
                }
            )
    return {
        "dataset": dataset,
        "csi_sha256": _sha256(root / registry.csi_filename),
        "num_samples": int(environments.numel()),
        "environment": environments.tolist(),
        "label": labels.astype(int).tolist(),
        "source_val_ratio": config.source_val_ratio,
        "target_adapt_ratio": config.target_adapt_ratio,
        "folds": folds,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dataset in DATASETS:
        out = OUTPUT_DIR / f"{dataset}.json"
        out.write_text(json.dumps(manifest(dataset)))
        logger.info("Wrote %s", out)


if __name__ == "__main__":
    main()
