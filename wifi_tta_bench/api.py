"""Public API for WiFi-TTA-Bench."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from pinn4csi.training.paper2_tta import (
    CANONICAL_TTA_METHODS,
    TTAExperimentConfig,
    build_prepared_tta_loaders,
    run_tta_suite,
)
from pinn4csi.utils.metrics import bootstrap_ci, cohens_d

AVAILABLE_DATASETS = ("widar_bvp", "ntufi_har", "signfi_top10")
DEFAULT_SPLITS = {
    "widar_bvp": [((0, 1), (2,)), ((0, 2), (1,)), ((1, 2), (0,))],
    "ntufi_har": [((0, 1), (2,)), ((0, 2), (1,)), ((1, 2), (0,))],
    "signfi_top10": [((0, 1), (2,)), ((0, 2), (1,)), ((1, 2), (0,))],
}


@dataclass
class BenchmarkDataset:
    """Loaded benchmark dataset ready for evaluation."""

    name: str
    prepared_root: Path
    splits: list[tuple[tuple[int, ...], tuple[int, ...]]]


@dataclass
class BenchmarkResult:
    """Result of evaluating one method on one dataset."""

    method: str
    dataset: str
    mean_gain: float
    ci_lower: float
    ci_upper: float
    cohens_d: float
    negative_adaptation_rate: float
    n_observations: int


def list_datasets() -> list[str]:
    """Return available dataset names."""
    return list(AVAILABLE_DATASETS)


def list_methods() -> list[str]:
    """Return available TTA method names."""
    return list(CANONICAL_TTA_METHODS)


def load_dataset(
    name: str,
    prepared_root: str | Path = "data/prepared",
) -> BenchmarkDataset:
    """Load a benchmark dataset by name.

    Args:
        name: Dataset name (widar_bvp, ntufi_har, signfi_top10).
        prepared_root: Path to prepared data directory.

    Returns:
        BenchmarkDataset ready for evaluate().
    """
    if name not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {AVAILABLE_DATASETS}")
    return BenchmarkDataset(
        name=name,
        prepared_root=Path(prepared_root),
        splits=DEFAULT_SPLITS[name],
    )


def evaluate(
    method: str,
    dataset: BenchmarkDataset,
    seeds: int = 5,
    source_epochs: int = 10,
    adaptation_steps: int = 5,
    batch_size: int = 64,
) -> BenchmarkResult:
    """Evaluate a TTA method on a benchmark dataset.

    Args:
        method: TTA method name from list_methods().
        dataset: Loaded dataset from load_dataset().
        seeds: Number of random seeds per fold.
        source_epochs: Source model training epochs.
        adaptation_steps: TTA adaptation steps.
        batch_size: Batch size for training and adaptation.

    Returns:
        BenchmarkResult with gain, CI, Cohen's d, and negative rate.
    """
    if method not in CANONICAL_TTA_METHODS:
        raise ValueError(
            f"Unknown method: {method}. Use list_methods() to see available."
        )

    config = TTAExperimentConfig(
        methods=(method, "no_adapt"),
        source_epochs=source_epochs,
        adaptation_steps=adaptation_steps,
        batch_size=batch_size,
        hidden_dim=64,
        num_layers=3,
        latent_dim=32,
        selective_confidence_threshold=0.2,
        selective_alignment_threshold=2.0,
    )

    gains: list[float] = []
    for train_envs, test_envs in dataset.splits:
        for seed in range(seeds):
            loaders = build_prepared_tta_loaders(
                dataset_name=dataset.name,
                prepared_root=dataset.prepared_root,
                seed=seed,
                batch_size=batch_size,
                source_val_ratio=0.2,
                target_adapt_ratio=0.5,
                train_env_ids=train_envs,
                test_env_ids=test_envs,
            )
            s1, s2, t1, t2, ishape, nc, meta = loaders
            rows = run_tta_suite(s1, s2, t1, t2, ishape, nc, meta, seed, config)
            for r in rows:
                if r.method == method:
                    gains.append(r.gain)

    g = torch.tensor(gains)
    noadapt = torch.zeros_like(g)
    pt, lo, hi = bootstrap_ci(g)
    d = float(cohens_d(g, noadapt))
    nr = float((g < 0).float().mean().item())

    return BenchmarkResult(
        method=method,
        dataset=dataset.name,
        mean_gain=pt,
        ci_lower=lo,
        ci_upper=hi,
        cohens_d=d,
        negative_adaptation_rate=nr,
        n_observations=len(gains),
    )
