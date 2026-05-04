# pyright: basic, reportMissingImports=false

"""Minimal Paper 2 training harness for domain-adaptation baseline comparisons."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from itertools import cycle
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from pinn4csi.data import (
    Paper1DatasetConfig,
    PreparedCSIBundle,
    PreparedCSIDataset,
    get_paper1_dataset_config,
    load_esp32_prepared_dataset,
    load_prepared_paper1_dataset,
    load_wifi6_prepared_dataset,
)
from pinn4csi.models import (
    DomainAdaptationBaselineConfig,
    Paper2DomainAdaptationBaseline,
    create_domain_adaptation_baseline,
    list_domain_adaptation_baselines,
)
from pinn4csi.utils.experiment import save_dataclass_rows_csv
from pinn4csi.utils.metrics import accuracy


@dataclass(frozen=True)
class Paper2BaselineExperimentConfig:
    """Configuration for the minimal Paper 2 baseline suite."""

    baseline_names: tuple[str, ...] = (
        "residual_source_only",
        "coral",
        "dann",
        "maml",
    )
    epochs: int = 5
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    num_layers: int = 3
    latent_dim: int = 32
    invariance_weight: float = 1.0
    physics_weight: float = 0.5
    domain_weight: float = 0.5
    meta_weight: float = 0.5
    meta_inner_lr: float = 0.1
    domain_hidden_dim: int = 32


@dataclass(frozen=True)
class Paper2BaselineResult:
    """Summary row for one trained baseline."""

    baseline_name: str
    best_epoch: int
    train_loss: float
    val_accuracy: float
    val_loss: float
    test_accuracy: float
    test_loss: float


@dataclass(frozen=True)
class Paper2PreparedDataConfig:
    """Prepared-data sources for Paper 2 multi-environment experiments."""

    prepared_root: Path | None = None
    paper1_dataset_names: tuple[str, ...] = ("ut_har",)
    esp32_prepared_dirs: tuple[Path, ...] = ()
    wifi6_prepared_dirs: tuple[Path, ...] = ()
    include_individual_bundles: bool = True
    include_combined_bundle: bool = True


@dataclass(frozen=True)
class Paper2MultiEnvironmentExperimentConfig:
    """Matrix configuration for leave-one-environment-out Paper 2 runs."""

    data: Paper2PreparedDataConfig
    baseline: Paper2BaselineExperimentConfig = field(
        default_factory=Paper2BaselineExperimentConfig
    )
    seeds: tuple[int, ...] = (0,)
    batch_size: int = 32
    num_workers: int = 0
    target_train_ratio: float = 0.5
    target_val_ratio: float = 0.25
    target_test_ratio: float = 0.25


@dataclass(frozen=True)
class Paper2MatrixEntry:
    """One leave-one-environment-out run specification."""

    dataset_name: str
    held_out_environment_id: int
    seed: int
    source_train_examples: int
    target_train_examples: int
    target_val_examples: int
    target_test_examples: int


@dataclass(frozen=True)
class Paper2MultiEnvironmentResultRow:
    """Structured row for Paper 2 table generation."""

    dataset_name: str
    held_out_environment_id: int
    seed: int
    baseline_name: str
    best_epoch: int
    train_loss: float
    val_accuracy: float
    val_loss: float
    test_accuracy: float
    test_loss: float
    source_train_examples: int
    target_train_examples: int
    target_val_examples: int
    target_test_examples: int


@dataclass(frozen=True)
class _NamedPreparedBundle:
    name: str
    bundle: PreparedCSIBundle


def train_domain_adaptation_epoch(
    model: Paper2DomainAdaptationBaseline,
    optimizer: Adam,
    source_loader: DataLoader[dict[str, Tensor]],
    target_loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> dict[str, float]:
    """Train one epoch with paired source/target batches."""
    model.train()
    totals = {
        "loss_total": 0.0,
        "loss_task": 0.0,
        "loss_invariance": 0.0,
        "loss_physics_residual": 0.0,
        "loss_domain": 0.0,
        "loss_meta": 0.0,
    }
    total_examples = 0
    target_batches = cycle(target_loader)

    for source_batch in source_loader:
        target_batch = next(target_batches)
        moved_source = _move_batch_to_device(source_batch, device)
        moved_target = _move_batch_to_device(target_batch, device)

        optimizer.zero_grad()
        losses = model.compute_batch_losses(
            source_x=moved_source["x"],
            source_labels=moved_source["label"],
            target_x=moved_target["x"],
            source_prior=_optional_prior(moved_source),
            target_prior=_optional_prior(moved_target),
            source_has_prior=moved_source["has_prior"],
            target_has_prior=moved_target["has_prior"],
        )
        losses["loss_total"].backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        batch_size = int(moved_source["label"].shape[0])
        for key in totals:
            totals[key] += float(losses[key].item()) * batch_size
        total_examples += batch_size

    return {key: value / max(total_examples, 1) for key, value in totals.items()}


def evaluate_domain_adaptation_baseline(
    model: Paper2DomainAdaptationBaseline,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> dict[str, float | int]:
    """Evaluate a baseline on a labeled split."""
    model.eval()
    total_loss = 0.0
    total_examples = 0
    logits_batches: list[Tensor] = []
    label_batches: list[Tensor] = []

    with torch.no_grad():
        for batch in loader:
            moved = _move_batch_to_device(batch, device)
            logits = model(
                moved["x"],
                prior=_optional_prior(moved),
                has_prior=moved["has_prior"],
            )["task_logits"]
            batch_size = int(moved["label"].shape[0])
            total_loss += (
                float(torch.nn.functional.cross_entropy(logits, moved["label"]).item())
                * batch_size
            )
            total_examples += batch_size
            logits_batches.append(logits.cpu())
            label_batches.append(moved["label"].cpu())

    logits_tensor = torch.cat(logits_batches, dim=0)
    labels_tensor = torch.cat(label_batches, dim=0)
    return {
        "accuracy": accuracy(logits_tensor, labels_tensor),
        "loss_total": total_loss / max(total_examples, 1),
        "num_examples": total_examples,
    }


def run_domain_adaptation_baselines(
    source_train_loader: DataLoader[dict[str, Tensor]],
    target_train_loader: DataLoader[dict[str, Tensor]],
    val_loader: DataLoader[dict[str, Tensor]],
    test_loader: DataLoader[dict[str, Tensor]],
    input_shape: tuple[int, ...],
    num_classes: int,
    config: Paper2BaselineExperimentConfig | None = None,
) -> list[Paper2BaselineResult]:
    """Train and evaluate the configured Paper 2 baseline suite."""
    experiment_config = config or Paper2BaselineExperimentConfig()
    _validate_baseline_names(experiment_config.baseline_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_config = DomainAdaptationBaselineConfig(
        hidden_dim=experiment_config.hidden_dim,
        num_layers=experiment_config.num_layers,
        latent_dim=experiment_config.latent_dim,
        invariance_weight=experiment_config.invariance_weight,
        physics_weight=experiment_config.physics_weight,
        domain_weight=experiment_config.domain_weight,
        meta_weight=experiment_config.meta_weight,
        meta_inner_lr=experiment_config.meta_inner_lr,
        domain_hidden_dim=experiment_config.domain_hidden_dim,
    )

    results: list[Paper2BaselineResult] = []
    for baseline_name in experiment_config.baseline_names:
        model = create_domain_adaptation_baseline(
            baseline_name=baseline_name,
            input_shape=input_shape,
            num_classes=num_classes,
            config=model_config,
        ).to(device)
        optimizer = Adam(model.parameters(), lr=experiment_config.learning_rate)

        best_epoch = -1
        best_accuracy = -1.0
        best_loss = float("inf")
        best_train_loss = float("inf")
        best_state: dict[str, Tensor] | None = None

        for epoch_idx in range(experiment_config.epochs):
            train_metrics = train_domain_adaptation_epoch(
                model=model,
                optimizer=optimizer,
                source_loader=source_train_loader,
                target_loader=target_train_loader,
                device=device,
            )
            val_metrics = evaluate_domain_adaptation_baseline(model, val_loader, device)
            val_accuracy = float(val_metrics["accuracy"])
            val_loss = float(val_metrics["loss_total"])
            if val_accuracy > best_accuracy or (
                np.isclose(val_accuracy, best_accuracy) and val_loss < best_loss
            ):
                best_epoch = epoch_idx
                best_accuracy = val_accuracy
                best_loss = val_loss
                best_train_loss = float(train_metrics["loss_total"])
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

        if best_state is None:
            raise RuntimeError(
                f"No validation checkpoint selected for baseline {baseline_name}."
            )

        model.load_state_dict(best_state)
        test_metrics = evaluate_domain_adaptation_baseline(model, test_loader, device)
        results.append(
            Paper2BaselineResult(
                baseline_name=baseline_name,
                best_epoch=best_epoch,
                train_loss=best_train_loss,
                val_accuracy=float(best_accuracy),
                val_loss=float(best_loss),
                test_accuracy=float(test_metrics["accuracy"]),
                test_loss=float(test_metrics["loss_total"]),
            )
        )

    return results


def build_paper2_leave_one_environment_out_matrix(
    config: Paper2MultiEnvironmentExperimentConfig,
) -> list[Paper2MatrixEntry]:
    bundles = _load_available_named_bundles(config.data)
    matrix_entries: list[Paper2MatrixEntry] = []

    for named_bundle in bundles:
        matrix_entries.extend(
            _build_matrix_entries_for_bundle(
                named_bundle=named_bundle,
                seeds=config.seeds,
                target_train_ratio=config.target_train_ratio,
                target_val_ratio=config.target_val_ratio,
                target_test_ratio=config.target_test_ratio,
            )
        )
    return matrix_entries


def run_paper2_leave_one_environment_out_matrix(
    config: Paper2MultiEnvironmentExperimentConfig,
) -> list[Paper2MultiEnvironmentResultRow]:
    _validate_target_ratios(
        config.target_train_ratio,
        config.target_val_ratio,
        config.target_test_ratio,
    )
    named_bundles = _load_available_named_bundles(config.data)
    device_matrix: list[Paper2MultiEnvironmentResultRow] = []

    for named_bundle in named_bundles:
        if named_bundle.bundle.environments is None:
            continue
        environment_ids = sorted(
            int(value)
            for value in torch.unique(named_bundle.bundle.environments).tolist()
        )

        for held_out_environment_id in environment_ids:
            for seed in config.seeds:
                split_indices = _build_leave_one_environment_out_split(
                    bundle=named_bundle.bundle,
                    held_out_environment_id=held_out_environment_id,
                    seed=seed,
                    target_train_ratio=config.target_train_ratio,
                    target_val_ratio=config.target_val_ratio,
                    target_test_ratio=config.target_test_ratio,
                )
                source_train_loader = _build_bundle_loader(
                    bundle=named_bundle.bundle,
                    indices=split_indices["source_train"],
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=config.num_workers,
                )
                target_train_loader = _build_bundle_loader(
                    bundle=named_bundle.bundle,
                    indices=split_indices["target_train"],
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=config.num_workers,
                )
                val_loader = _build_bundle_loader(
                    bundle=named_bundle.bundle,
                    indices=split_indices["target_val"],
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=config.num_workers,
                )
                test_loader = _build_bundle_loader(
                    bundle=named_bundle.bundle,
                    indices=split_indices["target_test"],
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=config.num_workers,
                )

                baseline_results = run_domain_adaptation_baselines(
                    source_train_loader=source_train_loader,
                    target_train_loader=target_train_loader,
                    val_loader=val_loader,
                    test_loader=test_loader,
                    input_shape=named_bundle.bundle.input_shape,
                    num_classes=named_bundle.bundle.num_classes,
                    config=config.baseline,
                )

                for baseline_result in baseline_results:
                    device_matrix.append(
                        Paper2MultiEnvironmentResultRow(
                            dataset_name=named_bundle.name,
                            held_out_environment_id=held_out_environment_id,
                            seed=seed,
                            baseline_name=baseline_result.baseline_name,
                            best_epoch=baseline_result.best_epoch,
                            train_loss=baseline_result.train_loss,
                            val_accuracy=baseline_result.val_accuracy,
                            val_loss=baseline_result.val_loss,
                            test_accuracy=baseline_result.test_accuracy,
                            test_loss=baseline_result.test_loss,
                            source_train_examples=int(
                                split_indices["source_train"].numel()
                            ),
                            target_train_examples=int(
                                split_indices["target_train"].numel()
                            ),
                            target_val_examples=int(
                                split_indices["target_val"].numel()
                            ),
                            target_test_examples=int(
                                split_indices["target_test"].numel()
                            ),
                        )
                    )
    return device_matrix


def save_paper2_multi_environment_results_csv(
    results: list[Paper2MultiEnvironmentResultRow],
    output_csv: str | Path,
) -> None:
    save_dataclass_rows_csv(
        results,
        output_csv,
        fieldnames=[field.name for field in fields(Paper2MultiEnvironmentResultRow)],
    )


def _move_batch_to_device(
    batch: dict[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _optional_prior(batch: dict[str, Tensor]) -> Tensor | None:
    if not bool(torch.any(batch["has_prior"]).item()):
        return None
    return batch["prior"]


def _validate_baseline_names(baseline_names: tuple[str, ...]) -> None:
    valid = set(list_domain_adaptation_baselines())
    for name in baseline_names:
        if name not in valid:
            raise ValueError(
                f"Unsupported baseline in experiment config: {name}. "
                f"Expected one of {sorted(valid)}."
            )


def _build_matrix_entries_for_bundle(
    named_bundle: _NamedPreparedBundle,
    seeds: tuple[int, ...],
    target_train_ratio: float,
    target_val_ratio: float,
    target_test_ratio: float,
) -> list[Paper2MatrixEntry]:
    environments = named_bundle.bundle.environments
    if environments is None:
        return []
    entries: list[Paper2MatrixEntry] = []
    for held_out_environment_id in sorted(
        int(value) for value in torch.unique(environments).tolist()
    ):
        for seed in seeds:
            split_indices = _build_leave_one_environment_out_split(
                bundle=named_bundle.bundle,
                held_out_environment_id=held_out_environment_id,
                seed=seed,
                target_train_ratio=target_train_ratio,
                target_val_ratio=target_val_ratio,
                target_test_ratio=target_test_ratio,
            )
            entries.append(
                Paper2MatrixEntry(
                    dataset_name=named_bundle.name,
                    held_out_environment_id=held_out_environment_id,
                    seed=seed,
                    source_train_examples=int(split_indices["source_train"].numel()),
                    target_train_examples=int(split_indices["target_train"].numel()),
                    target_val_examples=int(split_indices["target_val"].numel()),
                    target_test_examples=int(split_indices["target_test"].numel()),
                )
            )
    return entries


def _load_available_named_bundles(
    config: Paper2PreparedDataConfig,
) -> list[_NamedPreparedBundle]:
    bundles: list[_NamedPreparedBundle] = []
    prepared_root = config.prepared_root

    if prepared_root is not None:
        for dataset_name in config.paper1_dataset_names:
            dataset_dir = (
                prepared_root / get_paper1_dataset_config(dataset_name).directory_name
            )
            if not dataset_dir.exists():
                continue
            bundle = load_prepared_paper1_dataset(
                dataset_name=dataset_name,
                prepared_root=prepared_root,
            )
            bundles.append(_NamedPreparedBundle(name=bundle.config.name, bundle=bundle))

    for prepared_dir in config.esp32_prepared_dirs:
        if not prepared_dir.exists():
            continue
        loaded_esp32 = load_esp32_prepared_dataset(prepared_dir)
        bundle = _convert_collection_bundle(
            loaded_esp32.features,
            loaded_esp32.labels,
            loaded_esp32.environments,
            dataset_name=f"ESP32:{loaded_esp32.metadata.capture_id}",
        )
        bundles.append(_NamedPreparedBundle(name=bundle.config.name, bundle=bundle))

    for prepared_dir in config.wifi6_prepared_dirs:
        if not prepared_dir.exists():
            continue
        loaded_wifi6 = load_wifi6_prepared_dataset(prepared_dir)
        bundle = _convert_collection_bundle(
            loaded_wifi6.features,
            loaded_wifi6.labels,
            loaded_wifi6.environments,
            dataset_name=f"WiFi6:{loaded_wifi6.metadata.capture_id}",
        )
        bundles.append(_NamedPreparedBundle(name=bundle.config.name, bundle=bundle))

    env_bundles = [
        named
        for named in bundles
        if named.bundle.environments is not None
        and int(torch.unique(named.bundle.environments).numel()) >= 2
    ]
    if not env_bundles:
        return []

    selected: list[_NamedPreparedBundle] = []
    if config.include_individual_bundles:
        selected.extend(env_bundles)
    if config.include_combined_bundle and len(env_bundles) >= 2:
        combined = _try_combine_named_bundles(env_bundles)
        if combined is not None:
            selected.append(combined)
    return selected


def _convert_collection_bundle(
    features: Tensor,
    labels: Tensor,
    environments: Tensor,
    dataset_name: str,
) -> PreparedCSIBundle:
    config = Paper1DatasetConfig(
        name=dataset_name,
        directory_name=dataset_name.lower().replace(":", "_"),
        evaluation_mode="cross_environment",
        description=(
            "Prepared self-collected bundle normalized for Paper 2 experiments."
        ),
    )
    return PreparedCSIBundle(
        config=config,
        features=features.to(torch.float32),
        labels=labels.to(torch.long),
        environments=environments.to(torch.long),
        priors=None,
    )


def _try_combine_named_bundles(
    bundles: list[_NamedPreparedBundle],
) -> _NamedPreparedBundle | None:
    compatible = [bundles[0]]
    base_shape = bundles[0].bundle.input_shape
    base_classes = bundles[0].bundle.num_classes
    for named_bundle in bundles[1:]:
        if named_bundle.bundle.input_shape != base_shape:
            continue
        if named_bundle.bundle.num_classes != base_classes:
            continue
        compatible.append(named_bundle)

    if len(compatible) < 2:
        return None

    feature_batches: list[Tensor] = []
    label_batches: list[Tensor] = []
    env_batches: list[Tensor] = []
    offset = 0
    for named_bundle in compatible:
        bundle = named_bundle.bundle
        if bundle.environments is None:
            continue
        unique_envs = torch.unique(bundle.environments)
        remapped = torch.zeros_like(bundle.environments)
        for local_env_id in unique_envs.tolist():
            remapped[bundle.environments == local_env_id] = offset + int(local_env_id)
        offset = int(torch.max(remapped).item()) + 1

        feature_batches.append(bundle.features)
        label_batches.append(bundle.labels)
        env_batches.append(remapped)

    if not feature_batches:
        return None

    combined_name = "+".join(named.name for named in compatible)
    combined_config = Paper1DatasetConfig(
        name=f"combined:{combined_name}",
        directory_name="combined",
        evaluation_mode="cross_environment",
        description="Combined prepared bundles for multi-environment Paper 2 runs.",
    )
    combined_bundle = PreparedCSIBundle(
        config=combined_config,
        features=torch.cat(feature_batches, dim=0),
        labels=torch.cat(label_batches, dim=0),
        environments=torch.cat(env_batches, dim=0),
        priors=None,
    )
    return _NamedPreparedBundle(name=combined_config.name, bundle=combined_bundle)


def _validate_target_ratios(
    target_train_ratio: float,
    target_val_ratio: float,
    target_test_ratio: float,
) -> None:
    if min(target_train_ratio, target_val_ratio, target_test_ratio) <= 0:
        raise ValueError("Target split ratios must be positive.")
    if abs(target_train_ratio + target_val_ratio + target_test_ratio - 1.0) > 1e-6:
        raise ValueError("Target split ratios must sum to 1.0.")


def _build_leave_one_environment_out_split(
    bundle: PreparedCSIBundle,
    held_out_environment_id: int,
    seed: int,
    target_train_ratio: float,
    target_val_ratio: float,
    target_test_ratio: float,
) -> dict[str, Tensor]:
    environments = bundle.environments
    if environments is None:
        raise ValueError(
            "Bundle must include environments for leave-one-out splitting."
        )

    target_mask = environments == held_out_environment_id
    source_mask = ~target_mask
    source_indices = torch.nonzero(source_mask, as_tuple=False).squeeze(-1)
    target_indices = torch.nonzero(target_mask, as_tuple=False).squeeze(-1)
    if source_indices.numel() == 0:
        raise ValueError(
            "No source samples available after holding out env "
            f"{held_out_environment_id}."
        )
    if target_indices.numel() < 3:
        raise ValueError(
            "Need at least 3 target samples in held-out environment for train/val/test."
        )

    target_generator = torch.Generator().manual_seed(seed)
    shuffled_target = target_indices[
        torch.randperm(target_indices.numel(), generator=target_generator)
    ]
    train_count, val_count, test_count = _resolve_target_split_counts(
        num_samples=int(shuffled_target.numel()),
        target_train_ratio=target_train_ratio,
        target_val_ratio=target_val_ratio,
        target_test_ratio=target_test_ratio,
    )

    source_generator = torch.Generator().manual_seed(seed + 911)
    shuffled_source = source_indices[
        torch.randperm(source_indices.numel(), generator=source_generator)
    ]
    return {
        "source_train": shuffled_source,
        "target_train": shuffled_target[:train_count],
        "target_val": shuffled_target[train_count : train_count + val_count],
        "target_test": shuffled_target[
            train_count + val_count : train_count + val_count + test_count
        ],
    }


def _resolve_target_split_counts(
    num_samples: int,
    target_train_ratio: float,
    target_val_ratio: float,
    target_test_ratio: float,
) -> tuple[int, int, int]:
    total_ratio = target_train_ratio + target_val_ratio + target_test_ratio
    train_count = max(1, int(num_samples * (target_train_ratio / total_ratio)))
    val_count = max(1, int(num_samples * (target_val_ratio / total_ratio)))
    test_count = max(1, int(num_samples * (target_test_ratio / total_ratio)))

    if test_count <= 0:
        test_count = 1
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        else:
            val_count -= 1

    while train_count + val_count + test_count > num_samples:
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            test_count -= 1

    while train_count + val_count + test_count < num_samples:
        test_count += 1
    return train_count, val_count, test_count


def _build_bundle_loader(
    bundle: PreparedCSIBundle,
    indices: Tensor,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[dict[str, Tensor]]:
    subset = bundle.subset(indices)
    dataset = PreparedCSIDataset(subset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )
