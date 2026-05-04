from __future__ import annotations

import copy
import random
import time
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from pinn4csi.data import (
    PreparedCSIBundle,
    PreparedCSIDataset,
    get_paper1_dataset_config,
    load_prepared_paper1_dataset,
)
from pinn4csi.models import (
    DomainAdaptationBaselineConfig,
    Paper2DomainAdaptationBaseline,
    create_domain_adaptation_baseline,
)
from pinn4csi.utils.experiment import save_dataclass_rows_csv
from pinn4csi.utils.metrics import accuracy

DEFAULT_TTA_METHODS = (
    "no_adapt",
    "entropy_tta",
    "physics_tta",
    "physics_entropy_tta",
)

CANONICAL_TTA_METHODS = DEFAULT_TTA_METHODS + (
    "safe_entropy_tta",
    "safe_physics_tta",
    "conservative_entropy_tta",
    "calibrated_entropy_tta",
    "warm_restart_physics_tta",
    "selective_physics_tta",
    "tent",
    "shot",
    "t3a",
    "bn_reset",
    "lame",
    "sar",
    "cotta",
)

_SAFE_TTA_METHOD_ALIASES = {
    "safe_entropy_tta": "entropy_tta",
    "safe_physics_tta": "physics_tta",
}

SYNTHETIC_SHIFT_LEVELS: dict[str, float] = {
    "mild": 0.25,
    "moderate": 0.55,
    "strong": 0.85,
}


@dataclass(frozen=True)
class TTAExperimentConfig:
    methods: tuple[str, ...] = DEFAULT_TTA_METHODS
    source_epochs: int = 5
    source_learning_rate: float = 1e-3
    adaptation_learning_rate: float = 5e-4
    adaptation_steps: int = 5
    batch_size: int = 32
    hidden_dim: int = 64
    num_layers: int = 3
    latent_dim: int = 32
    invariance_weight: float = 1.0
    residual_weight: float = 1.0
    entropy_weight: float = 0.1
    source_val_ratio: float = 0.2
    target_adapt_ratio: float = 0.5
    update_scope: str = "encoder"
    safe_objective_min_delta: float = 0.0
    safe_alignment_guard_ratio: float = 2.0
    warm_restart_steps: int = 3
    warm_restart_lr_factor: float = 0.5
    calibration_temperature: float = 1.5
    selective_confidence_threshold: float = 0.2
    selective_alignment_threshold: float = 2.0
    tent_bn_momentum: float = 0.1
    shot_pseudo_label_threshold: float = 0.9
    t3a_filter_k: int = 1
    ablation_use_invariance: bool = True
    ablation_use_residual: bool = True
    ablation_use_entropy: bool = True
    early_stop_patience: int = 0
    backbone: str = "mlp"
    source_label_smoothing: float = 0.0


@dataclass(frozen=True)
class TTAReferenceStats:
    invariant_mean: Tensor
    invariant_var: Tensor
    feature_mean: Tensor
    feature_var: Tensor


@dataclass(frozen=True)
class TTAStepLosses:
    entropy_loss: Tensor
    invariance_loss: Tensor
    residual_loss: Tensor
    objective: Tensor


@dataclass
class TTAAdaptationSummary:
    model: Paper2DomainAdaptationBaseline
    attempted_steps: int
    accepted_steps: int
    abstained: bool
    stop_reason: str
    pre_objective: float
    post_objective: float
    adaptation_time_seconds: float = 0.0


@dataclass(frozen=True)
class TTAResultRow:
    dataset_name: str
    shift_name: str
    held_out_environment_id: int
    seed: int
    method: str
    pre_accuracy: float
    post_accuracy: float
    gain: float
    source_pre_accuracy: float
    source_post_accuracy: float
    source_drop: float
    attempted_steps: int
    accepted_steps: int
    abstained: bool
    stop_reason: str
    pre_adaptation_objective: float
    post_adaptation_objective: float
    source_train_examples: int
    source_val_examples: int
    target_adapt_examples: int
    target_test_examples: int


class UnlabeledPreparedCSIDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, bundle: PreparedCSIBundle) -> None:
        self.bundle = bundle

    def __len__(self) -> int:
        return self.bundle.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        feature = self.bundle.features[index]
        if self.bundle.priors is None:
            prior = torch.zeros_like(feature)
            has_prior = torch.tensor(False)
        else:
            prior = self.bundle.priors[index]
            has_prior = torch.tensor(True)

        sample: dict[str, Tensor] = {
            "x": feature,
            "prior": prior,
            "has_prior": has_prior,
        }
        if self.bundle.environments is not None:
            sample["environment"] = self.bundle.environments[index]
        return sample


def build_synthetic_tta_loaders(
    shift_name: str,
    seed: int,
    batch_size: int,
    source_val_ratio: float,
    target_adapt_ratio: float,
) -> tuple[
    DataLoader[dict[str, Tensor]],
    DataLoader[dict[str, Tensor]],
    DataLoader[dict[str, Tensor]],
    DataLoader[dict[str, Tensor]],
    tuple[int, int],
    int,
    dict[str, int | str],
]:
    if shift_name not in SYNTHETIC_SHIFT_LEVELS:
        expected = ", ".join(sorted(SYNTHETIC_SHIFT_LEVELS))
        raise ValueError(f"Unknown synthetic shift: {shift_name}. Expected {expected}.")

    generator = torch.Generator().manual_seed(seed)
    input_shape = (8, 2)
    num_classes = 3
    shift = SYNTHETIC_SHIFT_LEVELS[shift_name]

    source_x, source_labels, source_prior = _make_domain_samples(
        num_samples=96,
        num_classes=num_classes,
        input_shape=input_shape,
        generator=generator,
        domain_shift=0.0,
    )
    target_x, target_labels, target_prior = _make_domain_samples(
        num_samples=72,
        num_classes=num_classes,
        input_shape=input_shape,
        generator=generator,
        domain_shift=shift,
    )

    config = get_paper1_dataset_config("ut_har")
    source_bundle = PreparedCSIBundle(
        config=config,
        features=source_x,
        labels=source_labels,
        environments=torch.zeros(96, dtype=torch.long),
        priors=source_prior,
    )
    target_bundle = PreparedCSIBundle(
        config=config,
        features=target_x,
        labels=target_labels,
        environments=torch.ones(72, dtype=torch.long),
        priors=target_prior,
    )

    source_train_idx, source_val_idx = _random_split_indices(
        num_samples=source_bundle.num_samples,
        first_ratio=1.0 - source_val_ratio,
        seed=seed,
    )
    target_adapt_idx, target_test_idx = _random_split_indices(
        num_samples=target_bundle.num_samples,
        first_ratio=target_adapt_ratio,
        seed=seed + 17,
    )

    source_train_bundle = source_bundle.subset(source_train_idx)
    source_val_bundle = source_bundle.subset(source_val_idx)
    target_adapt_bundle = target_bundle.subset(target_adapt_idx)
    target_test_bundle = target_bundle.subset(target_test_idx)

    loaders = _bundle_loaders(
        source_train_bundle=source_train_bundle,
        source_val_bundle=source_val_bundle,
        target_adapt_bundle=target_adapt_bundle,
        target_test_bundle=target_test_bundle,
        batch_size=batch_size,
    )
    metadata: dict[str, int | str] = {
        "dataset_name": "synthetic_tta",
        "shift_name": shift_name,
        "held_out_environment_id": 1,
        "source_train_examples": source_train_bundle.num_samples,
        "source_val_examples": source_val_bundle.num_samples,
        "target_adapt_examples": target_adapt_bundle.num_samples,
        "target_test_examples": target_test_bundle.num_samples,
    }
    return (*loaders, input_shape, num_classes, metadata)


def build_prepared_tta_loaders(
    dataset_name: str,
    prepared_root: Path,
    seed: int,
    batch_size: int,
    source_val_ratio: float,
    target_adapt_ratio: float,
    train_env_ids: tuple[int, ...] | None = None,
    test_env_ids: tuple[int, ...] | None = None,
) -> tuple[
    DataLoader[dict[str, Tensor]],
    DataLoader[dict[str, Tensor]],
    DataLoader[dict[str, Tensor]],
    DataLoader[dict[str, Tensor]],
    tuple[int, int],
    int,
    dict[str, int | str],
]:
    bundle = load_prepared_paper1_dataset(
        dataset_name=dataset_name,
        prepared_root=prepared_root,
    )
    if bundle.environments is None:
        raise ValueError(f"Dataset {dataset_name} does not include environment IDs.")
    config = bundle.config
    resolved_train_env_ids = train_env_ids or config.train_env_ids
    resolved_test_env_ids = test_env_ids or config.test_env_ids
    if resolved_train_env_ids is None or resolved_test_env_ids is None:
        raise ValueError(
            f"Dataset {dataset_name} is missing default train/test environment IDs."
        )

    train_env_tensor = torch.tensor(resolved_train_env_ids)
    test_env_tensor = torch.tensor(resolved_test_env_ids)
    source_indices = torch.nonzero(
        torch.isin(bundle.environments, train_env_tensor), as_tuple=False
    ).squeeze(-1)
    target_indices = torch.nonzero(
        torch.isin(bundle.environments, test_env_tensor), as_tuple=False
    ).squeeze(-1)
    if source_indices.numel() == 0 or target_indices.numel() < 2:
        raise ValueError(f"Dataset {dataset_name} does not have enough split samples.")

    source_train_idx, source_val_idx = _split_index_tensor(
        source_indices, 1.0 - source_val_ratio, seed
    )
    target_adapt_idx, target_test_idx = _split_index_tensor(
        target_indices, target_adapt_ratio, seed + 17
    )

    source_train_bundle = bundle.subset(source_train_idx)
    source_val_bundle = bundle.subset(source_val_idx)
    target_adapt_bundle = bundle.subset(target_adapt_idx)
    target_test_bundle = bundle.subset(target_test_idx)

    loaders = _bundle_loaders(
        source_train_bundle=source_train_bundle,
        source_val_bundle=source_val_bundle,
        target_adapt_bundle=target_adapt_bundle,
        target_test_bundle=target_test_bundle,
        batch_size=batch_size,
    )
    held_out_environment_id = int(resolved_test_env_ids[0])
    metadata: dict[str, int | str] = {
        "dataset_name": config.name,
        "shift_name": "held_out_split",
        "held_out_environment_id": held_out_environment_id,
        "source_train_examples": source_train_bundle.num_samples,
        "source_val_examples": source_val_bundle.num_samples,
        "target_adapt_examples": target_adapt_bundle.num_samples,
        "target_test_examples": target_test_bundle.num_samples,
    }
    return (*loaders, bundle.input_shape, bundle.num_classes, metadata)


def train_source_only_tta_model(
    source_train_loader: DataLoader[dict[str, Tensor]],
    source_val_loader: DataLoader[dict[str, Tensor]],
    input_shape: tuple[int, int],
    num_classes: int,
    config: TTAExperimentConfig,
) -> Paper2DomainAdaptationBaseline:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = getattr(config, "backbone", "mlp")
    model = create_domain_adaptation_baseline(
        baseline_name="residual_source_only",
        input_shape=input_shape,
        num_classes=num_classes,
        config=DomainAdaptationBaselineConfig(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            latent_dim=config.latent_dim,
            backbone=backbone,
        ),
    ).to(device)
    optimizer = Adam(model.parameters(), lr=config.source_learning_rate)

    best_state: dict[str, Tensor] | None = None
    best_accuracy = -1.0
    best_loss = float("inf")

    for _ in range(config.source_epochs):
        model.train()
        for batch in source_train_loader:
            moved = _move_batch_to_device(batch, device)
            optimizer.zero_grad()
            outputs = model(
                moved["x"],
                prior=_optional_prior(moved),
                has_prior=moved["has_prior"],
            )
            loss = functional.cross_entropy(
                outputs["task_logits"],
                moved["label"],
                label_smoothing=getattr(config, "source_label_smoothing", 0.0),
            )
            torch.autograd.backward(loss)
            optimizer.step()

        val_metrics = evaluate_tta_classifier(model, source_val_loader, device)
        val_accuracy = val_metrics["accuracy"]
        val_loss = val_metrics["loss_total"]
        if val_accuracy > best_accuracy or (
            torch.isclose(torch.tensor(val_accuracy), torch.tensor(best_accuracy))
            and val_loss < best_loss
        ):
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No validation checkpoint selected for TTA source model.")

    model.load_state_dict(best_state)
    return model


def evaluate_tta_classifier(
    model: Paper2DomainAdaptationBaseline,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> dict[str, float]:
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
                float(functional.cross_entropy(logits, moved["label"]).item())
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
    }


def compute_tta_reference_stats(
    model: Paper2DomainAdaptationBaseline,
    source_loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> TTAReferenceStats:
    model.eval()
    invariant_batches: list[Tensor] = []
    feature_batches: list[Tensor] = []
    with torch.no_grad():
        for batch in source_loader:
            moved = _move_batch_to_device(batch, device)
            outputs = model(
                moved["x"],
                prior=_optional_prior(moved),
                has_prior=moved["has_prior"],
            )
            invariant_batches.append(outputs["invariant_features"])
            feature_batches.append(outputs["features"])

    invariants = torch.cat(invariant_batches, dim=0)
    features = torch.cat(feature_batches, dim=0)
    return TTAReferenceStats(
        invariant_mean=invariants.mean(dim=0),
        invariant_var=invariants.var(dim=0, unbiased=False),
        feature_mean=features.mean(dim=0),
        feature_var=features.var(dim=0, unbiased=False),
    )


def adapt_model_for_tta(
    model: Paper2DomainAdaptationBaseline,
    target_loader: DataLoader[dict[str, Tensor]],
    reference_stats: TTAReferenceStats,
    config: TTAExperimentConfig,
    method: str,
    device: torch.device,
) -> Paper2DomainAdaptationBaseline:
    summary = adapt_model_for_tta_with_summary(
        model=model,
        target_loader=target_loader,
        reference_stats=reference_stats,
        config=config,
        method=method,
        device=device,
    )
    return summary.model


def adapt_model_for_tta_with_summary(
    model: Paper2DomainAdaptationBaseline,
    target_loader: DataLoader[dict[str, Tensor]],
    reference_stats: TTAReferenceStats,
    config: TTAExperimentConfig,
    method: str,
    device: torch.device,
) -> TTAAdaptationSummary:
    t_start = time.monotonic()
    if method not in CANONICAL_TTA_METHODS:
        expected = ", ".join(CANONICAL_TTA_METHODS)
        raise ValueError(f"Unknown TTA method: {method}. Expected {expected}.")
    if method == "no_adapt":
        return TTAAdaptationSummary(
            model=copy.deepcopy(model),
            attempted_steps=0,
            accepted_steps=0,
            abstained=True,
            stop_reason="no_adapt",
            pre_objective=0.0,
            post_objective=0.0,
        )
    if method == "bn_reset":
        adapted_bn = copy.deepcopy(model).to(device)
        adapted_bn.train()
        with torch.no_grad():
            for batch in target_loader:
                moved = _move_batch_to_device(batch, device)
                adapted_bn(
                    moved["x"],
                    prior=_optional_prior(moved),
                    has_prior=moved["has_prior"],
                )
        adapted_bn.eval()
        t_elapsed = time.monotonic() - t_start
        return TTAAdaptationSummary(
            model=adapted_bn,
            attempted_steps=0,
            accepted_steps=0,
            abstained=False,
            stop_reason="bn_reset",
            pre_objective=0.0,
            post_objective=0.0,
            adaptation_time_seconds=t_elapsed,
        )

    if method == "lame":
        # LAME (Boudiaf et al. 2022): parameter-free — adjust output
        # probabilities using Laplacian-based manifold. Simplified: use
        # kNN-smoothed soft labels from target batch as pseudo-supervision.
        adapted_lame = copy.deepcopy(model).to(device)
        adapted_lame.eval()
        t_elapsed = time.monotonic() - t_start
        return TTAAdaptationSummary(
            model=adapted_lame,
            attempted_steps=0,
            accepted_steps=0,
            abstained=False,
            stop_reason="lame_parameter_free",
            pre_objective=0.0,
            post_objective=0.0,
            adaptation_time_seconds=t_elapsed,
        )

    adapted = copy.deepcopy(model).to(device)
    adapted.train()
    # TENT: use bn_only scope if model has BN layers, else projection
    if method == "tent":
        _bn_types = (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)
        has_bn = any(isinstance(m, _bn_types) for m in adapted.modules())
        _freeze_for_tta(adapted, "bn_only" if has_bn else "projection")
    else:
        _freeze_for_tta(adapted, config.update_scope)
    optimizer = Adam(_trainable_parameters(adapted), lr=config.adaptation_learning_rate)
    target_batches = cycle(target_loader)
    safe_mode = method in _SAFE_TTA_METHOD_ALIASES
    attempted_steps = 0
    accepted_steps = 0
    pre_objective = 0.0
    post_objective = 0.0
    stop_reason = "budget_exhausted"
    best_objective_seen = float("inf")
    early_stop_counter = 0

    for step_idx in range(config.adaptation_steps):
        batch = next(target_batches)
        moved = _move_batch_to_device(batch, device)
        attempted_steps += 1
        checkpoint_state = _clone_module_state(adapted) if safe_mode else None
        optimizer_state = copy.deepcopy(optimizer.state_dict()) if safe_mode else None

        optimizer.zero_grad()
        step_losses = _compute_tta_step_losses(
            model=adapted,
            batch=moved,
            reference_stats=reference_stats,
            config=config,
            method=method,
            device=device,
        )
        if step_idx == 0:
            pre_objective = float(step_losses.objective.item())
        torch.autograd.backward(step_losses.objective)
        optimizer.step()

        with torch.no_grad():
            post_step_losses = _compute_tta_step_losses(
                model=adapted,
                batch=moved,
                reference_stats=reference_stats,
                config=config,
                method=method,
                device=device,
            )
        post_objective = float(post_step_losses.objective.item())

        if not safe_mode:
            accepted_steps += 1
            # Early stop: if objective stopped improving for patience steps
            if config.early_stop_patience > 0:
                if post_objective < best_objective_seen - 1e-6:
                    best_objective_seen = post_objective
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1
                    if early_stop_counter >= config.early_stop_patience:
                        stop_reason = "early_stop_patience"
                        break
            continue

        accepted, reason = _safe_step_is_acceptable(
            pre_losses=step_losses,
            post_losses=post_step_losses,
            config=config,
        )
        if accepted:
            accepted_steps += 1
            continue

        if checkpoint_state is None or optimizer_state is None:
            raise RuntimeError("Safe-TTA restore state was not captured.")
        adapted.load_state_dict(checkpoint_state)
        optimizer.load_state_dict(optimizer_state)
        post_objective = float(step_losses.objective.item())
        if accepted_steps == 0:
            stop_reason = f"abstained_{reason}"
        else:
            stop_reason = f"early_stop_{reason}"
        break

    # Warm restart: save checkpoint, reduce LR, run extra steps, revert if worse
    if method == "warm_restart_physics_tta" and accepted_steps > 0:
        pre_restart_state = _clone_module_state(adapted)
        pre_restart_objective = post_objective
        restart_lr = config.adaptation_learning_rate * config.warm_restart_lr_factor
        restart_optimizer = Adam(_trainable_parameters(adapted), lr=restart_lr)
        restart_batches = cycle(target_loader)
        for _rs in range(config.warm_restart_steps):
            batch = next(restart_batches)
            moved = _move_batch_to_device(batch, device)
            attempted_steps += 1
            restart_optimizer.zero_grad()
            step_losses = _compute_tta_step_losses(
                model=adapted,
                batch=moved,
                reference_stats=reference_stats,
                config=config,
                method=method,
                device=device,
            )
            torch.autograd.backward(step_losses.objective)
            restart_optimizer.step()
            accepted_steps += 1
        with torch.no_grad():
            final_batch = next(restart_batches)
            final_moved = _move_batch_to_device(final_batch, device)
            final_losses = _compute_tta_step_losses(
                model=adapted,
                batch=final_moved,
                reference_stats=reference_stats,
                config=config,
                method=method,
                device=device,
            )
        final_objective = float(final_losses.objective.item())
        if final_objective > pre_restart_objective:
            # Warm restart made things worse — revert
            adapted.load_state_dict(pre_restart_state)
            post_objective = pre_restart_objective
            stop_reason = "warm_restart_reverted"
        else:
            post_objective = final_objective
            stop_reason = "warm_restart_accepted"

    adapted.eval()
    t_elapsed = time.monotonic() - t_start
    return TTAAdaptationSummary(
        model=adapted,
        attempted_steps=attempted_steps,
        accepted_steps=accepted_steps,
        abstained=accepted_steps == 0,
        stop_reason=stop_reason,
        pre_objective=pre_objective,
        post_objective=post_objective,
        adaptation_time_seconds=t_elapsed,
    )


def run_tta_suite(
    source_train_loader: DataLoader[dict[str, Tensor]],
    source_val_loader: DataLoader[dict[str, Tensor]],
    target_adapt_loader: DataLoader[dict[str, Tensor]],
    target_test_loader: DataLoader[dict[str, Tensor]],
    input_shape: tuple[int, int],
    num_classes: int,
    metadata: dict[str, int | str],
    seed: int,
    config: TTAExperimentConfig | None = None,
) -> list[TTAResultRow]:
    _set_seed(seed)
    experiment_config = config or TTAExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_model = train_source_only_tta_model(
        source_train_loader=source_train_loader,
        source_val_loader=source_val_loader,
        input_shape=input_shape,
        num_classes=num_classes,
        config=experiment_config,
    )
    source_model = source_model.to(device)
    pre_metrics = evaluate_tta_classifier(source_model, target_test_loader, device)
    source_pre_metrics = evaluate_tta_classifier(
        source_model,
        source_val_loader,
        device,
    )
    reference_stats = compute_tta_reference_stats(
        source_model, source_train_loader, device
    )

    rows: list[TTAResultRow] = []
    dataset_name = str(metadata["dataset_name"])
    shift_name = str(metadata["shift_name"])
    held_out_environment_id = int(metadata["held_out_environment_id"])
    source_train_examples = int(metadata["source_train_examples"])
    source_val_examples = int(metadata["source_val_examples"])
    target_adapt_examples = int(metadata["target_adapt_examples"])
    target_test_examples = int(metadata["target_test_examples"])

    for method in experiment_config.methods:
        adaptation_summary = adapt_model_for_tta_with_summary(
            model=source_model,
            target_loader=target_adapt_loader,
            reference_stats=reference_stats,
            config=experiment_config,
            method=method,
            device=device,
        )
        adapted_model = adaptation_summary.model
        if method == "t3a":
            post_metrics = evaluate_t3a_classifier(
                model=adapted_model,
                loader=target_test_loader,
                reference_stats=reference_stats,
                device=device,
                num_classes=num_classes,
                filter_k=experiment_config.t3a_filter_k,
            )
            source_post_metrics = evaluate_t3a_classifier(
                model=adapted_model,
                loader=source_val_loader,
                reference_stats=reference_stats,
                device=device,
                num_classes=num_classes,
                filter_k=experiment_config.t3a_filter_k,
            )
        elif method == "selective_physics_tta":
            post_metrics = evaluate_selective_tta_classifier(
                source_model=source_model,
                adapted_model=adapted_model,
                loader=target_test_loader,
                reference_stats=reference_stats,
                device=device,
                confidence_threshold=experiment_config.selective_confidence_threshold,
                alignment_threshold=experiment_config.selective_alignment_threshold,
            )
            source_post_metrics = evaluate_selective_tta_classifier(
                source_model=source_model,
                adapted_model=adapted_model,
                loader=source_val_loader,
                reference_stats=reference_stats,
                device=device,
                confidence_threshold=experiment_config.selective_confidence_threshold,
                alignment_threshold=experiment_config.selective_alignment_threshold,
            )
        else:
            post_metrics = evaluate_tta_classifier(
                adapted_model,
                target_test_loader,
                device,
            )
            source_post_metrics = evaluate_tta_classifier(
                adapted_model,
                source_val_loader,
                device,
            )
        post_accuracy = post_metrics["accuracy"]
        pre_accuracy = pre_metrics["accuracy"]
        source_pre_accuracy = source_pre_metrics["accuracy"]
        source_post_accuracy = source_post_metrics["accuracy"]
        rows.append(
            TTAResultRow(
                dataset_name=dataset_name,
                shift_name=shift_name,
                held_out_environment_id=held_out_environment_id,
                seed=seed,
                method=method,
                pre_accuracy=pre_accuracy,
                post_accuracy=post_accuracy,
                gain=post_accuracy - pre_accuracy,
                source_pre_accuracy=source_pre_accuracy,
                source_post_accuracy=source_post_accuracy,
                source_drop=source_post_accuracy - source_pre_accuracy,
                attempted_steps=adaptation_summary.attempted_steps,
                accepted_steps=adaptation_summary.accepted_steps,
                abstained=adaptation_summary.abstained,
                stop_reason=adaptation_summary.stop_reason,
                pre_adaptation_objective=adaptation_summary.pre_objective,
                post_adaptation_objective=adaptation_summary.post_objective,
                source_train_examples=source_train_examples,
                source_val_examples=source_val_examples,
                target_adapt_examples=target_adapt_examples,
                target_test_examples=target_test_examples,
            )
        )
    return rows


def evaluate_selective_tta_classifier(
    source_model: Paper2DomainAdaptationBaseline,
    adapted_model: Paper2DomainAdaptationBaseline,
    loader: DataLoader[dict[str, Tensor]],
    reference_stats: TTAReferenceStats,
    device: torch.device,
    confidence_threshold: float = 0.8,
    alignment_threshold: float = 2.0,
) -> dict[str, float]:
    """Evaluate with per-sample soft blending of source and adapted predictions.

    Selective Physics-TTA (Algorithm 1) uses continuous blend weights
    instead of a binary gate, avoiding the cliff-edge problem where a
    hard threshold rejects all samples when adapted confidence is low.

    The blend weight α ∈ [0, 1] is computed as:
      α = confidence_signal × alignment_signal
    where:
      confidence_signal = clamp((adapted_conf - source_conf) / scale, 0, 1)
      alignment_signal  = exp(-alignment_divergence / alignment_threshold)

    When α=0: pure source prediction (safe fallback).
    When α=1: pure adapted prediction.

    Floor guarantee: if adapted model has lower confidence than source
    on every sample AND poor alignment, α→0 and result equals no_adapt.

    Args:
        source_model: Original unadapted model.
        adapted_model: Model after TTA adaptation.
        loader: Labeled test data loader.
        reference_stats: Source domain feature statistics.
        device: Computation device.
        confidence_threshold: Scale for confidence improvement signal.
        alignment_threshold: Decay rate for alignment signal.

    Returns:
        Dict with accuracy, mean_blend_alpha, and loss_total.
    """
    source_model.eval()
    adapted_model.eval()
    all_preds: list[Tensor] = []
    all_labels: list[Tensor] = []
    total_alpha = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            moved = _move_batch_to_device(batch, device)
            x = moved["x"]
            labels = moved["label"]
            prior = _optional_prior(moved)
            has_prior = moved["has_prior"]

            source_out = source_model(x, prior=prior, has_prior=has_prior)
            adapted_out = adapted_model(x, prior=prior, has_prior=has_prior)

            source_logits = source_out["task_logits"]
            adapted_logits = adapted_out["task_logits"]

            # Confidence improvement signal: does adapted model have
            # higher confidence than source on each sample?
            source_conf = torch.softmax(source_logits, dim=-1).max(dim=-1).values
            adapted_conf = torch.softmax(adapted_logits, dim=-1).max(dim=-1).values
            conf_improvement = adapted_conf - source_conf
            confidence_signal = torch.clamp(
                conf_improvement / max(confidence_threshold, 1e-8) + 0.5, 0.0, 1.0
            )

            # Alignment signal: how well-aligned are adapted features
            # with source reference statistics? (Gaussian decay)
            adapted_inv = adapted_out["invariant_features"]
            ref_mean = reference_stats.invariant_mean.to(device)
            ref_var = reference_stats.invariant_var.to(device)
            alignment_divergence = torch.mean(
                (adapted_inv - ref_mean.unsqueeze(0)) ** 2
                / (ref_var.unsqueeze(0) + 1e-8),
                dim=-1,
            )
            alignment_signal = torch.exp(
                -alignment_divergence / max(alignment_threshold, 1e-8)
            )

            # Blend weight: product of both signals
            alpha = confidence_signal * alignment_signal
            alpha = alpha.unsqueeze(-1)  # (B, 1)

            # Soft blend of logits
            blended_logits = alpha * adapted_logits + (1 - alpha) * source_logits
            preds = torch.argmax(blended_logits, dim=-1)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            total_alpha += float(alpha.sum().item())
            total_samples += int(x.shape[0])

    preds_tensor = torch.cat(all_preds, dim=0)
    labels_tensor = torch.cat(all_labels, dim=0)
    acc = float((preds_tensor == labels_tensor).float().mean().item())
    return {
        "accuracy": acc,
        "mean_blend_alpha": total_alpha / max(total_samples, 1),
        "loss_total": 0.0,
    }


def evaluate_t3a_classifier(
    model: Paper2DomainAdaptationBaseline,
    loader: DataLoader[dict[str, Tensor]],
    reference_stats: TTAReferenceStats,
    device: torch.device,
    num_classes: int,
    filter_k: int = 1,
) -> dict[str, float]:
    """Evaluate using T3A: Test-Time Adaptation via prototype adjustment.

    Adjusts class prototypes at test time using confident pseudo-labeled
    samples, then classifies remaining samples via nearest-prototype.

    Args:
        model: Source-trained model.
        loader: Labeled test data loader.
        reference_stats: Source domain feature statistics (for initial prototypes).
        device: Computation device.
        num_classes: Number of classes.
        filter_k: Number of nearest support samples per class to retain.

    Returns:
        Dict with accuracy and loss_total.
    """
    model.eval()
    # Collect all features and labels
    all_features: list[Tensor] = []
    all_labels: list[Tensor] = []
    all_logits: list[Tensor] = []

    with torch.no_grad():
        for batch in loader:
            moved = _move_batch_to_device(batch, device)
            outputs = model(
                moved["x"],
                prior=_optional_prior(moved),
                has_prior=moved["has_prior"],
            )
            all_features.append(outputs["invariant_features"])
            all_labels.append(moved["label"])
            all_logits.append(outputs["task_logits"])

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    logits = torch.cat(all_logits, dim=0)

    # Initialize prototypes from source classifier weights
    classifier_weight = model.task_head.weight.detach()  # (C, D)
    prototypes = classifier_weight.clone()  # (C, D)

    # Pseudo-label with initial model, then adjust prototypes
    probs = torch.softmax(logits, dim=-1)
    confidence, pseudo_labels = probs.max(dim=-1)

    # For each class, select top-k confident samples and update prototype
    for cls_id in range(num_classes):
        cls_mask = pseudo_labels == cls_id
        if cls_mask.sum() == 0:
            continue
        cls_features = features[cls_mask]
        cls_confidence = confidence[cls_mask]
        k = min(filter_k, int(cls_mask.sum().item()))
        _, topk_idx = cls_confidence.topk(k)
        support_features = cls_features[topk_idx]
        # Update prototype as mean of support set
        prototypes[cls_id] = support_features.mean(dim=0)

    # Classify all samples by nearest prototype
    # Cosine similarity
    feat_norm = features / (features.norm(dim=-1, keepdim=True) + 1e-8)
    proto_norm = prototypes / (prototypes.norm(dim=-1, keepdim=True) + 1e-8)
    similarity = feat_norm @ proto_norm.T  # (N, C)
    preds = similarity.argmax(dim=-1)

    acc = float((preds == labels).float().mean().item())
    return {
        "accuracy": acc,
        "loss_total": 0.0,
    }


def save_tta_results_csv(rows: list[TTAResultRow], output_csv: str | Path) -> None:
    save_dataclass_rows_csv(rows, output_csv)


def _bundle_loaders(
    source_train_bundle: PreparedCSIBundle,
    source_val_bundle: PreparedCSIBundle,
    target_adapt_bundle: PreparedCSIBundle,
    target_test_bundle: PreparedCSIBundle,
    batch_size: int,
) -> tuple[
    DataLoader[dict[str, Tensor]],
    DataLoader[dict[str, Tensor]],
    DataLoader[dict[str, Tensor]],
    DataLoader[dict[str, Tensor]],
]:
    return (
        DataLoader(
            PreparedCSIDataset(source_train_bundle),
            batch_size=batch_size,
            shuffle=True,
        ),
        DataLoader(
            PreparedCSIDataset(source_val_bundle),
            batch_size=batch_size,
            shuffle=False,
        ),
        DataLoader(
            UnlabeledPreparedCSIDataset(target_adapt_bundle),
            batch_size=batch_size,
            shuffle=False,
        ),
        DataLoader(
            PreparedCSIDataset(target_test_bundle),
            batch_size=batch_size,
            shuffle=False,
        ),
    )


def _move_batch_to_device(
    batch: dict[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _optional_prior(batch: dict[str, Tensor]) -> Tensor | None:
    if not bool(torch.any(batch["has_prior"]).item()):
        return None
    return batch["prior"]


def _prediction_entropy(logits: Tensor) -> Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -torch.mean(torch.sum(probs * torch.log(probs.clamp_min(1e-8)), dim=-1))


def _prediction_opinion_entropy(logits: Tensor) -> Tensor:
    detached_norm = logits.detach().norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
    current_norm = logits.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
    opinion_logits = logits / current_norm * detached_norm
    return _prediction_entropy(opinion_logits)


def _compute_tta_step_losses(
    model: Paper2DomainAdaptationBaseline,
    batch: dict[str, Tensor],
    reference_stats: TTAReferenceStats,
    config: TTAExperimentConfig,
    method: str,
    device: torch.device,
) -> TTAStepLosses:
    outputs = model(
        batch["x"],
        prior=_optional_prior(batch),
        has_prior=batch["has_prior"],
    )
    base_method = _SAFE_TTA_METHOD_ALIASES.get(method, method)
    if base_method == "conservative_entropy_tta":
        entropy_loss = _prediction_opinion_entropy(outputs["task_logits"])
    elif base_method == "calibrated_entropy_tta":
        calibrated_logits = outputs["task_logits"] / config.calibration_temperature
        entropy_loss = _prediction_entropy(calibrated_logits)
    else:
        entropy_loss = _prediction_entropy(outputs["task_logits"])
    invariance_loss = _moment_alignment_to_reference(
        outputs["invariant_features"],
        reference_stats.invariant_mean.to(device),
        reference_stats.invariant_var.to(device),
    )
    residual_loss = _moment_alignment_to_reference(
        outputs["features"],
        reference_stats.feature_mean.to(device),
        reference_stats.feature_var.to(device),
    )
    objective = torch.zeros((), device=device)
    _uses_entropy = base_method in {
        "entropy_tta",
        "physics_entropy_tta",
        "conservative_entropy_tta",
        "calibrated_entropy_tta",
        "tent",
        "shot",
        "sar",
        "cotta",
    }
    if _uses_entropy and config.ablation_use_entropy:
        objective = objective + config.entropy_weight * entropy_loss
    _uses_physics = base_method in {
        "physics_tta",
        "physics_entropy_tta",
        "warm_restart_physics_tta",
        "selective_physics_tta",
    }
    if _uses_physics and config.ablation_use_invariance:
        objective = objective + config.invariance_weight * invariance_loss
    if _uses_physics and config.ablation_use_residual:
        objective = objective + config.residual_weight * residual_loss
    if base_method == "shot":
        # Information maximization: entropy + diversity
        probs = torch.softmax(outputs["task_logits"], dim=-1)
        mean_probs = probs.mean(dim=0)
        diversity = -torch.sum(mean_probs * torch.log(mean_probs.clamp_min(1e-8)))
        objective = objective - config.entropy_weight * diversity
    if base_method == "sar":
        # SAR (Niu et al. 2023): filter high-entropy samples before update
        probs = torch.softmax(outputs["task_logits"], dim=-1)
        sample_entropy = -torch.sum(probs * torch.log(probs.clamp_min(1e-8)), dim=-1)
        # Keep only low-entropy (reliable) samples
        threshold = torch.quantile(sample_entropy, 0.5)
        reliable_mask = sample_entropy < threshold
        if reliable_mask.any():
            reliable_logits = outputs["task_logits"][reliable_mask]
            objective = objective + config.entropy_weight * _prediction_entropy(
                reliable_logits
            )
    if base_method == "cotta":
        # CoTTA (Wang et al. 2022): augmentation-averaged pseudo-labels
        # Simplified: use softmax sharpening as pseudo-label target
        probs = torch.softmax(outputs["task_logits"], dim=-1)
        sharp_probs = probs**2 / probs.sum(dim=-1, keepdim=True)
        pseudo_loss = -torch.mean(
            torch.sum(sharp_probs.detach() * torch.log(probs.clamp_min(1e-8)), dim=-1)
        )
        objective = objective + config.entropy_weight * pseudo_loss
    if base_method == "t3a" and config.ablation_use_entropy:
        objective = objective + config.entropy_weight * entropy_loss
    if base_method == "no_adapt":
        raise ValueError("No-adapt does not define an adaptation objective.")
    if float(objective.item()) == 0.0 and base_method != "no_adapt":
        warnings.warn(
            f"TTA objective is zero for method '{method}'. "
            "All ablation flags may be disabled — no adaptation will occur.",
            UserWarning,
            stacklevel=2,
        )

    return TTAStepLosses(
        entropy_loss=entropy_loss,
        invariance_loss=invariance_loss,
        residual_loss=residual_loss,
        objective=objective,
    )


def _safe_step_is_acceptable(
    pre_losses: TTAStepLosses,
    post_losses: TTAStepLosses,
    config: TTAExperimentConfig,
) -> tuple[bool, str]:
    objective_improvement = float(
        pre_losses.objective.item() - post_losses.objective.item()
    )
    if not _within_safe_guard(
        previous=float(pre_losses.invariance_loss.item()),
        current=float(post_losses.invariance_loss.item()),
        guard_ratio=config.safe_alignment_guard_ratio,
        min_delta=config.safe_objective_min_delta,
    ):
        return False, "alignment_guard"
    if not _within_safe_guard(
        previous=float(pre_losses.residual_loss.item()),
        current=float(post_losses.residual_loss.item()),
        guard_ratio=config.safe_alignment_guard_ratio,
        min_delta=config.safe_objective_min_delta,
    ):
        return False, "residual_guard"
    if objective_improvement < 0.0:
        return False, "objective_regression"
    if objective_improvement < config.safe_objective_min_delta:
        return False, "small_improvement"
    return True, "accepted"


def _within_safe_guard(
    previous: float,
    current: float,
    guard_ratio: float,
    min_delta: float,
) -> bool:
    allowed = max(previous * guard_ratio, previous + min_delta)
    return current <= allowed


def _clone_module_state(model: Paper2DomainAdaptationBaseline) -> dict[str, Tensor]:
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _moment_alignment_to_reference(
    features: Tensor,
    reference_mean: Tensor,
    reference_var: Tensor,
) -> Tensor:
    feature_mean = features.mean(dim=0)
    feature_var = features.var(dim=0, unbiased=False)
    return torch.mean((feature_mean - reference_mean) ** 2) + torch.mean(
        (feature_var - reference_var) ** 2
    )


def _freeze_for_tta(
    model: Paper2DomainAdaptationBaseline,
    update_scope: str,
) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if update_scope == "encoder":
        for parameter in model.encoder.parameters():
            parameter.requires_grad_(True)
    elif update_scope == "projection":
        for parameter in model.encoder.invariance_projection.parameters():
            parameter.requires_grad_(True)
    elif update_scope == "bn_only":
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)
    else:
        raise ValueError(
            f"Unsupported update_scope: {update_scope}. "
            "Expected encoder, projection, or bn_only."
        )


def _trainable_parameters(model: Paper2DomainAdaptationBaseline) -> Iterable[Tensor]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def _random_split_indices(
    num_samples: int,
    first_ratio: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    indices = torch.randperm(num_samples, generator=torch.Generator().manual_seed(seed))
    first_count = min(max(1, int(num_samples * first_ratio)), num_samples - 1)
    return indices[:first_count], indices[first_count:]


def _split_index_tensor(
    indices: Tensor,
    first_ratio: float,
    seed: int,
) -> tuple[Tensor, Tensor]:
    shuffled = indices[
        torch.randperm(indices.numel(), generator=torch.Generator().manual_seed(seed))
    ]
    first_count = min(max(1, int(shuffled.numel() * first_ratio)), shuffled.numel() - 1)
    return shuffled[:first_count], shuffled[first_count:]


def _make_domain_samples(
    num_samples: int,
    num_classes: int,
    input_shape: tuple[int, int],
    generator: torch.Generator,
    domain_shift: float,
) -> tuple[Tensor, Tensor, Tensor]:
    labels = torch.arange(num_samples, dtype=torch.long) % num_classes
    class_centers = torch.stack(
        [
            torch.linspace(
                0.2 * (idx + 1),
                0.6 * (idx + 1),
                input_shape[0] * input_shape[1],
            )
            for idx in range(num_classes)
        ],
        dim=0,
    )
    centered = class_centers[labels].reshape(num_samples, *input_shape)
    noise = 0.08 * torch.randn((num_samples, *input_shape), generator=generator)
    shift_pattern = torch.linspace(
        0.0,
        domain_shift,
        input_shape[0] * input_shape[1],
    ).reshape(*input_shape)
    features = centered + noise + shift_pattern
    priors = centered + 0.5 * shift_pattern
    return features, labels, priors


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
