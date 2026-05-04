# pyright: basic, reportMissingImports=false

"""Run amplitude-only diagnosis experiments for UT_HAR.

This script evaluates six methods under two evaluation modes:
- ``iid``: random train/val/test split
- ``cross_env``: hold out environment ``2`` for test

NeurIPS-focused upgrades in this runner:
- stronger encoders (Conv1D residual stack, BiGRU)
- AdamW + cosine scheduling + gradient clipping
- early stopping with patience
- parameter count reporting
- richer paired statistics (sign test + paired permutation test + bootstrap CI)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

from pinn4csi.physics import compute_path_loss

METHODS: tuple[str, ...] = (
    "conv1d_baseline",
    "conv1d_amp_physics",
    "gru_baseline",
    "gru_amp_physics",
)
EVAL_MODES: tuple[str, ...] = ("iid", "cross_env")


@dataclass(frozen=True)
class ExperimentRow:
    """One result row for CSV export."""

    dataset: str
    method: str
    eval_mode: str
    seed: int
    best_epoch: int
    stopped_epoch: int
    train_examples: int
    val_examples: int
    test_examples: int
    parameter_count: int
    test_accuracy: float
    test_macro_f1: float
    test_loss: float
    generalization_gap_accuracy: float | None


def count_trainable_parameters(*modules: nn.Module) -> int:
    """Return the total trainable parameter count for modules."""

    return int(
        sum(
            param.numel()
            for module in modules
            for param in module.parameters()
            if param.requires_grad
        )
    )


class ConvResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm1d(channels)
        self.conv1 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.norm2 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(
            channels, channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = self.norm1(x)
        out = self.conv1(out)
        out = functional.gelu(out)
        out = self.dropout(out)
        out = self.norm2(out)
        out = self.conv2(out)
        out = functional.gelu(out)
        out = self.dropout(out)
        return out + residual


class Conv1DEncoder(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(num_features, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.block1 = ConvResidualBlock(
            channels=hidden_dim, kernel_size=5, dropout=dropout
        )
        self.block2 = ConvResidualBlock(
            channels=hidden_dim, kernel_size=3, dropout=dropout
        )
        self.block3 = ConvResidualBlock(
            channels=hidden_dim, kernel_size=3, dropout=dropout
        )

    def forward(self, x: Tensor) -> Tensor:
        encoded = self.stem(x.transpose(1, 2))
        encoded = self.block1(encoded)
        encoded = self.block2(encoded)
        encoded = self.block3(encoded)
        return encoded.transpose(1, 2)


class CNNGRUEncoder(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(250, hidden_dim, kernel_size=12, stride=3),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=2),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, stride=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
        )

        self.gru = nn.GRU(
            input_size=8,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            dropout=dropout if dropout > 0 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        conv_out = self.conv_block(x)  # (B, 250, 90) -> (B, H, 8)
        gru_out, _state = self.gru(conv_out)  # (B, H, 8) -> (B, H, H) batch_first
        return gru_out


class GRUEncoder(nn.Module):
    """BiGRU encoder over time for amplitude-only CSI tensors."""

    def __init__(self, num_features: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for bidirectional GRU.")
        self.gru = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            dropout=dropout if dropout > 0 else 0.0,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        encoded, _state = self.gru(x)
        return encoded


class SequenceClassifier(nn.Module):
    """Classifier with attention pooling over encoded sequences."""

    def __init__(self, hidden_dim: int, num_classes: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        pooled_mean = torch.mean(sequence, dim=1)
        attn_logits = self.attn(sequence).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1).unsqueeze(-1)
        pooled_attn = torch.sum(sequence * attn_weights, dim=1)
        pooled = torch.cat([pooled_mean, pooled_attn], dim=1)
        return self.head(pooled)


class AmplitudePhysicsHead(nn.Module):
    """Auxiliary amplitude head for detached amplitude supervision."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        bottleneck = max(16, hidden_dim // 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        return self.head(sequence)


def build_arg_parser() -> argparse.ArgumentParser:
    """Build argument parser for diagnosis experiments."""

    parser = argparse.ArgumentParser(
        description=(
            "Run amplitude-only diagnosis experiments with 6 methods and "
            "2 eval modes (iid/cross_env)."
        )
    )
    parser.add_argument("--prepared-root", type=Path, default=Path("data/prepared"))
    parser.add_argument("--dataset", type=str, default="ut_har")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--aug-weight", type=float, default=0.5)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--permutation-samples", type=int, default=20000)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/diagnosis_experiments.csv"),
    )
    return parser


def set_seed(seed: int) -> None:
    """Set deterministic random seeds for reproducible runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_ut_har_amplitude(
    prepared_root: Path,
    dataset: str,
) -> tuple[Tensor, Tensor, Tensor]:
    if dataset.lower() != "ut_har":
        raise ValueError("This diagnosis runner only supports --dataset ut_har.")

    dataset_dir = prepared_root / dataset
    csi = np.load(dataset_dir / "csi.npy")
    labels = np.load(dataset_dir / "labels.npy")
    environments = np.load(dataset_dir / "environments.npy")

    amplitudes = np.abs(csi).astype(np.float32)
    if amplitudes.ndim == 2:
        amplitudes = amplitudes[..., None]

    return (
        torch.from_numpy(amplitudes),
        torch.from_numpy(labels).long().flatten(),
        torch.from_numpy(environments).long().flatten(),
    )


def build_splits(
    labels: Tensor,
    environments: Tensor,
    seed: int,
    eval_mode: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, Tensor]:
    """Build indices for ``iid`` or ``cross_env`` evaluation modes."""

    if eval_mode not in EVAL_MODES:
        raise ValueError(f"Unsupported eval mode: {eval_mode}")

    generator = torch.Generator().manual_seed(seed)
    num_samples = int(labels.shape[0])

    if eval_mode == "iid":
        shuffled = torch.randperm(num_samples, generator=generator)
        train_end = int(num_samples * train_ratio)
        val_end = train_end + int(num_samples * val_ratio)
        return {
            "train": shuffled[:train_end],
            "val": shuffled[train_end:val_end],
            "test": shuffled[val_end:],
        }

    held_out_env = 2
    source_indices = torch.nonzero(
        environments != held_out_env,
        as_tuple=False,
    ).squeeze(-1)
    target_indices = torch.nonzero(
        environments == held_out_env,
        as_tuple=False,
    ).squeeze(-1)
    if source_indices.numel() == 0 or target_indices.numel() == 0:
        raise ValueError(
            "cross_env split requires source environments and held-out env=2 samples."
        )

    shuffled_source = source_indices[
        torch.randperm(source_indices.numel(), generator=generator)
    ]
    train_fraction = train_ratio / max(train_ratio + val_ratio, 1e-8)
    train_count = int(shuffled_source.numel() * train_fraction)
    if train_count <= 0:
        train_count = 1
    if train_count >= shuffled_source.numel():
        train_count = shuffled_source.numel() - 1
    return {
        "train": shuffled_source[:train_count],
        "val": shuffled_source[train_count:],
        "test": target_indices,
    }


def macro_f1_score(predictions: Tensor, targets: Tensor) -> float:
    """Compute macro F1 without external dependencies."""

    classes = sorted({int(value) for value in targets.tolist()})
    if not classes:
        return 0.0
    f1_values: list[float] = []
    for class_idx in classes:
        pred_positive = predictions == class_idx
        target_positive = targets == class_idx
        true_positive = int(torch.sum(pred_positive & target_positive).item())
        false_positive = int(torch.sum(pred_positive & ~target_positive).item())
        false_negative = int(torch.sum(~pred_positive & target_positive).item())
        precision = (
            true_positive / (true_positive + false_positive)
            if (true_positive + false_positive) > 0
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else 0.0
        )
        if precision + recall == 0.0:
            f1_values.append(0.0)
        else:
            f1_values.append(2.0 * precision * recall / (precision + recall))
    return float(sum(f1_values) / len(f1_values))


def evaluate_model(
    encoder: nn.Module,
    classifier: SequenceClassifier,
    loader: DataLoader[tuple[Tensor, ...]],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate classifier metrics on a dataloader."""

    encoder.eval()
    classifier.eval()
    total_loss = 0.0
    total_examples = 0
    prediction_batches: list[Tensor] = []
    label_batches: list[Tensor] = []

    with torch.no_grad():
        for batch_x, batch_y in loader:
            moved_x = batch_x.to(device)
            moved_y = batch_y.to(device)
            sequence = encoder(moved_x)
            logits = classifier(sequence)
            loss = functional.cross_entropy(logits, moved_y)
            batch_size = int(moved_y.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            prediction_batches.append(torch.argmax(logits, dim=1).cpu())
            label_batches.append(moved_y.cpu())

    predictions = torch.cat(prediction_batches, dim=0)
    labels = torch.cat(label_batches, dim=0)
    accuracy = float((predictions == labels).float().mean().item())
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1_score(predictions, labels),
        "loss": total_loss / max(total_examples, 1),
    }


def build_amplitude_physics_target(batch_x: Tensor) -> Tensor:
    """Build deterministic amplitude-only auxiliary target.

    Target is subcarrier-smoothed log-amplitude envelope over time,
    shape ``(B, T, 1)``.
    """

    smooth = functional.avg_pool1d(
        batch_x.transpose(1, 2),
        kernel_size=5,
        stride=1,
        padding=2,
    ).transpose(1, 2)
    return torch.log1p(torch.mean(smooth, dim=-1, keepdim=True))


def apply_physics_augmentation(batch_x: Tensor, step_index: int, seed: int) -> Tensor:
    """Apply deterministic Rayleigh fading + path-loss scaling augmentation."""

    cpu_generator = torch.Generator(device="cpu")
    cpu_generator.manual_seed(seed * 100_000 + step_index)

    fading_shape = (batch_x.shape[0], 1, batch_x.shape[-1])
    fading_real = torch.randn(
        fading_shape,
        generator=cpu_generator,
        dtype=batch_x.dtype,
        device="cpu",
    )
    fading_imag = torch.randn(
        fading_shape,
        generator=cpu_generator,
        dtype=batch_x.dtype,
        device="cpu",
    )
    rayleigh_fading = torch.sqrt(fading_real.square() + fading_imag.square() + 1e-8)
    rayleigh_fading = rayleigh_fading / np.sqrt(2.0)
    rayleigh_fading = rayleigh_fading.to(batch_x.device)

    base_distance = 1.0 + 0.15 * float(step_index % 7)
    distance_offsets = torch.linspace(
        0.0,
        0.9,
        steps=batch_x.shape[0],
        device=batch_x.device,
        dtype=batch_x.dtype,
    )
    distances = base_distance + distance_offsets
    path_loss_db = compute_path_loss(distances, frequency=2.4e9, n=2.0)
    path_gain = torch.pow(10.0, (-path_loss_db / 20.0)).view(-1, 1, 1)

    augmented = batch_x * rayleigh_fading
    augmented = augmented * path_gain
    return augmented


def train_and_evaluate_method(
    method: str,
    features: Tensor,
    labels: Tensor,
    split_indices: dict[str, Tensor],
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    aux_weight: float,
    aug_weight: float,
    grad_clip_norm: float,
    patience: int,
    min_delta: float,
    device: torch.device,
) -> dict[str, float | int]:
    """Train one method and return test metrics."""

    set_seed(seed)
    train_x = features[split_indices["train"]]
    train_y = labels[split_indices["train"]]
    val_x = features[split_indices["val"]]
    val_y = labels[split_indices["val"]]
    test_x = features[split_indices["test"]]
    test_y = labels[split_indices["test"]]
    num_features = int(train_x.shape[-1])
    num_classes = int(torch.max(labels).item()) + 1

    if method.startswith("cnn_gru"):
        encoder: nn.Module = CNNGRUEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    elif method.startswith("conv1d"):
        encoder: nn.Module = Conv1DEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    elif method.startswith("gru"):
        encoder = GRUEncoder(
            num_features=num_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    classifier = SequenceClassifier(
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        dropout=dropout,
    )
    auxiliary_head: AmplitudePhysicsHead | None = None
    if method.endswith("amp_physics"):
        auxiliary_head = AmplitudePhysicsHead(hidden_dim=hidden_dim)

    encoder = encoder.to(device)
    classifier = classifier.to(device)
    if auxiliary_head is not None:
        auxiliary_head = auxiliary_head.to(device)

    trainable_parameters = list(encoder.parameters()) + list(classifier.parameters())
    if auxiliary_head is not None:
        trainable_parameters += list(auxiliary_head.parameters())

    optimizer = AdamW(trainable_parameters, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(test_x, test_y),
        batch_size=batch_size,
        shuffle=False,
    )

    best_epoch = -1
    best_val_accuracy = -1.0
    best_state: dict[str, dict[str, Tensor]] = {}
    step_index = 0
    epochs_without_improvement = 0
    stopped_epoch = epochs - 1

    for epoch_idx in range(epochs):
        encoder.train()
        classifier.train()
        if auxiliary_head is not None:
            auxiliary_head.train()

        for batch_x, batch_y in train_loader:
            moved_x = batch_x.to(device)
            moved_y = batch_y.to(device)

            encoded_sequence = encoder(moved_x)
            logits = classifier(encoded_sequence)
            task_loss = functional.cross_entropy(logits, moved_y)
            total_loss = task_loss

            if method.endswith("augmented"):
                augmented_x = apply_physics_augmentation(
                    moved_x,
                    step_index=step_index,
                    seed=seed,
                )
                augmented_logits = classifier(encoder(augmented_x))
                augmented_loss = functional.cross_entropy(augmented_logits, moved_y)
                total_loss = total_loss + (aug_weight * augmented_loss)

            if auxiliary_head is not None:
                auxiliary_prediction = auxiliary_head(encoded_sequence.detach())
                auxiliary_target = build_amplitude_physics_target(moved_x)
                auxiliary_loss = functional.mse_loss(
                    auxiliary_prediction,
                    auxiliary_target,
                )
                total_loss = total_loss + (aux_weight * auxiliary_loss)

            optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(trainable_parameters, max_norm=grad_clip_norm)
            optimizer.step()
            step_index += 1

        scheduler.step()
        val_metrics = evaluate_model(encoder, classifier, val_loader, device)
        val_accuracy = float(val_metrics["accuracy"])
        if val_accuracy > (best_val_accuracy + min_delta):
            best_val_accuracy = val_accuracy
            best_epoch = epoch_idx
            epochs_without_improvement = 0
            best_state = {
                "encoder": {
                    key: value.detach().cpu().clone()
                    for key, value in encoder.state_dict().items()
                },
                "classifier": {
                    key: value.detach().cpu().clone()
                    for key, value in classifier.state_dict().items()
                },
            }
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            stopped_epoch = epoch_idx
            break

    if not best_state:
        raise RuntimeError("No validation checkpoint was captured.")

    encoder.load_state_dict(best_state["encoder"])
    classifier.load_state_dict(best_state["classifier"])
    encoder.to(device)
    classifier.to(device)
    test_metrics = evaluate_model(encoder, classifier, test_loader, device)

    return {
        "best_epoch": best_epoch,
        "stopped_epoch": stopped_epoch,
        "parameter_count": count_trainable_parameters(
            encoder,
            classifier,
            *([auxiliary_head] if auxiliary_head is not None else []),
        ),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_macro_f1": float(test_metrics["macro_f1"]),
        "test_loss": float(test_metrics["loss"]),
        "train_examples": int(train_x.shape[0]),
        "val_examples": int(val_x.shape[0]),
        "test_examples": int(test_x.shape[0]),
    }


def two_sided_sign_test_pvalue(wins: int, losses: int) -> float | None:
    """Compute two-sided sign-test p-value from win/loss counts."""

    decisive = wins + losses
    if decisive <= 0:
        return None
    tail = min(wins, losses)
    probability = sum(math.comb(decisive, value) for value in range(tail + 1))
    probability /= 2**decisive
    return float(min(1.0, 2.0 * probability))


def bootstrap_ci_mean(
    values: list[float],
    bootstrap_samples: int,
    seed: int,
) -> tuple[float, float] | None:
    """Estimate 95% bootstrap CI for the sample mean."""

    if not values:
        return None
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    means = np.empty(bootstrap_samples, dtype=np.float64)
    for idx in range(bootstrap_samples):
        sample = rng.choice(array, size=array.shape[0], replace=True)
        means[idx] = np.mean(sample)
    lower = float(np.percentile(means, 2.5))
    upper = float(np.percentile(means, 97.5))
    return lower, upper


def paired_permutation_pvalue(
    deltas: list[float],
    permutation_samples: int,
    seed: int,
) -> float | None:
    """Compute paired randomization p-value by sign flipping deltas."""

    if not deltas:
        return None
    observed = abs(float(np.mean(deltas)))
    if observed == 0.0:
        return 1.0

    array = np.asarray(deltas, dtype=np.float64)
    count = int(array.shape[0])
    exhaustive = 2**count
    exceed = 0
    total = 0

    if exhaustive <= 131072:
        for bits in itertools.product((-1.0, 1.0), repeat=count):
            signed = array * np.asarray(bits, dtype=np.float64)
            stat = abs(float(np.mean(signed)))
            if stat >= observed:
                exceed += 1
            total += 1
    else:
        rng = np.random.default_rng(seed)
        for _ in range(permutation_samples):
            signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=count)
            signed = array * signs
            stat = abs(float(np.mean(signed)))
            if stat >= observed:
                exceed += 1
            total += 1

    return float((exceed + 1) / (total + 1))


def append_generalization_gaps(rows: list[ExperimentRow]) -> list[ExperimentRow]:
    """Attach per-(method, seed) IID minus cross_env accuracy gap to rows."""

    lookup = {(row.method, row.seed, row.eval_mode): row for row in rows}
    with_gaps: list[ExperimentRow] = []
    for row in rows:
        iid_row = lookup.get((row.method, row.seed, "iid"))
        cross_row = lookup.get((row.method, row.seed, "cross_env"))
        gap = None
        if iid_row is not None and cross_row is not None:
            gap = iid_row.test_accuracy - cross_row.test_accuracy
        with_gaps.append(
            ExperimentRow(
                dataset=row.dataset,
                method=row.method,
                eval_mode=row.eval_mode,
                seed=row.seed,
                best_epoch=row.best_epoch,
                stopped_epoch=row.stopped_epoch,
                train_examples=row.train_examples,
                val_examples=row.val_examples,
                test_examples=row.test_examples,
                parameter_count=row.parameter_count,
                test_accuracy=row.test_accuracy,
                test_macro_f1=row.test_macro_f1,
                test_loss=row.test_loss,
                generalization_gap_accuracy=gap,
            )
        )
    return with_gaps


def build_paired_stats_summary(
    rows: list[ExperimentRow],
    bootstrap_samples: int,
    permutation_samples: int,
) -> list[dict[str, object]]:
    baseline_method = "conv1d_baseline"
    baseline = {
        (row.eval_mode, row.seed): row.test_accuracy
        for row in rows
        if row.method == baseline_method
    }
    if not baseline:
        fallback = METHODS[0]
        baseline = {
            (row.eval_mode, row.seed): row.test_accuracy
            for row in rows
            if row.method == fallback
        }
        baseline_method = fallback

    summary: list[dict[str, object]] = []

    for method in METHODS:
        if method == baseline_method:
            continue
        for eval_mode in EVAL_MODES:
            candidate = {
                row.seed: row.test_accuracy
                for row in rows
                if row.method == method and row.eval_mode == eval_mode
            }
            shared_seeds = sorted(
                seed for seed in candidate if (eval_mode, seed) in baseline
            )
            deltas = [
                candidate[seed] - baseline[(eval_mode, seed)] for seed in shared_seeds
            ]
            wins = sum(delta > 0.0 for delta in deltas)
            losses = sum(delta < 0.0 for delta in deltas)
            ties = len(deltas) - wins - losses
            sign_p = two_sided_sign_test_pvalue(wins=wins, losses=losses)
            perm_p = paired_permutation_pvalue(
                deltas=deltas,
                permutation_samples=permutation_samples,
                seed=abs(hash((method, eval_mode))) % (2**31 - 1),
            )
            ci = bootstrap_ci_mean(
                values=deltas,
                bootstrap_samples=bootstrap_samples,
                seed=abs(hash((method, eval_mode, "bootstrap"))) % (2**31 - 1),
            )
            summary.append(
                {
                    "method": method,
                    "eval_mode": eval_mode,
                    "reference_method": baseline_method,
                    "paired_seed_count": len(shared_seeds),
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "mean_delta_accuracy": float(np.mean(deltas)) if deltas else None,
                    "delta_accuracy_ci95": list(ci) if ci is not None else None,
                    "sign_test_pvalue": sign_p,
                    "paired_permutation_pvalue": perm_p,
                }
            )
    return summary


def build_method_eval_summary(
    rows: list[ExperimentRow],
    bootstrap_samples: int,
) -> list[dict[str, object]]:
    """Build per-method/per-mode aggregate summary with CIs."""

    summary: list[dict[str, object]] = []
    for method in METHODS:
        for eval_mode in EVAL_MODES:
            values = [
                row.test_accuracy
                for row in rows
                if row.method == method and row.eval_mode == eval_mode
            ]
            ci = bootstrap_ci_mean(
                values=values,
                bootstrap_samples=bootstrap_samples,
                seed=abs(hash((method, eval_mode, "acc_ci"))) % (2**31 - 1),
            )
            summary.append(
                {
                    "method": method,
                    "eval_mode": eval_mode,
                    "count": len(values),
                    "mean_accuracy": float(np.mean(values)) if values else None,
                    "std_accuracy": float(np.std(values, ddof=1))
                    if len(values) > 1
                    else None,
                    "accuracy_ci95": list(ci) if ci is not None else None,
                }
            )
    return summary


def parse_seeds(raw_seeds: str) -> list[int]:
    """Parse comma-separated integer seeds."""

    return [int(token.strip()) for token in raw_seeds.split(",") if token.strip()]


def main() -> None:
    """Execute diagnosis experiments and write CSV plus summary JSON."""

    args = build_arg_parser().parse_args()
    seeds = parse_seeds(args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features, labels, environments = load_ut_har_amplitude(
        prepared_root=args.prepared_root,
        dataset=args.dataset,
    )

    rows: list[ExperimentRow] = []
    for method in METHODS:
        for eval_mode in EVAL_MODES:
            for seed in seeds:
                split_indices = build_splits(
                    labels=labels,
                    environments=environments,
                    seed=seed,
                    eval_mode=eval_mode,
                )
                metrics = train_and_evaluate_method(
                    method=method,
                    features=features,
                    labels=labels,
                    split_indices=split_indices,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden_dim,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    dropout=args.dropout,
                    aux_weight=args.aux_weight,
                    aug_weight=args.aug_weight,
                    grad_clip_norm=args.grad_clip_norm,
                    patience=args.patience,
                    min_delta=args.min_delta,
                    device=device,
                )
                row = ExperimentRow(
                    dataset=args.dataset,
                    method=method,
                    eval_mode=eval_mode,
                    seed=seed,
                    best_epoch=int(metrics["best_epoch"]),
                    stopped_epoch=int(metrics["stopped_epoch"]),
                    train_examples=int(metrics["train_examples"]),
                    val_examples=int(metrics["val_examples"]),
                    test_examples=int(metrics["test_examples"]),
                    parameter_count=int(metrics["parameter_count"]),
                    test_accuracy=float(metrics["test_accuracy"]),
                    test_macro_f1=float(metrics["test_macro_f1"]),
                    test_loss=float(metrics["test_loss"]),
                    generalization_gap_accuracy=None,
                )
                rows.append(row)
                print(
                    f"{method} | eval={eval_mode} seed={seed} "
                    f"acc={row.test_accuracy:.4f} f1={row.test_macro_f1:.4f} "
                    f"best_epoch={row.best_epoch} stop_epoch={row.stopped_epoch}"
                )

    rows_with_gaps = append_generalization_gaps(rows)
    paired_stats_summary = build_paired_stats_summary(
        rows=rows_with_gaps,
        bootstrap_samples=args.bootstrap_samples,
        permutation_samples=args.permutation_samples,
    )
    method_eval_summary = build_method_eval_summary(
        rows=rows_with_gaps,
        bootstrap_samples=args.bootstrap_samples,
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=list(asdict(rows_with_gaps[0]).keys()),
        )
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows_with_gaps)

    for method in METHODS:
        method_rows = [row for row in rows_with_gaps if row.method == method]
        iid_scores = [
            row.test_accuracy for row in method_rows if row.eval_mode == "iid"
        ]
        cross_scores = [
            row.test_accuracy for row in method_rows if row.eval_mode == "cross_env"
        ]
        if iid_scores and cross_scores:
            print(
                f"gap[{method}] iid-cross_env="
                f"{float(np.mean(iid_scores) - np.mean(cross_scores)):+.4f}"
            )

    for entry in paired_stats_summary:
        print(
            "paired-stats "
            f"{entry['method']} vs {entry['reference_method']} "
            f"eval={entry['eval_mode']} n={entry['paired_seed_count']} "
            f"wins={entry['wins']} losses={entry['losses']} ties={entry['ties']} "
            f"mean_delta={entry['mean_delta_accuracy']} "
            f"sign_p={entry['sign_test_pvalue']} "
            f"perm_p={entry['paired_permutation_pvalue']}"
        )

    summary_payload = {
        "dataset": args.dataset,
        "methods": list(METHODS),
        "eval_modes": list(EVAL_MODES),
        "seeds": seeds,
        "training_config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_dim": args.hidden_dim,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "aux_weight": args.aux_weight,
            "aug_weight": args.aug_weight,
            "grad_clip_norm": args.grad_clip_norm,
            "patience": args.patience,
            "min_delta": args.min_delta,
        },
        "output_csv": str(args.output_csv),
        "method_eval_summary": method_eval_summary,
        "paired_stats_vs_conv1d_baseline": paired_stats_summary,
    }
    summary_path = args.output_csv.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    print(f"Results written to: {args.output_csv}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
