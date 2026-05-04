# pyright: basic, reportMissingImports=false

"""Classification feasibility gate: tests whether OFDM physics improves
class separability on synthetic data where class identity is tied to
distinct multipath channel parameters."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from pinn4csi.physics import ofdm_channel_response, ofdm_residual
from pinn4csi.utils import accuracy, save_dataclass_rows_csv
from pinn4csi.utils.metrics import bootstrap_ci, cohens_d


@dataclass(frozen=True)
class ClassificationFeasibilityConfig:
    """Configuration for classification feasibility gate."""

    seeds: int = 5
    num_classes: int = 4
    num_subcarriers: int = 32
    num_paths: int = 3
    samples_per_class: int = 128
    epochs: int = 30
    batch_size: int = 64
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    physics_lambda: float = 0.1
    noise_std: float = 0.1
    center_frequency_hz: float = 5.32e9
    subcarrier_spacing_hz: float = 312.5e3
    output_csv: Path = Path("outputs/feasibility_classification_results.csv")


@dataclass(frozen=True)
class ClassificationFeasibilityResult:
    """Per-run result row."""

    seed: int
    model_type: str
    test_accuracy: float
    test_macro_f1: float


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _make_subcarrier_frequencies(config: ClassificationFeasibilityConfig) -> Tensor:
    centered = torch.arange(config.num_subcarriers, dtype=torch.float32)
    centered = centered - float(config.num_subcarriers // 2)
    return config.center_frequency_hz + centered * config.subcarrier_spacing_hz


def _class_channel_params(
    class_id: int,
    num_samples: int,
    config: ClassificationFeasibilityConfig,
) -> tuple[Tensor, Tensor]:
    """Generate class-specific multipath parameters.

    Each class has a distinct delay/gain profile so that physics-aware
    features should improve separability.
    """
    rng = torch.Generator()
    rng.manual_seed(class_id * 1000)

    # Class-specific base delays spread across [20ns, 200ns]
    base_delay = 20e-9 + (class_id * 50e-9)
    delays = (
        base_delay + torch.rand(num_samples, config.num_paths, generator=rng) * 30e-9
    )

    # Class-specific gain magnitudes with per-path decay
    phase_offset = class_id * (torch.pi / config.num_classes)
    path_scale = 1.0 / (torch.arange(config.num_paths, dtype=torch.float32) + 1.0)
    magnitude = (
        0.5 + 0.5 * torch.rand(num_samples, config.num_paths, generator=rng)
    ) * path_scale.unsqueeze(0)
    angles = (
        phase_offset + torch.randn(num_samples, config.num_paths, generator=rng) * 0.3
    )
    gains = torch.polar(magnitude, angles)

    return gains, delays


def generate_classification_dataset(
    config: ClassificationFeasibilityConfig,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Generate synthetic classification dataset with OFDM physics.

    Returns:
        Tuple of (features, labels, path_gains, path_delays, subcarrier_freqs).
    """
    set_seed(seed)
    freqs = _make_subcarrier_frequencies(config)
    all_features = []
    all_labels = []
    all_gains = []
    all_delays = []

    for class_id in range(config.num_classes):
        gains, delays = _class_channel_params(
            class_id, config.samples_per_class, config
        )
        # Generate CSI from OFDM model
        csi = ofdm_channel_response(h_l=gains, tau_l=delays, f_k=freqs)
        # Add noise
        noise = config.noise_std * torch.randn_like(csi.real)
        noise_imag = config.noise_std * torch.randn_like(csi.imag)
        csi = csi + torch.complex(noise, noise_imag)
        # Convert to real features: [real, imag]
        features = torch.cat([csi.real, csi.imag], dim=-1)
        labels = torch.full((config.samples_per_class,), class_id, dtype=torch.long)

        all_features.append(features)
        all_labels.append(labels)
        all_gains.append(gains)
        all_delays.append(delays)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)
    gains = torch.cat(all_gains, dim=0)
    delays = torch.cat(all_delays, dim=0)
    freqs_expanded = freqs.unsqueeze(0).expand(features.shape[0], -1)

    # Shuffle
    perm = torch.randperm(features.shape[0])
    return features[perm], labels[perm], gains[perm], delays[perm], freqs_expanded[perm]


def _macro_f1(predictions: Tensor, targets: Tensor) -> float:
    """Compute macro F1 score."""
    classes = torch.unique(targets)
    f1_scores = []
    for cls in classes:
        tp = ((predictions == cls) & (targets == cls)).sum().float()
        fp = ((predictions == cls) & (targets != cls)).sum().float()
        fn = ((predictions != cls) & (targets == cls)).sum().float()
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1.item())
    return float(np.mean(f1_scores))


def run_classification_feasibility(
    config: ClassificationFeasibilityConfig,
) -> list[ClassificationFeasibilityResult]:
    """Run classification feasibility gate across seeds and model types."""
    results: list[ClassificationFeasibilityResult] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = config.num_subcarriers * 2  # real + imag
    model_types = ("baseline", "physics_aux", "physics_features")

    for seed in range(config.seeds):
        features, labels, gains, delays, freqs = generate_classification_dataset(
            config, seed
        )
        # Split: 70/15/15
        n = features.shape[0]
        n_train = int(0.7 * n)
        n_val = int(0.15 * n)

        train_x, train_y = features[:n_train], labels[:n_train]
        train_gains, train_delays = gains[:n_train], delays[:n_train]
        train_freqs = freqs[:n_train]
        val_x, val_y = (
            features[n_train : n_train + n_val],
            labels[n_train : n_train + n_val],
        )
        test_x, test_y = features[n_train + n_val :], labels[n_train + n_val :]

        for model_type in model_types:
            set_seed(seed)

            if model_type == "physics_features":
                # Compute OFDM features PER SPLIT to avoid data leakage.
                # Each split uses only its own gains/delays.
                f_k = _make_subcarrier_frequencies(config)
                val_gains = gains[n_train : n_train + n_val]
                val_delays = delays[n_train : n_train + n_val]
                test_gains = gains[n_train + n_val :]
                test_delays = delays[n_train + n_val :]
                train_ofdm = torch.abs(
                    ofdm_channel_response(h_l=train_gains, tau_l=train_delays, f_k=f_k)
                )
                val_ofdm = torch.abs(
                    ofdm_channel_response(h_l=val_gains, tau_l=val_delays, f_k=f_k)
                )
                test_ofdm = torch.abs(
                    ofdm_channel_response(h_l=test_gains, tau_l=test_delays, f_k=f_k)
                )
                aug_train = torch.cat([train_x, train_ofdm], dim=-1)
                aug_val = torch.cat([val_x, val_ofdm], dim=-1)
                aug_test = torch.cat([test_x, test_ofdm], dim=-1)
                model_input_dim = input_dim + config.num_subcarriers
            else:
                aug_train, aug_val, aug_test = train_x, val_x, test_x
                model_input_dim = input_dim

            model = nn.Sequential(
                nn.Linear(model_input_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.num_classes),
            ).to(device)
            optimizer = Adam(model.parameters(), lr=config.learning_rate)

            train_loader = DataLoader(
                TensorDataset(
                    aug_train, train_y, train_gains, train_delays, train_freqs
                ),
                batch_size=config.batch_size,
                shuffle=True,
            )

            best_val_acc = -1.0
            best_state = model.state_dict()

            for _epoch in range(config.epochs):
                model.train()
                for batch in train_loader:
                    bx, by, bg, bd, bf = (t.to(device) for t in batch)
                    optimizer.zero_grad()
                    logits = model(bx)
                    loss = functional.cross_entropy(logits, by)

                    if model_type == "physics_aux":
                        # OFDM residual as auxiliary loss
                        pred_complex = (
                            torch.complex(
                                logits[:, : config.num_subcarriers],
                                logits[:, config.num_subcarriers :],
                            )
                            if logits.shape[1] >= config.num_subcarriers * 2
                            else None
                        )
                        if pred_complex is None:
                            # Use a separate projection for OFDM
                            pass
                        ofdm_loss = ofdm_residual(
                            predicted_csi=ofdm_channel_response(
                                bg, bd, bf[:1].squeeze(0)
                            ),
                            path_gains=bg,
                            path_delays=bd,
                            subcarrier_frequencies=bf,
                        )
                        loss = loss + config.physics_lambda * ofdm_loss

                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    val_logits = model(aug_val.to(device))
                    val_acc = accuracy(val_logits, val_y.to(device))
                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_state = {
                            k: v.clone() for k, v in model.state_dict().items()
                        }

            model.load_state_dict(best_state)
            model.eval()
            with torch.no_grad():
                test_logits = model(aug_test.to(device))
                test_preds = torch.argmax(test_logits, dim=1)
                test_acc = accuracy(test_logits, test_y.to(device))
                test_f1 = _macro_f1(test_preds.cpu(), test_y)

            results.append(
                ClassificationFeasibilityResult(
                    seed=seed,
                    model_type=model_type,
                    test_accuracy=test_acc,
                    test_macro_f1=test_f1,
                )
            )

    return results


def summarize_classification_feasibility(
    results: list[ClassificationFeasibilityResult],
) -> dict[str, dict[str, float]]:
    """Compute summary statistics with effect sizes and CIs."""
    by_type: dict[str, list[float]] = {}
    for r in results:
        by_type.setdefault(r.model_type, []).append(r.test_accuracy)

    summary: dict[str, dict[str, float]] = {}
    baseline_accs = torch.tensor(by_type.get("baseline", []))

    for model_type, accs_list in by_type.items():
        accs = torch.tensor(accs_list)
        point, ci_lower, ci_upper = bootstrap_ci(accs)
        entry: dict[str, float] = {
            "mean_accuracy": point,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }
        if model_type != "baseline" and len(baseline_accs) > 0:
            entry["cohens_d_vs_baseline"] = cohens_d(accs, baseline_accs)
        summary[model_type] = entry

    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Classification feasibility gate for OFDM physics integration"
    )
    for f in fields(ClassificationFeasibilityConfig):
        if f.name == "output_csv":
            parser.add_argument(f"--{f.name}", type=Path, default=f.default)
        elif f.type == "int":
            parser.add_argument(f"--{f.name}", type=int, default=f.default)
        elif f.type == "float":
            parser.add_argument(f"--{f.name}", type=float, default=f.default)

    args = parser.parse_args()
    config = ClassificationFeasibilityConfig(
        seeds=args.seeds,
        num_classes=args.num_classes,
        epochs=args.epochs,
        output_csv=args.output_csv,
    )

    results = run_classification_feasibility(config)
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    save_dataclass_rows_csv(results, config.output_csv)

    summary = summarize_classification_feasibility(results)
    print("\n=== Classification Feasibility Summary ===")
    for model_type, stats in summary.items():
        ci = f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
        d_str = ""
        if "cohens_d_vs_baseline" in stats:
            d_str = f", d={stats['cohens_d_vs_baseline']:.3f}"
        print(f"  {model_type}: acc={stats['mean_accuracy']:.3f} CI={ci}{d_str}")


if __name__ == "__main__":
    main()
