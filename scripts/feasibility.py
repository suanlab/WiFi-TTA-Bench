# pyright: basic, reportMissingImports=false

from __future__ import annotations

import argparse
import copy
import math
import random
from dataclasses import dataclass, fields
from pathlib import Path
from typing import cast

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from pinn4csi.models import PINN
from pinn4csi.physics import (
    ofdm_channel_response,
    ofdm_residual,
    subcarrier_correlation_loss,
)
from pinn4csi.utils import parse_csv_floats, save_dataclass_rows_csv


@dataclass(frozen=True)
class FeasibilityConfig:
    """Configuration for the OFDM pivot feasibility re-run."""

    seeds: int = 3
    lambdas: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0, 10.0)
    epochs: int = 40
    batch_size: int = 64
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    num_layers: int = 3
    input_dim: int = 6
    num_paths: int = 3
    num_subcarriers: int = 32
    train_samples: int = 512
    val_samples: int = 128
    test_samples: int = 512
    center_frequency_hz: float = 5.32e9
    subcarrier_spacing_hz: float = 312.5e3
    noise_std: float = 0.05
    nuisance_scale: float = 0.35
    correlation_weight: float = 0.1
    lambda_min: float = 1e-6
    lambda_max: float = 1e6
    adaptive_eps: float = 1e-8
    output_csv: Path = Path("outputs/feasibility_ofdm_results.csv")


@dataclass(frozen=True)
class ExperimentResult:
    """Per-run record written to the feasibility CSV."""

    seed: int
    model_type: str
    initial_lambda: float
    best_epoch: int
    val_rmse: float
    test_rmse: float
    val_nmse: float
    test_nmse: float
    final_lambda: float


class SyntheticOFDMDataset(Dataset[tuple[Tensor, Tensor, dict[str, Tensor]]]):
    """Synthetic CSI dataset with OFDM physics metadata per sample."""

    def __init__(
        self,
        features: Tensor,
        targets: Tensor,
        path_gains: Tensor,
        path_delays: Tensor,
        subcarrier_frequencies: Tensor,
    ) -> None:
        super().__init__()
        self.features = features
        self.targets = targets
        self.path_gains_real = path_gains.real
        self.path_gains_imag = path_gains.imag
        self.path_delays = path_delays
        self.subcarrier_frequencies = subcarrier_frequencies

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        physics = {
            "path_gains_real": self.path_gains_real[index],
            "path_gains_imag": self.path_gains_imag[index],
            "path_delays": self.path_delays[index],
            "subcarrier_frequencies": self.subcarrier_frequencies[index],
        }
        return self.features[index], self.targets[index], physics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _make_subcarrier_frequencies(config: FeasibilityConfig) -> Tensor:
    centered = torch.arange(config.num_subcarriers, dtype=torch.float32)
    centered = centered - float(config.num_subcarriers // 2)
    return config.center_frequency_hz + centered * config.subcarrier_spacing_hz


def _features_to_channel_params(
    features: Tensor,
    config: FeasibilityConfig,
) -> tuple[Tensor, Tensor]:
    channel_features = features[:, :3]

    coeff_real = torch.tensor(
        [[0.9, -0.2, 0.4], [-0.3, 0.8, 0.1], [0.4, 0.2, -0.7]],
        dtype=features.dtype,
    )[: config.num_paths]
    coeff_imag = torch.tensor(
        [[-0.5, 0.7, 0.3], [0.6, -0.4, 0.5], [0.2, 0.3, 0.6]],
        dtype=features.dtype,
    )[: config.num_paths]
    coeff_delay = torch.tensor(
        [[0.6, -0.1, 0.5], [-0.2, 0.7, 0.3], [0.4, 0.4, -0.6]],
        dtype=features.dtype,
    )[: config.num_paths]

    gains_real = torch.tanh(channel_features @ coeff_real.T)
    gains_imag = torch.tanh(channel_features @ coeff_imag.T)
    path_scale = (
        1.0 / (torch.arange(config.num_paths, dtype=features.dtype) + 1.0)
    ).unsqueeze(0)
    path_gains = torch.complex(gains_real, gains_imag) * path_scale

    delay_logits = channel_features @ coeff_delay.T
    path_delays = 20e-9 + 180e-9 * torch.sigmoid(delay_logits)
    return path_gains, path_delays


def _environment_nuisance(
    features: Tensor,
    subcarrier_frequencies: Tensor,
    environment: str,
    config: FeasibilityConfig,
) -> Tensor:
    nuisance_features = features[:, 3:]
    if nuisance_features.shape[1] < 2:
        nuisance_features = torch.nn.functional.pad(
            nuisance_features,
            (0, 2 - nuisance_features.shape[1]),
        )

    normalized_freq = subcarrier_frequencies - subcarrier_frequencies.mean()
    normalized_freq = normalized_freq / (torch.max(torch.abs(normalized_freq)) + 1e-6)

    if environment == "A":
        amp = 0.35 * nuisance_features[:, 0:1] - 0.25 * nuisance_features[:, 1:2]
        phase = 1.8 * nuisance_features[:, 0:1] + 0.7 * nuisance_features[:, 1:2]
    elif environment == "B":
        amp = -0.30 * nuisance_features[:, 0:1] + 0.40 * nuisance_features[:, 1:2]
        phase = -1.4 * nuisance_features[:, 0:1] + 1.1 * nuisance_features[:, 1:2]
    else:
        raise ValueError(f"Unknown environment: {environment}")

    amp_profile = amp * torch.tanh(normalized_freq.unsqueeze(0))
    phase_profile = phase * normalized_freq.unsqueeze(0)
    return config.nuisance_scale * torch.complex(amp_profile, phase_profile)


def _complex_to_real_imag(csi: Tensor) -> Tensor:
    return torch.cat([csi.real, csi.imag], dim=1)


def _real_imag_to_complex(stacked: Tensor, num_subcarriers: int) -> Tensor:
    real = stacked[:, :num_subcarriers]
    imag = stacked[:, num_subcarriers:]
    return torch.complex(real, imag)


def _analytical_ofdm_prior(
    physics: dict[str, Tensor],
    device: torch.device,
) -> Tensor:
    path_gains = torch.complex(
        physics["path_gains_real"].to(device),
        physics["path_gains_imag"].to(device),
    )
    path_delays = physics["path_delays"].to(device)
    frequencies = physics["subcarrier_frequencies"].to(device)
    return ofdm_channel_response(
        h_l=path_gains,
        tau_l=path_delays,
        f_k=frequencies,
    )


def _predict_stacked_response(
    model: nn.Module,
    x_device: Tensor,
    physics: dict[str, Tensor],
    model_type: str,
) -> Tensor:
    network_output = cast(Tensor, model(x_device))
    if model_type in {"baseline", "ofdm_pinn"}:
        return network_output
    if model_type == "ofdm_residual":
        prior_complex = _analytical_ofdm_prior(physics=physics, device=x_device.device)
        prior_stacked = _complex_to_real_imag(prior_complex)
        return prior_stacked + network_output
    raise ValueError(f"Unknown model_type: {model_type}")


def generate_environment_dataset(
    num_samples: int,
    environment: str,
    seed: int,
    config: FeasibilityConfig,
) -> SyntheticOFDMDataset:
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(num_samples, config.input_dim, generator=generator)

    path_gains, path_delays = _features_to_channel_params(features, config)
    frequencies = _make_subcarrier_frequencies(config)
    clean_csi = torch.zeros(num_samples, config.num_subcarriers, dtype=torch.complex64)
    for idx in range(num_samples):
        per_sample_gains = path_gains[idx]
        per_sample_delays = path_delays[idx]
        phase = -2.0 * math.pi * per_sample_delays.unsqueeze(-1) * frequencies
        steering = torch.exp(torch.complex(torch.zeros_like(phase), phase))
        clean_csi[idx] = torch.sum(per_sample_gains.unsqueeze(-1) * steering, dim=0)

    nuisance = _environment_nuisance(features, frequencies, environment, config)
    noise_real = torch.randn(num_samples, config.num_subcarriers, generator=generator)
    noise_imag = torch.randn(num_samples, config.num_subcarriers, generator=generator)
    noise = config.noise_std * torch.complex(noise_real, noise_imag)
    observed_csi = clean_csi + nuisance + noise

    targets = _complex_to_real_imag(observed_csi)
    subcarrier_frequencies = frequencies.unsqueeze(0).repeat(num_samples, 1)

    return SyntheticOFDMDataset(
        features=features,
        targets=targets,
        path_gains=path_gains,
        path_delays=path_delays,
        subcarrier_frequencies=subcarrier_frequencies,
    )


def _subset_dataset(
    dataset: SyntheticOFDMDataset, indices: Tensor
) -> SyntheticOFDMDataset:
    return SyntheticOFDMDataset(
        features=dataset.features[indices],
        targets=dataset.targets[indices],
        path_gains=torch.complex(
            dataset.path_gains_real[indices],
            dataset.path_gains_imag[indices],
        ),
        path_delays=dataset.path_delays[indices],
        subcarrier_frequencies=dataset.subcarrier_frequencies[indices],
    )


def build_cross_environment_loaders(
    seed: int,
    config: FeasibilityConfig,
) -> tuple[
    DataLoader[tuple[Tensor, Tensor, dict[str, Tensor]]],
    DataLoader[tuple[Tensor, Tensor, dict[str, Tensor]]],
    DataLoader[tuple[Tensor, Tensor, dict[str, Tensor]]],
]:
    env_a = generate_environment_dataset(
        num_samples=config.train_samples + config.val_samples,
        environment="A",
        seed=seed,
        config=config,
    )
    env_b = generate_environment_dataset(
        num_samples=config.test_samples,
        environment="B",
        seed=seed + 100_000,
        config=config,
    )

    generator = torch.Generator().manual_seed(seed + 7)
    permutation = torch.randperm(len(env_a), generator=generator)
    train_idx = permutation[: config.train_samples]
    val_idx = permutation[config.train_samples :]

    train_dataset = _subset_dataset(env_a, train_idx)
    val_dataset = _subset_dataset(env_a, val_idx)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(env_b, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def _grad_norm(loss: Tensor, model: nn.Module, eps: float) -> Tensor:
    params = [param for param in model.parameters() if param.requires_grad]
    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    squared_norm = torch.zeros((), device=loss.device)
    for grad in grads:
        if grad is not None:
            squared_norm = squared_norm + torch.sum(grad * grad)
    return torch.sqrt(squared_norm + eps)


def evaluate_csi(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor, dict[str, Tensor]]],
    device: torch.device,
    model_type: str,
    num_subcarriers: int,
) -> tuple[float, float]:
    model.eval()
    predictions: list[Tensor] = []
    targets: list[Tensor] = []

    with torch.no_grad():
        for x, y, physics in loader:
            x_device = x.to(device)
            y_pred = _predict_stacked_response(
                model=model,
                x_device=x_device,
                physics=physics,
                model_type=model_type,
            ).cpu()
            predictions.append(y_pred)
            targets.append(y.cpu())

    pred = torch.cat(predictions, dim=0)
    target = torch.cat(targets, dim=0)
    pred_complex = _real_imag_to_complex(pred, num_subcarriers)
    target_complex = _real_imag_to_complex(target, num_subcarriers)

    mse = torch.mean(torch.abs(pred_complex - target_complex) ** 2)
    rmse = float(torch.sqrt(mse).item())
    target_power = torch.mean(torch.abs(target_complex) ** 2)
    nmse = float((mse / target_power).item()) if target_power.item() > 0 else 0.0
    return rmse, nmse


def run_single_experiment(
    seed: int,
    model_type: str,
    initial_lambda: float,
    adaptive_lambda: bool,
    config: FeasibilityConfig,
) -> ExperimentResult:
    """Train one baseline, OFDM-PINN, or residual-prior run."""
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = build_cross_environment_loaders(
        seed, config
    )

    model = PINN(
        in_features=config.input_dim,
        out_features=2 * config.num_subcarriers,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        activation="relu",
    ).to(device)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    current_lambda = float(initial_lambda)
    best_state: dict[str, Tensor] | None = None
    best_epoch = -1
    best_val_nmse = math.inf
    best_val_rmse = math.inf

    for epoch_idx in range(config.epochs):
        model.train()
        for x, y, physics in train_loader:
            x_device = x.to(device)
            y_device = y.to(device)
            optimizer.zero_grad()

            predictions = _predict_stacked_response(
                model=model,
                x_device=x_device,
                physics=physics,
                model_type=model_type,
            )
            loss_data = loss_fn(predictions, y_device)

            if model_type == "ofdm_pinn":
                pred_complex = _real_imag_to_complex(
                    predictions,
                    config.num_subcarriers,
                )
                path_gains = torch.complex(
                    physics["path_gains_real"].to(device),
                    physics["path_gains_imag"].to(device),
                )
                path_delays = physics["path_delays"].to(device)
                frequencies = physics["subcarrier_frequencies"].to(device)
                loss_physics = ofdm_residual(
                    predicted_csi=pred_complex,
                    path_gains=path_gains,
                    path_delays=path_delays,
                    subcarrier_frequencies=frequencies,
                ) + config.correlation_weight * subcarrier_correlation_loss(
                    pred_complex
                )

                if adaptive_lambda:
                    data_norm = _grad_norm(loss_data, model, config.adaptive_eps)
                    physics_norm = _grad_norm(loss_physics, model, config.adaptive_eps)
                    ratio = data_norm / (physics_norm + config.adaptive_eps)
                    clamped = torch.clamp(
                        ratio,
                        min=config.lambda_min,
                        max=config.lambda_max,
                    )
                    current_lambda = float(clamped.detach().item())
            elif model_type in {"baseline", "ofdm_residual"}:
                loss_physics = torch.zeros((), device=device)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            loss_total = loss_data + current_lambda * loss_physics
            loss_total.backward()
            optimizer.step()

        val_rmse, val_nmse = evaluate_csi(
            model,
            val_loader,
            device,
            model_type=model_type,
            num_subcarriers=config.num_subcarriers,
        )
        if val_nmse < best_val_nmse:
            best_val_nmse = val_nmse
            best_val_rmse = val_rmse
            best_epoch = epoch_idx
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("No validation checkpoint selected.")

    model.load_state_dict(best_state)
    test_rmse, test_nmse = evaluate_csi(
        model,
        test_loader,
        device,
        model_type=model_type,
        num_subcarriers=config.num_subcarriers,
    )

    return ExperimentResult(
        seed=seed,
        model_type=model_type,
        initial_lambda=initial_lambda,
        best_epoch=best_epoch,
        val_rmse=best_val_rmse,
        test_rmse=test_rmse,
        val_nmse=best_val_nmse,
        test_nmse=test_nmse,
        final_lambda=current_lambda,
    )


def run_feasibility(config: FeasibilityConfig) -> list[ExperimentResult]:
    """Execute baseline, OFDM-PINN, and residual-prior runs across seeds."""
    results: list[ExperimentResult] = []
    for seed in range(config.seeds):
        baseline_result = run_single_experiment(
            seed=seed,
            model_type="baseline",
            initial_lambda=0.0,
            adaptive_lambda=False,
            config=config,
        )
        results.append(baseline_result)

        residual_result = run_single_experiment(
            seed=seed,
            model_type="ofdm_residual",
            initial_lambda=0.0,
            adaptive_lambda=False,
            config=config,
        )
        results.append(residual_result)

        for initial_lambda in config.lambdas:
            ofdm_result = run_single_experiment(
                seed=seed,
                model_type="ofdm_pinn",
                initial_lambda=initial_lambda,
                adaptive_lambda=True,
                config=config,
            )
            results.append(ofdm_result)

    return results


def save_results_csv(results: list[ExperimentResult], output_csv: Path) -> None:
    """Write machine-readable per-run results for comparison analysis."""
    save_dataclass_rows_csv(
        results,
        output_csv,
        fieldnames=[field.name for field in fields(ExperimentResult)],
    )


def summarize_results(results: list[ExperimentResult]) -> dict[str, float]:
    """Summarize validation-selected performance for all feasibility modes."""

    def _select_best_per_seed(rows: list[ExperimentResult]) -> list[ExperimentResult]:
        best_rows: list[ExperimentResult] = []
        for seed in sorted({row.seed for row in rows}):
            seed_rows = [row for row in rows if row.seed == seed]
            best_rows.append(min(seed_rows, key=lambda row: row.val_nmse))
        return best_rows

    baseline_rows = [row for row in results if row.model_type == "baseline"]
    ofdm_pinn_rows = [row for row in results if row.model_type == "ofdm_pinn"]
    residual_rows = [row for row in results if row.model_type == "ofdm_residual"]
    if not baseline_rows:
        raise ValueError("Missing baseline rows in results.")
    if not ofdm_pinn_rows:
        raise ValueError("Missing OFDM-PINN rows in results.")
    if not residual_rows:
        raise ValueError("Missing OFDM-residual rows in results.")

    baseline_mean_test_nmse = float(np.mean([row.test_nmse for row in baseline_rows]))
    baseline_mean_test_rmse = float(np.mean([row.test_rmse for row in baseline_rows]))

    best_ofdm_pinn_per_seed = _select_best_per_seed(ofdm_pinn_rows)
    best_residual_per_seed = _select_best_per_seed(residual_rows)

    best_ofdm_pinn_mean_test_nmse = float(
        np.mean([row.test_nmse for row in best_ofdm_pinn_per_seed])
    )
    best_ofdm_pinn_mean_test_rmse = float(
        np.mean([row.test_rmse for row in best_ofdm_pinn_per_seed])
    )
    best_residual_mean_test_nmse = float(
        np.mean([row.test_nmse for row in best_residual_per_seed])
    )
    best_residual_mean_test_rmse = float(
        np.mean([row.test_rmse for row in best_residual_per_seed])
    )
    return {
        "baseline_mean_test_nmse": baseline_mean_test_nmse,
        "best_ofdm_pinn_mean_test_nmse": best_ofdm_pinn_mean_test_nmse,
        "best_ofdm_residual_mean_test_nmse": best_residual_mean_test_nmse,
        "delta_ofdm_pinn_test_nmse": best_ofdm_pinn_mean_test_nmse
        - baseline_mean_test_nmse,
        "delta_ofdm_residual_test_nmse": best_residual_mean_test_nmse
        - baseline_mean_test_nmse,
        "baseline_mean_test_rmse": baseline_mean_test_rmse,
        "best_ofdm_pinn_mean_test_rmse": best_ofdm_pinn_mean_test_rmse,
        "best_ofdm_residual_mean_test_rmse": best_residual_mean_test_rmse,
        "delta_ofdm_pinn_test_rmse": best_ofdm_pinn_mean_test_rmse
        - baseline_mean_test_rmse,
        "delta_ofdm_residual_test_rmse": best_residual_mean_test_rmse
        - baseline_mean_test_rmse,
    }


def parse_lambdas(raw: str) -> tuple[float, ...]:
    """Parse comma-separated lambda values from CLI text input."""
    return parse_csv_floats(raw)


def build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI parser for OFDM feasibility rerun."""
    parser = argparse.ArgumentParser(
        description="Option B OFDM feasibility rerun for PINN4CSI"
    )
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--lambdas", type=str, default="0.001,0.01,0.1,1.0,10.0")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--input-dim", type=int, default=6)
    parser.add_argument("--num-paths", type=int, default=3)
    parser.add_argument("--num-subcarriers", type=int, default=32)
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=128)
    parser.add_argument("--test-samples", type=int, default=512)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/feasibility_ofdm_results.csv"),
    )
    return parser


def main() -> None:
    """Run OFDM pivot feasibility script and print compact summary."""
    parser = build_arg_parser()
    args = parser.parse_args()
    config = FeasibilityConfig(
        seeds=args.seeds,
        lambdas=parse_lambdas(args.lambdas),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        input_dim=args.input_dim,
        num_paths=args.num_paths,
        num_subcarriers=args.num_subcarriers,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        test_samples=args.test_samples,
        output_csv=args.output_csv,
    )

    results = run_feasibility(config)
    save_results_csv(results, config.output_csv)
    summary = summarize_results(results)

    print(f"Results written to: {config.output_csv}")
    print(f"Baseline mean test NMSE: {summary['baseline_mean_test_nmse']:.4f}")
    print(
        f"Best OFDM-PINN mean test NMSE: {summary['best_ofdm_pinn_mean_test_nmse']:.4f}"
    )
    print(
        "Best OFDM-residual mean test NMSE: "
        f"{summary['best_ofdm_residual_mean_test_nmse']:.4f}"
    )
    print(
        f"Delta (OFDM-PINN - baseline) NMSE: {summary['delta_ofdm_pinn_test_nmse']:.4f}"
    )
    print(
        "Delta (OFDM-residual - baseline) NMSE: "
        f"{summary['delta_ofdm_residual_test_nmse']:.4f}"
    )
    print(f"Baseline mean test RMSE: {summary['baseline_mean_test_rmse']:.4f}")
    print(
        f"Best OFDM-PINN mean test RMSE: {summary['best_ofdm_pinn_mean_test_rmse']:.4f}"
    )
    print(
        "Best OFDM-residual mean test RMSE: "
        f"{summary['best_ofdm_residual_mean_test_rmse']:.4f}"
    )
    print(
        f"Delta (OFDM-PINN - baseline) RMSE: {summary['delta_ofdm_pinn_test_rmse']:.4f}"
    )
    print(
        "Delta (OFDM-residual - baseline) RMSE: "
        f"{summary['delta_ofdm_residual_test_rmse']:.4f}"
    )


if __name__ == "__main__":
    main()
