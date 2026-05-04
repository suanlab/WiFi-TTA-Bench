# pyright: basic, reportMissingImports=false

"""Paper 3 synthetic validation: Neural Operator (DeepONet) for
environment-parametric CSI prediction with OFDM physics constraints."""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from pinn4csi.models import PhysicsInformedDeepONet
from pinn4csi.physics import ofdm_channel_response
from pinn4csi.utils import nmse, save_dataclass_rows_csv


@dataclass(frozen=True)
class NeuralOperatorExperimentConfig:
    """Configuration for Paper 3 synthetic experiments."""

    seeds: tuple[int, ...] = tuple(range(5))
    environment_dim: int = 6
    num_subcarriers: int = 32
    num_paths: int = 3
    train_samples: int = 256
    test_samples: int = 128
    epochs: int = 30
    batch_size: int = 32
    hidden_dim: int = 64
    latent_dim: int = 64
    learning_rate: float = 1e-3
    ofdm_weight: float = 0.1
    correlation_weight: float = 0.05
    center_frequency_hz: float = 5.32e9
    subcarrier_spacing_hz: float = 312.5e3
    output_csv: Path = Path("outputs/neural_operator_results.csv")


@dataclass(frozen=True)
class NeuralOperatorResult:
    """Per-run result row."""

    seed: int
    model_type: str
    test_rmse: float
    test_nmse_db: float


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_subcarrier_frequencies(config: NeuralOperatorExperimentConfig) -> Tensor:
    centered = torch.arange(config.num_subcarriers, dtype=torch.float32)
    centered = centered - float(config.num_subcarriers // 2)
    return config.center_frequency_hz + centered * config.subcarrier_spacing_hz


def _generate_dataset(
    num_samples: int,
    config: NeuralOperatorExperimentConfig,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Generate synthetic environment → CSI dataset.

    Returns:
        (env_features, query_coords, target_response, path_gains, path_delays)
    """
    _set_seed(seed)
    freqs = _make_subcarrier_frequencies(config)

    # Random environment features
    env_features = torch.randn(num_samples, config.environment_dim)

    # Deterministic mapping: env_features → channel params
    gain_proj = torch.randn(config.environment_dim, config.num_paths * 2)
    delay_proj = torch.randn(config.environment_dim, config.num_paths)

    gain_vals = torch.tanh(env_features @ gain_proj)
    gains_real = gain_vals[:, : config.num_paths]
    gains_imag = gain_vals[:, config.num_paths :]
    path_scale = 1.0 / (torch.arange(config.num_paths, dtype=torch.float32) + 1.0)
    path_gains = torch.complex(gains_real, gains_imag) * path_scale.unsqueeze(0)

    delay_logits = env_features @ delay_proj
    path_delays = 20e-9 + 180e-9 * torch.sigmoid(delay_logits)

    # Generate target CSI
    target_complex = ofdm_channel_response(h_l=path_gains, tau_l=path_delays, f_k=freqs)
    noise = 0.02 * torch.randn_like(target_complex.real)
    target_complex = target_complex + torch.complex(noise, noise)
    target_response = torch.stack([target_complex.real, target_complex.imag], dim=-1)

    # Query coordinates: normalized subcarrier frequencies
    query_coords = freqs.unsqueeze(-1)  # (num_subcarriers, 1)
    query_coords = (query_coords - query_coords.mean()) / (query_coords.std() + 1e-8)

    return env_features, query_coords, target_response, path_gains, path_delays


def run_neural_operator_experiments(
    config: NeuralOperatorExperimentConfig,
) -> list[NeuralOperatorResult]:
    """Run Paper 3 synthetic experiments."""
    results: list[NeuralOperatorResult] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    freqs = _make_subcarrier_frequencies(config)

    for seed in config.seeds:
        (train_env, query_coords, train_target, train_gains, train_delays) = (
            _generate_dataset(config.train_samples, config, seed)
        )
        (test_env, _, test_target, test_gains, test_delays) = _generate_dataset(
            config.test_samples, config, seed + 1000
        )
        freqs_batch_train = freqs.unsqueeze(0).expand(config.train_samples, -1)

        for model_type in ("baseline", "physics_informed"):
            _set_seed(seed)
            model = PhysicsInformedDeepONet(
                environment_dim=config.environment_dim,
                query_dim=1,
                hidden_dim=config.hidden_dim,
                latent_dim=config.latent_dim,
                output_channels=2,
            ).to(device)
            optimizer = Adam(model.parameters(), lr=config.learning_rate)

            train_loader = DataLoader(
                TensorDataset(
                    train_env,
                    train_target,
                    train_gains.real,
                    train_gains.imag,
                    train_delays,
                    freqs_batch_train,
                ),
                batch_size=config.batch_size,
                shuffle=True,
            )
            qc = query_coords.to(device)

            for _epoch in range(config.epochs):
                model.train()
                for batch in train_loader:
                    b_env, b_target, b_gr, b_gi, b_del, b_freq = (
                        t.to(device) for t in batch
                    )
                    optimizer.zero_grad()
                    pred = model(b_env, qc)

                    if model_type == "physics_informed":
                        physics = {
                            "path_gains_real": b_gr,
                            "path_gains_imag": b_gi,
                            "path_delays": b_del,
                            "subcarrier_frequencies": b_freq,
                        }
                        losses = model.compute_losses(
                            pred,
                            b_target,
                            physics=physics,
                            ofdm_weight=config.ofdm_weight,
                            correlation_weight=config.correlation_weight,
                        )
                    else:
                        losses = model.compute_losses(pred, b_target)

                    losses["loss_total"].backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                test_pred = model(test_env.to(device), qc)
                test_target_d = test_target.to(device)
                mse = torch.mean((test_pred - test_target_d) ** 2).item()
                rmse = mse**0.5
                nmse_val = nmse(test_pred, test_target_d)
                nmse_db = 10 * np.log10(max(nmse_val, 1e-12))

            results.append(
                NeuralOperatorResult(
                    seed=seed,
                    model_type=model_type,
                    test_rmse=rmse,
                    test_nmse_db=nmse_db,
                )
            )

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Paper 3: Neural Operator Experiments")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--output_csv", type=Path, default=Path("outputs/neural_operator_results.csv")
    )
    args = parser.parse_args()

    config = NeuralOperatorExperimentConfig(
        epochs=args.epochs,
        output_csv=args.output_csv,
    )
    results = run_neural_operator_experiments(config)
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    save_dataclass_rows_csv(results, config.output_csv)

    print("\n=== Neural Operator Results ===")
    for r in results:
        print(
            f"  seed={r.seed} {r.model_type}: "
            f"RMSE={r.test_rmse:.4f} NMSE={r.test_nmse_db:.1f} dB"
        )


if __name__ == "__main__":
    main()
