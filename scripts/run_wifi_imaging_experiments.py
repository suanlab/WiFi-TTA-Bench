# pyright: basic, reportMissingImports=false

"""Paper 4 synthetic validation: WiFi Imaging via Helmholtz-constrained PINN.
Generates synthetic permittivity maps and field solutions, then evaluates
inverse reconstruction with and without Helmholtz PDE constraints."""

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

from pinn4csi.models import WiFiImagingPINN
from pinn4csi.utils import nmse, save_dataclass_rows_csv


@dataclass(frozen=True)
class WiFiImagingExperimentConfig:
    """Configuration for Paper 4 synthetic experiments."""

    seeds: tuple[int, ...] = tuple(range(5))
    csi_feature_dim: int = 64
    grid_resolution: int = 8
    train_samples: int = 256
    test_samples: int = 128
    epochs: int = 30
    batch_size: int = 32
    hidden_dim: int = 64
    latent_dim: int = 32
    frequency_ghz: float = 2.4
    learning_rate: float = 1e-3
    helmholtz_weight: float = 0.1
    output_csv: Path = Path("outputs/wifi_imaging_results.csv")


@dataclass(frozen=True)
class WiFiImagingResult:
    """Per-run result row."""

    seed: int
    model_type: str
    field_rmse: float
    field_nmse_db: float
    permittivity_rmse: float


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _generate_imaging_dataset(
    num_samples: int,
    config: WiFiImagingExperimentConfig,
    seed: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Generate synthetic WiFi imaging dataset.

    Creates random CSI features, grid coordinates, field values, and
    permittivity maps for inverse problem training/evaluation.

    Returns:
        (csi_features, coordinates, field_targets, permittivity_targets)
    """
    _set_seed(seed)
    resolution = config.grid_resolution
    num_points = resolution * resolution

    # Random CSI features (simulating encoded CSI measurements)
    csi_features = torch.randn(num_samples, config.csi_feature_dim)

    # Grid coordinates normalized to [-1, 1]
    x = torch.linspace(-1, 1, resolution)
    y = torch.linspace(-1, 1, resolution)
    gx, gy = torch.meshgrid(x, y, indexing="ij")
    coords = torch.stack([gx.flatten(), gy.flatten()], dim=-1)  # (num_points, 2)
    coords = coords.unsqueeze(0).expand(num_samples, -1, -1)  # (N, P, 2)

    # Synthetic permittivity maps (smooth random fields)
    # Use a simple parametric model: sum of Gaussians
    permittivity = torch.ones(num_samples, num_points)
    for _ in range(3):
        center = torch.randn(num_samples, 1, 2) * 0.5
        sigma = 0.3 + 0.2 * torch.rand(num_samples, 1, 1)
        dist_sq = torch.sum((coords - center) ** 2, dim=-1)
        bump = torch.exp(-dist_sq / (2 * sigma.squeeze(-1) ** 2))
        permittivity = permittivity + 2.0 * bump

    # Synthetic field values (related to permittivity via simple model)
    # Not a true Helmholtz solution but provides a learnable mapping
    frequency_hz = config.frequency_ghz * 1e9
    k0 = 2 * torch.pi * frequency_hz / 3e8
    wavenumber_map = k0 * torch.sqrt(permittivity)
    # Standing wave approximation
    field = torch.sin(wavenumber_map * 0.1) * torch.exp(-0.1 * permittivity)
    # Add noise
    field = field + 0.02 * torch.randn_like(field)

    return csi_features, coords, field, permittivity


def run_wifi_imaging_experiments(
    config: WiFiImagingExperimentConfig,
) -> list[WiFiImagingResult]:
    """Run Paper 4 synthetic experiments."""
    results: list[WiFiImagingResult] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frequency_hz = config.frequency_ghz * 1e9

    for seed in config.seeds:
        train_csi, train_coords, train_field, train_perm = _generate_imaging_dataset(
            config.train_samples, config, seed
        )
        test_csi, test_coords, test_field, test_perm = _generate_imaging_dataset(
            config.test_samples, config, seed + 1000
        )

        for model_type in ("baseline", "helmholtz_pinn"):
            _set_seed(seed)
            model = WiFiImagingPINN(
                csi_feature_dim=config.csi_feature_dim,
                coordinate_dim=2,
                hidden_dim=config.hidden_dim,
                latent_dim=config.latent_dim,
            ).to(device)
            optimizer = Adam(model.parameters(), lr=config.learning_rate)

            train_loader = DataLoader(
                TensorDataset(train_csi, train_coords, train_field, train_perm),
                batch_size=config.batch_size,
                shuffle=True,
            )

            for _epoch in range(config.epochs):
                model.train()
                for batch in train_loader:
                    b_csi, b_coords, b_field, b_perm = (t.to(device) for t in batch)
                    optimizer.zero_grad()

                    if model_type == "helmholtz_pinn":
                        b_coords_grad = b_coords.detach().requires_grad_(True)
                        losses = model.compute_losses(
                            csi_features=b_csi,
                            coordinates=b_coords_grad,
                            frequency=frequency_hz,
                            field_target=b_field,
                            permittivity_target=b_perm,
                            lambda_physics=config.helmholtz_weight,
                        )
                    else:
                        losses = model.compute_losses(
                            csi_features=b_csi,
                            coordinates=b_coords,
                            frequency=frequency_hz,
                            field_target=b_field,
                            permittivity_target=b_perm,
                            lambda_physics=0.0,
                        )

                    losses["loss_total"].backward()
                    optimizer.step()

            model.eval()
            with torch.no_grad():
                test_pred = model(test_csi.to(device), test_coords.to(device))
                pred_field = test_pred["field"]
                pred_perm = test_pred["permittivity"]

                field_mse = torch.mean((pred_field - test_field.to(device)) ** 2).item()
                field_rmse = field_mse**0.5
                field_nmse_val = nmse(pred_field, test_field.to(device))
                field_nmse_db = 10 * np.log10(max(field_nmse_val, 1e-12))

                perm_mse = torch.mean((pred_perm - test_perm.to(device)) ** 2).item()
                perm_rmse = perm_mse**0.5

            results.append(
                WiFiImagingResult(
                    seed=seed,
                    model_type=model_type,
                    field_rmse=field_rmse,
                    field_nmse_db=field_nmse_db,
                    permittivity_rmse=perm_rmse,
                )
            )

    return results


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Paper 4: WiFi Imaging Experiments")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--output_csv", type=Path, default=Path("outputs/wifi_imaging_results.csv")
    )
    args = parser.parse_args()

    config = WiFiImagingExperimentConfig(
        epochs=args.epochs,
        output_csv=args.output_csv,
    )
    results = run_wifi_imaging_experiments(config)
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    save_dataclass_rows_csv(results, config.output_csv)

    print("\n=== WiFi Imaging Results ===")
    for r in results:
        print(
            f"  seed={r.seed} {r.model_type}: "
            f"field_RMSE={r.field_rmse:.4f} "
            f"field_NMSE={r.field_nmse_db:.1f} dB "
            f"perm_RMSE={r.permittivity_rmse:.4f}"
        )


if __name__ == "__main__":
    main()
