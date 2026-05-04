# pyright: basic, reportMissingImports=false

"""Training entry point for WiFi imaging PINN branch.

Trains a WiFiImagingPINN model on prepared WiFi imaging CSI data with:
- Deterministic seed fixing
- Checkpoint save for best validation loss
- Explicit loss logging (field, permittivity, physics)
- Support for smoke-train runs on mock/prepared data
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader

from pinn4csi.data.wifi_imaging_dataset import (
    create_wifi_imaging_splits,
    load_wifi_imaging_prepared_dataset,
)
from pinn4csi.models.wifi_imager import WiFiImagingPINN
from pinn4csi.utils.device import get_device

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainConfig:
    """Configuration for WiFi imaging PINN training."""

    prepared_root: Path
    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    hidden_dim: int = 64
    latent_dim: int = 64
    num_layers: int = 2
    lambda_field: float = 1.0
    lambda_permittivity: float = 1.0
    lambda_physics: float = 0.1
    seed: int = 42
    checkpoint_dir: Path = Path("outputs/wifi_imager")
    log_interval: int = 10


def set_seed(seed: int) -> None:
    """Set deterministic seed for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_model(
    config: TrainConfig,
    csi_feature_dim: int,
    coordinate_dim: int,
    device: torch.device,
) -> WiFiImagingPINN:
    """Create WiFiImagingPINN model.

    Args:
        config: Training configuration.
        csi_feature_dim: CSI feature dimension from dataset.
        coordinate_dim: Coordinate dimension from dataset.
        device: Device to place model on.

    Returns:
        Initialized WiFiImagingPINN model.
    """
    model = WiFiImagingPINN(
        csi_feature_dim=csi_feature_dim,
        coordinate_dim=coordinate_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
        num_layers=config.num_layers,
    )
    model.to(device)
    return model


def train_epoch(
    model: WiFiImagingPINN,
    train_loader: DataLoader[dict[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    """Run one training epoch.

    Args:
        model: WiFiImagingPINN model.
        train_loader: Training data loader.
        optimizer: Optimizer.
        config: Training configuration.
        device: Device to use.
        epoch: Current epoch number.

    Returns:
        Dictionary with loss metrics.
    """
    model.train()
    total_loss = 0.0
    total_loss_field = 0.0
    total_loss_permittivity = 0.0
    total_loss_physics = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        csi_features = batch["csi_features"].to(device)
        query_coordinates = batch["query_coordinates"].to(device)
        field_target = batch["field_target"].to(device)
        permittivity_target = batch["permittivity_target"].to(device)
        frequency_hz = batch["frequency_hz"].to(device)

        optimizer.zero_grad()

        losses = model.compute_losses(
            csi_features=csi_features,
            coordinates=query_coordinates,
            frequency=frequency_hz,
            field_target=field_target,
            permittivity_target=permittivity_target,
            lambda_field=config.lambda_field,
            lambda_permittivity=config.lambda_permittivity,
            lambda_physics=config.lambda_physics,
        )

        loss_total = losses["loss_total"]
        loss_total.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        total_loss += loss_total.item()
        total_loss_field += losses["loss_field"].item()
        total_loss_permittivity += losses["loss_permittivity"].item()
        total_loss_physics += losses["loss_physics"].item()
        num_batches += 1

        if (batch_idx + 1) % config.log_interval == 0:
            logger.info(
                f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                f"loss_field={losses['loss_field'].item():.4f} "
                f"loss_perm={losses['loss_permittivity'].item():.4f} "
                f"loss_phys={losses['loss_physics'].item():.4f} "
                f"loss_total={loss_total.item():.4f}"
            )

    return {
        "loss_total": total_loss / num_batches,
        "loss_field": total_loss_field / num_batches,
        "loss_permittivity": total_loss_permittivity / num_batches,
        "loss_physics": total_loss_physics / num_batches,
    }


def eval_epoch(
    model: WiFiImagingPINN,
    val_loader: DataLoader[dict[str, Tensor]],
    config: TrainConfig,
    device: torch.device,
) -> dict[str, float]:
    """Run one validation epoch.

    Args:
        model: WiFiImagingPINN model.
        val_loader: Validation data loader.
        config: Training configuration.
        device: Device to use.

    Returns:
        Dictionary with loss metrics.
    """
    model.eval()
    total_loss = 0.0
    total_loss_field = 0.0
    total_loss_permittivity = 0.0
    total_loss_physics = 0.0
    num_batches = 0

    for batch in val_loader:
        csi_features = batch["csi_features"].to(device)
        query_coordinates = batch["query_coordinates"].to(device)
        field_target = batch["field_target"].to(device)
        permittivity_target = batch["permittivity_target"].to(device)
        frequency_hz = batch["frequency_hz"].to(device)

        # Compute losses without updating model parameters
        # (gradients still needed for physics loss computation)
        losses = model.compute_losses(
            csi_features=csi_features,
            coordinates=query_coordinates,
            frequency=frequency_hz,
            field_target=field_target,
            permittivity_target=permittivity_target,
            lambda_field=config.lambda_field,
            lambda_permittivity=config.lambda_permittivity,
            lambda_physics=config.lambda_physics,
        )

        total_loss += losses["loss_total"].item()
        total_loss_field += losses["loss_field"].item()
        total_loss_permittivity += losses["loss_permittivity"].item()
        total_loss_physics += losses["loss_physics"].item()
        num_batches += 1

    return {
        "loss_total": total_loss / num_batches,
        "loss_field": total_loss_field / num_batches,
        "loss_permittivity": total_loss_permittivity / num_batches,
        "loss_physics": total_loss_physics / num_batches,
    }


def save_checkpoint(
    model: WiFiImagingPINN,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    checkpoint_dir: Path,
    name: str = "best_model",
) -> Path:
    """Save model checkpoint.

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        epoch: Current epoch.
        val_loss: Validation loss.
        checkpoint_dir: Directory to save checkpoint.
        name: Checkpoint name (without extension).

    Returns:
        Path to saved checkpoint.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        },
        checkpoint_path,
    )
    logger.info(f"Checkpoint saved to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    """Main training entry point."""
    parser = argparse.ArgumentParser(
        description="Train WiFi imaging PINN on prepared CSI data"
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        required=True,
        help="Path to prepared WiFi imaging data directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden dimension (default: 64)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=64,
        help="Latent dimension (default: 64)",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of MLP layers (default: 2)",
    )
    parser.add_argument(
        "--lambda-field",
        type=float,
        default=1.0,
        help="Field loss weight (default: 1.0)",
    )
    parser.add_argument(
        "--lambda-permittivity",
        type=float,
        default=1.0,
        help="Permittivity loss weight (default: 1.0)",
    )
    parser.add_argument(
        "--lambda-physics",
        type=float,
        default=0.1,
        help="Physics loss weight (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("outputs/wifi_imager"),
        help="Checkpoint directory (default: outputs/wifi_imager)",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Logging interval in batches (default: 10)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = TrainConfig(
        prepared_root=args.prepared_root,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_layers=args.num_layers,
        lambda_field=args.lambda_field,
        lambda_permittivity=args.lambda_permittivity,
        lambda_physics=args.lambda_physics,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        log_interval=args.log_interval,
    )

    logger.info(f"Training config: {config}")

    set_seed(config.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    logger.info(f"Loading prepared data from {config.prepared_root}")
    bundle = load_wifi_imaging_prepared_dataset(config.prepared_root)
    logger.info(
        f"Loaded {bundle.num_samples} samples with "
        f"{bundle.csi_feature_dim} CSI features, "
        f"coordinate_dim={bundle.metadata.coordinate_dim}"
    )

    splits = create_wifi_imaging_splits(bundle, seed=config.seed)
    logger.info(
        f"Train: {len(splits.train)} samples, "
        f"Val: {len(splits.val)} samples, "
        f"Test: {len(splits.test)} samples"
    )

    train_loader = DataLoader(
        splits.train,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        splits.val,
        batch_size=config.batch_size,
        shuffle=False,
    )

    model = create_model(
        config,
        csi_feature_dim=bundle.csi_feature_dim,
        coordinate_dim=bundle.metadata.coordinate_dim,
        device=device,
    )
    logger.info("Created WiFiImagingPINN model")

    optimizer = Adam(model.parameters(), lr=config.learning_rate)

    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(1, config.epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, config, device, epoch
        )
        val_metrics = eval_epoch(model, val_loader, config, device)

        logger.info(
            f"Epoch {epoch}/{config.epochs} | "
            f"train_loss={train_metrics['loss_total']:.4f} | "
            f"val_loss={val_metrics['loss_total']:.4f} | "
            f"val_field={val_metrics['loss_field']:.4f} | "
            f"val_perm={val_metrics['loss_permittivity']:.4f} | "
            f"val_phys={val_metrics['loss_physics']:.4f}"
        )

        if val_metrics["loss_total"] < best_val_loss:
            best_val_loss = val_metrics["loss_total"]
            best_epoch = epoch
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_val_loss,
                config.checkpoint_dir,
                name="best_model",
            )

    logger.info(
        f"Training complete. Best validation loss: {best_val_loss:.4f} "
        f"at epoch {best_epoch}"
    )


if __name__ == "__main__":
    main()
