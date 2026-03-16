"""Training script with Hydra configuration."""

import logging

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from pinn4csi.training.trainer import BaseTrainer
from pinn4csi.utils.device import get_device

logger = logging.getLogger(__name__)


def create_synthetic_dataset(
    num_samples: int = 100,
    num_subcarriers: int = 52,
    num_features: int = 2,
    num_classes: int = 10,
) -> TensorDataset:
    """Create synthetic CSI dataset for testing.

    Args:
        num_samples: Number of samples.
        num_subcarriers: Number of subcarriers.
        num_features: Number of features per subcarrier (e.g., amplitude + phase).
        num_classes: Number of output classes.

    Returns:
        TensorDataset with (x, y) pairs.
    """
    x = torch.randn(num_samples, num_subcarriers * num_features)
    y = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(x, y)


def create_simple_model(
    in_features: int,
    out_features: int,
    hidden_dim: int = 64,
    num_layers: int = 4,
) -> nn.Module:
    """Create a simple MLP model.

    Args:
        in_features: Input feature dimension.
        out_features: Output feature dimension.
        hidden_dim: Hidden layer dimension.
        num_layers: Number of hidden layers.

    Returns:
        PyTorch model.
    """
    layers = []
    prev_dim = in_features

    for _ in range(num_layers):
        layers.append(nn.Linear(prev_dim, hidden_dim))
        layers.append(nn.ReLU())
        prev_dim = hidden_dim

    layers.append(nn.Linear(prev_dim, out_features))

    return nn.Sequential(*layers)


@hydra.main(version_base=None, config_path="../pinn4csi/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration.
    """
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format=cfg.logging.format,
    )
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Get device
    device = get_device(cuda=cfg.device.cuda)
    logger.info(f"Using device: {device}")

    # Create synthetic dataset
    logger.info("Creating synthetic dataset...")
    dataset = create_synthetic_dataset(
        num_samples=cfg.data.get("synthetic_samples", 100),
        num_subcarriers=cfg.data.get("synthetic_num_subcarriers", 52),
        num_features=cfg.data.get("synthetic_num_features", 2),
        num_classes=cfg.model.out_features,
    )

    # Split dataset
    train_size = int(cfg.data.train_split * len(dataset))
    val_size = int(cfg.data.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
    )

    logger.info(
        f"Dataset split: train={len(train_dataset)}, "
        f"val={len(val_dataset)}, test={len(test_dataset)}"
    )

    # Create model
    logger.info("Creating model...")
    # Adjust in_features based on synthetic data (subcarriers * features)
    actual_in_features = cfg.data.get("synthetic_num_subcarriers", 52) * cfg.data.get(
        "synthetic_num_features", 2
    )
    model = create_simple_model(
        in_features=actual_in_features,
        out_features=cfg.model.out_features,
        hidden_dim=cfg.model.hidden_dim,
        num_layers=cfg.model.num_layers,
    )

    # Create optimizer and loss function
    optimizer = Adam(model.parameters(), lr=cfg.trainer.learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    # Create trainer
    trainer = BaseTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        checkpoint_dir=cfg.trainer.checkpoint_dir,
    )

    # Training loop
    logger.info(f"Starting training for {cfg.trainer.max_epochs} epochs...")
    for epoch in range(cfg.trainer.max_epochs):
        train_metrics = trainer.train_epoch(train_loader)
        val_metrics = trainer.eval_epoch(val_loader)

        logger.info(
            f"Epoch {epoch + 1}/{cfg.trainer.max_epochs} | "
            f"Train loss_data={train_metrics['loss_data']:.4f} "
            f"loss_physics={train_metrics['loss_physics']:.4f} "
            f"loss_total={train_metrics['loss_total']:.4f} | "
            f"Val loss_total={val_metrics['loss_total']:.4f}"
        )

        # Save checkpoint every epoch
        if (epoch + 1) % max(1, cfg.trainer.max_epochs // 5) == 0:
            trainer.save_checkpoint(f"epoch_{epoch + 1}")

    # Save final checkpoint
    trainer.save_checkpoint("final")
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
