"""Base training loop for PINN models."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pinn4csi.utils.device import get_device

logger = logging.getLogger(__name__)


class BaseTrainer:
    """Base trainer for neural network models.

    Handles training/evaluation loops, checkpointing, and loss logging.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        device: torch.device | None = None,
        checkpoint_dir: str | Path = "outputs",
    ):
        """Initialize trainer.

        Args:
            model: Neural network model to train.
            optimizer: Optimizer for model parameters.
            loss_fn: Loss function.
            device: Device to use (cuda/cpu). If None, auto-select.
            checkpoint_dir: Directory to save checkpoints.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device or get_device()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.to(self.device)
        self.epoch = 0
        self.best_loss = float("inf")

    def train_epoch(
        self, train_loader: DataLoader[tuple[Tensor, Tensor]]
    ) -> dict[str, float]:
        """Run one training epoch.

        Args:
            train_loader: DataLoader for training data.

        Returns:
            Dictionary with loss metrics: loss_total, loss_data, loss_physics.
        """
        self.model.train()
        total_loss = 0.0
        total_loss_data = 0.0
        total_loss_physics = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            # Handle both (x, y) and (x, y, coords) formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    x, y = batch
                    coords = None
                elif len(batch) == 3:
                    x, y, coords = batch
                else:
                    raise ValueError(f"Unexpected batch format: {len(batch)} elements")
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")

            x = x.to(self.device)
            y = y.to(self.device)
            if coords is not None:
                coords = coords.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            y_pred = self.model(x)

            # Compute loss
            loss_data = self.loss_fn(y_pred, y)
            loss_physics = torch.tensor(0.0, device=self.device)
            loss_total = loss_data + loss_physics

            # Backward pass
            loss_total.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss_total.item()
            total_loss_data += loss_data.item()
            total_loss_physics += loss_physics.item()
            num_batches += 1

            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                logger.info(
                    f"Epoch {self.epoch} [{batch_idx + 1}/{len(train_loader)}] "
                    f"loss_data={loss_data.item():.4f} "
                    f"loss_physics={loss_physics.item():.4f} "
                    f"loss_total={loss_total.item():.4f}"
                )

        self.epoch += 1

        return {
            "loss_total": total_loss / num_batches,
            "loss_data": total_loss_data / num_batches,
            "loss_physics": total_loss_physics / num_batches,
        }

    def eval_epoch(
        self, eval_loader: DataLoader[tuple[Tensor, Tensor]]
    ) -> dict[str, float]:
        """Run one evaluation epoch.

        Args:
            eval_loader: DataLoader for evaluation data.

        Returns:
            Dictionary with loss metrics.
        """
        self.model.eval()
        total_loss = 0.0
        total_loss_data = 0.0
        total_loss_physics = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_loader:
                # Handle both (x, y) and (x, y, coords) formats
                if isinstance(batch, (list, tuple)):
                    if len(batch) == 2:
                        x, y = batch
                    elif len(batch) == 3:
                        x, y, _ = batch
                    else:
                        msg = f"Unexpected batch format: {len(batch)} elements"
                        raise ValueError(msg)
                else:
                    raise ValueError(f"Unexpected batch type: {type(batch)}")

                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(x)
                loss_data = self.loss_fn(y_pred, y)
                loss_physics = torch.tensor(0.0, device=self.device)
                loss_total = loss_data + loss_physics

                total_loss += loss_total.item()
                total_loss_data += loss_data.item()
                total_loss_physics += loss_physics.item()
                num_batches += 1

        return {
            "loss_total": total_loss / num_batches,
            "loss_data": total_loss_data / num_batches,
            "loss_physics": total_loss_physics / num_batches,
        }

    def save_checkpoint(self, name: str = "checkpoint") -> Path:
        """Save model checkpoint.

        Args:
            name: Checkpoint name (without extension).

        Returns:
            Path to saved checkpoint.
        """
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
            },
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load model checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
