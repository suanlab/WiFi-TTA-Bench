"""PINN-specific trainer with path-loss physics regularization."""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from pinn4csi.physics import compute_path_loss
from pinn4csi.training.trainer import BaseTrainer

logger = logging.getLogger(__name__)


class PINNTrainer(BaseTrainer):
    """Trainer that combines data loss with path-loss physics residual loss.

    Batch contract for training/evaluation is:
    `(x, y, physics)` where `physics` is a dict containing:
    - `distance`: Tensor, shape (batch,) in meters, strictly positive
    - `frequency`: Tensor, shape (batch,) in Hz, strictly positive
    - `tx_power_dbm`: Tensor, shape (batch,) in dBm
    Optional:
    - `path_loss_exponent`: Tensor, shape (batch,) or scalar-like
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.Module,
        device: torch.device | None = None,
        checkpoint_dir: str | Path = "outputs",
        initial_lambda_physics: float = 1.0,
        adaptive_lambda: bool = True,
        adaptive_eps: float = 1e-8,
        lambda_min: float = 1e-6,
        lambda_max: float = 1e6,
    ) -> None:
        """Initialize PINN trainer.

        Args:
            model: Neural network model to train.
            optimizer: Optimizer for model parameters.
            loss_fn: Data loss function.
            device: Device to use (cuda/cpu). If None, auto-select.
            checkpoint_dir: Directory to save checkpoints.
            initial_lambda_physics: Initial physics loss weight.
            adaptive_lambda: If True, update lambda using gradient normalization.
            adaptive_eps: Numerical stability epsilon for gradient normalization.
            lambda_min: Minimum allowed value for adaptive lambda.
            lambda_max: Maximum allowed value for adaptive lambda.
        """
        super().__init__(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )
        self.lambda_physics = float(initial_lambda_physics)
        self.adaptive_lambda = adaptive_lambda
        self.adaptive_eps = adaptive_eps
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max

    def _extract_physics_inputs(
        self, physics: dict[str, Tensor]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Extract required physics tensors from a batch dictionary.

        Args:
            physics: Physics metadata dictionary from batch.

        Returns:
            Distance, frequency, transmit power, and path-loss exponent tensors.

        Raises:
            ValueError: If required physics fields are missing.
        """
        required = ("distance", "frequency", "tx_power_dbm")
        missing = [key for key in required if key not in physics]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Missing required physics metadata keys: {missing_str}")

        distance = physics["distance"].to(self.device)
        frequency = physics["frequency"].to(self.device)
        tx_power_dbm = physics["tx_power_dbm"].to(self.device)
        path_loss_exponent = physics.get(
            "path_loss_exponent",
            torch.full_like(distance, 2.0, device=self.device),
        ).to(self.device)

        return distance, frequency, tx_power_dbm, path_loss_exponent

    def _physics_loss(self, predictions: Tensor, physics: dict[str, Tensor]) -> Tensor:
        """Compute path-loss residual loss.

        Assumes the first prediction channel corresponds to received power in dBm.
        Handles per-sample frequency and path-loss exponent for heterogeneous batches.

        Args:
            predictions: Model output. Shape: (batch, out_features).
            physics: Physics metadata dictionary.

        Returns:
            Mean squared residual between predicted and path-loss implied power.
        """
        distance, frequency, tx_power_dbm, path_loss_exponent = (
            self._extract_physics_inputs(physics)
        )

        predicted_rx_dbm = predictions[:, 0]

        # Compute path loss per sample to handle heterogeneous batches
        # Loop over batch to compute path loss for each sample with its own
        # frequency and path_loss_exponent
        batch_size = distance.shape[0]
        expected_path_loss_list = []
        for i in range(batch_size):
            pl = compute_path_loss(
                distance=distance[i : i + 1],
                frequency=float(frequency[i].item()),
                n=float(path_loss_exponent[i].item()),
            )
            expected_path_loss_list.append(pl)
        expected_path_loss = torch.cat(expected_path_loss_list, dim=0)

        expected_rx_dbm = tx_power_dbm - expected_path_loss
        return torch.mean((predicted_rx_dbm - expected_rx_dbm) ** 2)

    def _grad_norm(self, loss: Tensor) -> Tensor:
        """Compute L2 norm of gradients for current model parameters."""
        params = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        gradients = torch.autograd.grad(
            loss,
            params,
            retain_graph=True,
            allow_unused=True,
        )
        squared_norm = torch.zeros((), device=self.device)
        for grad in gradients:
            if grad is not None:
                squared_norm = squared_norm + torch.sum(grad * grad)
        return torch.sqrt(squared_norm + self.adaptive_eps)

    def _update_lambda(self, loss_data: Tensor, loss_physics: Tensor) -> None:
        """Update physics weight using gradient normalization."""
        if not self.adaptive_lambda:
            return

        data_norm = self._grad_norm(loss_data)
        physics_norm = self._grad_norm(loss_physics)

        ratio = data_norm / (physics_norm + self.adaptive_eps)
        clamped = torch.clamp(ratio, min=self.lambda_min, max=self.lambda_max)
        self.lambda_physics = float(clamped.detach().item())

    def train_epoch(
        self, train_loader: DataLoader[tuple[Tensor, Tensor]]
    ) -> dict[str, float]:
        """Run one PINN training epoch."""
        self.model.train()
        total_loss = 0.0
        total_loss_data = 0.0
        total_loss_physics = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            if not isinstance(batch, (list, tuple)) or len(batch) != 3:
                msg = "PINNTrainer expects batch format: (x, y, physics_dict)"
                raise ValueError(msg)

            x, y, physics = batch
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(x)

            loss_data = self.loss_fn(predictions, y)
            loss_physics = self._physics_loss(predictions, physics)
            self._update_lambda(loss_data, loss_physics)

            loss_total = loss_data + self.lambda_physics * loss_physics
            loss_total.backward()
            self.optimizer.step()

            total_loss += float(loss_total.item())
            total_loss_data += float(loss_data.item())
            total_loss_physics += float(loss_physics.item())
            num_batches += 1

            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                logger.info(
                    f"Epoch {self.epoch} [{batch_idx + 1}/{len(train_loader)}] "
                    f"loss_data={loss_data.item():.4f} "
                    f"loss_physics={loss_physics.item():.4f} "
                    f"lambda_physics={self.lambda_physics:.4f} "
                    f"loss_total={loss_total.item():.4f}"
                )

        self.epoch += 1

        return {
            "loss_total": total_loss / num_batches,
            "loss_data": total_loss_data / num_batches,
            "loss_physics": total_loss_physics / num_batches,
            "lambda_physics": self.lambda_physics,
        }

    def eval_epoch(
        self, eval_loader: DataLoader[tuple[Tensor, Tensor]]
    ) -> dict[str, float]:
        """Run one PINN evaluation epoch."""
        self.model.eval()
        total_loss = 0.0
        total_loss_data = 0.0
        total_loss_physics = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in eval_loader:
                if not isinstance(batch, (list, tuple)) or len(batch) != 3:
                    msg = "PINNTrainer expects batch format: (x, y, physics_dict)"
                    raise ValueError(msg)

                x, y, physics = batch
                x = x.to(self.device)
                y = y.to(self.device)

                predictions = self.model(x)
                loss_data = self.loss_fn(predictions, y)
                loss_physics = self._physics_loss(predictions, physics)
                loss_total = loss_data + self.lambda_physics * loss_physics

                total_loss += float(loss_total.item())
                total_loss_data += float(loss_data.item())
                total_loss_physics += float(loss_physics.item())
                num_batches += 1

        return {
            "loss_total": total_loss / num_batches,
            "loss_data": total_loss_data / num_batches,
            "loss_physics": total_loss_physics / num_batches,
            "lambda_physics": self.lambda_physics,
        }

    def save_checkpoint(self, name: str = "checkpoint") -> Path:
        """Save model checkpoint, including PINN-specific state."""
        checkpoint_path = self.checkpoint_dir / f"{name}.pt"
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_loss": self.best_loss,
                "lambda_physics": self.lambda_physics,
                "adaptive_lambda": self.adaptive_lambda,
            },
            checkpoint_path,
        )
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Load model checkpoint, including PINN-specific state."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        self.lambda_physics = float(
            checkpoint.get("lambda_physics", self.lambda_physics)
        )
        self.adaptive_lambda = bool(
            checkpoint.get("adaptive_lambda", self.adaptive_lambda)
        )
        logger.info(f"Checkpoint loaded from {checkpoint_path}")
