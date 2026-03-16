"""Tests for training module."""

import tempfile

import pytest
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from pinn4csi.training.trainer import BaseTrainer
from pinn4csi.utils.device import get_device
from pinn4csi.utils.metrics import accuracy, f1_score, nmse


class TestDeviceManagement:
    """Test device selection utilities."""

    def test_get_device_auto_select(self) -> None:
        """Test automatic device selection."""
        device = get_device(cuda=None)
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "cuda")

    def test_get_device_force_cpu(self) -> None:
        """Test forcing CPU device."""
        device = get_device(cuda=False)
        assert device.type == "cpu"

    def test_get_device_force_cuda_unavailable(self) -> None:
        """Test forcing CUDA when unavailable."""
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError):
                get_device(cuda=True)
        else:
            device = get_device(cuda=True)
            assert device.type == "cuda"


class TestMetrics:
    """Test evaluation metrics."""

    def test_accuracy_binary(self) -> None:
        """Test binary classification accuracy."""
        predictions = torch.tensor([0, 1, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 0, 1])
        acc = accuracy(predictions, targets)
        assert acc == 0.8  # 4/5 correct

    def test_accuracy_multiclass(self) -> None:
        """Test multiclass accuracy."""
        predictions = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.3, 0.7]])
        targets = torch.tensor([1, 0, 1])
        acc = accuracy(predictions, targets)
        assert acc == 1.0  # All correct

    def test_nmse(self) -> None:
        """Test NMSE metric."""
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.0, 2.0, 3.0])
        nmse_val = nmse(predictions, targets)
        assert nmse_val == 0.0

    def test_nmse_nonzero(self) -> None:
        """Test NMSE with non-zero error."""
        predictions = torch.tensor([1.0, 2.0, 3.0])
        targets = torch.tensor([1.1, 2.1, 3.1])
        nmse_val = nmse(predictions, targets)
        assert nmse_val > 0.0

    def test_f1_score(self) -> None:
        """Test F1 score."""
        predictions = torch.tensor([0.9, 0.1, 0.8, 0.2])
        targets = torch.tensor([1, 0, 1, 0])
        f1 = f1_score(predictions, targets)
        assert 0.0 <= f1 <= 1.0


class TestBaseTrainer:
    """Test base trainer functionality."""

    @pytest.fixture
    def simple_model(self) -> nn.Module:
        """Create a simple model for testing."""
        return nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    @pytest.fixture
    def simple_dataset(self) -> TensorDataset:
        """Create a simple dataset for testing."""
        x = torch.randn(32, 10)
        y = torch.randint(0, 5, (32,))
        return TensorDataset(x, y)

    def test_trainer_initialization(self, simple_model: nn.Module) -> None:
        """Test trainer initialization."""
        optimizer = Adam(simple_model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        trainer = BaseTrainer(
            model=simple_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device("cpu"),
        )
        assert trainer.epoch == 0
        assert trainer.best_loss == float("inf")

    def test_train_epoch(
        self, simple_model: nn.Module, simple_dataset: TensorDataset
    ) -> None:
        """Test training one epoch."""
        optimizer = Adam(simple_model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        trainer = BaseTrainer(
            model=simple_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device("cpu"),
        )

        train_loader = DataLoader(simple_dataset, batch_size=8)
        metrics = trainer.train_epoch(train_loader)

        assert "loss_total" in metrics
        assert "loss_data" in metrics
        assert "loss_physics" in metrics
        assert metrics["loss_total"] > 0
        assert trainer.epoch == 1

    def test_eval_epoch(
        self, simple_model: nn.Module, simple_dataset: TensorDataset
    ) -> None:
        """Test evaluation epoch."""
        optimizer = Adam(simple_model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        trainer = BaseTrainer(
            model=simple_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device("cpu"),
        )

        eval_loader = DataLoader(simple_dataset, batch_size=8)
        metrics = trainer.eval_epoch(eval_loader)

        assert "loss_total" in metrics
        assert "loss_data" in metrics
        assert "loss_physics" in metrics
        assert metrics["loss_total"] > 0

    def test_checkpoint_save_load(
        self, simple_model: nn.Module, simple_dataset: TensorDataset
    ) -> None:
        """Test checkpoint save and load."""
        optimizer = Adam(simple_model.parameters())
        loss_fn = nn.CrossEntropyLoss()

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = BaseTrainer(
                model=simple_model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                device=torch.device("cpu"),
                checkpoint_dir=tmpdir,
            )

            # Train one epoch
            train_loader = DataLoader(simple_dataset, batch_size=8)
            trainer.train_epoch(train_loader)

            # Save checkpoint
            checkpoint_path = trainer.save_checkpoint("test")
            assert checkpoint_path.exists()

            # Create new trainer and load checkpoint
            new_model = nn.Sequential(
                nn.Linear(10, 32),
                nn.ReLU(),
                nn.Linear(32, 5),
            )
            new_optimizer = Adam(new_model.parameters())
            new_trainer = BaseTrainer(
                model=new_model,
                optimizer=new_optimizer,
                loss_fn=loss_fn,
                device=torch.device("cpu"),
                checkpoint_dir=tmpdir,
            )

            new_trainer.load_checkpoint(checkpoint_path)
            assert new_trainer.epoch == 1

    def test_loss_component_logging(
        self, simple_model: nn.Module, simple_dataset: TensorDataset
    ) -> None:
        """Test that loss components are logged separately."""
        optimizer = Adam(simple_model.parameters())
        loss_fn = nn.CrossEntropyLoss()
        trainer = BaseTrainer(
            model=simple_model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device("cpu"),
        )

        train_loader = DataLoader(simple_dataset, batch_size=8)
        metrics = trainer.train_epoch(train_loader)

        # Verify all components are present
        assert "loss_data" in metrics
        assert "loss_physics" in metrics
        assert "loss_total" in metrics

        # Verify loss_total = loss_data + loss_physics
        expected_total = metrics["loss_data"] + metrics["loss_physics"]
        assert abs(metrics["loss_total"] - expected_total) < 1e-5


@pytest.mark.slow
class TestTrainingSmoke:
    """Smoke tests for full training pipeline."""

    def test_smoke_training_synthetic_data(self) -> None:
        """Test full training loop with synthetic data."""
        # Create synthetic data
        x = torch.randn(64, 52)
        y = torch.randint(0, 10, (64,))
        dataset = TensorDataset(x, y)

        # Create model
        model = nn.Sequential(
            nn.Linear(52, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

        # Create trainer
        optimizer = Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        trainer = BaseTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=torch.device("cpu"),
        )

        # Train for 1 epoch
        train_loader = DataLoader(dataset, batch_size=16)
        metrics = trainer.train_epoch(train_loader)

        assert metrics["loss_total"] > 0
        assert trainer.epoch == 1
