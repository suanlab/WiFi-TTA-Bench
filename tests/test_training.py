"""Tests for training module."""

import tempfile

import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset, TensorDataset

from pinn4csi.models import PINN
from pinn4csi.physics import compute_path_loss
from pinn4csi.training.pinn_trainer import PINNTrainer
from pinn4csi.training.trainer import BaseTrainer
from pinn4csi.utils.device import get_device
from pinn4csi.utils.metrics import accuracy, bootstrap_ci, cohens_d, f1_score, nmse


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

    def test_cohens_d_identical_groups(self) -> None:
        """Cohen's d should be ~0 for identical distributions."""
        a = torch.randn(50)
        d = cohens_d(a, a.clone())
        assert abs(d) < 0.01

    def test_cohens_d_known_separation(self) -> None:
        """Cohen's d ~1.0 for groups separated by 1 std."""
        torch.manual_seed(0)
        a = torch.randn(500) + 1.0
        b = torch.randn(500)
        d = cohens_d(a, b)
        assert 0.8 < d < 1.2

    def test_cohens_d_small_sample(self) -> None:
        """Cohen's d returns 0.0 for insufficient samples."""
        a = torch.tensor([1.0])
        b = torch.tensor([2.0])
        assert cohens_d(a, b) == 0.0

    def test_bootstrap_ci_contains_true_mean(self) -> None:
        """Bootstrap CI should contain the sample mean."""
        torch.manual_seed(42)
        values = torch.randn(100) + 5.0
        point, lower, upper = bootstrap_ci(values)
        assert lower <= point <= upper
        assert lower < 5.5
        assert upper > 4.5

    def test_bootstrap_ci_single_value(self) -> None:
        """Bootstrap CI with single value returns same for all."""
        values = torch.tensor([3.0])
        point, lower, upper = bootstrap_ci(values)
        assert point == lower == upper == 3.0


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


class SyntheticPINNDataset(Dataset[tuple[Tensor, Tensor, dict[str, Tensor]]]):
    """Synthetic dataset that follows PINNTrainer batch contract."""

    def __init__(
        self,
        features: Tensor,
        targets: Tensor,
        distance: Tensor,
        frequency: Tensor,
        tx_power_dbm: Tensor,
        path_loss_exponent: Tensor,
    ) -> None:
        self.features = features
        self.targets = targets
        self.distance = distance
        self.frequency = frequency
        self.tx_power_dbm = tx_power_dbm
        self.path_loss_exponent = path_loss_exponent

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        physics = {
            "distance": self.distance[index],
            "frequency": self.frequency[index],
            "tx_power_dbm": self.tx_power_dbm[index],
            "path_loss_exponent": self.path_loss_exponent[index],
        }
        return self.features[index], self.targets[index], physics


def _make_pinn_dataset(num_samples: int = 64, tx_power_offset: float = 0.0) -> Dataset:
    """Create synthetic regression data aligned with path-loss physics."""
    features = torch.randn(num_samples, 4)
    distance = torch.linspace(1.0, 20.0, num_samples)
    frequency = torch.full((num_samples,), 2.4e9)
    path_loss_exponent = torch.full((num_samples,), 2.0)
    tx_power_dbm = torch.full((num_samples,), 20.0 + tx_power_offset)

    path_loss = compute_path_loss(distance=distance, frequency=2.4e9, n=2.0)
    target_rx = tx_power_dbm - path_loss + 0.3 * features[:, 0]
    targets = target_rx.unsqueeze(-1)

    return SyntheticPINNDataset(
        features=features,
        targets=targets,
        distance=distance,
        frequency=frequency,
        tx_power_dbm=tx_power_dbm,
        path_loss_exponent=path_loss_exponent,
    )


class TestPINNTrainer:
    """Test PINN trainer loss computation and adaptive weighting."""

    def test_loss_dict_has_required_keys_and_finite_values(self) -> None:
        """Training returns finite data/physics/total losses."""
        model = PINN(in_features=4, out_features=1, hidden_dim=16, num_layers=2)
        optimizer = Adam(model.parameters(), lr=1e-2)
        trainer = PINNTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.MSELoss(),
            device=torch.device("cpu"),
            initial_lambda_physics=0.5,
            adaptive_lambda=True,
        )
        loader = DataLoader(_make_pinn_dataset(), batch_size=16, shuffle=False)

        metrics = trainer.train_epoch(loader)

        for key in ("loss_data", "loss_physics", "loss_total", "lambda_physics"):
            assert key in metrics
            assert torch.isfinite(torch.tensor(metrics[key]))

    def test_adaptive_lambda_changes_across_epochs(self) -> None:
        """Adaptive lambda changes as gradient norms evolve."""
        model = PINN(in_features=4, out_features=1, hidden_dim=16, num_layers=2)
        optimizer = Adam(model.parameters(), lr=5e-3)
        trainer = PINNTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.MSELoss(),
            device=torch.device("cpu"),
            initial_lambda_physics=1.0,
            adaptive_lambda=True,
        )

        loader_epoch_1 = DataLoader(
            _make_pinn_dataset(tx_power_offset=0.0), batch_size=16
        )
        loader_epoch_2 = DataLoader(
            _make_pinn_dataset(tx_power_offset=6.0), batch_size=16
        )

        lambda_before = trainer.lambda_physics
        trainer.train_epoch(loader_epoch_1)
        lambda_after_first = trainer.lambda_physics
        trainer.train_epoch(loader_epoch_2)
        lambda_after_second = trainer.lambda_physics

        assert lambda_after_first != lambda_before
        assert lambda_after_second != lambda_after_first

    def test_checkpoint_roundtrip_preserves_lambda(self) -> None:
        """Checkpoint save/load keeps PINN-specific lambda state."""
        model = PINN(in_features=4, out_features=1, hidden_dim=16, num_layers=2)
        optimizer = Adam(model.parameters(), lr=1e-2)

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = PINNTrainer(
                model=model,
                optimizer=optimizer,
                loss_fn=nn.MSELoss(),
                device=torch.device("cpu"),
                checkpoint_dir=tmpdir,
            )
            loader = DataLoader(_make_pinn_dataset(), batch_size=16)
            trainer.train_epoch(loader)
            expected_lambda = trainer.lambda_physics
            checkpoint_path = trainer.save_checkpoint("pinn")

            new_model = PINN(in_features=4, out_features=1, hidden_dim=16, num_layers=2)
            new_optimizer = Adam(new_model.parameters(), lr=1e-2)
            new_trainer = PINNTrainer(
                model=new_model,
                optimizer=new_optimizer,
                loss_fn=nn.MSELoss(),
                device=torch.device("cpu"),
                checkpoint_dir=tmpdir,
            )
            new_trainer.load_checkpoint(checkpoint_path)

            assert new_trainer.lambda_physics == expected_lambda

    def test_heterogeneous_frequency_and_exponent_batch(self) -> None:
        """Test PINN trainer handles per-sample frequency and path_loss_exponent."""
        # Create dataset with varying frequency and path_loss_exponent per sample
        num_samples = 32
        features = torch.randn(num_samples, 4)
        distance = torch.linspace(1.0, 20.0, num_samples)

        # Heterogeneous: frequency varies per sample
        frequency = torch.tensor(
            [2.4e9 if i % 2 == 0 else 5.0e9 for i in range(num_samples)]
        )
        # Heterogeneous: path_loss_exponent varies per sample
        path_loss_exponent = torch.tensor(
            [2.0 if i % 3 == 0 else 2.5 for i in range(num_samples)]
        )
        tx_power_dbm = torch.full((num_samples,), 20.0)

        # Compute targets using per-sample physics parameters
        targets_list = []
        for i in range(num_samples):
            pl = compute_path_loss(
                distance=distance[i : i + 1],
                frequency=float(frequency[i].item()),
                n=float(path_loss_exponent[i].item()),
            )
            target_rx = tx_power_dbm[i] - pl + 0.1 * features[i, 0]
            targets_list.append(target_rx)
        targets = torch.cat(targets_list, dim=0).unsqueeze(-1)

        dataset = SyntheticPINNDataset(
            features=features,
            targets=targets,
            distance=distance,
            frequency=frequency,
            tx_power_dbm=tx_power_dbm,
            path_loss_exponent=path_loss_exponent,
        )

        model = PINN(in_features=4, out_features=1, hidden_dim=16, num_layers=2)
        optimizer = Adam(model.parameters(), lr=1e-2)
        trainer = PINNTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.MSELoss(),
            device=torch.device("cpu"),
            initial_lambda_physics=0.5,
            adaptive_lambda=False,
        )

        loader = DataLoader(dataset, batch_size=8, shuffle=False)
        metrics = trainer.train_epoch(loader)

        # Verify training completes with finite losses
        assert torch.isfinite(torch.tensor(metrics["loss_data"]))
        assert torch.isfinite(torch.tensor(metrics["loss_physics"]))
        assert torch.isfinite(torch.tensor(metrics["loss_total"]))
        assert metrics["loss_total"] > 0
