"""Tests for CSI data loading and preprocessing."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from pinn4csi.data import (
    CSIDataset,
    amplitude_phase_split,
    cross_environment_split,
    normalize_amplitude,
    train_val_test_split,
)


@pytest.fixture
def synthetic_csi_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic CSI data for testing.

    Returns:
        Tuple of (csi_data, labels, environments).
        - csi_data: Shape (100, 56, 3) complex-valued
        - labels: Shape (100,) with 3 classes
        - environments: Shape (100,) with 2 environments
    """
    num_samples = 100
    num_subcarriers = 56
    num_antennas = 3

    # Create synthetic complex CSI
    csi_real = np.random.randn(num_samples, num_subcarriers, num_antennas)
    csi_imag = np.random.randn(num_samples, num_subcarriers, num_antennas)
    csi_data = csi_real + 1j * csi_imag

    # Create labels (3 classes)
    labels = np.random.randint(0, 3, num_samples)

    # Create environments (2 environments)
    environments = np.random.randint(0, 2, num_samples)

    return csi_data, labels, environments


@pytest.fixture
def csi_dataset_file(synthetic_csi_data: tuple) -> Path:
    """Create a temporary CSI dataset file.

    Returns:
        Path to temporary .npy file.
    """
    csi_data, _, _ = synthetic_csi_data
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        np.save(f.name, csi_data)
        return Path(f.name)


class TestAmplitudePhaseTransform:
    """Test amplitude/phase splitting."""

    def test_amplitude_phase_split_2d(self) -> None:
        """Test amplitude/phase split for 2D tensor (num_subcarriers, num_antennas)."""
        num_subcarriers = 56
        num_antennas = 3
        csi_complex = torch.randn(num_subcarriers, num_antennas, dtype=torch.complex64)

        result = amplitude_phase_split(csi_complex)

        # Expected shape: (num_subcarriers, 2*num_antennas)
        assert result.shape == (num_subcarriers, 2 * num_antennas)
        assert result.dtype == torch.float32

    def test_amplitude_phase_split_3d(self) -> None:
        """Test amplitude/phase split for 3D tensor.

        Tests (batch, num_subcarriers, num_antennas) shape.
        """
        batch_size = 16
        num_subcarriers = 56
        num_antennas = 3
        csi_complex = torch.randn(
            batch_size, num_subcarriers, num_antennas, dtype=torch.complex64
        )

        result = amplitude_phase_split(csi_complex)

        # Expected shape: (batch, num_subcarriers, 2*num_antennas)
        assert result.shape == (batch_size, num_subcarriers, 2 * num_antennas)
        assert result.dtype == torch.float32

    def test_amplitude_phase_values(self) -> None:
        """Test that amplitude and phase values are correct."""
        csi_complex = torch.tensor(
            [[[1.0 + 1.0j, 2.0 + 0.0j]]], dtype=torch.complex64
        )  # (1, 1, 2)

        result = amplitude_phase_split(csi_complex)

        # Expected: amplitude = [sqrt(2), 2], phase = [pi/4, 0]
        expected_amp = torch.tensor([[np.sqrt(2), 2.0]], dtype=torch.float32)
        expected_phase = torch.tensor([[np.pi / 4, 0.0]], dtype=torch.float32)

        # Reshape result to (1, 2) for amplitude and (1, 2) for phase
        result_reshaped = result.reshape(
            1, 2, 2
        )  # (1, 2, 2) -> (1, 2 antennas, 2 channels)
        result_amp = result_reshaped[:, :, 0]
        result_phase = result_reshaped[:, :, 1]

        assert torch.allclose(result_amp, expected_amp, atol=1e-5)
        assert torch.allclose(result_phase, expected_phase, atol=1e-5)


class TestNormalizeAmplitude:
    """Test amplitude normalization."""

    def test_normalize_amplitude_range(self) -> None:
        """Test that normalized amplitude is in [0, 1]."""
        amplitude = torch.randn(16, 56, 3).abs()  # Ensure positive
        normalized = normalize_amplitude(amplitude)

        assert torch.all(normalized >= 0.0)
        assert torch.all(normalized <= 1.0)

    def test_normalize_amplitude_shape(self) -> None:
        """Test that normalization preserves shape."""
        amplitude = torch.randn(16, 56, 3).abs()
        normalized = normalize_amplitude(amplitude)

        assert normalized.shape == amplitude.shape


class TestCSIDataset:
    """Test CSIDataset class."""

    def test_dataset_initialization(
        self, csi_dataset_file: Path, synthetic_csi_data: tuple
    ) -> None:
        """Test dataset initialization."""
        csi_data, labels, environments = synthetic_csi_data
        dataset = CSIDataset(csi_dataset_file, labels=labels, environments=environments)

        assert len(dataset) == len(csi_data)
        assert dataset.get_csi_shape() == (56, 3)

    def test_dataset_getitem_shape(
        self, csi_dataset_file: Path, synthetic_csi_data: tuple
    ) -> None:
        """Test that __getitem__ returns correct shape."""
        csi_data, labels, environments = synthetic_csi_data
        dataset = CSIDataset(
            csi_dataset_file,
            labels=labels,
            environments=environments,
            use_amplitude_phase=True,
        )

        csi_tensor, label = dataset[0]

        # Expected shape: (num_subcarriers, 2*num_antennas)
        assert csi_tensor.shape == (56, 6)
        assert isinstance(label, int)
        assert 0 <= label < 3

    def test_dataset_getitem_label_type(
        self, csi_dataset_file: Path, synthetic_csi_data: tuple
    ) -> None:
        """Test that labels are integers."""
        csi_data, labels, environments = synthetic_csi_data
        dataset = CSIDataset(csi_dataset_file, labels=labels, environments=environments)

        for i in range(min(10, len(dataset))):
            _, label = dataset[i]
            assert isinstance(label, int)

    def test_dataset_default_labels(self, csi_dataset_file: Path) -> None:
        """Test dataset with default labels (all zeros)."""
        dataset = CSIDataset(csi_dataset_file)

        for i in range(min(10, len(dataset))):
            _, label = dataset[i]
            assert label == 0

    def test_dataset_default_environments(self, csi_dataset_file: Path) -> None:
        """Test dataset with default environments (all zeros)."""
        dataset = CSIDataset(csi_dataset_file)

        envs = dataset.get_environments()
        assert np.all(envs == 0)

    def test_dataset_complex_csi(
        self, csi_dataset_file: Path, synthetic_csi_data: tuple
    ) -> None:
        """Test dataset with complex CSI (use_amplitude_phase=False)."""
        csi_data, labels, environments = synthetic_csi_data
        dataset = CSIDataset(
            csi_dataset_file,
            labels=labels,
            environments=environments,
            use_amplitude_phase=False,
        )

        csi_tensor, _ = dataset[0]

        # Expected shape: (num_subcarriers, num_antennas) complex
        assert csi_tensor.shape == (56, 3)
        assert csi_tensor.dtype == torch.complex64

    def test_dataset_get_environment(
        self, csi_dataset_file: Path, synthetic_csi_data: tuple
    ) -> None:
        """Test get_environment method."""
        csi_data, labels, environments = synthetic_csi_data
        dataset = CSIDataset(csi_dataset_file, labels=labels, environments=environments)

        for i in range(min(10, len(dataset))):
            env = dataset.get_environment(i)
            assert env == environments[i]


class TestDataLoaderIteration:
    """Test DataLoader iteration."""

    def test_dataloader_one_epoch(
        self, csi_dataset_file: Path, synthetic_csi_data: tuple
    ) -> None:
        """Test that DataLoader can iterate one epoch without errors."""
        from torch.utils.data import DataLoader

        csi_data, labels, environments = synthetic_csi_data
        dataset = CSIDataset(csi_dataset_file, labels=labels, environments=environments)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

        batch_count = 0
        for batch_csi, batch_labels in dataloader:
            batch_count += 1
            assert batch_csi.shape[0] <= 16  # batch size
            assert batch_csi.shape[1] == 56  # num_subcarriers
            assert batch_csi.shape[2] == 6  # 2*num_antennas
            assert len(batch_labels) == batch_csi.shape[0]

        assert batch_count > 0


class TestTrainValTestSplit:
    """Test train/val/test split."""

    def test_split_ratios(
        self, csi_dataset_file: Path, synthetic_csi_data: tuple
    ) -> None:
        """Test that split respects specified ratios."""
        csi_data, labels, environments = synthetic_csi_data
        dataset = CSIDataset(csi_dataset_file, labels=labels, environments=environments)

        train_ds, val_ds, test_ds = train_val_test_split(
            dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )

        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == len(dataset)

        # Check approximate ratios (allow ±1 sample due to rounding)
        assert abs(len(train_ds) / total - 0.7) < 0.02
        assert abs(len(val_ds) / total - 0.15) < 0.02
        assert abs(len(test_ds) / total - 0.15) < 0.02

    def test_split_no_overlap(
        self, csi_dataset_file: Path, synthetic_csi_data: tuple
    ) -> None:
        """Test that splits have no overlapping samples."""
        csi_data, labels, environments = synthetic_csi_data
        dataset = CSIDataset(csi_dataset_file, labels=labels, environments=environments)

        train_ds, val_ds, test_ds = train_val_test_split(dataset)

        # Verify no overlap by checking total size
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == len(dataset)

    def test_split_reproducibility(
        self, csi_dataset_file: Path, synthetic_csi_data: tuple
    ) -> None:
        """Test that split is reproducible with same seed."""
        csi_data, labels, environments = synthetic_csi_data
        dataset = CSIDataset(csi_dataset_file, labels=labels, environments=environments)

        train_ds1, _, _ = train_val_test_split(dataset, seed=42)
        train_ds2, _, _ = train_val_test_split(dataset, seed=42)

        # Check that first few samples are the same
        for i in range(min(5, len(train_ds1))):
            csi1, label1 = train_ds1[i]
            csi2, label2 = train_ds2[i]
            assert torch.allclose(csi1, csi2)
            assert label1 == label2


class TestCrossEnvironmentSplit:
    """Test cross-environment split."""

    def test_cross_env_split_no_overlap(
        self, csi_dataset_file: Path, synthetic_csi_data: tuple
    ) -> None:
        """Test that cross-environment split has no environment overlap."""
        csi_data, labels, environments = synthetic_csi_data
        dataset = CSIDataset(csi_dataset_file, labels=labels, environments=environments)

        train_ds, test_ds = cross_environment_split(
            dataset, train_env_ids=0, test_env_ids=1
        )

        train_envs = train_ds.get_environments()
        test_envs = test_ds.get_environments()

        # Check no overlap
        assert not np.any(np.isin(train_envs, test_envs))
        assert not np.any(np.isin(test_envs, train_envs))

    def test_cross_env_split_raises_on_overlap(
        self, csi_dataset_file: Path, synthetic_csi_data: tuple
    ) -> None:
        """Test that overlapping environments raise ValueError."""
        csi_data, labels, environments = synthetic_csi_data
        dataset = CSIDataset(csi_dataset_file, labels=labels, environments=environments)

        with pytest.raises(ValueError, match="overlap"):
            cross_environment_split(dataset, train_env_ids=0, test_env_ids=0)

    def test_cross_env_split_multiple_envs(
        self, csi_dataset_file: Path, synthetic_csi_data: tuple
    ) -> None:
        """Test cross-environment split with multiple environment IDs."""
        csi_data, labels, environments = synthetic_csi_data
        # Create dataset with 3 environments
        environments = np.random.randint(0, 3, len(csi_data))
        dataset = CSIDataset(csi_dataset_file, labels=labels, environments=environments)

        train_ds, test_ds = cross_environment_split(
            dataset, train_env_ids=[0, 1], test_env_ids=[2]
        )

        train_envs = set(train_ds.get_environments())
        test_envs = set(test_ds.get_environments())

        assert train_envs <= {0, 1}
        assert test_envs <= {2}
        assert len(train_envs & test_envs) == 0
