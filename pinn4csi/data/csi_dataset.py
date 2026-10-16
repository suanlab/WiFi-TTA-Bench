"""PyTorch Dataset for CSI data."""

from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from pinn4csi.data.transforms import amplitude_phase_split


class CSIDataset(Dataset[tuple[Tensor, int]]):
    """PyTorch Dataset for WiFi CSI data.

    Supports loading CSI data in various formats and provides flexible preprocessing.
    CSI shape convention: (num_packets, num_subcarriers, num_antennas) — complex-valued.
    """

    def __init__(
        self,
        data_path: str | Path,
        labels: np.ndarray | None = None,
        environments: np.ndarray | None = None,
        use_amplitude_phase: bool = True,
        transform: Callable[[Tensor], Tensor] | None = None,
    ):
        """Initialize CSI dataset.

        Args:
            data_path: Path to CSI data file (.npy, .pt, or .npz).
            labels: Integer class labels. Shape: (num_samples,).
                If None, assumes single-class dataset (label=0).
            environments: Environment IDs for cross-environment split.
                Shape: (num_samples,). If None, assumes single env (env=0).
            use_amplitude_phase: If True, convert complex CSI to
                amplitude+phase channels. If False, keep complex-valued CSI.
            transform: Optional callable to apply to each sample.
        """
        self.data_path = Path(data_path)
        self.use_amplitude_phase = use_amplitude_phase
        self.transform = transform

        # Load CSI data
        if self.data_path.suffix == ".npy":
            self.csi_data = np.load(self.data_path)
        elif self.data_path.suffix == ".pt":
            self.csi_data = torch.load(self.data_path).numpy()
        elif self.data_path.suffix == ".npz":
            npz = np.load(self.data_path)
            # Assume 'csi' key contains the data
            self.csi_data = npz["csi"] if "csi" in npz else npz[list(npz.keys())[0]]
        else:
            raise ValueError(f"Unsupported file format: {self.data_path.suffix}")

        # Ensure CSI is complex-valued
        if not np.iscomplexobj(self.csi_data):
            # If real-valued, assume it's amplitude; convert to complex
            self.csi_data = self.csi_data.astype(np.complex64)

        num_samples = self.csi_data.shape[0]

        # Set labels
        if labels is None:
            self.labels = np.zeros(num_samples, dtype=np.int64)
        else:
            self.labels = np.asarray(labels, dtype=np.int64)
            if len(self.labels) != num_samples:
                raise ValueError(
                    f"labels length {len(self.labels)} != data samples {num_samples}"
                )

        # Set environments
        if environments is None:
            self.environments = np.zeros(num_samples, dtype=np.int64)
        else:
            self.environments = np.asarray(environments, dtype=np.int64)
            if len(self.environments) != num_samples:
                raise ValueError(
                    f"environments length {len(self.environments)} != "
                    f"data samples {num_samples}"
                )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.csi_data)

    def __getitem__(self, idx: int) -> tuple[Tensor, int]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Tuple of (csi_tensor, label).
            - csi_tensor: Shape (num_subcarriers, features) where
              features=2*num_antennas if use_amplitude_phase=True, else
              (num_subcarriers, num_antennas) complex.
            - label: Integer class label.
        """
        csi_complex = torch.from_numpy(self.csi_data[idx]).to(torch.complex64)

        # Convert to amplitude+phase if requested
        if self.use_amplitude_phase:
            csi_tensor = amplitude_phase_split(csi_complex)
        else:
            csi_tensor = csi_complex

        # Apply transform if provided
        if self.transform is not None:
            csi_tensor = self.transform(csi_tensor)

        label = int(self.labels[idx])
        return csi_tensor, label

    def get_environment(self, idx: int) -> int:
        """Get environment ID for a sample.

        Args:
            idx: Sample index.

        Returns:
            Environment ID.
        """
        return int(self.environments[idx])

    def get_environments(self) -> np.ndarray:
        """Get all environment IDs.

        Returns:
            Array of environment IDs. Shape: (num_samples,).
        """
        return np.asarray(self.environments.copy())

    def get_labels(self) -> np.ndarray:
        """Get all labels.

        Returns:
            Array of labels. Shape: (num_samples,).
        """
        return np.asarray(self.labels.copy())

    def get_csi_shape(self) -> tuple[int, ...]:
        """Get shape of CSI data (num_subcarriers, num_antennas).

        Returns:
            Tuple of (num_subcarriers, num_antennas).
        """
        return tuple(self.csi_data.shape[1:])


def train_val_test_split(
    dataset: CSIDataset,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[CSIDataset, CSIDataset, CSIDataset]:
    """Split dataset into train/val/test subsets.

    Args:
        dataset: CSIDataset to split.
        train_ratio: Fraction for training (default 0.7).
        val_ratio: Fraction for validation (default 0.15).
        test_ratio: Fraction for testing (default 0.15).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
    """
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and 0 < test_ratio < 1):
        raise ValueError("Ratios must be in (0, 1)")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(
            f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
        )

    num_samples = len(dataset)
    indices = np.arange(num_samples)
    rng = np.random.RandomState(seed)
    rng.shuffle(indices)

    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_dataset = _create_subset(dataset, train_indices)
    val_dataset = _create_subset(dataset, val_indices)
    test_dataset = _create_subset(dataset, test_indices)

    return train_dataset, val_dataset, test_dataset


def cross_environment_split(
    dataset: CSIDataset,
    train_env_ids: list[int] | int,
    test_env_ids: list[int] | int,
) -> tuple[CSIDataset, CSIDataset]:
    """Split dataset by environment (no overlap).

    Args:
        dataset: CSIDataset with environment metadata.
        train_env_ids: Environment ID(s) for training.
        test_env_ids: Environment ID(s) for testing.

    Returns:
        Tuple of (train_dataset, test_dataset) with no environment overlap.

    Raises:
        ValueError: If train and test environments overlap.
    """
    # Convert to list if int
    train_list: list[int] = (
        [train_env_ids] if isinstance(train_env_ids, int) else train_env_ids
    )
    test_list: list[int] = (
        [test_env_ids] if isinstance(test_env_ids, int) else test_env_ids
    )

    train_env_set = set(train_list)
    test_env_set = set(test_list)

    if train_env_set & test_env_set:
        raise ValueError(
            f"Train and test environments overlap: {train_env_set & test_env_set}"
        )

    environments = dataset.get_environments()

    train_indices = np.where(np.isin(environments, train_list))[0]
    test_indices = np.where(np.isin(environments, test_list))[0]

    train_dataset = _create_subset(dataset, train_indices)
    test_dataset = _create_subset(dataset, test_indices)

    return train_dataset, test_dataset


def _create_subset(dataset: CSIDataset, indices: np.ndarray) -> CSIDataset:
    """Create a subset of a CSIDataset.

    Args:
        dataset: Original dataset.
        indices: Array of indices to include.

    Returns:
        New CSIDataset with selected samples.
    """
    subset_csi = dataset.csi_data[indices]
    subset_labels = dataset.labels[indices]
    subset_envs = dataset.environments[indices]

    # Create a new dataset by directly setting attributes
    # to avoid reloading from file
    subset = object.__new__(CSIDataset)
    subset.data_path = dataset.data_path
    subset.csi_data = subset_csi
    subset.labels = subset_labels
    subset.environments = subset_envs
    subset.use_amplitude_phase = dataset.use_amplitude_phase
    subset.transform = dataset.transform
    return subset
