# pyright: basic, reportMissingImports=false

"""Prepared dataset registry and helpers for Paper 1 experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from pinn4csi.data.transforms import amplitude_phase_split

EvaluationMode = Literal["train_val_test", "cross_environment"]


@dataclass(frozen=True)
class Paper1DatasetConfig:
    """Configuration for a named Paper 1 dataset target.

    Args:
        name: Display name used in results.
        directory_name: Directory under the prepared data root.
        evaluation_mode: Split strategy for this dataset target.
        description: Short dataset description.
        csi_filename: Prepared CSI tensor file.
        labels_filename: Prepared label file.
        environments_filename: Optional environment-id file.
        priors_filename: Optional prior reconstruction file.
        train_env_ids: Default train environment IDs for cross-environment runs.
        test_env_ids: Default test environment IDs for cross-environment runs.
        use_amplitude_phase: If True, convert complex CSI to amplitude/phase.
    """

    name: str
    directory_name: str
    evaluation_mode: EvaluationMode
    description: str
    csi_filename: str = "csi.npy"
    labels_filename: str = "labels.npy"
    environments_filename: str | None = "environments.npy"
    priors_filename: str | None = "priors.npy"
    train_env_ids: tuple[int, ...] | None = None
    test_env_ids: tuple[int, ...] | None = None
    use_amplitude_phase: bool = True


PAPER1_DATASET_REGISTRY: dict[str, Paper1DatasetConfig] = {
    "signfi": Paper1DatasetConfig(
        name="SignFi",
        directory_name="signfi",
        evaluation_mode="train_val_test",
        description=(
            "Prepared SignFi classification target using CSI tensors plus labels, "
            "with optional priors and environment metadata."
        ),
    ),
    "ut_har": Paper1DatasetConfig(
        name="UT_HAR",
        directory_name="ut_har",
        evaluation_mode="cross_environment",
        description=(
            "Prepared UT_HAR activity-recognition target with explicit environment "
            "IDs for cross-environment evaluation."
        ),
        train_env_ids=(0,),
        test_env_ids=(1,),
    ),
    "widar_bvp": Paper1DatasetConfig(
        name="Widar_BVP",
        directory_name="widar_bvp",
        evaluation_mode="cross_environment",
        description=(
            "Prepared Widar3.0 BVP benchmark using room IDs as environments and "
            "shared gestures across rooms."
        ),
        priors_filename=None,
        train_env_ids=(0, 1),
        test_env_ids=(2,),
        use_amplitude_phase=False,
    ),
    "ntufi_har": Paper1DatasetConfig(
        name="NTU-Fi_HAR",
        directory_name="ntufi_har",
        evaluation_mode="cross_environment",
        description=(
            "NTU-Fi HAR activity recognition. The three environments are ordered "
            "partitions: each class's recordings in order, cut into thirds."
        ),
        priors_filename=None,
        train_env_ids=(0, 1),
        test_env_ids=(2,),
        use_amplitude_phase=False,
    ),
    "signfi_top10": Paper1DatasetConfig(
        name="SignFi_Top10",
        directory_name="signfi_top10",
        evaluation_mode="cross_environment",
        description=(
            "Ten SignFi sign gestures. The three environments are ordered "
            "partitions of each class's recordings, not separate sessions."
        ),
        priors_filename=None,
        train_env_ids=(0, 1),
        test_env_ids=(2,),
        use_amplitude_phase=False,
    ),
}


@dataclass(frozen=True)
class PreparedCSIBundle:
    config: Paper1DatasetConfig
    features: Tensor
    labels: Tensor
    environments: Tensor | None = None
    priors: Tensor | None = None

    @property
    def num_samples(self) -> int:
        return int(self.features.shape[0])

    @property
    def input_shape(self) -> tuple[int, int]:
        if self.features.dim() != 3:
            raise ValueError(
                "Expected features with shape (N, S, F), got "
                f"{tuple(self.features.shape)}"
            )
        return int(self.features.shape[1]), int(self.features.shape[2])

    @property
    def num_classes(self) -> int:
        return int(torch.max(self.labels).item()) + 1

    def subset(self, indices: Tensor) -> PreparedCSIBundle:
        priors = self.priors[indices] if self.priors is not None else None
        environments = (
            self.environments[indices] if self.environments is not None else None
        )
        return PreparedCSIBundle(
            config=self.config,
            features=self.features[indices],
            labels=self.labels[indices],
            environments=environments,
            priors=priors,
        )


@dataclass(frozen=True)
class Paper1Splits:
    train: PreparedCSIDataset
    val: PreparedCSIDataset
    test: PreparedCSIDataset
    eval_mode: EvaluationMode
    train_environment_ids: tuple[int, ...] | None = None
    test_environment_ids: tuple[int, ...] | None = None


class PreparedCSIDataset(Dataset[dict[str, Tensor]]):
    def __init__(self, bundle: PreparedCSIBundle) -> None:
        self.bundle = bundle

    def __len__(self) -> int:
        return self.bundle.num_samples

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        feature = self.bundle.features[index]
        label = self.bundle.labels[index]
        if self.bundle.priors is None:
            prior = torch.zeros_like(feature)
            has_prior = torch.tensor(False)
        else:
            prior = self.bundle.priors[index]
            has_prior = torch.tensor(True)

        sample: dict[str, Tensor] = {
            "x": feature,
            "label": label,
            "prior": prior,
            "has_prior": has_prior,
        }
        if self.bundle.environments is not None:
            sample["environment"] = self.bundle.environments[index]
        return sample


def get_paper1_dataset_config(name: str) -> Paper1DatasetConfig:
    key = name.lower()
    if key not in PAPER1_DATASET_REGISTRY:
        valid = ", ".join(sorted(PAPER1_DATASET_REGISTRY))
        raise ValueError(f"Unknown Paper 1 dataset: {name}. Expected one of: {valid}")
    return PAPER1_DATASET_REGISTRY[key]


def list_paper1_datasets() -> tuple[str, ...]:
    return tuple(sorted(PAPER1_DATASET_REGISTRY))


def load_prepared_paper1_dataset(
    dataset_name: str,
    prepared_root: str | Path,
    use_amplitude_phase: bool | None = None,
) -> PreparedCSIBundle:
    """Load a prepared dataset directory for the Paper 1 harness.

    Prepared directory contract:
    - `<root>/<dataset>/csi.(npy|npz|pt)`
    - `<root>/<dataset>/labels.(npy|npz|pt)`
    - optional `<root>/<dataset>/environments.(npy|npz|pt)`
    - optional `<root>/<dataset>/priors.(npy|npz|pt)`
    """
    config = get_paper1_dataset_config(dataset_name)
    dataset_dir = Path(prepared_root) / config.directory_name
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Prepared dataset directory not found for {config.name}: {dataset_dir}"
        )

    csi = _prepare_csi_tensor(
        load_array_file(dataset_dir / config.csi_filename),
        use_amplitude_phase=config.use_amplitude_phase
        if use_amplitude_phase is None
        else use_amplitude_phase,
    )
    labels = _prepare_labels(load_array_file(dataset_dir / config.labels_filename))
    environments = _maybe_load_index_tensor(dataset_dir, config.environments_filename)
    priors_array = _maybe_load_array(dataset_dir, config.priors_filename)
    priors = None
    if priors_array is not None:
        priors = _prepare_csi_tensor(
            priors_array,
            use_amplitude_phase=config.use_amplitude_phase
            if use_amplitude_phase is None
            else use_amplitude_phase,
        )

    _validate_bundle_shapes(
        config=config,
        features=csi,
        labels=labels,
        environments=environments,
        priors=priors,
    )
    return PreparedCSIBundle(
        config=config,
        features=csi,
        labels=labels,
        environments=environments,
        priors=priors,
    )


def create_paper1_splits(
    bundle: PreparedCSIBundle,
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    evaluation_mode: EvaluationMode | None = None,
    train_env_ids: tuple[int, ...] | None = None,
    test_env_ids: tuple[int, ...] | None = None,
) -> Paper1Splits:
    """Create train/val/test splits from a prepared bundle."""
    _validate_ratios(train_ratio, val_ratio, test_ratio)
    effective_mode = evaluation_mode or bundle.config.evaluation_mode

    resolved_train_envs: tuple[int, ...] | None = None
    resolved_test_envs: tuple[int, ...] | None = None

    if effective_mode == "train_val_test":
        indices = _shuffle_indices(bundle.num_samples, seed)
        train_end = int(bundle.num_samples * train_ratio)
        val_end = train_end + int(bundle.num_samples * val_ratio)
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
    else:
        environments = bundle.environments
        if environments is None:
            raise ValueError(
                f"Dataset {bundle.config.name} requires environment metadata for "
                "cross-environment evaluation."
            )
        resolved_train_envs = train_env_ids or bundle.config.train_env_ids
        resolved_test_envs = test_env_ids or bundle.config.test_env_ids
        if resolved_train_envs is None or resolved_test_envs is None:
            raise ValueError(
                f"Dataset {bundle.config.name} is missing train/test environment IDs."
            )

        train_mask = torch.isin(environments, torch.tensor(resolved_train_envs))
        test_mask = torch.isin(environments, torch.tensor(resolved_test_envs))
        train_pool = torch.nonzero(train_mask, as_tuple=False).squeeze(-1)
        test_indices = torch.nonzero(test_mask, as_tuple=False).squeeze(-1)
        if train_pool.numel() == 0:
            raise ValueError(
                f"No samples found for train environments {resolved_train_envs} in "
                f"{bundle.config.name}."
            )
        if test_indices.numel() == 0:
            raise ValueError(
                f"No samples found for test environments {resolved_test_envs} in "
                f"{bundle.config.name}."
            )

        shuffled_pool = train_pool[
            torch.randperm(
                train_pool.numel(),
                generator=torch.Generator().manual_seed(seed),
            )
        ]
        train_fraction = train_ratio / (train_ratio + val_ratio)
        train_cutoff = int(shuffled_pool.numel() * train_fraction)
        train_indices = shuffled_pool[:train_cutoff]
        val_indices = shuffled_pool[train_cutoff:]

    return Paper1Splits(
        train=PreparedCSIDataset(bundle.subset(train_indices)),
        val=PreparedCSIDataset(bundle.subset(val_indices)),
        test=PreparedCSIDataset(bundle.subset(test_indices)),
        eval_mode=effective_mode,
        train_environment_ids=resolved_train_envs,
        test_environment_ids=resolved_test_envs,
    )


def create_mock_paper1_prepared_data(
    prepared_root: str | Path,
    samples_per_class: int = 18,
    num_subcarriers: int = 16,
    num_classes: int = 3,
    seed: int = 7,
) -> None:
    """Create tiny prepared-data fixtures for SignFi and UT_HAR smoke runs."""
    root = Path(prepared_root)
    root.mkdir(parents=True, exist_ok=True)

    signfi_dir = root / PAPER1_DATASET_REGISTRY["signfi"].directory_name
    signfi_dir.mkdir(parents=True, exist_ok=True)
    signfi_csi, signfi_labels, signfi_priors = _make_mock_dataset_arrays(
        samples_per_class=samples_per_class,
        num_subcarriers=num_subcarriers,
        num_classes=num_classes,
        seed=seed,
        environment_shift=0.15,
    )
    np.save(signfi_dir / "csi.npy", signfi_csi)
    np.save(signfi_dir / "labels.npy", signfi_labels)
    np.save(signfi_dir / "priors.npy", signfi_priors)

    ut_har_dir = root / PAPER1_DATASET_REGISTRY["ut_har"].directory_name
    ut_har_dir.mkdir(parents=True, exist_ok=True)
    ut_har_csi, ut_har_labels, ut_har_priors, ut_har_envs = _make_mock_cross_env_arrays(
        samples_per_class=samples_per_class,
        num_subcarriers=num_subcarriers,
        num_classes=num_classes,
        seed=seed + 11,
    )
    np.save(ut_har_dir / "csi.npy", ut_har_csi)
    np.save(ut_har_dir / "labels.npy", ut_har_labels)
    np.save(ut_har_dir / "priors.npy", ut_har_priors)
    np.save(ut_har_dir / "environments.npy", ut_har_envs)


def load_array_file(path: str | Path, array_key: str | None = None) -> np.ndarray:
    """Load a numpy or torch array from disk.

    Args:
        path: Source array file. Supports `.npy`, `.npz`, and `.pt`.
        array_key: Optional key to extract from `.npz` archives.

    Returns:
        Loaded array.
    """
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Required prepared-data file not found: {resolved_path}"
        )

    if resolved_path.suffix == ".npy":
        return cast(np.ndarray, np.load(resolved_path))
    if resolved_path.suffix == ".npz":
        npz = np.load(resolved_path)
        if len(npz.files) == 0:
            raise ValueError(f"No arrays found in npz file: {resolved_path}")
        if array_key is not None:
            if array_key not in npz.files:
                raise ValueError(
                    f"Array key '{array_key}' not found in {resolved_path}; "
                    f"available keys: {', '.join(npz.files)}"
                )
            return cast(np.ndarray, npz[array_key])

        for preferred_key in (
            "csi",
            "features",
            "labels",
            "environments",
            "priors",
        ):
            if preferred_key in npz.files:
                return cast(np.ndarray, npz[preferred_key])
        return cast(np.ndarray, npz[npz.files[0]])
    if resolved_path.suffix == ".pt":
        loaded = torch.load(resolved_path, map_location="cpu")
        if isinstance(loaded, Tensor):
            return loaded.detach().cpu().numpy()
        raise TypeError(f"Expected Tensor in {resolved_path}, got {type(loaded)}")
    raise ValueError(f"Unsupported prepared-data file format: {resolved_path.suffix}")


def save_prepared_paper1_dataset(
    dataset_name: str,
    output_root: str | Path,
    features: np.ndarray,
    labels: np.ndarray,
    environments: np.ndarray | None = None,
    priors: np.ndarray | None = None,
    metadata: dict[str, object] | None = None,
) -> Path:
    """Write user-provided arrays into the Paper 1 prepared-data contract.

    Args:
        dataset_name: Registry dataset key, such as `signfi` or `ut_har`.
        output_root: Root directory that will receive `<output_root>/<dataset>`.
        features: User CSI/features array. Shape: `(N, S)` or `(N, S, F)`.
        labels: User labels array. Shape: `(N,)`.
        environments: Optional environment IDs. Shape: `(N,)`.
        priors: Optional prior CSI/features array. Shape must match `features`.
        metadata: Optional JSON-serializable metadata to persist as `metadata.json`.

    Returns:
        Path to the prepared dataset directory.
    """
    config = get_paper1_dataset_config(dataset_name)
    prepared_features = _prepare_csi_tensor(features, use_amplitude_phase=False)
    prepared_labels = _prepare_labels(labels)

    prepared_environments = None
    if environments is not None:
        prepared_environments = torch.as_tensor(
            environments,
            dtype=torch.long,
        ).flatten()

    prepared_priors = None
    if priors is not None:
        prepared_priors = _prepare_csi_tensor(priors, use_amplitude_phase=False)

    _validate_bundle_shapes(
        config=config,
        features=prepared_features,
        labels=prepared_labels,
        environments=prepared_environments,
        priors=prepared_priors,
    )

    if config.evaluation_mode == "cross_environment" and prepared_environments is None:
        raise ValueError(
            f"Dataset {config.name} requires environment metadata for preparation."
        )
    if prepared_environments is not None:
        _validate_environment_ids(config, prepared_environments)

    dataset_dir = Path(output_root) / config.directory_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.save(dataset_dir / config.csi_filename, features)
    np.save(dataset_dir / config.labels_filename, prepared_labels.cpu().numpy())
    if prepared_environments is not None and config.environments_filename is not None:
        np.save(
            dataset_dir / config.environments_filename,
            prepared_environments.cpu().numpy(),
        )
    if (
        prepared_priors is not None
        and config.priors_filename is not None
        and priors is not None
    ):
        np.save(dataset_dir / config.priors_filename, priors.copy())

    manifest = {
        "dataset_key": dataset_name.lower(),
        "dataset_name": config.name,
        "evaluation_mode": config.evaluation_mode,
        "num_samples": int(prepared_features.shape[0]),
        "raw_feature_shape": list(features.shape),
        "prepared_feature_shape": list(prepared_features.shape),
        "num_classes": int(torch.unique(prepared_labels).numel()),
        "has_environments": prepared_environments is not None,
        "has_priors": prepared_priors is not None,
        "train_environment_ids": list(config.train_env_ids or ()),
        "test_environment_ids": list(config.test_env_ids or ()),
        "user_metadata": metadata or {},
    }
    metadata_path = dataset_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return dataset_dir


def _validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    if min(train_ratio, val_ratio, test_ratio) <= 0:
        raise ValueError("train_ratio, val_ratio, and test_ratio must all be > 0")
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio, val_ratio, and test_ratio must sum to 1.0")


def _shuffle_indices(num_samples: int, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_samples, generator=generator)


def _maybe_load_array(directory: Path, filename: str | None) -> np.ndarray | None:
    if filename is None:
        return None
    path = directory / filename
    if not path.exists():
        return None
    return load_array_file(path)


def _maybe_load_index_tensor(directory: Path, filename: str | None) -> Tensor | None:
    array = _maybe_load_array(directory, filename)
    if array is None:
        return None
    return torch.as_tensor(array, dtype=torch.long)


def _prepare_csi_tensor(array: np.ndarray, use_amplitude_phase: bool) -> Tensor:
    tensor = torch.from_numpy(array)
    if torch.is_complex(tensor):
        complex_tensor = tensor.to(torch.complex64)
        if use_amplitude_phase:
            if complex_tensor.dim() == 2:
                amplitude = torch.abs(complex_tensor)
                phase = torch.angle(complex_tensor)
                return torch.stack([amplitude, phase], dim=-1)
            if complex_tensor.dim() == 3:
                return amplitude_phase_split(complex_tensor)
            raise ValueError(
                "Complex CSI tensors must have shape (N, S) or (N, S, A), got "
                f"{tuple(complex_tensor.shape)}"
            )

        if complex_tensor.dim() == 2:
            return torch.view_as_real(complex_tensor).to(torch.float32)
        if complex_tensor.dim() == 3:
            return (
                torch.view_as_real(complex_tensor)
                .flatten(start_dim=-2)
                .to(torch.float32)
            )
        raise ValueError(
            "Complex CSI tensors must have shape (N, S) or (N, S, A), got "
            f"{tuple(complex_tensor.shape)}"
        )

    real_tensor = tensor.to(torch.float32)
    if real_tensor.dim() == 2:
        return real_tensor.unsqueeze(-1)
    if real_tensor.dim() == 3:
        return real_tensor
    raise ValueError(
        "Real CSI tensors must have shape (N, S) or (N, S, F), got "
        f"{tuple(real_tensor.shape)}"
    )


def _prepare_labels(array: np.ndarray) -> Tensor:
    labels = torch.as_tensor(array, dtype=torch.long).flatten()
    if labels.dim() != 1:
        raise ValueError(f"Expected 1D labels, got shape {tuple(labels.shape)}")
    return labels


def _validate_bundle_shapes(
    config: Paper1DatasetConfig,
    features: Tensor,
    labels: Tensor,
    environments: Tensor | None,
    priors: Tensor | None,
) -> None:
    num_samples = int(features.shape[0])
    if int(labels.shape[0]) != num_samples:
        raise ValueError(
            f"Label count mismatch for {config.name}: "
            f"{labels.shape[0]} != {num_samples}"
        )
    if environments is not None and int(environments.shape[0]) != num_samples:
        raise ValueError(
            f"Environment count mismatch for {config.name}: "
            f"{environments.shape[0]} != {num_samples}"
        )
    if priors is not None and priors.shape != features.shape:
        raise ValueError(
            f"Prior shape mismatch for {config.name}: {tuple(priors.shape)} != "
            f"{tuple(features.shape)}"
        )


def _validate_environment_ids(
    config: Paper1DatasetConfig,
    environments: Tensor,
) -> None:
    if config.train_env_ids is None and config.test_env_ids is None:
        return

    available_envs = set(torch.unique(environments).tolist())
    missing_train = set(config.train_env_ids or ()).difference(available_envs)
    missing_test = set(config.test_env_ids or ()).difference(available_envs)
    if missing_train or missing_test:
        raise ValueError(
            f"Prepared environments for {config.name} are missing configured IDs: "
            f"train={sorted(missing_train)}, test={sorted(missing_test)}"
        )


def _make_mock_dataset_arrays(
    samples_per_class: int,
    num_subcarriers: int,
    num_classes: int,
    seed: int,
    environment_shift: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    csi: list[np.ndarray] = []
    labels: list[int] = []
    priors: list[np.ndarray] = []
    base_axis = np.linspace(-1.0, 1.0, num_subcarriers, dtype=np.float32)
    for class_idx in range(num_classes):
        template_real = np.sin((class_idx + 1) * np.pi * base_axis)
        template_imag = np.cos((class_idx + 1.5) * np.pi * base_axis)
        template = template_real + 1j * template_imag
        for sample_idx in range(samples_per_class):
            scale = 1.0 + 0.2 * class_idx + 0.03 * sample_idx
            phase_shift = np.exp(1j * environment_shift * (sample_idx % 4))
            noise = 0.08 * (
                rng.standard_normal(num_subcarriers)
                + 1j * rng.standard_normal(num_subcarriers)
            )
            prior = (scale * template).astype(np.complex64)
            observed = (prior * phase_shift + noise).astype(np.complex64)
            priors.append(prior)
            csi.append(observed)
            labels.append(class_idx)
    return (
        np.stack(csi).astype(np.complex64),
        np.asarray(labels, dtype=np.int64),
        np.stack(priors).astype(np.complex64),
    )


def _make_mock_cross_env_arrays(
    samples_per_class: int,
    num_subcarriers: int,
    num_classes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    env0_csi, env0_labels, env0_priors = _make_mock_dataset_arrays(
        samples_per_class=samples_per_class,
        num_subcarriers=num_subcarriers,
        num_classes=num_classes,
        seed=seed,
        environment_shift=0.05,
    )
    env1_csi, env1_labels, env1_priors = _make_mock_dataset_arrays(
        samples_per_class=samples_per_class,
        num_subcarriers=num_subcarriers,
        num_classes=num_classes,
        seed=seed + 1,
        environment_shift=0.28,
    )
    csi = np.concatenate([env0_csi, env1_csi], axis=0)
    labels = np.concatenate([env0_labels, env1_labels], axis=0)
    priors = np.concatenate([env0_priors, env1_priors], axis=0)
    environments = np.concatenate(
        [
            np.zeros(env0_labels.shape[0], dtype=np.int64),
            np.ones(env1_labels.shape[0], dtype=np.int64),
        ]
    )
    return csi, labels, priors, environments
