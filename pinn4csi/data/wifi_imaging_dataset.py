# pyright: basic, reportMissingImports=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportUntypedBaseClass=false, reportMissingSuperCall=false, reportImplicitStringConcatenation=false

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from pinn4csi.data.prepared_collection import (
    load_metadata_json,
    load_prepared_array,
    parse_id_name_map,
    require_float,
    require_int,
    require_string,
)

WiFiImagingRepresentation = Literal["amplitude", "amplitude_phase"]


@dataclass(frozen=True)
class CoordinateNormalizationStats:
    minimum: Tensor
    maximum: Tensor


@dataclass(frozen=True)
class FieldNormalizationStats:
    minimum: float
    maximum: float


@dataclass(frozen=True)
class WiFiImagingMetadata:
    source: str
    protocol_version: str
    capture_id: str
    task_name: str
    environment_names: dict[int, str]
    coordinate_dim: int
    num_subcarriers: int
    num_tx_rx_pairs: int
    grid_shape: tuple[int, ...] | None
    frequency_hz: float | None


@dataclass(frozen=True)
class WiFiImagingPreparedBundle:
    metadata: WiFiImagingMetadata
    csi_features: Tensor
    tx_rx_positions: Tensor
    query_coordinates: Tensor
    field_targets: Tensor
    permittivity_targets: Tensor
    environment_ids: Tensor
    frequencies_hz: Tensor
    coordinate_stats: dict[int, CoordinateNormalizationStats]
    field_stats: dict[int, FieldNormalizationStats]
    representation: WiFiImagingRepresentation

    @property
    def num_samples(self) -> int:
        return int(self.csi_features.shape[0])

    @property
    def num_pairs(self) -> int:
        return int(self.csi_features.shape[1])

    @property
    def pair_feature_dim(self) -> int:
        return int(self.csi_features.shape[2])

    @property
    def csi_feature_dim(self) -> int:
        return self.num_pairs * self.pair_feature_dim

    def denormalize_field(
        self, field_values: Tensor, environment_ids: Tensor
    ) -> Tensor:
        if field_values.shape[0] != environment_ids.shape[0]:
            raise ValueError(
                "field_values and environment_ids must share batch size. "
                f"Got {field_values.shape[0]} and {environment_ids.shape[0]}"
            )

        restored = field_values.clone()
        for raw_environment_id in sorted(set(environment_ids.tolist())):
            environment_id = int(raw_environment_id)
            stats = self.field_stats[environment_id]
            mask = environment_ids == environment_id
            scale = stats.maximum - stats.minimum
            if scale <= 1e-8:
                restored[mask] = stats.minimum
            else:
                restored[mask] = (restored[mask] * scale) + stats.minimum
        return restored


class WiFiImagingDataset(Dataset[dict[str, Tensor]]):
    def __init__(
        self,
        bundle: WiFiImagingPreparedBundle,
        indices: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.bundle = bundle
        self.indices = (
            torch.arange(bundle.num_samples, dtype=torch.long)
            if indices is None
            else indices.to(dtype=torch.long)
        )
        if self.indices.dim() != 1:
            raise ValueError(
                f"indices must be 1D, got shape {tuple(self.indices.shape)}"
            )

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        sample_index = int(self.indices[index].item())
        pair_features = self.bundle.csi_features[sample_index]
        return {
            "sample_index": torch.tensor(sample_index, dtype=torch.long),
            "csi_features": pair_features.reshape(-1),
            "pair_features": pair_features,
            "tx_rx_positions": self.bundle.tx_rx_positions[sample_index],
            "query_coordinates": self.bundle.query_coordinates[sample_index],
            "field_target": self.bundle.field_targets[sample_index],
            "permittivity_target": self.bundle.permittivity_targets[sample_index],
            "environment_id": self.bundle.environment_ids[sample_index],
            "frequency_hz": self.bundle.frequencies_hz[sample_index],
        }


@dataclass(frozen=True)
class WiFiImagingSplits:
    train: WiFiImagingDataset
    val: WiFiImagingDataset
    test: WiFiImagingDataset
    train_environment_ids: tuple[int, ...]
    val_environment_ids: tuple[int, ...]
    test_environment_ids: tuple[int, ...]


def load_wifi_imaging_prepared_dataset(
    prepared_root: str | Path,
    representation: WiFiImagingRepresentation = "amplitude",
) -> WiFiImagingPreparedBundle:
    dataset_dir = Path(prepared_root)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"WiFi imaging prepared-data directory not found: {dataset_dir}"
        )

    metadata = _parse_metadata(load_metadata_json(dataset_dir / "metadata.json"))
    csi_array = _load_required_array(dataset_dir, ("csi.npy", "csi_measurements.npy"))
    tx_rx_positions = _load_required_array(dataset_dir, ("tx_rx_positions.npy",))
    query_coordinates = _load_required_array(dataset_dir, ("query_coordinates.npy",))
    field_target = _load_required_array(dataset_dir, ("field.npy", "field_target.npy"))
    permittivity = _load_required_array(
        dataset_dir,
        ("permittivity.npy", "permittivity_map.npy"),
    )
    environment_ids = _load_required_array(
        dataset_dir,
        ("environments.npy", "environment_ids.npy"),
    )
    frequencies = _load_optional_array(
        dataset_dir,
        ("frequencies.npy", "frequency_hz.npy"),
    )

    return _prepare_bundle(
        metadata=metadata,
        csi_array=csi_array,
        tx_rx_positions=tx_rx_positions,
        query_coordinates=query_coordinates,
        field_target=field_target,
        permittivity=permittivity,
        environment_ids=environment_ids,
        frequencies=frequencies,
        representation=representation,
    )


def create_wifi_imaging_splits(
    bundle: WiFiImagingPreparedBundle,
    *,
    train_environment_ids: tuple[int, ...] | list[int] | None = None,
    val_environment_ids: tuple[int, ...] | list[int] | None = None,
    test_environment_ids: tuple[int, ...] | list[int] | None = None,
    seed: int = 42,
    val_environments: int = 1,
    test_environments: int = 1,
) -> WiFiImagingSplits:
    available_environment_ids = sorted(
        {int(environment_id) for environment_id in bundle.environment_ids.tolist()}
    )
    if len(available_environment_ids) < 3:
        raise ValueError(
            "WiFi imaging splits require at least 3 environments for train/val/test. "
            f"Got {available_environment_ids}"
        )

    rng = np.random.default_rng(seed)
    shuffled_environment_ids = available_environment_ids.copy()
    rng.shuffle(shuffled_environment_ids)

    resolved_val = _normalize_environment_ids(val_environment_ids)
    if resolved_val is None:
        resolved_val = tuple(sorted(shuffled_environment_ids[:val_environments]))

    resolved_test = _normalize_environment_ids(test_environment_ids)
    if resolved_test is None:
        remaining = [
            environment_id
            for environment_id in shuffled_environment_ids
            if environment_id not in set(resolved_val)
        ]
        resolved_test = tuple(sorted(remaining[:test_environments]))

    resolved_train = _normalize_environment_ids(train_environment_ids)
    if resolved_train is None:
        reserved = set(resolved_val) | set(resolved_test)
        resolved_train = tuple(
            environment_id
            for environment_id in available_environment_ids
            if environment_id not in reserved
        )

    _validate_environment_splits(
        available_environment_ids=tuple(available_environment_ids),
        train_environment_ids=resolved_train,
        val_environment_ids=resolved_val,
        test_environment_ids=resolved_test,
    )
    return WiFiImagingSplits(
        train=WiFiImagingDataset(
            bundle,
            indices=_select_environment_indices(bundle.environment_ids, resolved_train),
        ),
        val=WiFiImagingDataset(
            bundle,
            indices=_select_environment_indices(bundle.environment_ids, resolved_val),
        ),
        test=WiFiImagingDataset(
            bundle,
            indices=_select_environment_indices(bundle.environment_ids, resolved_test),
        ),
        train_environment_ids=resolved_train,
        val_environment_ids=resolved_val,
        test_environment_ids=resolved_test,
    )


def _parse_metadata(metadata: dict[str, object]) -> WiFiImagingMetadata:
    raw_grid_shape = metadata.get("grid_shape")
    grid_shape: tuple[int, ...] | None = None
    if raw_grid_shape is not None:
        if not isinstance(raw_grid_shape, list) or len(raw_grid_shape) == 0:
            raise ValueError("metadata.json key 'grid_shape' must be a non-empty list")
        parsed_grid_shape: list[int] = []
        for raw_value in raw_grid_shape:
            if isinstance(raw_value, bool) or not isinstance(raw_value, int):
                raise ValueError("metadata.json key 'grid_shape' must contain integers")
            parsed_grid_shape.append(raw_value)
        grid_shape = tuple(parsed_grid_shape)

    frequency_hz: float | None = None
    if "frequency_hz" in metadata:
        frequency_hz = require_float(metadata, "frequency_hz")

    return WiFiImagingMetadata(
        source=require_string(metadata, "source"),
        protocol_version=require_string(metadata, "protocol_version"),
        capture_id=require_string(metadata, "capture_id"),
        task_name=require_string(metadata, "task_name"),
        environment_names=parse_id_name_map(metadata, "environment_names"),
        coordinate_dim=require_int(metadata, "coordinate_dim"),
        num_subcarriers=require_int(metadata, "num_subcarriers"),
        num_tx_rx_pairs=require_int(metadata, "num_tx_rx_pairs"),
        grid_shape=grid_shape,
        frequency_hz=frequency_hz,
    )


def _load_required_array(directory: Path, candidates: tuple[str, ...]) -> np.ndarray:
    array = _load_optional_array(directory, candidates)
    if array is None:
        joined_names = ", ".join(candidates)
        raise FileNotFoundError(
            "Required prepared-data file not found in "
            f"{directory}: one of {joined_names}"
        )
    return array


def _load_optional_array(
    directory: Path,
    candidates: tuple[str, ...],
) -> np.ndarray | None:
    for candidate in candidates:
        candidate_path = directory / candidate
        if candidate_path.exists():
            return load_prepared_array(candidate_path)
    return None


def _prepare_bundle(
    *,
    metadata: WiFiImagingMetadata,
    csi_array: np.ndarray,
    tx_rx_positions: np.ndarray,
    query_coordinates: np.ndarray,
    field_target: np.ndarray,
    permittivity: np.ndarray,
    environment_ids: np.ndarray,
    frequencies: np.ndarray | None,
    representation: WiFiImagingRepresentation,
) -> WiFiImagingPreparedBundle:
    if csi_array.ndim != 3:
        raise ValueError(
            "CSI prepared array must have shape (N, P, K), got "
            f"{tuple(csi_array.shape)}"
        )
    num_samples, num_pairs, num_subcarriers = csi_array.shape
    if num_pairs != metadata.num_tx_rx_pairs:
        raise ValueError(
            "num_tx_rx_pairs mismatch between metadata and CSI array. "
            f"Got {metadata.num_tx_rx_pairs} and {num_pairs}"
        )
    if num_subcarriers != metadata.num_subcarriers:
        raise ValueError(
            "num_subcarriers mismatch between metadata and CSI array. "
            f"Got {metadata.num_subcarriers} and {num_subcarriers}"
        )

    tx_rx_tensor = _prepare_tx_rx_positions(
        tx_rx_positions,
        num_samples=num_samples,
        num_pairs=num_pairs,
        coordinate_dim=metadata.coordinate_dim,
    )
    query_tensor = _prepare_query_coordinates(
        query_coordinates,
        num_samples=num_samples,
        coordinate_dim=metadata.coordinate_dim,
    )
    field_tensor = _prepare_scalar_array(field_target, num_samples, "field_target")
    permittivity_tensor = _prepare_scalar_array(
        permittivity, num_samples, "permittivity"
    )
    environment_tensor = _prepare_environment_ids(environment_ids, num_samples)
    frequency_tensor = _prepare_frequencies(frequencies, metadata, num_samples)

    if query_tensor.shape[1] != field_tensor.shape[1]:
        raise ValueError(
            "query_coordinates and field_target must share num_query_points. "
            f"Got {query_tensor.shape[1]} and {field_tensor.shape[1]}"
        )
    if query_tensor.shape[1] != permittivity_tensor.shape[1]:
        raise ValueError(
            "query_coordinates and permittivity must share num_query_points. "
            f"Got {query_tensor.shape[1]} and {permittivity_tensor.shape[1]}"
        )
    if (permittivity_tensor < 1.0).any():
        minimum_value = float(permittivity_tensor.min().item())
        raise ValueError(f"permittivity must be >= 1.0, got minimum {minimum_value}")

    missing_environment_ids = sorted(
        set(environment_tensor.tolist()) - set(metadata.environment_names)
    )
    if missing_environment_ids:
        raise ValueError(
            "metadata.json environment_names missing IDs for WiFi imaging data: "
            f"{missing_environment_ids}"
        )

    csi_tensor = _prepare_csi_features(csi_array, representation)
    normalized_queries, normalized_positions, coordinate_stats = (
        _normalize_coordinates_per_environment(
            query_tensor,
            tx_rx_tensor,
            environment_tensor,
        )
    )
    normalized_field, field_stats = _normalize_field_per_environment(
        field_tensor,
        environment_tensor,
    )
    return WiFiImagingPreparedBundle(
        metadata=metadata,
        csi_features=csi_tensor,
        tx_rx_positions=normalized_positions,
        query_coordinates=normalized_queries,
        field_targets=normalized_field,
        permittivity_targets=permittivity_tensor,
        environment_ids=environment_tensor,
        frequencies_hz=frequency_tensor,
        coordinate_stats=coordinate_stats,
        field_stats=field_stats,
        representation=representation,
    )


def _prepare_tx_rx_positions(
    array: np.ndarray,
    *,
    num_samples: int,
    num_pairs: int,
    coordinate_dim: int,
) -> Tensor:
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0).expand(num_samples, -1, -1, -1)
    if tensor.dim() != 4:
        raise ValueError(
            "tx_rx_positions must have shape (N, P, 2, D) or (P, 2, D), got "
            f"{tuple(tensor.shape)}"
        )
    if tuple(tensor.shape[:2]) != (num_samples, num_pairs):
        raise ValueError(
            "tx_rx_positions must match CSI sample and pair counts. Got "
            f"{tuple(tensor.shape[:2])} vs {(num_samples, num_pairs)}"
        )
    if tensor.shape[2] != 2:
        raise ValueError(
            "tx_rx_positions must encode transmitter and receiver coordinates, got "
            f"shape {tuple(tensor.shape)}"
        )
    if tensor.shape[3] != coordinate_dim:
        raise ValueError(
            "tx_rx_positions coordinate dimension mismatch. Got "
            f"{tensor.shape[3]} and {coordinate_dim}"
        )
    return tensor


def _prepare_query_coordinates(
    array: np.ndarray,
    *,
    num_samples: int,
    coordinate_dim: int,
) -> Tensor:
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.dim() == 2:
        tensor = tensor.unsqueeze(0).expand(num_samples, -1, -1)
    if tensor.dim() != 3:
        raise ValueError(
            "query_coordinates must have shape (N, Q, D) or (Q, D), got "
            f"{tuple(tensor.shape)}"
        )
    if tensor.shape[0] != num_samples:
        raise ValueError(
            "query_coordinates sample count mismatch. Got "
            f"{tensor.shape[0]} and {num_samples}"
        )
    if tensor.shape[2] != coordinate_dim:
        raise ValueError(
            "query_coordinates coordinate dimension mismatch. Got "
            f"{tensor.shape[2]} and {coordinate_dim}"
        )
    return tensor


def _prepare_scalar_array(
    array: np.ndarray,
    num_samples: int,
    field_name: str,
) -> Tensor:
    tensor = torch.as_tensor(array, dtype=torch.float32)
    if tensor.dim() != 2:
        raise ValueError(
            f"{field_name} must have shape (N, Q), got {tuple(tensor.shape)}"
        )
    if tensor.shape[0] != num_samples:
        raise ValueError(
            f"{field_name} sample count mismatch. Got {tensor.shape[0]} "
            f"and {num_samples}"
        )
    return tensor


def _prepare_environment_ids(array: np.ndarray, num_samples: int) -> Tensor:
    tensor = torch.as_tensor(array, dtype=torch.long)
    if tensor.dim() != 1:
        raise ValueError(f"environment_ids must be 1D, got shape {tuple(tensor.shape)}")
    if tensor.shape[0] != num_samples:
        raise ValueError(
            "environment_ids sample count mismatch. Got "
            f"{tensor.shape[0]} and {num_samples}"
        )
    return tensor


def _prepare_frequencies(
    frequencies: np.ndarray | None,
    metadata: WiFiImagingMetadata,
    num_samples: int,
) -> Tensor:
    if frequencies is None:
        if metadata.frequency_hz is None:
            raise ValueError(
                "Prepared WiFi imaging data must provide frequency_hz in metadata "
                "or frequencies.npy"
            )
        return torch.full((num_samples,), metadata.frequency_hz, dtype=torch.float32)

    tensor = torch.as_tensor(frequencies, dtype=torch.float32)
    if tensor.dim() == 0:
        return tensor.repeat(num_samples)
    if tensor.dim() != 1 or tensor.shape[0] != num_samples:
        raise ValueError(
            f"frequencies must be scalar or shape (N,), got {tuple(tensor.shape)}"
        )
    if (tensor <= 0.0).any():
        minimum_value = float(tensor.min().item())
        raise ValueError(f"frequencies must be positive, got minimum {minimum_value}")
    return tensor


def _prepare_csi_features(
    csi_array: np.ndarray,
    representation: str,
) -> Tensor:
    if np.iscomplexobj(csi_array):
        complex_array = np.asarray(csi_array, dtype=np.complex64)
        amplitude = np.abs(complex_array).astype(np.float32, copy=False)
        phase = np.angle(complex_array).astype(np.float32, copy=False)
    else:
        complex_array = None
        amplitude = np.asarray(csi_array, dtype=np.float32)
        phase = None

    amplitude_mean = amplitude.mean(axis=-1, keepdims=True)
    amplitude_std = amplitude.std(axis=-1, keepdims=True)
    amplitude_normalized = (amplitude - amplitude_mean) / (amplitude_std + 1e-8)

    if representation == "amplitude":
        return torch.as_tensor(amplitude_normalized, dtype=torch.float32)
    if representation == "amplitude_phase":
        if complex_array is None or phase is None:
            raise ValueError(
                "amplitude_phase representation requires complex-valued CSI input"
            )
        combined = np.concatenate([amplitude_normalized, phase / np.pi], axis=-1)
        return torch.as_tensor(combined, dtype=torch.float32)
    raise ValueError(f"Unsupported WiFi imaging representation: {representation}")


def _normalize_coordinates_per_environment(
    query_coordinates: Tensor,
    tx_rx_positions: Tensor,
    environment_ids: Tensor,
) -> tuple[Tensor, Tensor, dict[int, CoordinateNormalizationStats]]:
    normalized_queries = query_coordinates.clone()
    normalized_positions = tx_rx_positions.clone()
    stats: dict[int, CoordinateNormalizationStats] = {}

    for raw_environment_id in sorted(set(environment_ids.tolist())):
        environment_id = int(raw_environment_id)
        mask = environment_ids == environment_id
        environment_queries = query_coordinates[mask]
        environment_positions = tx_rx_positions[mask].reshape(
            -1, tx_rx_positions.shape[-1]
        )
        stacked_coordinates = torch.cat(
            [
                environment_queries.reshape(-1, environment_queries.shape[-1]),
                environment_positions,
            ],
            dim=0,
        )
        minimum = stacked_coordinates.min(dim=0).values
        maximum = stacked_coordinates.max(dim=0).values
        scale = maximum - minimum
        normalized_queries[mask] = (
            2.0 * (environment_queries - minimum) / (scale + 1e-8)
        ) - 1.0
        normalized_positions[mask] = (
            2.0 * (tx_rx_positions[mask] - minimum) / (scale + 1e-8)
        ) - 1.0
        stats[environment_id] = CoordinateNormalizationStats(
            minimum=minimum,
            maximum=maximum,
        )

    return normalized_queries, normalized_positions, stats


def _normalize_field_per_environment(
    field_target: Tensor,
    environment_ids: Tensor,
) -> tuple[Tensor, dict[int, FieldNormalizationStats]]:
    normalized_field = field_target.clone()
    stats: dict[int, FieldNormalizationStats] = {}
    for raw_environment_id in sorted(set(environment_ids.tolist())):
        environment_id = int(raw_environment_id)
        mask = environment_ids == environment_id
        environment_field = field_target[mask]
        minimum = float(environment_field.min().item())
        maximum = float(environment_field.max().item())
        scale = maximum - minimum
        if scale <= 1e-8:
            normalized_field[mask] = 0.0
        else:
            normalized_field[mask] = (environment_field - minimum) / (scale + 1e-8)
        stats[environment_id] = FieldNormalizationStats(
            minimum=minimum,
            maximum=maximum,
        )
    return normalized_field, stats


def _normalize_environment_ids(
    environment_ids: tuple[int, ...] | list[int] | None,
) -> tuple[int, ...] | None:
    if environment_ids is None:
        return None
    return tuple(sorted(int(environment_id) for environment_id in environment_ids))


def _validate_environment_splits(
    *,
    available_environment_ids: tuple[int, ...],
    train_environment_ids: tuple[int, ...],
    val_environment_ids: tuple[int, ...],
    test_environment_ids: tuple[int, ...],
) -> None:
    if not train_environment_ids:
        raise ValueError("train_environment_ids must be non-empty")
    if not val_environment_ids:
        raise ValueError("val_environment_ids must be non-empty")
    if not test_environment_ids:
        raise ValueError("test_environment_ids must be non-empty")

    available = set(available_environment_ids)
    train_set = set(train_environment_ids)
    val_set = set(val_environment_ids)
    test_set = set(test_environment_ids)
    if (
        not train_set <= available
        or not val_set <= available
        or not test_set <= available
    ):
        raise ValueError(
            "Split environment IDs must come from available environments. Got "
            f"train={train_environment_ids}, val={val_environment_ids}, "
            f"test={test_environment_ids}, available={available_environment_ids}"
        )
    if train_set & val_set or train_set & test_set or val_set & test_set:
        raise ValueError(
            "Train/val/test environment IDs must be disjoint. Got "
            f"train={train_environment_ids}, val={val_environment_ids}, "
            f"test={test_environment_ids}"
        )


def _select_environment_indices(
    environment_ids: Tensor,
    selected_environment_ids: tuple[int, ...],
) -> Tensor:
    mask = torch.isin(
        environment_ids,
        torch.tensor(selected_environment_ids, dtype=environment_ids.dtype),
    )
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError(
            f"No samples found for environment IDs {selected_environment_ids}"
        )
    return indices.to(dtype=torch.long)
