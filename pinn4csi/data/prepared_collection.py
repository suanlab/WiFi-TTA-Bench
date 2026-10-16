# pyright: basic, reportMissingImports=false, reportUnreachable=false

"""Shared helpers for prepared self-collected CSI datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Literal, TypeVar, cast

import numpy as np
import torch
from torch import Tensor

from pinn4csi.data.transforms import amplitude_phase_split

CollectionRepresentation = Literal[
    "complex",
    "amplitude",
    "amplitude_phase",
    "real_imag",
]


@dataclass(frozen=True)
class CollectionMetadataBase:
    source: str
    protocol_version: str
    capture_id: str
    task_name: str
    environment_names: dict[int, str]
    label_names: dict[int, str]
    num_subcarriers: int
    num_rx_antennas: int
    center_frequency_hz: float
    bandwidth_mhz: float


MetadataT = TypeVar("MetadataT", bound=CollectionMetadataBase)


@dataclass(frozen=True)
class PreparedCollectionBundle(Generic[MetadataT]):
    metadata: MetadataT
    features: Tensor
    labels: Tensor
    environments: Tensor
    timestamps: Tensor | None
    representation: CollectionRepresentation

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
        return len(self.metadata.label_names)


def load_metadata_json(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Required metadata file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in metadata file: {path}")
    return cast(dict[str, object], loaded)


def load_prepared_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Required prepared-data file not found: {path}")

    if path.suffix == ".npy":
        return cast(np.ndarray, np.load(path))
    if path.suffix == ".npz":
        npz = np.load(path)
        if len(npz.files) == 0:
            raise ValueError(f"No arrays found in npz file: {path}")
        key = "csi" if "csi" in npz.files else npz.files[0]
        return cast(np.ndarray, npz[key])
    if path.suffix == ".pt":
        loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, Tensor):
            return loaded.detach().cpu().numpy()
        raise TypeError(f"Expected Tensor in {path}, got {type(loaded)}")
    raise ValueError(f"Unsupported prepared-data file format: {path.suffix}")


def maybe_load_prepared_array(
    directory: Path,
    filename: str | None,
) -> np.ndarray | None:
    if filename is None:
        return None
    path = directory / filename
    if not path.exists():
        return None
    return load_prepared_array(path)


def require_string(metadata: dict[str, object], key: str) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"metadata.json key '{key}' must be a non-empty string")
    return value


def require_int(metadata: dict[str, object], key: str) -> int:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"metadata.json key '{key}' must be an integer")
    return value


def require_float(metadata: dict[str, object], key: str) -> float:
    value = metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"metadata.json key '{key}' must be numeric")
    return float(value)


def parse_id_name_map(metadata: dict[str, object], key: str) -> dict[int, str]:
    value = metadata.get(key)
    if not isinstance(value, dict) or len(value) == 0:
        raise ValueError(f"metadata.json key '{key}' must be a non-empty object")

    parsed: dict[int, str] = {}
    for raw_id, raw_name in value.items():
        if not isinstance(raw_id, str) or not raw_id.isdigit():
            raise ValueError(
                f"metadata.json key '{key}' must use stringified integer IDs"
            )
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError(
                f"metadata.json key '{key}' values must be non-empty strings"
            )
        parsed[int(raw_id)] = raw_name
    return parsed


def prepare_index_tensor(array: np.ndarray, field_name: str) -> Tensor:
    if array.ndim != 1:
        raise ValueError(f"Expected 1D {field_name}, got shape {tuple(array.shape)}")
    return torch.as_tensor(array, dtype=torch.long)


def prepare_timestamp_tensor(array: np.ndarray | None) -> Tensor | None:
    if array is None:
        return None
    if array.ndim != 1:
        raise ValueError(f"Expected 1D timestamps, got shape {tuple(array.shape)}")
    return torch.as_tensor(array, dtype=torch.float64)


def prepare_csi_tensor(
    array: np.ndarray,
    representation: CollectionRepresentation,
) -> Tensor:
    if array.ndim != 3:
        raise ValueError(
            f"Prepared CSI arrays must have shape (N, S, A), got {tuple(array.shape)}"
        )
    if not np.iscomplexobj(array):
        raise ValueError(
            "Prepared CSI arrays must be complex-valued with shape (N, S, A)"
        )

    complex_tensor = torch.from_numpy(array).to(torch.complex64)
    normalized_representation = cast(str, representation)
    if normalized_representation == "complex":
        return complex_tensor
    if normalized_representation == "amplitude":
        return torch.abs(complex_tensor)
    if normalized_representation == "amplitude_phase":
        return amplitude_phase_split(complex_tensor)
    if normalized_representation == "real_imag":
        return (
            torch.view_as_real(complex_tensor).flatten(start_dim=-2).to(torch.float32)
        )
    raise ValueError(f"Unsupported representation: {normalized_representation}")


def validate_prepared_capture(
    source_name: str,
    metadata: CollectionMetadataBase,
    csi_array: np.ndarray,
    labels: Tensor,
    environments: Tensor,
    timestamps: Tensor | None,
) -> None:
    if csi_array.ndim != 3:
        raise ValueError(
            f"{source_name} prepared CSI must have shape (N, S, A), got "
            f"{tuple(csi_array.shape)}"
        )
    if not np.iscomplexobj(csi_array):
        raise ValueError(f"{source_name} prepared CSI must be complex-valued")

    num_samples, num_subcarriers, num_rx_antennas = csi_array.shape
    if int(labels.shape[0]) != num_samples:
        raise ValueError(
            "Label count mismatch for "
            f"{source_name}: {labels.shape[0]} != {num_samples}"
        )
    if int(environments.shape[0]) != num_samples:
        raise ValueError(
            "Environment count mismatch for "
            f"{source_name}: {environments.shape[0]} != {num_samples}"
        )
    if timestamps is not None and int(timestamps.shape[0]) != num_samples:
        raise ValueError(
            f"Timestamp count mismatch for {source_name}: "
            f"{timestamps.shape[0]} != {num_samples}"
        )

    if metadata.num_subcarriers != num_subcarriers:
        raise ValueError(
            f"num_subcarriers mismatch for {source_name}: "
            f"metadata={metadata.num_subcarriers}, csi={num_subcarriers}"
        )
    if metadata.num_rx_antennas != num_rx_antennas:
        raise ValueError(
            f"num_rx_antennas mismatch for {source_name}: "
            f"metadata={metadata.num_rx_antennas}, csi={num_rx_antennas}"
        )

    missing_labels = sorted(set(labels.tolist()) - set(metadata.label_names))
    if missing_labels:
        raise ValueError(
            f"metadata.json label_names missing IDs for {source_name}: {missing_labels}"
        )

    missing_envs = sorted(set(environments.tolist()) - set(metadata.environment_names))
    if missing_envs:
        raise ValueError(
            "metadata.json environment_names missing IDs for "
            f"{source_name}: {missing_envs}"
        )
