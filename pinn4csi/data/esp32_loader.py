# pyright: basic, reportMissingImports=false

"""Prepared-data loader for ESP32 CSI captures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pinn4csi.data.prepared_collection import (
    CollectionMetadataBase,
    CollectionRepresentation,
    PreparedCollectionBundle,
    load_metadata_json,
    load_prepared_array,
    maybe_load_prepared_array,
    parse_id_name_map,
    prepare_csi_tensor,
    prepare_index_tensor,
    prepare_timestamp_tensor,
    require_float,
    require_int,
    require_string,
    validate_prepared_capture,
)


@dataclass(frozen=True)
class ESP32CaptureMetadata(CollectionMetadataBase):
    board: str
    firmware_version: str
    channel: int
    phase_quality: str


def load_esp32_prepared_dataset(
    prepared_root: str | Path,
    representation: CollectionRepresentation = "amplitude",
) -> PreparedCollectionBundle[ESP32CaptureMetadata]:
    dataset_dir = Path(prepared_root)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"ESP32 prepared-data directory not found: {dataset_dir}"
        )

    metadata = _parse_esp32_metadata(
        load_metadata_json(dataset_dir / "metadata.json")
    )
    csi_array = load_prepared_array(dataset_dir / "csi.npy")
    labels = prepare_index_tensor(
        load_prepared_array(dataset_dir / "labels.npy"),
        "labels",
    )
    environments = prepare_index_tensor(
        load_prepared_array(dataset_dir / "environments.npy"),
        "environments",
    )
    timestamps = prepare_timestamp_tensor(
        maybe_load_prepared_array(dataset_dir, "timestamps.npy")
    )

    validate_prepared_capture(
        source_name="ESP32",
        metadata=metadata,
        csi_array=csi_array,
        labels=labels,
        environments=environments,
        timestamps=timestamps,
    )
    return PreparedCollectionBundle(
        metadata=metadata,
        features=prepare_csi_tensor(csi_array, representation=representation),
        labels=labels,
        environments=environments,
        timestamps=timestamps,
        representation=representation,
    )


def _parse_esp32_metadata(metadata: dict[str, object]) -> ESP32CaptureMetadata:
    source = require_string(metadata, "source")
    if source != "esp32":
        raise ValueError(f"metadata.json source must be 'esp32', got {source!r}")

    return ESP32CaptureMetadata(
        source=source,
        protocol_version=require_string(metadata, "protocol_version"),
        capture_id=require_string(metadata, "capture_id"),
        task_name=require_string(metadata, "task_name"),
        environment_names=parse_id_name_map(metadata, "environment_names"),
        label_names=parse_id_name_map(metadata, "label_names"),
        num_subcarriers=require_int(metadata, "num_subcarriers"),
        num_rx_antennas=require_int(metadata, "num_rx_antennas"),
        center_frequency_hz=require_float(metadata, "center_frequency_hz"),
        bandwidth_mhz=require_float(metadata, "bandwidth_mhz"),
        board=require_string(metadata, "board"),
        firmware_version=require_string(metadata, "firmware_version"),
        channel=require_int(metadata, "channel"),
        phase_quality=require_string(metadata, "phase_quality"),
    )
