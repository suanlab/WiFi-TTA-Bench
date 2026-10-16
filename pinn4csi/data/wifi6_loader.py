# pyright: basic, reportMissingImports=false

"""Prepared-data loader for WiFi 6 CSI captures."""

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
class WiFi6CaptureMetadata(CollectionMetadataBase):
    receiver: str
    chipset: str
    standard: str
    num_tx_streams: int


def load_wifi6_prepared_dataset(
    prepared_root: str | Path,
    representation: CollectionRepresentation = "amplitude_phase",
) -> PreparedCollectionBundle[WiFi6CaptureMetadata]:
    dataset_dir = Path(prepared_root)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"WiFi 6 prepared-data directory not found: {dataset_dir}"
        )

    metadata = _parse_wifi6_metadata(
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
        source_name="WiFi 6",
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


def _parse_wifi6_metadata(metadata: dict[str, object]) -> WiFi6CaptureMetadata:
    source = require_string(metadata, "source")
    if source != "wifi6":
        raise ValueError(f"metadata.json source must be 'wifi6', got {source!r}")

    standard = require_string(metadata, "standard")
    if standard != "802.11ax":
        raise ValueError(
            f"metadata.json standard must be '802.11ax', got {standard!r}"
        )

    return WiFi6CaptureMetadata(
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
        receiver=require_string(metadata, "receiver"),
        chipset=require_string(metadata, "chipset"),
        standard=standard,
        num_tx_streams=require_int(metadata, "num_tx_streams"),
    )
