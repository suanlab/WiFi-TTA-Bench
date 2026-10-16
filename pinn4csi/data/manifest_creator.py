# pyright: basic, reportMissingImports=false

"""Utilities for creating valid metadata.json manifests for prepared CSI captures."""

from __future__ import annotations

from typing import Any


def parse_id_name_mapping(mapping_str: str) -> dict[str, str]:
    """Parse 'id:name,id:name,...' format into dict[str, str].

    Args:
        mapping_str: Comma-separated 'id:name' pairs.

    Returns:
        Dictionary with string IDs as keys and names as values.

    Raises:
        ValueError: If format is invalid or IDs are not numeric strings.
    """
    if not mapping_str.strip():
        raise ValueError("Mapping string cannot be empty")

    result: dict[str, str] = {}
    for pair in mapping_str.split(","):
        pair = pair.strip()
        if ":" not in pair:
            raise ValueError(f"Invalid mapping format '{pair}': expected 'id:name'")
        id_str, name = pair.split(":", 1)
        id_str = id_str.strip()
        name = name.strip()

        if not id_str.isdigit():
            raise ValueError(f"Invalid ID '{id_str}': must be a non-negative integer")
        if not name:
            raise ValueError(f"Invalid name for ID {id_str}: cannot be empty")

        result[id_str] = name

    return result


def create_esp32_manifest(
    capture_id: str,
    task_name: str,
    environment_names: dict[str, str],
    label_names: dict[str, str],
    num_subcarriers: int = 32,
    num_rx_antennas: int = 2,
    center_frequency_hz: float = 5.18e9,
    bandwidth_mhz: float = 20.0,
    board: str = "ESP32-S3",
    firmware_version: str = "esp-csi-tool-1.2",
    channel: int = 36,
    phase_quality: str = "noisy",
) -> dict[str, Any]:
    """Create ESP32 metadata manifest.

    Args:
        capture_id: Unique session identifier.
        task_name: Task name (e.g., 'presence', 'gesture').
        environment_names: Mapping of environment IDs to names.
        label_names: Mapping of label IDs to names.
        num_subcarriers: Number of OFDM subcarriers (default: 32).
        num_rx_antennas: Number of RX antennas (default: 2).
        center_frequency_hz: Center frequency in Hz (default: 5.18 GHz).
        bandwidth_mhz: Bandwidth in MHz (default: 20).
        board: ESP32 board model (default: 'ESP32-S3').
        firmware_version: Firmware version string (default: 'esp-csi-tool-1.2').
        channel: WiFi channel number (default: 36).
        phase_quality: Phase quality assessment (default: 'noisy').

    Returns:
        Dictionary ready for JSON serialization.
    """
    return {
        "source": "esp32",
        "protocol_version": "1.0",
        "capture_id": capture_id,
        "task_name": task_name,
        "environment_names": environment_names,
        "label_names": label_names,
        "num_subcarriers": num_subcarriers,
        "num_rx_antennas": num_rx_antennas,
        "center_frequency_hz": center_frequency_hz,
        "bandwidth_mhz": bandwidth_mhz,
        "board": board,
        "firmware_version": firmware_version,
        "channel": channel,
        "phase_quality": phase_quality,
    }


def create_wifi6_manifest(
    capture_id: str,
    task_name: str,
    environment_names: dict[str, str],
    label_names: dict[str, str],
    num_subcarriers: int = 32,
    num_rx_antennas: int = 2,
    center_frequency_hz: float = 5.32e9,
    bandwidth_mhz: float = 80.0,
    receiver: str = "intel-ax210",
    chipset: str = "AX210",
    num_tx_streams: int = 2,
) -> dict[str, Any]:
    """Create WiFi 6 metadata manifest.

    Args:
        capture_id: Unique session identifier.
        task_name: Task name (e.g., 'presence', 'gesture').
        environment_names: Mapping of environment IDs to names.
        label_names: Mapping of label IDs to names.
        num_subcarriers: Number of OFDM subcarriers (default: 32).
        num_rx_antennas: Number of RX antennas (default: 2).
        center_frequency_hz: Center frequency in Hz (default: 5.32 GHz).
        bandwidth_mhz: Bandwidth in MHz (default: 80).
        receiver: Receiver model (default: 'intel-ax210').
        chipset: Chipset name (default: 'AX210').
        num_tx_streams: Number of TX streams (default: 2).

    Returns:
        Dictionary ready for JSON serialization.
    """
    return {
        "source": "wifi6",
        "protocol_version": "1.0",
        "capture_id": capture_id,
        "task_name": task_name,
        "environment_names": environment_names,
        "label_names": label_names,
        "num_subcarriers": num_subcarriers,
        "num_rx_antennas": num_rx_antennas,
        "center_frequency_hz": center_frequency_hz,
        "bandwidth_mhz": bandwidth_mhz,
        "receiver": receiver,
        "chipset": chipset,
        "standard": "802.11ax",
        "num_tx_streams": num_tx_streams,
    }
