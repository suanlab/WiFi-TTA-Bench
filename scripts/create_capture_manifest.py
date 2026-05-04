#!/usr/bin/env python3
# pyright: basic, reportMissingImports=false

"""CLI tool to create valid metadata.json manifests for ESP32 and WiFi 6 captures.

This tool generates a metadata.json file that conforms to the prepared-data contract
expected by pinn4csi.data.esp32_loader and pinn4csi.data.wifi6_loader.

Usage:
    python scripts/create_capture_manifest.py \\
        --source esp32 \\
        --output-dir /path/to/capture \\
        --capture-id my-session-001 \\
        --task-name presence \\
        --environments "0:lab,1:corridor,2:office" \\
        --labels "0:empty,1:occupied"

    python scripts/create_capture_manifest.py \\
        --source wifi6 \\
        --output-dir /path/to/capture \\
        --capture-id ax-office-001 \\
        --task-name gesture \\
        --environments "0:lab,1:corridor,2:office" \\
        --labels "0:idle,1:swipe,2:push"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pinn4csi.data.manifest_creator import (
    create_esp32_manifest,
    create_wifi6_manifest,
    parse_id_name_mapping,
)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Create valid metadata.json for ESP32 or WiFi 6 CSI captures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # ESP32 capture
  python scripts/create_capture_manifest.py \\
    --source esp32 \\
    --output-dir ./data/esp32/my_session \\
    --capture-id esp32-lab-001 \\
    --task-name presence \\
    --environments "0:lab,1:corridor,2:office" \\
    --labels "0:empty,1:occupied"

  # WiFi 6 capture
  python scripts/create_capture_manifest.py \\
    --source wifi6 \\
    --output-dir ./data/wifi6/my_session \\
    --capture-id ax-office-001 \\
    --task-name gesture \\
    --environments "0:lab,1:corridor,2:office" \\
    --labels "0:idle,1:swipe,2:push"
        """,
    )

    parser.add_argument(
        "--source",
        required=True,
        choices=["esp32", "wifi6"],
        help="Capture source type.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory where metadata.json will be written.",
    )
    parser.add_argument(
        "--capture-id",
        required=True,
        help="Unique capture session identifier.",
    )
    parser.add_argument(
        "--task-name",
        required=True,
        help="Task name (e.g., 'presence', 'gesture', 'activity').",
    )
    parser.add_argument(
        "--environments",
        required=True,
        help="Environment ID-to-name mapping: '0:lab,1:corridor,2:office'.",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="Label ID-to-name mapping: '0:empty,1:occupied'.",
    )

    # ESP32-specific options
    parser.add_argument(
        "--board",
        default="ESP32-S3",
        help="ESP32 board model (default: ESP32-S3).",
    )
    parser.add_argument(
        "--firmware-version",
        default="esp-csi-tool-1.2",
        help="Firmware version (default: esp-csi-tool-1.2).",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=36,
        help="WiFi channel number (default: 36).",
    )
    parser.add_argument(
        "--phase-quality",
        default="noisy",
        help="Phase quality assessment (default: noisy).",
    )

    # WiFi 6-specific options
    parser.add_argument(
        "--receiver",
        default="intel-ax210",
        help="Receiver model (default: intel-ax210).",
    )
    parser.add_argument(
        "--chipset",
        default="AX210",
        help="Chipset name (default: AX210).",
    )
    parser.add_argument(
        "--num-tx-streams",
        type=int,
        default=2,
        help="Number of TX streams (default: 2).",
    )

    # Common options
    parser.add_argument(
        "--num-subcarriers",
        type=int,
        default=32,
        help="Number of OFDM subcarriers (default: 32).",
    )
    parser.add_argument(
        "--num-rx-antennas",
        type=int,
        default=2,
        help="Number of RX antennas (default: 2).",
    )
    parser.add_argument(
        "--center-frequency-hz",
        type=float,
        help="Center frequency in Hz (default: 5.18e9 for ESP32, 5.32e9 for WiFi6).",
    )
    parser.add_argument(
        "--bandwidth-mhz",
        type=float,
        help="Bandwidth in MHz (default: 20 for ESP32, 80 for WiFi6).",
    )

    args = parser.parse_args()

    # Parse environment and label mappings
    try:
        environment_names = parse_id_name_mapping(args.environments)
        label_names = parse_id_name_mapping(args.labels)
    except ValueError as e:
        print(f"Error parsing mappings: {e}", file=sys.stderr)
        return 1

    # Set defaults for frequency and bandwidth based on source
    center_frequency_hz = args.center_frequency_hz
    bandwidth_mhz = args.bandwidth_mhz

    if args.source == "esp32":
        if center_frequency_hz is None:
            center_frequency_hz = 5.18e9
        if bandwidth_mhz is None:
            bandwidth_mhz = 20.0

        manifest = create_esp32_manifest(
            capture_id=args.capture_id,
            task_name=args.task_name,
            environment_names=environment_names,
            label_names=label_names,
            num_subcarriers=args.num_subcarriers,
            num_rx_antennas=args.num_rx_antennas,
            center_frequency_hz=center_frequency_hz,
            bandwidth_mhz=bandwidth_mhz,
            board=args.board,
            firmware_version=args.firmware_version,
            channel=args.channel,
            phase_quality=args.phase_quality,
        )
    else:  # wifi6
        if center_frequency_hz is None:
            center_frequency_hz = 5.32e9
        if bandwidth_mhz is None:
            bandwidth_mhz = 80.0

        manifest = create_wifi6_manifest(
            capture_id=args.capture_id,
            task_name=args.task_name,
            environment_names=environment_names,
            label_names=label_names,
            num_subcarriers=args.num_subcarriers,
            num_rx_antennas=args.num_rx_antennas,
            center_frequency_hz=center_frequency_hz,
            bandwidth_mhz=bandwidth_mhz,
            receiver=args.receiver,
            chipset=args.chipset,
            num_tx_streams=args.num_tx_streams,
        )

    # Create output directory and write manifest
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.json"
    try:
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"✓ Manifest written to: {metadata_path}")
        return 0
    except OSError as e:
        print(f"Error writing manifest: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
