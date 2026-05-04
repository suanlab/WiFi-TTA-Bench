#!/usr/bin/env python3
# pyright: basic, reportMissingImports=false

"""CLI tool to validate individual ESP32 or WiFi 6 capture bundles.

This tool checks that a capture directory contains all required files
(metadata.json, csi.npy, labels.npy, environments.npy) and that they
conform to the prepared-data contract expected by the loaders.

Usage:
    python scripts/validate_capture_bundle.py \\
        --source esp32 \\
        --bundle-dir /path/to/capture

    python scripts/validate_capture_bundle.py \\
        --source wifi6 \\
        --bundle-dir /path/to/capture \\
        --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pinn4csi.data import load_esp32_prepared_dataset, load_wifi6_prepared_dataset


def validate_bundle(
    source: str,
    bundle_dir: str | Path,
    quiet: bool = False,
) -> int:
    """Validate a capture bundle and print results.

    Args:
        source: "esp32" or "wifi6"
        bundle_dir: Path to the bundle directory
        quiet: If True, only print on failure

    Returns:
        0 on success, 1 on failure
    """
    bundle_path = Path(bundle_dir)

    if not bundle_path.exists():
        print(f"✗ Bundle directory not found: {bundle_path}", file=sys.stderr)
        return 1

    if not bundle_path.is_dir():
        print(f"✗ Path is not a directory: {bundle_path}", file=sys.stderr)
        return 1

    try:
        if source == "esp32":
            bundle_obj = load_esp32_prepared_dataset(bundle_path)
        elif source == "wifi6":
            bundle_obj = load_wifi6_prepared_dataset(bundle_path)  # type: ignore[assignment]
        else:
            print(f"✗ Unknown source: {source}", file=sys.stderr)
            return 1
    except (FileNotFoundError, ValueError, TypeError) as e:
        print(f"✗ Validation failed: {e}", file=sys.stderr)
        return 1

    if not quiet:
        print(f"✓ Bundle validation passed: {bundle_path}")
        print(f"  Source: {bundle_obj.metadata.source}")
        print(f"  Capture ID: {bundle_obj.metadata.capture_id}")
        print(f"  Task: {bundle_obj.metadata.task_name}")
        print(f"  Num samples: {bundle_obj.num_samples}")
        print(f"  CSI shape: {tuple(bundle_obj.features.shape)}")
        print(f"  Representation: {bundle_obj.representation}")
        print(f"  Num subcarriers: {bundle_obj.metadata.num_subcarriers}")
        print(f"  Num RX antennas: {bundle_obj.metadata.num_rx_antennas}")
        print(f"  Environments: {bundle_obj.metadata.environment_names}")
        print(f"  Labels: {bundle_obj.metadata.label_names}")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate an individual ESP32 or WiFi 6 capture bundle.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate ESP32 capture
  python scripts/validate_capture_bundle.py \\
    --source esp32 \\
    --bundle-dir ./data/esp32/my_session

  # Validate WiFi 6 capture (quiet mode)
  python scripts/validate_capture_bundle.py \\
    --source wifi6 \\
    --bundle-dir ./data/wifi6/my_session \\
    --quiet
        """,
    )

    parser.add_argument(
        "--source",
        required=True,
        choices=["esp32", "wifi6"],
        help="Capture source type.",
    )
    parser.add_argument(
        "--bundle-dir",
        required=True,
        type=Path,
        help="Path to the capture bundle directory.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only print output on failure.",
    )

    args = parser.parse_args()

    return validate_bundle(
        source=args.source,
        bundle_dir=args.bundle_dir,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
