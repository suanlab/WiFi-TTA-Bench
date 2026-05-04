# pyright: basic, reportMissingImports=false

"""Tests for create_capture_manifest CLI and manifest generation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pinn4csi.data import load_esp32_prepared_dataset, load_wifi6_prepared_dataset


def _make_complex_csi(
    num_samples: int = 6,
    num_subcarriers: int = 32,
    num_antennas: int = 2,
) -> np.ndarray:
    """Create synthetic complex CSI array."""
    real = np.random.randn(num_samples, num_subcarriers, num_antennas)
    imag = np.random.randn(num_samples, num_subcarriers, num_antennas)
    return (real + 1j * imag).astype(np.complex64)


class TestParseIdNameMapping:
    """Test the parse_id_name_mapping function."""

    def test_parse_valid_mapping(self) -> None:
        """Test parsing valid environment/label mappings."""
        from pinn4csi.data.manifest_creator import parse_id_name_mapping

        result = parse_id_name_mapping("0:lab,1:corridor,2:office")
        assert result == {"0": "lab", "1": "corridor", "2": "office"}

    def test_parse_single_entry(self) -> None:
        """Test parsing single entry."""
        from pinn4csi.data.manifest_creator import parse_id_name_mapping

        result = parse_id_name_mapping("0:empty")
        assert result == {"0": "empty"}

    def test_parse_with_whitespace(self) -> None:
        """Test parsing with extra whitespace."""
        from pinn4csi.data.manifest_creator import parse_id_name_mapping

        result = parse_id_name_mapping("  0 : lab , 1 : corridor  ")
        assert result == {"0": "lab", "1": "corridor"}

    def test_parse_invalid_format_missing_colon(self) -> None:
        """Test error on missing colon."""
        from pinn4csi.data.manifest_creator import parse_id_name_mapping

        with pytest.raises(ValueError, match="expected 'id:name'"):
            parse_id_name_mapping("0-lab")

    def test_parse_invalid_id_not_numeric(self) -> None:
        """Test error on non-numeric ID."""
        from pinn4csi.data.manifest_creator import parse_id_name_mapping

        with pytest.raises(ValueError, match="must be a non-negative integer"):
            parse_id_name_mapping("a:lab")

    def test_parse_empty_name(self) -> None:
        """Test error on empty name."""
        from pinn4csi.data.manifest_creator import parse_id_name_mapping

        with pytest.raises(ValueError, match="cannot be empty"):
            parse_id_name_mapping("0:")

    def test_parse_empty_string(self) -> None:
        """Test error on empty mapping string."""
        from pinn4csi.data.manifest_creator import parse_id_name_mapping

        with pytest.raises(ValueError, match="cannot be empty"):
            parse_id_name_mapping("")


class TestCreateESP32Manifest:
    """Test ESP32 manifest generation."""

    def test_create_esp32_manifest_defaults(self) -> None:
        """Test ESP32 manifest with default values."""
        from pinn4csi.data.manifest_creator import create_esp32_manifest

        manifest = create_esp32_manifest(
            capture_id="esp32-lab-001",
            task_name="presence",
            environment_names={"0": "lab", "1": "corridor"},
            label_names={"0": "empty", "1": "occupied"},
        )

        assert manifest["source"] == "esp32"
        assert manifest["protocol_version"] == "1.0"
        assert manifest["capture_id"] == "esp32-lab-001"
        assert manifest["task_name"] == "presence"
        assert manifest["environment_names"] == {"0": "lab", "1": "corridor"}
        assert manifest["label_names"] == {"0": "empty", "1": "occupied"}
        assert manifest["num_subcarriers"] == 32
        assert manifest["num_rx_antennas"] == 2
        assert manifest["center_frequency_hz"] == 5.18e9
        assert manifest["bandwidth_mhz"] == 20.0
        assert manifest["board"] == "ESP32-S3"
        assert manifest["firmware_version"] == "esp-csi-tool-1.2"
        assert manifest["channel"] == 36
        assert manifest["phase_quality"] == "noisy"

    def test_create_esp32_manifest_custom_values(self) -> None:
        """Test ESP32 manifest with custom values."""
        from pinn4csi.data.manifest_creator import create_esp32_manifest

        manifest = create_esp32_manifest(
            capture_id="esp32-custom",
            task_name="gesture",
            environment_names={"0": "lab"},
            label_names={"0": "idle", "1": "wave"},
            num_subcarriers=64,
            num_rx_antennas=4,
            center_frequency_hz=2.4e9,
            bandwidth_mhz=40.0,
            board="ESP32-C6",
            firmware_version="custom-1.0",
            channel=11,
            phase_quality="good",
        )

        assert manifest["num_subcarriers"] == 64
        assert manifest["num_rx_antennas"] == 4
        assert manifest["center_frequency_hz"] == 2.4e9
        assert manifest["bandwidth_mhz"] == 40.0
        assert manifest["board"] == "ESP32-C6"
        assert manifest["firmware_version"] == "custom-1.0"
        assert manifest["channel"] == 11
        assert manifest["phase_quality"] == "good"


class TestCreateWiFi6Manifest:
    """Test WiFi 6 manifest generation."""

    def test_create_wifi6_manifest_defaults(self) -> None:
        """Test WiFi 6 manifest with default values."""
        from pinn4csi.data.manifest_creator import create_wifi6_manifest

        manifest = create_wifi6_manifest(
            capture_id="ax-office-001",
            task_name="gesture",
            environment_names={"0": "lab", "1": "office"},
            label_names={"0": "idle", "1": "swipe"},
        )

        assert manifest["source"] == "wifi6"
        assert manifest["protocol_version"] == "1.0"
        assert manifest["capture_id"] == "ax-office-001"
        assert manifest["task_name"] == "gesture"
        assert manifest["environment_names"] == {"0": "lab", "1": "office"}
        assert manifest["label_names"] == {"0": "idle", "1": "swipe"}
        assert manifest["num_subcarriers"] == 32
        assert manifest["num_rx_antennas"] == 2
        assert manifest["center_frequency_hz"] == 5.32e9
        assert manifest["bandwidth_mhz"] == 80.0
        assert manifest["receiver"] == "intel-ax210"
        assert manifest["chipset"] == "AX210"
        assert manifest["standard"] == "802.11ax"
        assert manifest["num_tx_streams"] == 2

    def test_create_wifi6_manifest_custom_values(self) -> None:
        """Test WiFi 6 manifest with custom values."""
        from pinn4csi.data.manifest_creator import create_wifi6_manifest

        manifest = create_wifi6_manifest(
            capture_id="ax-custom",
            task_name="activity",
            environment_names={"0": "lab"},
            label_names={"0": "sitting", "1": "standing"},
            num_subcarriers=128,
            num_rx_antennas=8,
            center_frequency_hz=6.0e9,
            bandwidth_mhz=160.0,
            receiver="custom-receiver",
            chipset="CUSTOM",
            num_tx_streams=4,
        )

        assert manifest["num_subcarriers"] == 128
        assert manifest["num_rx_antennas"] == 8
        assert manifest["center_frequency_hz"] == 6.0e9
        assert manifest["bandwidth_mhz"] == 160.0
        assert manifest["receiver"] == "custom-receiver"
        assert manifest["chipset"] == "CUSTOM"
        assert manifest["num_tx_streams"] == 4


class TestCLIIntegration:
    """Integration tests for the CLI script."""

    def test_cli_esp32_creates_valid_manifest(self, tmp_path: Path) -> None:
        """Test CLI creates valid ESP32 manifest that loaders accept."""
        output_dir = tmp_path / "esp32_session"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/create_capture_manifest.py",
                "--source",
                "esp32",
                "--output-dir",
                str(output_dir),
                "--capture-id",
                "esp32-test-001",
                "--task-name",
                "presence",
                "--environments",
                "0:lab,1:corridor,2:office",
                "--labels",
                "0:empty,1:occupied",
            ],
            cwd="/projects/PINN4CSI",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert (output_dir / "metadata.json").exists()

        # Verify manifest is valid JSON
        with (output_dir / "metadata.json").open() as f:
            manifest = json.load(f)

        assert manifest["source"] == "esp32"
        assert manifest["capture_id"] == "esp32-test-001"
        assert manifest["task_name"] == "presence"

    def test_cli_wifi6_creates_valid_manifest(self, tmp_path: Path) -> None:
        """Test CLI creates valid WiFi 6 manifest that loaders accept."""
        output_dir = tmp_path / "wifi6_session"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/create_capture_manifest.py",
                "--source",
                "wifi6",
                "--output-dir",
                str(output_dir),
                "--capture-id",
                "ax-test-001",
                "--task-name",
                "gesture",
                "--environments",
                "0:lab,1:corridor,2:office",
                "--labels",
                "0:idle,1:swipe,2:push",
            ],
            cwd="/projects/PINN4CSI",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert (output_dir / "metadata.json").exists()

        # Verify manifest is valid JSON
        with (output_dir / "metadata.json").open() as f:
            manifest = json.load(f)

        assert manifest["source"] == "wifi6"
        assert manifest["capture_id"] == "ax-test-001"
        assert manifest["task_name"] == "gesture"

    def test_cli_esp32_manifest_accepted_by_loader(self, tmp_path: Path) -> None:
        """Test generated ESP32 manifest is accepted by the loader."""
        session_dir = tmp_path / "esp32_session"

        # Create manifest via CLI
        result = subprocess.run(
            [
                sys.executable,
                "scripts/create_capture_manifest.py",
                "--source",
                "esp32",
                "--output-dir",
                str(session_dir),
                "--capture-id",
                "esp32-loader-test",
                "--task-name",
                "presence",
                "--environments",
                "0:lab,1:corridor",
                "--labels",
                "0:empty,1:occupied",
                "--num-subcarriers",
                "32",
                "--num-rx-antennas",
                "2",
            ],
            cwd="/projects/PINN4CSI",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Create dummy aligned arrays
        csi = _make_complex_csi(num_samples=6, num_subcarriers=32, num_antennas=2)
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
        environments = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)

        np.save(session_dir / "csi.npy", csi)
        np.save(session_dir / "labels.npy", labels)
        np.save(session_dir / "environments.npy", environments)

        # Verify loader accepts the manifest
        bundle = load_esp32_prepared_dataset(session_dir)
        assert bundle.metadata.capture_id == "esp32-loader-test"
        assert bundle.metadata.task_name == "presence"
        assert bundle.num_samples == 6

    def test_cli_wifi6_manifest_accepted_by_loader(self, tmp_path: Path) -> None:
        """Test generated WiFi 6 manifest is accepted by the loader."""
        session_dir = tmp_path / "wifi6_session"

        # Create manifest via CLI
        result = subprocess.run(
            [
                sys.executable,
                "scripts/create_capture_manifest.py",
                "--source",
                "wifi6",
                "--output-dir",
                str(session_dir),
                "--capture-id",
                "ax-loader-test",
                "--task-name",
                "gesture",
                "--environments",
                "0:lab,1:office",
                "--labels",
                "0:idle,1:swipe,2:push",
                "--num-subcarriers",
                "32",
                "--num-rx-antennas",
                "2",
            ],
            cwd="/projects/PINN4CSI",
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

        # Create dummy aligned arrays
        csi = _make_complex_csi(num_samples=6, num_subcarriers=32, num_antennas=2)
        labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        environments = np.asarray([0, 0, 1, 1, 0, 1], dtype=np.int64)

        np.save(session_dir / "csi.npy", csi)
        np.save(session_dir / "labels.npy", labels)
        np.save(session_dir / "environments.npy", environments)

        # Verify loader accepts the manifest
        bundle = load_wifi6_prepared_dataset(session_dir)
        assert bundle.metadata.capture_id == "ax-loader-test"
        assert bundle.metadata.task_name == "gesture"
        assert bundle.num_samples == 6

    def test_cli_invalid_source(self, tmp_path: Path) -> None:
        """Test CLI rejects invalid source."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/create_capture_manifest.py",
                "--source",
                "invalid",
                "--output-dir",
                str(tmp_path),
                "--capture-id",
                "test",
                "--task-name",
                "test",
                "--environments",
                "0:lab",
                "--labels",
                "0:empty",
            ],
            cwd="/projects/PINN4CSI",
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_cli_invalid_mapping_format(self, tmp_path: Path) -> None:
        """Test CLI rejects invalid mapping format."""
        result = subprocess.run(
            [
                sys.executable,
                "scripts/create_capture_manifest.py",
                "--source",
                "esp32",
                "--output-dir",
                str(tmp_path),
                "--capture-id",
                "test",
                "--task-name",
                "test",
                "--environments",
                "invalid-format",
                "--labels",
                "0:empty",
            ],
            cwd="/projects/PINN4CSI",
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "Error parsing mappings" in result.stderr
