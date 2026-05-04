# pyright: basic, reportMissingImports=false

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def _make_complex_csi(
    num_samples: int = 6,
    num_subcarriers: int = 32,
    num_antennas: int = 2,
) -> np.ndarray:
    real = np.random.randn(num_samples, num_subcarriers, num_antennas)
    imag = np.random.randn(num_samples, num_subcarriers, num_antennas)
    return (real + 1j * imag).astype(np.complex64)


def _write_prepared_capture(
    session_dir: Path,
    metadata: dict[str, object],
    csi: np.ndarray,
    labels: np.ndarray,
    environments: np.ndarray,
    timestamps: np.ndarray | None = None,
) -> Path:
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    np.save(session_dir / "csi.npy", csi)
    np.save(session_dir / "labels.npy", labels)
    np.save(session_dir / "environments.npy", environments)
    if timestamps is not None:
        np.save(session_dir / "timestamps.npy", timestamps)
    return session_dir


def _esp32_metadata() -> dict[str, object]:
    return {
        "source": "esp32",
        "protocol_version": "1.0",
        "capture_id": "esp32-lab-001",
        "task_name": "presence",
        "environment_names": {"0": "lab", "1": "corridor", "2": "classroom"},
        "label_names": {"0": "empty", "1": "occupied"},
        "num_subcarriers": 32,
        "num_rx_antennas": 2,
        "center_frequency_hz": 5.18e9,
        "bandwidth_mhz": 20.0,
        "board": "ESP32-S3",
        "firmware_version": "esp-csi-tool-1.2",
        "channel": 36,
        "phase_quality": "noisy",
    }


def _wifi6_metadata() -> dict[str, object]:
    return {
        "source": "wifi6",
        "protocol_version": "1.0",
        "capture_id": "ax-office-001",
        "task_name": "gesture",
        "environment_names": {"0": "lab", "1": "corridor", "2": "office"},
        "label_names": {"0": "idle", "1": "swipe", "2": "push"},
        "num_subcarriers": 32,
        "num_rx_antennas": 2,
        "center_frequency_hz": 5.32e9,
        "bandwidth_mhz": 80.0,
        "receiver": "intel-ax210",
        "chipset": "AX210",
        "standard": "802.11ax",
        "num_tx_streams": 2,
    }


class TestValidateCaptureBundle:
    """Test the validate_capture_bundle CLI."""

    def test_cli_validates_esp32_bundle(self, tmp_path: Path) -> None:
        """Test that CLI successfully validates a valid ESP32 bundle."""
        csi = _make_complex_csi()
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
        environments = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        session_dir = _write_prepared_capture(
            tmp_path / "esp32_session",
            metadata=_esp32_metadata(),
            csi=csi,
            labels=labels,
            environments=environments,
        )

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_capture_bundle.py",
                "--source",
                "esp32",
                "--bundle-dir",
                str(session_dir),
            ],
            cwd="/projects/PINN4CSI",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "✓ Bundle validation passed" in result.stdout
        assert "esp32-lab-001" in result.stdout
        assert "presence" in result.stdout
        assert "6" in result.stdout

    def test_cli_validates_wifi6_bundle(self, tmp_path: Path) -> None:
        """Test that CLI successfully validates a valid WiFi 6 bundle."""
        csi = _make_complex_csi()
        labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        environments = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        session_dir = _write_prepared_capture(
            tmp_path / "wifi6_session",
            metadata=_wifi6_metadata(),
            csi=csi,
            labels=labels,
            environments=environments,
        )

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_capture_bundle.py",
                "--source",
                "wifi6",
                "--bundle-dir",
                str(session_dir),
            ],
            cwd="/projects/PINN4CSI",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "✓ Bundle validation passed" in result.stdout
        assert "ax-office-001" in result.stdout
        assert "gesture" in result.stdout

    def test_cli_quiet_mode_on_success(self, tmp_path: Path) -> None:
        """Test that --quiet suppresses output on success."""
        csi = _make_complex_csi()
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
        environments = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        session_dir = _write_prepared_capture(
            tmp_path / "esp32_quiet",
            metadata=_esp32_metadata(),
            csi=csi,
            labels=labels,
            environments=environments,
        )

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_capture_bundle.py",
                "--source",
                "esp32",
                "--bundle-dir",
                str(session_dir),
                "--quiet",
            ],
            cwd="/projects/PINN4CSI",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert result.stdout == ""

    def test_cli_rejects_missing_directory(self, tmp_path: Path) -> None:
        """Test that CLI fails gracefully for missing directory."""
        missing_dir = tmp_path / "nonexistent"

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_capture_bundle.py",
                "--source",
                "esp32",
                "--bundle-dir",
                str(missing_dir),
            ],
            cwd="/projects/PINN4CSI",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "not found" in result.stderr

    def test_cli_rejects_missing_metadata(self, tmp_path: Path) -> None:
        """Test that CLI fails when metadata.json is missing."""
        csi = _make_complex_csi()
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
        environments = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        session_dir = tmp_path / "esp32_no_metadata"
        session_dir.mkdir(parents=True, exist_ok=True)
        np.save(session_dir / "csi.npy", csi)
        np.save(session_dir / "labels.npy", labels)
        np.save(session_dir / "environments.npy", environments)

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_capture_bundle.py",
                "--source",
                "esp32",
                "--bundle-dir",
                str(session_dir),
            ],
            cwd="/projects/PINN4CSI",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "Validation failed" in result.stderr

    def test_cli_rejects_non_complex_csi(self, tmp_path: Path) -> None:
        """Test that CLI fails when CSI is not complex-valued."""
        csi = np.random.randn(6, 32, 2).astype(np.float32)
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
        environments = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        session_dir = _write_prepared_capture(
            tmp_path / "esp32_real_csi",
            metadata=_esp32_metadata(),
            csi=csi,
            labels=labels,
            environments=environments,
        )

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_capture_bundle.py",
                "--source",
                "esp32",
                "--bundle-dir",
                str(session_dir),
            ],
            cwd="/projects/PINN4CSI",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "complex" in result.stderr

    def test_cli_rejects_mismatched_sample_count(self, tmp_path: Path) -> None:
        """Test that CLI fails when sample counts don't match."""
        csi = _make_complex_csi(num_samples=6)
        labels = np.asarray([0, 1, 0], dtype=np.int64)  # Only 3 labels
        environments = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        session_dir = _write_prepared_capture(
            tmp_path / "esp32_mismatch",
            metadata=_esp32_metadata(),
            csi=csi,
            labels=labels,
            environments=environments,
        )

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_capture_bundle.py",
                "--source",
                "esp32",
                "--bundle-dir",
                str(session_dir),
            ],
            cwd="/projects/PINN4CSI",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "mismatch" in result.stderr

    def test_cli_rejects_missing_label_mapping(self, tmp_path: Path) -> None:
        """Test that CLI fails when label mapping is incomplete."""
        csi = _make_complex_csi()
        labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)  # Has label 2
        environments = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        metadata = _esp32_metadata()
        metadata["label_names"] = {"0": "empty", "1": "occupied"}  # Missing 2
        session_dir = _write_prepared_capture(
            tmp_path / "esp32_bad_labels",
            metadata=metadata,
            csi=csi,
            labels=labels,
            environments=environments,
        )

        result = subprocess.run(
            [
                sys.executable,
                "scripts/validate_capture_bundle.py",
                "--source",
                "esp32",
                "--bundle-dir",
                str(session_dir),
            ],
            cwd="/projects/PINN4CSI",
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "label_names missing" in result.stderr
