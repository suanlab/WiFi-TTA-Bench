# pyright: basic, reportMissingImports=false

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pinn4csi.data import load_esp32_prepared_dataset, load_wifi6_prepared_dataset


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


class TestESP32PreparedLoader:
    def test_loads_amplitude_features_and_metadata(self, tmp_path: Path) -> None:
        csi = _make_complex_csi()
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
        environments = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        timestamps = np.linspace(0.0, 0.5, len(labels), dtype=np.float64)
        session_dir = _write_prepared_capture(
            tmp_path / "esp32_session",
            metadata=_esp32_metadata(),
            csi=csi,
            labels=labels,
            environments=environments,
            timestamps=timestamps,
        )

        bundle = load_esp32_prepared_dataset(session_dir)

        assert bundle.features.shape == (6, 32, 2)
        assert bundle.features.dtype == torch.float32
        assert bundle.representation == "amplitude"
        assert bundle.labels.tolist() == labels.tolist()
        assert bundle.environments.tolist() == environments.tolist()
        assert bundle.timestamps is not None
        assert bundle.metadata.board == "ESP32-S3"
        assert bundle.metadata.environment_names[2] == "classroom"

    def test_rejects_non_complex_csi(self, tmp_path: Path) -> None:
        csi = np.random.randn(6, 32, 2).astype(np.float32)
        labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
        environments = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        session_dir = _write_prepared_capture(
            tmp_path / "esp32_bad",
            metadata=_esp32_metadata(),
            csi=csi,
            labels=labels,
            environments=environments,
        )

        with pytest.raises(ValueError, match="complex-valued"):
            load_esp32_prepared_dataset(session_dir)


class TestWiFi6PreparedLoader:
    def test_loads_amplitude_phase_features(self, tmp_path: Path) -> None:
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

        bundle = load_wifi6_prepared_dataset(session_dir)

        assert bundle.features.shape == (6, 32, 4)
        assert bundle.features.dtype == torch.float32
        assert bundle.representation == "amplitude_phase"
        assert bundle.metadata.standard == "802.11ax"
        assert bundle.metadata.num_tx_streams == 2
        assert bundle.input_shape == (32, 4)

    def test_rejects_missing_environment_mapping(self, tmp_path: Path) -> None:
        csi = _make_complex_csi()
        labels = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
        environments = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
        metadata = _wifi6_metadata()
        metadata["environment_names"] = {"0": "lab", "1": "corridor"}
        session_dir = _write_prepared_capture(
            tmp_path / "wifi6_bad",
            metadata=metadata,
            csi=csi,
            labels=labels,
            environments=environments,
        )

        with pytest.raises(ValueError, match="environment_names missing IDs"):
            load_wifi6_prepared_dataset(session_dir)
