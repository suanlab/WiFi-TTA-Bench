# pyright: basic, reportMissingImports=false

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from pinn4csi.data import create_wifi_imaging_splits, load_wifi_imaging_prepared_dataset


def _write_prepared_imaging_capture(
    dataset_dir: Path,
    *,
    use_frequency_file: bool = False,
) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source": "mock_wifi_imaging",
        "protocol_version": "1.0",
        "capture_id": "mock-imaging-001",
        "task_name": "wifi_imaging",
        "environment_names": {"0": "env_0", "1": "env_1", "2": "env_2"},
        "coordinate_dim": 2,
        "num_subcarriers": 4,
        "num_tx_rx_pairs": 3,
        "grid_shape": [2, 3],
    }
    if not use_frequency_file:
        metadata["frequency_hz"] = 2.4e9

    csi = np.zeros((6, 3, 4), dtype=np.complex64)
    tx_rx_positions = np.zeros((6, 3, 2, 2), dtype=np.float32)
    query_coordinates = np.zeros((6, 6, 2), dtype=np.float32)
    field = np.zeros((6, 6), dtype=np.float32)
    permittivity = np.zeros((6, 6), dtype=np.float32)
    environments = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    frequencies = np.linspace(2.4e9, 2.45e9, 6, dtype=np.float32)

    base_query = np.asarray(
        [
            [-0.6, -0.5],
            [0.0, -0.5],
            [0.6, -0.5],
            [-0.6, 0.5],
            [0.0, 0.5],
            [0.6, 0.5],
        ],
        dtype=np.float32,
    )
    for sample_index in range(6):
        environment_id = int(environments[sample_index])
        sample_shift = 0.05 * sample_index
        for pair_index in range(3):
            tx_rx_positions[sample_index, pair_index, 0] = np.asarray(
                [-0.8 + 0.3 * pair_index + 0.1 * environment_id, -0.7 + sample_shift],
                dtype=np.float32,
            )
            tx_rx_positions[sample_index, pair_index, 1] = np.asarray(
                [-0.5 + 0.3 * pair_index + 0.1 * environment_id, 0.7 - sample_shift],
                dtype=np.float32,
            )

        query_coordinates[sample_index] = base_query + np.asarray(
            [0.1 * environment_id, -0.03 * sample_index],
            dtype=np.float32,
        )
        amplitude_code = 0.2 * environment_id + 0.1 * sample_index
        phase_code = -0.15 * environment_id + 0.05 * sample_index
        for pair_index in range(3):
            subcarrier_axis = np.arange(4, dtype=np.float32)
            real = amplitude_code + 0.05 * pair_index + 0.02 * subcarrier_axis
            imag = phase_code - 0.04 * pair_index + 0.03 * subcarrier_axis
            csi[sample_index, pair_index] = real + 1j * imag

        x_coord = query_coordinates[sample_index, :, 0]
        y_coord = query_coordinates[sample_index, :, 1]
        field[sample_index] = (
            0.4
            + 0.3 * environment_id
            + 0.15 * np.sin(np.pi * x_coord)
            + 0.07 * np.cos(np.pi * y_coord)
            + 0.02 * sample_index
        ).astype(np.float32)
        permittivity[sample_index] = (
            1.5
            + 0.2 * environment_id
            + 0.4 * (x_coord > 0.0).astype(np.float32)
            + 0.1 * (y_coord > 0.0).astype(np.float32)
        ).astype(np.float32)

    (dataset_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    np.save(dataset_dir / "csi.npy", csi)
    np.save(dataset_dir / "tx_rx_positions.npy", tx_rx_positions)
    np.save(dataset_dir / "query_coordinates.npy", query_coordinates)
    np.save(dataset_dir / "field.npy", field)
    np.save(dataset_dir / "permittivity.npy", permittivity)
    np.save(dataset_dir / "environments.npy", environments)
    if use_frequency_file:
        np.save(dataset_dir / "frequencies.npy", frequencies)
    return dataset_dir


def test_loader_normalizes_prepared_imaging_data(tmp_path: Path) -> None:
    dataset_dir = _write_prepared_imaging_capture(tmp_path / "prepared")

    bundle = load_wifi_imaging_prepared_dataset(dataset_dir)

    assert bundle.csi_features.shape == (6, 3, 4)
    assert bundle.query_coordinates.shape == (6, 6, 2)
    assert bundle.tx_rx_positions.shape == (6, 3, 2, 2)
    assert bundle.field_targets.shape == (6, 6)
    assert bundle.permittivity_targets.shape == (6, 6)
    assert bundle.environment_ids.tolist() == [0, 0, 1, 1, 2, 2]
    assert bundle.frequencies_hz.shape == (6,)
    assert bundle.csi_feature_dim == 12
    assert torch.allclose(
        bundle.csi_features.mean(dim=-1), torch.zeros(6, 3), atol=1e-5
    )
    assert torch.all(bundle.query_coordinates <= 1.0 + 1e-6)
    assert torch.all(bundle.query_coordinates >= -1.0 - 1e-6)
    assert torch.all(bundle.tx_rx_positions <= 1.0 + 1e-6)
    assert torch.all(bundle.tx_rx_positions >= -1.0 - 1e-6)
    assert torch.all(bundle.field_targets >= -1e-6)
    assert torch.all(bundle.field_targets <= 1.0 + 1e-6)
    assert torch.all(bundle.permittivity_targets >= 1.0)

    restored = bundle.denormalize_field(
        bundle.field_targets[:2],
        bundle.environment_ids[:2],
    )
    raw_field = torch.as_tensor(
        np.load(dataset_dir / "field.npy")[:2], dtype=torch.float32
    )
    assert torch.allclose(restored, raw_field, atol=1e-5)


def test_loader_supports_amplitude_phase_and_frequency_file(tmp_path: Path) -> None:
    dataset_dir = _write_prepared_imaging_capture(
        tmp_path / "prepared_freq_file",
        use_frequency_file=True,
    )

    bundle = load_wifi_imaging_prepared_dataset(
        dataset_dir,
        representation="amplitude_phase",
    )

    assert bundle.csi_features.shape == (6, 3, 8)
    assert bundle.representation == "amplitude_phase"
    expected_frequencies = torch.as_tensor(
        np.linspace(2.4e9, 2.45e9, 6, dtype=np.float32)
    )
    assert torch.allclose(bundle.frequencies_hz, expected_frequencies)


def test_environment_splits_are_deterministic_and_disjoint(tmp_path: Path) -> None:
    dataset_dir = _write_prepared_imaging_capture(tmp_path / "prepared_split")
    bundle = load_wifi_imaging_prepared_dataset(dataset_dir)

    first = create_wifi_imaging_splits(bundle, seed=7)
    second = create_wifi_imaging_splits(bundle, seed=7)

    assert first.train_environment_ids == second.train_environment_ids
    assert first.val_environment_ids == second.val_environment_ids
    assert first.test_environment_ids == second.test_environment_ids
    assert set(first.train_environment_ids).isdisjoint(first.val_environment_ids)
    assert set(first.train_environment_ids).isdisjoint(first.test_environment_ids)
    assert set(first.val_environment_ids).isdisjoint(first.test_environment_ids)

    train_envs = set(first.train.bundle.environment_ids[first.train.indices].tolist())
    val_envs = set(first.val.bundle.environment_ids[first.val.indices].tolist())
    test_envs = set(first.test.bundle.environment_ids[first.test.indices].tolist())
    assert train_envs == set(first.train_environment_ids)
    assert val_envs == set(first.val_environment_ids)
    assert test_envs == set(first.test_environment_ids)

    sample = first.test[0]
    assert sample["csi_features"].shape == (12,)
    assert sample["pair_features"].shape == (3, 4)


def test_loader_rejects_missing_environment_mapping(tmp_path: Path) -> None:
    dataset_dir = _write_prepared_imaging_capture(tmp_path / "prepared_bad_mapping")
    metadata_path = dataset_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["environment_names"] = {"0": "env_0", "1": "env_1"}
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="environment_names missing IDs"):
        load_wifi_imaging_prepared_dataset(dataset_dir)
