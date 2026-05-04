# pyright: basic, reportMissingImports=false

import importlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import torch

from pinn4csi.data import load_prepared_paper1_dataset, save_prepared_paper1_dataset


def _load_prepare_data_module() -> ModuleType:
    script_path = Path(__file__).parent.parent / "scripts" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_data_module", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_prepare_data_main() -> Any:
    module = _load_prepare_data_module()
    return module.main


def _make_source_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base = np.linspace(-1.0, 1.0, 6, dtype=np.float32)
    features = []
    priors = []
    labels = []
    environments = []
    for environment_id in (0, 1):
        for class_idx in range(3):
            template = np.sin((class_idx + 1) * base) + 1j * np.cos(base)
            prior = (1.0 + 0.1 * environment_id) * template
            observed = prior * np.exp(1j * 0.05 * environment_id)
            features.append(observed.astype(np.complex64))
            priors.append(prior.astype(np.complex64))
            labels.append(class_idx)
            environments.append(environment_id)
    return (
        np.stack(features).astype(np.complex64),
        np.asarray(labels, dtype=np.int64),
        np.stack(priors).astype(np.complex64),
        np.asarray(environments, dtype=np.int64),
    )


def test_save_prepared_paper1_dataset_writes_valid_bundle(tmp_path: Path) -> None:
    features, labels, priors, environments = _make_source_arrays()

    dataset_dir = save_prepared_paper1_dataset(
        dataset_name="ut_har",
        output_root=tmp_path,
        features=features,
        labels=labels,
        environments=environments,
        priors=priors,
        metadata={"source": "local_arrays", "task": "paper1"},
    )

    bundle = load_prepared_paper1_dataset("ut_har", tmp_path)
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))

    assert dataset_dir == tmp_path / "ut_har"
    assert bundle.features.shape == (6, 6, 2)
    assert bundle.priors is not None
    assert bundle.priors.shape == (6, 6, 2)
    assert bundle.environments is not None
    assert bundle.environments.tolist() == environments.tolist()
    assert metadata["dataset_name"] == "UT_HAR"
    assert metadata["prepared_feature_shape"] == [6, 6, 2]
    assert metadata["user_metadata"]["source"] == "local_arrays"


def test_save_prepared_paper1_dataset_rejects_missing_cross_env_metadata(
    tmp_path: Path,
) -> None:
    features, labels, _, _ = _make_source_arrays()

    with pytest.raises(ValueError, match="requires environment metadata"):
        save_prepared_paper1_dataset(
            dataset_name="ut_har",
            output_root=tmp_path,
            features=features,
            labels=labels,
        )


def test_prepare_data_script_loads_custom_keys_and_writes_manifest(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    features, labels, _, environments = _make_source_arrays()
    features_path = tmp_path / "features.npz"
    labels_path = tmp_path / "labels.pt"
    environments_path = tmp_path / "environments.npz"
    metadata_path = tmp_path / "metadata.json"
    output_dir = tmp_path / "prepared"

    np.savez(features_path, features_block=features)
    torch.save(torch.as_tensor(labels), labels_path)
    np.savez(environments_path, env_ids=environments)
    metadata_path.write_text(
        json.dumps({"collector": "manual_export", "split": "room_holdout"}),
        encoding="utf-8",
    )

    prepare_data_main = _load_prepare_data_main()
    prepare_data_main(
        [
            "--dataset",
            "ut_har",
            "--features",
            str(features_path),
            "--features-key",
            "features_block",
            "--labels",
            str(labels_path),
            "--environments",
            str(environments_path),
            "--environments-key",
            "env_ids",
            "--metadata",
            str(metadata_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    bundle = load_prepared_paper1_dataset("ut_har", output_dir)
    manifest = json.loads(
        (output_dir / "ut_har" / "metadata.json").read_text(encoding="utf-8")
    )
    stdout = capsys.readouterr().out

    assert bundle.num_samples == 6
    assert bundle.environments is not None
    assert bundle.environments.tolist() == environments.tolist()
    assert manifest["user_metadata"]["collector"] == "manual_export"
    assert manifest["user_metadata"]["source_files"]["features"] == str(features_path)
    assert "Prepared dataset written to:" in stdout


def test_prepare_data_script_ut_har_layout_adapter(tmp_path: Path) -> None:
    source_root = tmp_path / "UT_HAR"
    data_dir = source_root / "data"
    label_dir = source_root / "label"
    data_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    env0_features = np.ones((3, 6), dtype=np.float32)
    env1_features = np.full((3, 6), 2.0, dtype=np.float32)
    env0_labels = np.array([0, 1, 2], dtype=np.int64)
    env1_labels = np.array([0, 2, 1], dtype=np.int64)

    with (data_dir / "room0.csv").open("wb") as file_obj:
        np.save(file_obj, env0_features)
    with (data_dir / "room1.csv").open("wb") as file_obj:
        np.save(file_obj, env1_features)
    with (label_dir / "room0.csv").open("wb") as file_obj:
        np.save(file_obj, env0_labels)
    with (label_dir / "room1.csv").open("wb") as file_obj:
        np.save(file_obj, env1_labels)

    output_dir = tmp_path / "prepared"
    prepare_data_main = _load_prepare_data_main()
    prepare_data_main(
        [
            "--dataset",
            "ut_har",
            "--source-format",
            "ut_har_layout",
            "--source-root",
            str(source_root),
            "--output-dir",
            str(output_dir),
        ]
    )

    bundle = load_prepared_paper1_dataset(
        "ut_har",
        output_dir,
        use_amplitude_phase=False,
    )
    manifest = json.loads(
        (output_dir / "ut_har" / "metadata.json").read_text(encoding="utf-8")
    )

    assert bundle.num_samples == 6
    assert bundle.environments is not None
    assert bundle.environments.tolist() == [0, 0, 0, 1, 1, 1]
    assert manifest["user_metadata"]["source_files"]["source_format"] == "ut_har_layout"
    assert manifest["user_metadata"]["source_files"]["source_root"] == str(source_root)


def test_prepare_data_script_signfi_mat_missing_scipy(tmp_path: Path) -> None:
    module = _load_prepare_data_module()
    mat_path = tmp_path / "dataset_lab_276_dl.mat"
    mat_path.write_bytes(b"placeholder")

    def _missing_import(name: str) -> Any:
        if name == "scipy.io":
            raise ImportError("scipy not installed")
        return importlib.import_module(name)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module.importlib, "import_module", _missing_import)
    try:
        with pytest.raises(RuntimeError, match="requires scipy"):
            module.main(
                [
                    "--dataset",
                    "signfi",
                    "--source-format",
                    "signfi_mat",
                    "--mat-file",
                    str(mat_path),
                    "--output-dir",
                    str(tmp_path / "prepared"),
                ]
            )
    finally:
        monkeypatch.undo()


def test_prepare_data_script_signfi_mat_with_stubbed_loader(tmp_path: Path) -> None:
    module = _load_prepare_data_module()
    mat_path = tmp_path / "dataset_lab_276_dl.mat"
    mat_path.write_bytes(b"placeholder")

    class _ScipyIoStub:
        @staticmethod
        def loadmat(_path: Path) -> dict[str, np.ndarray]:
            return {
                "features": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                "labels": np.array([0, 1], dtype=np.int64),
            }

    def _import_stub(name: str) -> Any:
        if name == "scipy.io":
            return _ScipyIoStub
        return importlib.import_module(name)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(module.importlib, "import_module", _import_stub)
    output_dir = tmp_path / "prepared"

    try:
        module.main(
            [
                "--dataset",
                "signfi",
                "--source-format",
                "signfi_mat",
                "--mat-file",
                str(mat_path),
                "--output-dir",
                str(output_dir),
            ]
        )
    finally:
        monkeypatch.undo()

    bundle = load_prepared_paper1_dataset(
        "signfi",
        output_dir,
        use_amplitude_phase=False,
    )
    manifest = json.loads(
        (output_dir / "signfi" / "metadata.json").read_text(encoding="utf-8")
    )

    assert bundle.num_samples == 2
    assert bundle.labels.tolist() == [0, 1]
    assert manifest["user_metadata"]["source_files"]["source_format"] == "signfi_mat"
    assert manifest["user_metadata"]["source_files"]["mat_file"] == str(mat_path)
