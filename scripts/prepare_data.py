# pyright: basic, reportMissingImports=false

"""Prepare user-supplied Paper 1 arrays into the repo's prepared-data layout."""

from __future__ import annotations

import argparse
import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pinn4csi.data import load_array_file, save_prepared_paper1_dataset

SourceFormat = str

SIGNFI_FEATURE_CANDIDATE_KEYS = (
    "csi",
    "features",
    "x",
    "data",
    "dataset",
)
SIGNFI_LABEL_CANDIDATE_KEYS = (
    "label",
    "labels",
    "y",
    "target",
    "targets",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert local user-provided arrays/files into the prepared-data "
            "contract consumed by Paper 1 experiments."
        )
    )
    parser.add_argument("--dataset", required=True, choices=("signfi", "ut_har"))
    parser.add_argument(
        "--source-format",
        default="generic",
        choices=("generic", "ut_har_layout", "signfi_mat"),
        help=(
            "Input source layout. 'generic' expects --features/--labels files; "
            "'ut_har_layout' expects SenseFi-style UT_HAR directory with "
            "data/*.csv and label/*.csv NumPy payloads; 'signfi_mat' expects "
            "a SignFi .mat file via --mat-file."
        ),
    )
    parser.add_argument("--features", type=Path, default=None)
    parser.add_argument("--labels", type=Path, default=None)
    parser.add_argument("--environments", type=Path, default=None)
    parser.add_argument("--priors", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Root directory for source-format adapters such as ut_har_layout.",
    )
    parser.add_argument(
        "--mat-file",
        type=Path,
        default=None,
        help="SignFi .mat file path when using --source-format signfi_mat.",
    )
    parser.add_argument("--features-key", type=str, default=None)
    parser.add_argument("--labels-key", type=str, default=None)
    parser.add_argument("--environments-key", type=str, default=None)
    parser.add_argument("--priors-key", type=str, default=None)
    return parser


def _load_metadata(path: Path | None) -> dict[str, object]:
    if path is None:
        return {}

    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected JSON object in metadata file: {path}")
    return dict(loaded)


def _load_signfi_mat(
    mat_file: Path,
    features_key: str | None,
    labels_key: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        scipy_io = importlib.import_module("scipy.io")
    except ImportError as exc:
        raise RuntimeError(
            "SignFi MAT parsing requires scipy. Install it with 'pip install scipy' "
            "or provide generic arrays with --source-format generic."
        ) from exc

    try:
        loaded = scipy_io.loadmat(mat_file)
    except NotImplementedError as exc:
        raise RuntimeError(
            "This MAT file format is not supported by scipy.io.loadmat "
            "(likely MATLAB v7.3/HDF5). Export a v7 MAT file or pre-convert to "
            "NumPy arrays and use --source-format generic."
        ) from exc

    data = {
        key: value
        for key, value in loaded.items()
        if not key.startswith("__") and isinstance(value, np.ndarray)
    }
    if not data:
        raise ValueError(f"No ndarray entries found in MAT file: {mat_file}")

    if features_key is not None:
        if features_key not in data:
            available = ", ".join(sorted(data))
            raise ValueError(
                f"Features key '{features_key}' not found in {mat_file}; "
                f"available keys: {available}"
            )
        features = data[features_key]
    else:
        features = _pick_signfi_features(data)

    if labels_key is not None:
        if labels_key not in data:
            available = ", ".join(sorted(data))
            raise ValueError(
                f"Labels key '{labels_key}' not found in {mat_file}; "
                f"available keys: {available}"
            )
        labels = data[labels_key]
    else:
        labels = _pick_signfi_labels(data, num_samples=int(features.shape[0]))

    return features, labels


def _pick_signfi_features(data: dict[str, np.ndarray]) -> np.ndarray:
    lowered = {key.lower(): key for key in data}
    for candidate in SIGNFI_FEATURE_CANDIDATE_KEYS:
        if candidate in lowered:
            return data[lowered[candidate]]

    for value in data.values():
        if value.ndim >= 2 and np.issubdtype(value.dtype, np.number):
            return value

    raise ValueError("Failed to infer SignFi features array from MAT file contents")


def _pick_signfi_labels(data: dict[str, np.ndarray], num_samples: int) -> np.ndarray:
    lowered = {key.lower(): key for key in data}
    for candidate in SIGNFI_LABEL_CANDIDATE_KEYS:
        if candidate in lowered:
            return data[lowered[candidate]]

    for value in data.values():
        flattened = np.asarray(value).reshape(-1)
        if flattened.shape[0] == num_samples:
            return flattened

    raise ValueError(
        "Failed to infer SignFi labels array from MAT file contents; pass --labels-key"
    )


def _load_ut_har_layout(source_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data_dir = source_root / "data"
    label_dir = source_root / "label"
    if not data_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(
            "Expected UT_HAR source root with 'data/' and 'label/' directories, got "
            f"{source_root}"
        )

    data_files = sorted(data_dir.glob("*.csv"))
    label_files = sorted(label_dir.glob("*.csv"))
    if not data_files or not label_files:
        raise FileNotFoundError(
            "UT_HAR layout requires data/*.csv and label/*.csv files containing "
            "NumPy arrays serialized by np.save."
        )
    if len(data_files) != len(label_files):
        raise ValueError(
            "UT_HAR layout has mismatched data/label file counts: "
            f"{len(data_files)} data files vs {len(label_files)} label files"
        )

    feature_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    env_blocks: list[np.ndarray] = []

    for env_id, (data_file, label_file) in enumerate(
        zip(data_files, label_files, strict=False)
    ):
        features = np.load(data_file)
        labels = np.load(label_file)

        flattened_labels = np.asarray(labels).reshape(-1)
        if int(features.shape[0]) != int(flattened_labels.shape[0]):
            raise ValueError(
                "UT_HAR file pair sample mismatch for "
                f"{data_file.name}/{label_file.name}: {features.shape[0]} "
                f"!= {flattened_labels.shape[0]}"
            )

        feature_blocks.append(features)
        label_blocks.append(flattened_labels)
        env_blocks.append(
            np.full(flattened_labels.shape[0], env_id, dtype=np.int64),
        )

    if len(feature_blocks) < 2:
        raise ValueError(
            "UT_HAR cross-environment preparation expects at least two data/label "
            "file pairs so environment IDs can include train/test domains."
        )

    return (
        np.concatenate(feature_blocks, axis=0),
        np.concatenate(label_blocks, axis=0),
        np.concatenate(env_blocks, axis=0),
    )


def _resolve_inputs(
    *,
    source_format: SourceFormat,
    features: Path | None,
    labels: Path | None,
    environments: Path | None,
    priors: Path | None,
    source_root: Path | None,
    mat_file: Path | None,
    features_key: str | None,
    labels_key: str | None,
    environments_key: str | None,
    priors_key: str | None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None, dict[str, Any]
]:
    source_files: dict[str, Any] = {
        "source_format": source_format,
    }
    if source_format == "generic":
        if features is None or labels is None:
            raise ValueError(
                "--source-format generic requires both --features and --labels"
            )
        resolved_features = load_array_file(features, array_key=features_key)
        resolved_labels = load_array_file(labels, array_key=labels_key)
        resolved_environments = None
        if environments is not None:
            resolved_environments = load_array_file(
                environments,
                array_key=environments_key,
            )

        resolved_priors = None
        if priors is not None:
            resolved_priors = load_array_file(priors, array_key=priors_key)

        source_files.update(
            {
                "features": str(features),
                "labels": str(labels),
                "environments": str(environments) if environments is not None else None,
                "priors": str(priors) if priors is not None else None,
            }
        )
        return (
            resolved_features,
            resolved_labels,
            resolved_environments,
            resolved_priors,
            source_files,
        )

    if source_format == "ut_har_layout":
        if source_root is None:
            raise ValueError(
                "--source-format ut_har_layout requires --source-root pointing to "
                "the UT_HAR directory."
            )
        resolved_features, resolved_labels, resolved_environments = _load_ut_har_layout(
            source_root,
        )
        source_files.update(
            {
                "source_root": str(source_root),
                "data_dir": str(source_root / "data"),
                "label_dir": str(source_root / "label"),
            }
        )
        resolved_priors = None
        return (
            resolved_features,
            resolved_labels,
            resolved_environments,
            resolved_priors,
            source_files,
        )

    if source_format == "signfi_mat":
        if mat_file is None:
            raise ValueError(
                "--source-format signfi_mat requires --mat-file pointing to a SignFi "
                "MAT file."
            )
        resolved_features, resolved_labels = _load_signfi_mat(
            mat_file=mat_file,
            features_key=features_key,
            labels_key=labels_key,
        )
        resolved_environments = None
        if environments is not None:
            resolved_environments = load_array_file(
                environments,
                array_key=environments_key,
            )
        resolved_priors = None
        if priors is not None:
            resolved_priors = load_array_file(priors, array_key=priors_key)
        source_files.update(
            {
                "mat_file": str(mat_file),
                "features_key": features_key,
                "labels_key": labels_key,
                "environments": str(environments) if environments is not None else None,
                "priors": str(priors) if priors is not None else None,
            }
        )
        return (
            resolved_features,
            resolved_labels,
            resolved_environments,
            resolved_priors,
            source_files,
        )

    raise ValueError(f"Unsupported source format: {source_format}")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    if args.source_format == "ut_har_layout" and args.dataset != "ut_har":
        raise ValueError("--source-format ut_har_layout can only be used with ut_har")
    if args.source_format == "signfi_mat" and args.dataset != "signfi":
        raise ValueError("--source-format signfi_mat can only be used with signfi")

    metadata = _load_metadata(args.metadata)
    features, labels, environments, priors, source_files = _resolve_inputs(
        source_format=args.source_format,
        features=args.features,
        labels=args.labels,
        environments=args.environments,
        priors=args.priors,
        source_root=args.source_root,
        mat_file=args.mat_file,
        features_key=args.features_key,
        labels_key=args.labels_key,
        environments_key=args.environments_key,
        priors_key=args.priors_key,
    )
    source_files["metadata"] = str(args.metadata) if args.metadata is not None else None
    metadata["source_files"] = source_files

    dataset_dir = save_prepared_paper1_dataset(
        dataset_name=args.dataset,
        output_root=args.output_dir,
        features=features,
        labels=labels,
        environments=environments,
        priors=priors,
        metadata=metadata,
    )

    print(f"Prepared dataset written to: {dataset_dir}")
    print(f"Source format: {args.source_format}")
    if args.source_format == "generic":
        print(f"Features source: {args.features}")
        print(f"Labels source: {args.labels}")
    if args.source_format == "ut_har_layout":
        print(f"UT_HAR source root: {args.source_root}")
    if args.source_format == "signfi_mat":
        print(f"SignFi MAT source: {args.mat_file}")
    if args.environments is not None:
        print(f"Environments source: {args.environments}")
    if args.priors is not None:
        print(f"Priors source: {args.priors}")
    if args.metadata is not None:
        print(f"Metadata source: {args.metadata}")


if __name__ == "__main__":
    main()
