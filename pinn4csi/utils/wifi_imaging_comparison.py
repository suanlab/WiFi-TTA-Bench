from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

BASELINE_DISPLAY_ORDER = (
    "wifi_pinn",
    "backprojection",
    "newrf",
    "gsrf",
)

_FIELD_NMSE_ALIASES = (
    "field_nmse",
    "field_nmse_mean",
    "nmse_field",
)
_FIELD_NMSE_STD_ALIASES = ("field_nmse_std",)
_PERMITTIVITY_NMSE_ALIASES = (
    "permittivity_nmse",
    "permittivity_nmse_mean",
    "epsilon_nmse",
    "eps_nmse",
    "nmse_permittivity",
)
_PERMITTIVITY_NMSE_STD_ALIASES = ("permittivity_nmse_std", "epsilon_nmse_std")
_PHYSICS_LOSS_ALIASES = (
    "physics_loss",
    "physics_loss_mean",
    "helmholtz_residual",
    "helmholtz_residual_mean",
    "r_h",
)
_PHYSICS_LOSS_STD_ALIASES = (
    "physics_loss_std",
    "helmholtz_residual_std",
    "r_h_std",
)
_SPLIT_ALIASES = ("split", "evaluation_split", "eval_split")
_ENVIRONMENT_ALIASES = (
    "environment",
    "environment_id",
    "environment_name",
    "env",
    "env_id",
)
_SEED_ALIASES = ("seed", "random_seed")
_CHECKPOINT_ALIASES = (
    "checkpoint",
    "checkpoint_name",
    "checkpoint_path",
    "model_checkpoint",
)
_NUM_SAMPLES_ALIASES = ("num_samples", "sample_count")
_NUM_ENVIRONMENTS_ALIASES = ("num_environments", "environment_count")
_ROW_CONTAINER_KEYS = ("rows", "results", "records", "items")


@dataclass(frozen=True)
class BaselineArtifact:
    baseline_name: str
    artifact_path: Path


@dataclass(frozen=True)
class NormalizedComparisonRow:
    baseline_name: str
    split: str
    environment: str
    seed: int | None
    checkpoint: str | None
    source_artifact_path: str
    source_format: str
    metric_scope: str
    field_nmse: float | None
    field_nmse_std: float | None
    permittivity_nmse: float | None
    permittivity_nmse_std: float | None
    physics_loss: float | None
    physics_loss_std: float | None
    num_samples: int | None
    num_environments: int | None


@dataclass(frozen=True)
class AggregateComparisonRow:
    baseline_name: str
    split: str
    environment: str
    num_rows: int
    field_nmse_mean: float | None
    field_nmse_std: float | None
    field_nmse_count: int
    permittivity_nmse_mean: float | None
    permittivity_nmse_std: float | None
    permittivity_nmse_count: int
    physics_loss_mean: float | None
    physics_loss_std: float | None
    physics_loss_count: int


def parse_baseline_artifact(raw: str) -> BaselineArtifact:
    baseline_name, separator, raw_path = raw.partition("=")
    if separator != "=" or not baseline_name.strip() or not raw_path.strip():
        raise ValueError(
            "Artifact arguments must use the form 'baseline_name=/path/to/artifact'."
        )

    return BaselineArtifact(
        baseline_name=normalize_baseline_name(baseline_name),
        artifact_path=Path(raw_path).expanduser(),
    )


def normalize_baseline_name(name: str) -> str:
    normalized = _normalize_key(name)
    if not normalized:
        raise ValueError(f"Invalid baseline name: {name!r}")
    return normalized


def load_comparison_rows(
    artifact: BaselineArtifact,
    *,
    default_split: str = "test",
) -> list[NormalizedComparisonRow]:
    artifact_path = artifact.artifact_path
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found: {artifact_path}")

    suffix = artifact_path.suffix.lower()
    if suffix == ".csv":
        raw_rows = _load_csv_rows(artifact_path)
        source_format = "csv"
    elif suffix == ".json":
        raw_rows = _load_json_rows(artifact_path)
        source_format = "json"
    else:
        raise ValueError(
            f"Unsupported artifact format for {artifact_path}. Expected .json or .csv."
        )

    if not raw_rows:
        raise ValueError(f"Artifact contains no comparison rows: {artifact_path}")

    return [
        _normalize_row(
            row,
            artifact=artifact,
            default_split=default_split,
            source_format=source_format,
        )
        for row in raw_rows
    ]


def aggregate_comparison_rows(
    rows: Iterable[NormalizedComparisonRow],
) -> list[AggregateComparisonRow]:
    grouped_rows: dict[tuple[str, str, str], list[NormalizedComparisonRow]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped_rows[(row.baseline_name, row.split, row.environment)].append(row)

    aggregate_rows: list[AggregateComparisonRow] = []
    for group_key in sorted(grouped_rows, key=_aggregate_sort_key):
        group = grouped_rows[group_key]
        baseline_name, split, environment = group_key
        field_mean, field_std, field_count = _mean_std(
            row.field_nmse for row in group if row.field_nmse is not None
        )
        perm_mean, perm_std, perm_count = _mean_std(
            row.permittivity_nmse for row in group if row.permittivity_nmse is not None
        )
        physics_mean, physics_std, physics_count = _mean_std(
            row.physics_loss for row in group if row.physics_loss is not None
        )
        aggregate_rows.append(
            AggregateComparisonRow(
                baseline_name=baseline_name,
                split=split,
                environment=environment,
                num_rows=len(group),
                field_nmse_mean=field_mean,
                field_nmse_std=field_std,
                field_nmse_count=field_count,
                permittivity_nmse_mean=perm_mean,
                permittivity_nmse_std=perm_std,
                permittivity_nmse_count=perm_count,
                physics_loss_mean=physics_mean,
                physics_loss_std=physics_std,
                physics_loss_count=physics_count,
            )
        )

    return aggregate_rows


def render_comparison_summary(rows: Iterable[AggregateComparisonRow]) -> str:
    aggregate_rows = list(rows)
    if not aggregate_rows:
        return "No comparison rows were loaded."

    lines = ["WiFi imaging comparison summary"]
    for row in aggregate_rows:
        metrics: list[str] = []
        metrics.append(
            _format_metric(
                name="field_nmse",
                mean=row.field_nmse_mean,
                std=row.field_nmse_std,
                count=row.field_nmse_count,
            )
        )
        metrics.append(
            _format_metric(
                name="permittivity_nmse",
                mean=row.permittivity_nmse_mean,
                std=row.permittivity_nmse_std,
                count=row.permittivity_nmse_count,
            )
        )
        metrics.append(
            _format_metric(
                name="physics_loss",
                mean=row.physics_loss_mean,
                std=row.physics_loss_std,
                count=row.physics_loss_count,
            )
        )
        line = (
            f"- {row.baseline_name} split={row.split} environment={row.environment} "
            f"rows={row.num_rows}; "
            f"{', '.join(metrics)}"
        )
        lines.append(line)
    return "\n".join(lines)


def _aggregate_sort_key(group_key: tuple[str, str, str]) -> tuple[int, str, str, str]:
    baseline_name, split, environment = group_key
    try:
        baseline_index = BASELINE_DISPLAY_ORDER.index(baseline_name)
    except ValueError:
        baseline_index = len(BASELINE_DISPLAY_ORDER)
    return (baseline_index, baseline_name, split, environment)


def _load_csv_rows(artifact_path: Path) -> list[dict[str, object]]:
    with artifact_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _load_json_rows(artifact_path: Path) -> list[dict[str, object]]:
    with artifact_path.open("r", encoding="utf-8") as handle:
        payload = cast(object, json.load(handle))

    if isinstance(payload, list):
        values = cast(list[object], payload)
        return _require_mapping_list(values, artifact_path)

    mapping = _require_mapping(payload, artifact_path)
    for key in _ROW_CONTAINER_KEYS:
        container = mapping.get(key)
        if isinstance(container, list):
            values = cast(list[object], container)
            return _require_mapping_list(values, artifact_path)
    return [mapping]


def _require_mapping(value: object, artifact_path: Path) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(
            f"Expected mapping rows in {artifact_path}, got {type(value).__name__}."
        )
    mapping = cast(Mapping[object, object], value)
    return _stringify_mapping(mapping)


def _require_mapping_list(
    values: list[object],
    artifact_path: Path,
) -> list[dict[str, object]]:
    return [_require_mapping(value, artifact_path) for value in values]


def _stringify_mapping(value: Mapping[object, object]) -> dict[str, object]:
    normalized: dict[str, object] = {}
    for key, item in value.items():
        normalized[str(key)] = item
    return normalized


def _normalize_row(
    row: Mapping[str, object],
    *,
    artifact: BaselineArtifact,
    default_split: str,
    source_format: str,
) -> NormalizedComparisonRow:
    flat_row = _flatten_mapping(row)
    field_nmse = _get_float(flat_row, _FIELD_NMSE_ALIASES)
    permittivity_nmse = _get_float(flat_row, _PERMITTIVITY_NMSE_ALIASES)
    physics_loss = _get_float(flat_row, _PHYSICS_LOSS_ALIASES)

    if field_nmse is None and permittivity_nmse is None and physics_loss is None:
        message = (
            f"Artifact row from {artifact.artifact_path} does not expose any supported "
            f"comparison metrics."
        )
        raise ValueError(message)

    split = _get_string(flat_row, _SPLIT_ALIASES) or default_split
    environment = _get_string(flat_row, _ENVIRONMENT_ALIASES) or "all"

    return NormalizedComparisonRow(
        baseline_name=artifact.baseline_name,
        split=split,
        environment=environment,
        seed=_get_int(flat_row, _SEED_ALIASES),
        checkpoint=_get_string(flat_row, _CHECKPOINT_ALIASES),
        source_artifact_path=str(artifact.artifact_path),
        source_format=source_format,
        metric_scope=_infer_metric_scope(flat_row),
        field_nmse=field_nmse,
        field_nmse_std=_get_float(flat_row, _FIELD_NMSE_STD_ALIASES),
        permittivity_nmse=permittivity_nmse,
        permittivity_nmse_std=_get_float(flat_row, _PERMITTIVITY_NMSE_STD_ALIASES),
        physics_loss=physics_loss,
        physics_loss_std=_get_float(flat_row, _PHYSICS_LOSS_STD_ALIASES),
        num_samples=_get_int(flat_row, _NUM_SAMPLES_ALIASES),
        num_environments=_get_int(flat_row, _NUM_ENVIRONMENTS_ALIASES),
    )


def _flatten_mapping(row: Mapping[str, object]) -> dict[str, object]:
    flat_row: dict[str, object] = {}
    _flatten_mapping_into(row, flat_row)
    return flat_row


def _flatten_mapping_into(
    row: Mapping[str, object],
    flat_row: dict[str, object],
    *,
    prefix: str = "",
) -> None:
    for key, value in row.items():
        normalized_key = _normalize_key(key)
        full_key = normalized_key if not prefix else f"{prefix}_{normalized_key}"
        if isinstance(value, Mapping):
            nested_mapping = _stringify_mapping(cast(Mapping[object, object], value))
            _flatten_mapping_into(nested_mapping, flat_row, prefix=full_key)
            continue
        flat_row[full_key] = value
        if not prefix and normalized_key not in flat_row:
            flat_row[normalized_key] = value


def _normalize_key(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized)
    return normalized.strip("_")


def _get_float(row: Mapping[str, object], aliases: Iterable[str]) -> float | None:
    value = _get_value(row, aliases)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid numeric comparison metrics.")
    if isinstance(value, (int, float)):
        return float(value)
    return float(str(value))


def _get_int(row: Mapping[str, object], aliases: Iterable[str]) -> int | None:
    value = _get_value(row, aliases)
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean values are not valid integer metadata fields.")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return int(str(value))


def _get_string(row: Mapping[str, object], aliases: Iterable[str]) -> str | None:
    value = _get_value(row, aliases)
    if value is None:
        return None
    rendered = str(value).strip()
    if not rendered:
        return None
    return rendered


def _get_value(row: Mapping[str, object], aliases: Iterable[str]) -> object | None:
    for alias in aliases:
        normalized_alias = _normalize_key(alias)
        if normalized_alias in row:
            return row[normalized_alias]
        suffix = f"_{normalized_alias}"
        for key, value in row.items():
            if key.endswith(suffix):
                return value
    return None


def _infer_metric_scope(row: Mapping[str, object]) -> str:
    if any(key.endswith("_std") or key.endswith("_mean") for key in row):
        return "aggregate"
    if "sample_index" in row:
        return "sample"
    return "row"


def _mean_std(values: Iterable[float]) -> tuple[float | None, float | None, int]:
    resolved_values = [value for value in values]
    count = len(resolved_values)
    if count == 0:
        return None, None, 0

    mean_value = sum(resolved_values) / count
    if count == 1:
        return mean_value, 0.0, 1

    variance = sum((value - mean_value) ** 2 for value in resolved_values) / (count - 1)
    return mean_value, variance**0.5, count


def _format_metric(
    *,
    name: str,
    mean: float | None,
    std: float | None,
    count: int,
) -> str:
    if mean is None or std is None or count == 0:
        return f"{name}=n/a"
    return f"{name}={mean:.6f}±{std:.6f} (n={count})"
