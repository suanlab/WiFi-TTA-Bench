from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import Field, is_dataclass
from pathlib import Path
from typing import Protocol, TypeGuard


class _DataclassInstance(Protocol):
    __dataclass_fields__: dict[str, Field[object]]


def _is_dataclass_instance(value: object) -> TypeGuard[_DataclassInstance]:
    return is_dataclass(value) and not isinstance(value, type)


def parse_csv_items(raw: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not values:
        raise ValueError("Expected at least one comma-separated value.")
    return values


def parse_csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(value) for value in parse_csv_items(raw))


def parse_csv_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(value) for value in parse_csv_items(raw))


def save_dataclass_rows_csv(
    rows: Sequence[object],
    output_csv: str | Path,
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_fieldnames = list(fieldnames) if fieldnames is not None else None
    if rows:
        first_row = rows[0]
        if not _is_dataclass_instance(first_row):
            raise TypeError("Expected dataclass instances when writing CSV rows.")
        if resolved_fieldnames is None:
            resolved_fieldnames = list(first_row.__dataclass_fields__)
    elif resolved_fieldnames is None:
        raise ValueError("fieldnames are required when writing an empty CSV.")

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=resolved_fieldnames)
        writer.writeheader()
        for row in rows:
            if not _is_dataclass_instance(row):
                raise TypeError("Expected dataclass instances when writing CSV rows.")
            writer.writerow(
                {
                    field_name: getattr(row, field_name)
                    for field_name in resolved_fieldnames
                }
            )


def save_json_file(
    payload: object,
    output_json: str | Path,
    *,
    indent: int = 2,
    sort_keys: bool = True,
) -> None:
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, sort_keys=sort_keys)
