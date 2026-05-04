# pyright: basic, reportMissingImports=false

import json
from dataclasses import dataclass
from pathlib import Path

import pinn4csi
from pinn4csi.utils import (
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_items,
    save_dataclass_rows_csv,
    save_json_file,
)


@dataclass(frozen=True)
class _ExampleRow:
    name: str
    value: int


def test_top_level_package_exports_framework_namespaces() -> None:
    assert pinn4csi.__version__ == "0.1.0"
    assert pinn4csi.data is not None
    assert pinn4csi.models is not None
    assert pinn4csi.physics is not None
    assert pinn4csi.training is not None
    assert pinn4csi.utils is not None


def test_shared_experiment_parsers_handle_common_cli_inputs() -> None:
    assert parse_csv_items("signfi, ut_har") == ("signfi", "ut_har")
    assert parse_csv_ints("0, 2,4") == (0, 2, 4)
    assert parse_csv_floats("0.1, 1.0,2.5") == (0.1, 1.0, 2.5)


def test_shared_experiment_parsers_reject_empty_input() -> None:
    try:
        parse_csv_items(" ,  ")
    except ValueError as exc:
        assert "comma-separated" in str(exc)
    else:
        raise AssertionError("Expected parse_csv_items to reject empty input.")


def test_shared_experiment_artifact_helpers_write_expected_files(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "results" / "rows.csv"
    json_path = tmp_path / "results" / "analysis.json"
    rows = [_ExampleRow(name="baseline", value=1), _ExampleRow(name="pinn", value=2)]

    save_dataclass_rows_csv(rows, csv_path)
    save_json_file({"num_rows": len(rows)}, json_path)

    assert csv_path.exists()
    assert json_path.exists()
    assert csv_path.read_text(encoding="utf-8").splitlines() == [
        "name,value",
        "baseline,1",
        "pinn,2",
    ]
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"num_rows": 2}
