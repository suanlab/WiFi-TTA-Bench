# pyright: basic, reportMissingImports=false

import csv
import json
import subprocess
import sys
from pathlib import Path

from pinn4csi.utils.wifi_imaging_comparison import (
    aggregate_comparison_rows,
    load_comparison_rows,
    parse_baseline_artifact,
)


def _write_wifi_pinn_aggregate_json(path: Path) -> Path:
    payload = {
        "num_samples": 6,
        "num_environments": 2,
        "field_nmse_mean": 0.12,
        "field_nmse_std": 0.01,
        "permittivity_nmse_mean": 0.22,
        "permittivity_nmse_std": 0.02,
        "physics_loss_mean": 0.005,
        "physics_loss_std": 0.001,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_external_csv(path: Path) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "environment_id",
                "split",
                "seed",
                "field_nmse",
                "permittivity_nmse",
                "helmholtz_residual",
                "checkpoint",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "environment_id": 1,
                "split": "test",
                "seed": 42,
                "field_nmse": 0.31,
                "permittivity_nmse": 0.41,
                "helmholtz_residual": 0.11,
                "checkpoint": "epoch_10.pt",
            }
        )
        writer.writerow(
            {
                "environment_id": 1,
                "split": "test",
                "seed": 123,
                "field_nmse": 0.29,
                "permittivity_nmse": 0.43,
                "helmholtz_residual": 0.09,
                "checkpoint": "epoch_12.pt",
            }
        )
    return path


def _write_external_json(path: Path) -> Path:
    payload = {
        "results": [
            {
                "metadata": {
                    "env": "env_2",
                    "split": "test",
                    "seed": 7,
                    "checkpoint_path": "runs/gsrf.ckpt",
                },
                "metrics": {
                    "field_nmse": 0.27,
                    "epsilon_nmse": 0.38,
                    "helmholtz_residual": 0.15,
                },
            }
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def test_load_comparison_rows_normalizes_json_and_csv_inputs(tmp_path: Path) -> None:
    wifi_pinn_artifact = parse_baseline_artifact(
        f"wifi_pinn={_write_wifi_pinn_aggregate_json(tmp_path / 'wifi.json')}"
    )
    newrf_artifact = parse_baseline_artifact(
        f"newrf={_write_external_csv(tmp_path / 'newrf.csv')}"
    )

    wifi_rows = load_comparison_rows(wifi_pinn_artifact)
    newrf_rows = load_comparison_rows(newrf_artifact)

    assert len(wifi_rows) == 1
    assert wifi_rows[0].baseline_name == "wifi_pinn"
    assert wifi_rows[0].metric_scope == "aggregate"
    assert wifi_rows[0].split == "test"
    assert wifi_rows[0].environment == "all"
    assert wifi_rows[0].field_nmse == 0.12
    assert wifi_rows[0].permittivity_nmse == 0.22
    assert wifi_rows[0].physics_loss == 0.005

    assert len(newrf_rows) == 2
    assert newrf_rows[0].baseline_name == "newrf"
    assert newrf_rows[0].environment == "1"
    assert newrf_rows[0].seed == 42
    assert newrf_rows[0].checkpoint == "epoch_10.pt"
    assert newrf_rows[0].physics_loss == 0.11


def test_aggregate_comparison_rows_groups_metrics_by_baseline(tmp_path: Path) -> None:
    newrf_artifact = parse_baseline_artifact(
        f"newrf={_write_external_csv(tmp_path / 'newrf.csv')}"
    )
    gsrf_artifact = parse_baseline_artifact(
        f"gsrf={_write_external_json(tmp_path / 'gsrf.json')}"
    )

    aggregate_rows = aggregate_comparison_rows(
        [
            *load_comparison_rows(newrf_artifact),
            *load_comparison_rows(gsrf_artifact),
        ]
    )

    assert len(aggregate_rows) == 2

    newrf_row = aggregate_rows[0]
    assert newrf_row.baseline_name == "newrf"
    assert newrf_row.environment == "1"
    assert newrf_row.field_nmse_count == 2
    assert newrf_row.permittivity_nmse_count == 2
    assert newrf_row.physics_loss_count == 2
    assert newrf_row.field_nmse_mean == 0.30
    assert newrf_row.physics_loss_mean == 0.10

    gsrf_row = aggregate_rows[1]
    assert gsrf_row.baseline_name == "gsrf"
    assert gsrf_row.environment == "env_2"
    assert gsrf_row.field_nmse_mean == 0.27
    assert gsrf_row.permittivity_nmse_mean == 0.38


def test_compare_wifi_imaging_baselines_cli_writes_outputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    output_dir = tmp_path / "comparison_outputs"
    wifi_path = _write_wifi_pinn_aggregate_json(tmp_path / "wifi_eval.json")
    newrf_path = _write_external_csv(tmp_path / "newrf.csv")
    gsrf_path = _write_external_json(tmp_path / "gsrf.json")

    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "compare_wifi_imaging_baselines.py"),
            "--artifact",
            f"wifi_pinn={wifi_path}",
            "--artifact",
            f"newrf={newrf_path}",
            "--artifact",
            f"gsrf={gsrf_path}",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )

    normalized_csv = output_dir / "wifi_imaging_comparison_rows.csv"
    aggregate_csv = output_dir / "wifi_imaging_comparison_aggregates.csv"
    summary_json = output_dir / "wifi_imaging_comparison_summary.json"
    summary_txt = output_dir / "wifi_imaging_comparison_summary.txt"

    assert normalized_csv.exists()
    assert aggregate_csv.exists()
    assert summary_json.exists()
    assert summary_txt.exists()
    assert "WiFi imaging comparison summary" in completed.stdout

    summary_payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert len(summary_payload["aggregate_rows"]) == 3
    assert summary_payload["aggregate_rows"][0]["baseline_name"] == "wifi_pinn"
