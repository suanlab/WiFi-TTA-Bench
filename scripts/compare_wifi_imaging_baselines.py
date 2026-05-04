# pyright: basic, reportMissingImports=false

"""Compare WiFi imaging baseline artifacts on a shared metric schema."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from pinn4csi.utils import save_dataclass_rows_csv, save_json_file
from pinn4csi.utils.wifi_imaging_comparison import (
    aggregate_comparison_rows,
    load_comparison_rows,
    parse_baseline_artifact,
    render_comparison_summary,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize WiFi imaging evaluation artifacts from in-repo and external "
            "baselines into a shared comparison table."
        )
    )
    parser.add_argument(
        "--artifact",
        action="append",
        required=True,
        help=(
            "Baseline artifact in the form 'baseline_name=/path/to/results.json'. "
            "Repeat for multiple artifacts. Supported explicit names include "
            "wifi_pinn, backprojection, newrf, and gsrf."
        ),
    )
    parser.add_argument(
        "--default-split",
        type=str,
        default="test",
        help="Fallback split name when an artifact omits one (default: test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper4_imaging/comparison"),
        help=(
            "Directory for normalized CSV/JSON artifacts and text summary "
            "(default: outputs/paper4_imaging/comparison)."
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    comparison_rows = []
    parsed_artifacts = [parse_baseline_artifact(raw) for raw in args.artifact]
    for artifact in parsed_artifacts:
        comparison_rows.extend(
            load_comparison_rows(artifact, default_split=args.default_split)
        )

    aggregate_rows = aggregate_comparison_rows(comparison_rows)
    summary_text = render_comparison_summary(aggregate_rows)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    normalized_csv = output_dir / "wifi_imaging_comparison_rows.csv"
    aggregate_csv = output_dir / "wifi_imaging_comparison_aggregates.csv"
    summary_json = output_dir / "wifi_imaging_comparison_summary.json"
    summary_txt = output_dir / "wifi_imaging_comparison_summary.txt"

    save_dataclass_rows_csv(comparison_rows, normalized_csv)
    save_dataclass_rows_csv(aggregate_rows, aggregate_csv)
    save_json_file(
        {
            "artifacts": [
                {
                    "baseline_name": artifact.baseline_name,
                    "artifact_path": str(artifact.artifact_path),
                }
                for artifact in parsed_artifacts
            ],
            "normalized_rows": [asdict(row) for row in comparison_rows],
            "aggregate_rows": [asdict(row) for row in aggregate_rows],
            "summary": summary_text,
        },
        summary_json,
    )
    summary_txt.write_text(summary_text + "\n", encoding="utf-8")

    print(summary_text)
    print(f"Normalized rows CSV: {normalized_csv}")
    print(f"Aggregate CSV: {aggregate_csv}")
    print(f"Summary JSON: {summary_json}")
    print(f"Summary text: {summary_txt}")


if __name__ == "__main__":
    main()
