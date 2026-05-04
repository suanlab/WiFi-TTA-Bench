from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from statistics import mean


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build Paper 2 pooled gain-vs-drift artifacts"
    )
    parser.add_argument(
        "--synthetic-csv",
        type=Path,
        required=True,
        help="Synthetic pooled CSV from run_tta_experiments.py",
    )
    parser.add_argument(
        "--widar-csv",
        action="append",
        type=Path,
        required=True,
        help="Repeat for each Widar CSV to pool.",
    )
    parser.add_argument(
        "--output-prefix",
        type=Path,
        default=Path("outputs/paper2_tta_gain_drift"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    synthetic_rows = load_rows([args.synthetic_csv])
    widar_rows = load_rows(args.widar_csv)

    synthetic_summary = summarize_scope(
        rows=synthetic_rows,
        scope_name="synthetic_pooled",
    )
    widar_summary = summarize_scope(
        rows=widar_rows,
        scope_name="widar_pooled",
    )
    summaries = synthetic_summary + widar_summary

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    write_summary_csv(summaries, args.output_prefix.with_suffix(".csv"))
    write_summary_markdown(summaries, args.output_prefix.with_suffix(".md"))
    save_gain_drift_figure(summaries, args.output_prefix.with_suffix(".svg"))


def load_rows(csv_paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in csv_paths:
        with path.open() as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def summarize_scope(
    rows: list[dict[str, str]], scope_name: str
) -> list[dict[str, str | float]]:
    methods = sorted({row["method"] for row in rows})
    summary_rows: list[dict[str, str | float]] = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        gains = [float(row["gain"]) for row in method_rows]
        source_drops = [float(row["source_drop"]) for row in method_rows]
        negative_gains = [-gain for gain in gains if gain < 0.0]
        summary_rows.append(
            {
                "scope": scope_name,
                "method": method,
                "mean_gain": mean(gains),
                "source_drop": mean(source_drops),
                "harm_rate": sum(gain < 0.0 for gain in gains) / max(len(gains), 1),
                "negative_tail_severity": (
                    mean(negative_gains) if negative_gains else 0.0
                ),
            }
        )
    return summary_rows


def write_summary_csv(rows: list[dict[str, str | float]], path: Path) -> None:
    fieldnames = [
        "scope",
        "method",
        "mean_gain",
        "source_drop",
        "harm_rate",
        "negative_tail_severity",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_markdown(rows: list[dict[str, str | float]], path: Path) -> None:
    lines = [
        "# Paper 2 Gain-vs-Drift Summary",
        "",
        "| Scope | Method | Mean Gain | Source Drop | Harm Rate | "
        "Negative-Tail Severity |",
        "|-------|--------|-----------|-------------|-----------|------------------------|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row['scope']} | {row['method']} | {float(row['mean_gain']):+.4f} | "
            f"{float(row['source_drop']):+.4f} | {float(row['harm_rate']):.2f} | "
            f"{float(row['negative_tail_severity']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_gain_drift_figure(
    rows: list[dict[str, str | float]],
    path: Path,
) -> None:
    scopes = ["synthetic_pooled", "widar_pooled"]
    colors = {
        "no_adapt": "#4c566a",
        "entropy_tta": "#bf616a",
        "conservative_entropy_tta": "#d08770",
        "physics_tta": "#5e81ac",
        "safe_physics_tta": "#2e8b57",
    }
    labels = {
        "no_adapt": "No Adapt",
        "entropy_tta": "Entropy",
        "conservative_entropy_tta": "Conservative Entropy",
        "physics_tta": "Physics",
        "safe_physics_tta": "Safe Physics",
    }

    width = 1200
    height = 520
    panel_width = 500
    panel_height = 340
    margin_left = 80
    margin_top = 90
    panel_gap = 70

    all_x = [float(row["source_drop"]) for row in rows]
    all_y = [float(row["mean_gain"]) for row in rows]
    min_x = min(all_x + [0.0]) - 0.01
    max_x = max(all_x + [0.0]) + 0.01
    min_y = min(all_y + [0.0]) - 0.02
    max_y = max(all_y + [0.0]) + 0.02

    def scale_x(value: float, panel_index: int) -> float:
        panel_origin_x = margin_left + panel_index * (panel_width + panel_gap)
        return panel_origin_x + (value - min_x) / max(max_x - min_x, 1e-8) * panel_width

    def scale_y(value: float) -> float:
        return (
            margin_top
            + panel_height
            - ((value - min_y) / max(max_y - min_y, 1e-8) * panel_height)
        )

    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white"/>',
        (
            '<text x="600" y="36" text-anchor="middle" font-size="24" '
            'font-family="Arial">Paper 2 TTA Gain-vs-Drift Tradeoff</text>'
        ),
    ]

    for panel_index, scope in enumerate(scopes):
        panel_origin_x = margin_left + panel_index * (panel_width + panel_gap)
        panel_origin_y = margin_top
        scope_rows = [row for row in rows if row["scope"] == scope]
        lines.extend(
            [
                (
                    f'<rect x="{panel_origin_x}" y="{panel_origin_y}" '
                    f'width="{panel_width}" height="{panel_height}" '
                    'fill="none" stroke="#444" stroke-width="1"/>'
                ),
                (
                    f'<text x="{panel_origin_x + panel_width / 2}" '
                    f'y="{panel_origin_y - 18}" text-anchor="middle" '
                    'font-size="18" font-family="Arial">'
                    f"{html.escape(scope.replace('_', ' ').title())}</text>"
                ),
                (
                    f'<line x1="{scale_x(0.0, panel_index)}" '
                    f'y1="{panel_origin_y}" '
                    f'x2="{scale_x(0.0, panel_index)}" '
                    f'y2="{panel_origin_y + panel_height}" '
                    'stroke="#999" stroke-dasharray="6 4"/>'
                ),
                (
                    f'<line x1="{panel_origin_x}" y1="{scale_y(0.0)}" '
                    f'x2="{panel_origin_x + panel_width}" '
                    f'y2="{scale_y(0.0)}" '
                    'stroke="#999" stroke-dasharray="6 4"/>'
                ),
                (
                    f'<text x="{panel_origin_x + panel_width / 2}" '
                    f'y="{panel_origin_y + panel_height + 34}" '
                    'text-anchor="middle" font-size="14" '
                    'font-family="Arial">Mean Source Drop</text>'
                ),
            ]
        )
        if panel_index == 0:
            lines.append(
                f'<text x="24" y="{panel_origin_y + panel_height / 2}" '
                f'transform="rotate(-90 24 {panel_origin_y + panel_height / 2})" '
                'text-anchor="middle" font-size="14" '
                'font-family="Arial">Mean Target Gain</text>'
            )
        for row in scope_rows:
            method = str(row["method"])
            x = scale_x(float(row["source_drop"]), panel_index)
            y = scale_y(float(row["mean_gain"]))
            color = colors.get(method, "#333333")
            label = labels.get(method, method)
            lines.extend(
                [
                    (
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="8" '
                        f'fill="{color}" stroke="black" stroke-width="1"/>'
                    ),
                    (
                        f'<text x="{x + 10:.2f}" y="{y - 10:.2f}" '
                        'font-size="11" font-family="Arial">'
                        f"{html.escape(label)}</text>"
                    ),
                ]
            )

    lines.append("</svg>")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
