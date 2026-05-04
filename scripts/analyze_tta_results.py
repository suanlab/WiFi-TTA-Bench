from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ComparisonRow:
    scope: str
    compare_method: str
    reference_method: str
    paired_mean_delta: float
    paired_std_delta: float
    wins: int
    total: int
    bootstrap_ci_low: float
    bootstrap_ci_high: float
    permutation_p: float
    compare_harm_rate: float
    reference_harm_rate: float
    compare_negative_tail_severity: float
    reference_negative_tail_severity: float
    compare_source_drop_mean: float
    reference_source_drop_mean: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze paired TTA result CSV files")
    parser.add_argument("--csv", action="append", required=True)
    parser.add_argument(
        "--compare-method",
        action="append",
        required=True,
        help="Method to compare, repeat for multiple methods.",
    )
    parser.add_argument(
        "--reference-method",
        action="append",
        required=True,
        help="Reference method, repeat for multiple baselines.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/paper2_tta_analysis.json"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = load_rows([Path(path) for path in args.csv])
    comparison_rows = []

    scopes = ["pooled", *sorted({scope_key(row) for row in rows})]
    for scope in scopes:
        if scope == "pooled":
            scoped_rows = rows
        else:
            scoped_rows = [row for row in rows if scope_key(row) == scope]
        for compare_method in args.compare_method:
            for reference_method in args.reference_method:
                comparison_rows.append(
                    analyze_comparison(
                        scoped_rows,
                        scope=scope,
                        compare_method=compare_method,
                        reference_method=reference_method,
                    )
                )

    payload = {
        "comparisons": [asdict(row) for row in comparison_rows],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    for row in comparison_rows:
        print(
            f"{row.scope} | {row.compare_method} vs {row.reference_method} | "
            f"delta={row.paired_mean_delta:+.4f} ± {row.paired_std_delta:.4f} | "
            f"wins={row.wins}/{row.total} | "
            f"CI=[{row.bootstrap_ci_low:+.4f}, {row.bootstrap_ci_high:+.4f}] | "
            f"p={row.permutation_p:.4f} | "
            f"harm={row.compare_harm_rate:.2f} vs {row.reference_harm_rate:.2f} | "
            f"tail={row.compare_negative_tail_severity:.4f} "
            f"vs {row.reference_negative_tail_severity:.4f} | "
            f"source_drop={row.compare_source_drop_mean:+.4f} "
            f"vs {row.reference_source_drop_mean:+.4f}"
        )


def load_rows(csv_paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in csv_paths:
        with path.open() as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def scope_key(row: dict[str, str]) -> str:
    dataset_name = row["dataset_name"]
    shift_name = row["shift_name"]
    held_out_environment_id = row["held_out_environment_id"]
    if shift_name == "held_out_split":
        return dataset_name + f":room{held_out_environment_id}"
    return dataset_name + f":{shift_name}:room{held_out_environment_id}"


def analyze_comparison(
    rows: list[dict[str, str]],
    scope: str,
    compare_method: str,
    reference_method: str,
) -> ComparisonRow:
    paired_keys = sorted(
        {
            pairing_key(row)
            for row in rows
            if row["method"] in {compare_method, reference_method}
        }
    )
    compare_gains = []
    reference_gains = []
    compare_source_drop = []
    reference_source_drop = []

    for pair_key in paired_keys:
        compare_row = single_row(rows, compare_method, pair_key)
        reference_row = single_row(rows, reference_method, pair_key)
        compare_gains.append(float(compare_row["gain"]))
        reference_gains.append(float(reference_row["gain"]))
        compare_source_drop.append(float(compare_row["source_drop"]))
        reference_source_drop.append(float(reference_row["source_drop"]))

    compare_arr = np.array(compare_gains, dtype=np.float64)
    reference_arr = np.array(reference_gains, dtype=np.float64)
    deltas = compare_arr - reference_arr
    ci_low, ci_high = bootstrap_ci(deltas)
    p_value = paired_permutation_p(deltas)

    return ComparisonRow(
        scope=scope,
        compare_method=compare_method,
        reference_method=reference_method,
        paired_mean_delta=float(deltas.mean()),
        paired_std_delta=float(deltas.std()),
        wins=int((deltas > 0).sum()),
        total=int(deltas.size),
        bootstrap_ci_low=ci_low,
        bootstrap_ci_high=ci_high,
        permutation_p=p_value,
        compare_harm_rate=harm_rate(compare_arr),
        reference_harm_rate=harm_rate(reference_arr),
        compare_negative_tail_severity=negative_tail_severity(compare_arr),
        reference_negative_tail_severity=negative_tail_severity(reference_arr),
        compare_source_drop_mean=float(np.mean(compare_source_drop)),
        reference_source_drop_mean=float(np.mean(reference_source_drop)),
    )


def single_row(
    rows: list[dict[str, str]],
    method: str,
    pair_key_value: tuple[str, str, str, str],
) -> dict[str, str]:
    dataset_name, shift_name, env_str, seed_str = pair_key_value
    matched = [
        row
        for row in rows
        if row["method"] == method
        and row["dataset_name"] == dataset_name
        and row["shift_name"] == shift_name
        and row["held_out_environment_id"] == env_str
        and row["seed"] == seed_str
    ]
    if len(matched) != 1:
        raise ValueError(
            "Expected one row for "
            f"method={method}, dataset={dataset_name}, shift={shift_name}, "
            f"seed={seed_str}, env={env_str}."
        )
    return matched[0]


def pairing_key(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (
        row["dataset_name"],
        row["shift_name"],
        row["held_out_environment_id"],
        row["seed"],
    )


def harm_rate(gains: np.ndarray) -> float:
    return float(np.mean(gains < 0.0))


def negative_tail_severity(gains: np.ndarray) -> float:
    harmful = gains[gains < 0.0]
    if harmful.size == 0:
        return 0.0
    return float(np.mean(-harmful))


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(n_boot):
        indices = rng.integers(0, deltas.size, size=deltas.size)
        samples.append(float(deltas[indices].mean()))
    ci = np.quantile(np.array(samples), [0.025, 0.975])
    return float(ci[0]), float(ci[1])


def paired_permutation_p(deltas: np.ndarray, n_perm: int = 10000) -> float:
    rng = np.random.default_rng(42)
    observed = abs(float(deltas.mean()))
    permuted = []
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=deltas.size)
        permuted.append(abs(float(np.mean(deltas * signs))))
    return float(np.mean(np.array(permuted) >= observed))


if __name__ == "__main__":
    main()
