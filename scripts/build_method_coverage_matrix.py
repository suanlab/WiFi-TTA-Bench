# pyright: basic, reportMissingImports=false

"""Rebuild outputs/coverage/method_coverage_matrix.json from real artifacts.

CHECK.md Critical 3: the previous matrix hard-coded SAR/CoTTA/LAME as absent on
NTU-Fi and SignFi, contradicting the extended runs in outputs/new_methods/.
This script scans the canonical per-dataset summaries plus the extended
SAR/CoTTA/LAME summaries and emits a fresh matrix. It is called by
reproduce_all_results.py so the matrix never goes stale again.

Output:
    outputs/coverage/method_coverage_matrix.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS: dict[str, dict[str, str]] = {
    "widar_bvp": {
        "canonical": "outputs/widar_full_tta/method_summary.json",
        "extras": "outputs/new_methods/widar_lame_sar_cotta.json",
    },
    "ntufi_har": {
        "canonical": "outputs/ntufi_tta/method_summary.json",
        "extras": "outputs/new_methods/ntufi_har_lame_sar_cotta.json",
    },
    "signfi_top10": {
        "canonical": "outputs/signfi_tta/method_summary.json",
        "extras": "outputs/new_methods/signfi_top10_lame_sar_cotta.json",
    },
}

INTERNAL_VARIANTS = {"physics_entropy_tta"}


def _methods_in(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path) as f:
        d = json.load(f)
    return {
        k for k, v in d.items() if isinstance(v, dict) and k not in INTERNAL_VARIANTS
    }


def build(output: Path) -> dict[str, object]:
    coverage: dict[str, list[str]] = {}
    for dataset, paths in DATASETS.items():
        methods = set()
        for key in ("canonical", "extras"):
            p = paths.get(key)
            if p:
                methods |= _methods_in(Path(p))
        coverage[dataset] = sorted(methods)

    all_methods = sorted(set().union(*coverage.values()))
    matrix = {
        "methods": all_methods,
        "coverage": {
            ds: {m: (m in coverage[ds]) for m in all_methods} for ds in coverage
        },
        "totals": {ds: len(coverage[ds]) for ds in coverage},
        "sources": {
            ds: {
                "canonical": paths["canonical"],
                "extras": paths.get("extras"),
            }
            for ds, paths in DATASETS.items()
        },
        "excluded_internal_variants": sorted(INTERNAL_VARIANTS),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(matrix, f, indent=2)
    return matrix


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/coverage/method_coverage_matrix.json"),
    )
    args = parser.parse_args()
    matrix = build(args.output)
    print(f"\nSaved: {args.output}")
    print(f"All methods ({len(matrix['methods'])}): {matrix['methods']}")
    for ds, count in matrix["totals"].items():
        present = [m for m, ok in matrix["coverage"][ds].items() if ok]
        print(f"  {ds}: {count} methods -> {present}")


if __name__ == "__main__":
    main()
