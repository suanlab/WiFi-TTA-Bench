# pyright: basic, reportMissingImports=false

"""Compare released result artifacts against a fresh reproduction.

Walks every numeric leaf of each JSON artifact in a baseline tree and the
matching file in a reproduction tree, and reports per file the number of
numeric fields compared, how many differ, and the largest absolute and relative
difference. Paths that changed name between the two trees are mapped
explicitly. Fields that legitimately vary between runs (timings, dates) are
ignored.

Usage:
    python scripts/compare_artifacts.py BASELINE_OUTPUTS REPRO_OUTPUTS \
        --report outputs/reproduction_audit/report.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from pinn4csi.utils import save_json_file

# Baseline path -> reproduction path, for artifacts whose file name changed.
RENAMED = {
    "new_methods/widar_lame_sar_cotta.json": (
        "new_methods/widar_bvp_lame_sar_cotta.json"
    ),
    "new_methods/superseded_2026-05-04/widar_lame_sar_cotta.json": (
        "new_methods/widar_bvp_lame_sar_cotta.json"
    ),
    "new_methods/superseded_2026-05-04/ntufi_har_lame_sar_cotta.json": (
        "new_methods/ntufi_har_lame_sar_cotta.json"
    ),
    "new_methods/superseded_2026-05-04/signfi_top10_lame_sar_cotta.json": (
        "new_methods/signfi_top10_lame_sar_cotta.json"
    ),
    "new_methods/superseded_2026-05-04/widar_mlpbn_faithful_sar_cotta.json": (
        "new_methods/widar_mlpbn_faithful_sar_cotta.json"
    ),
    "final_ablations/superseded_2026-04/arch_ablation_with_ci.json": (
        "final_ablations/arch_ablation_with_ci.json"
    ),
    "final_ablations/superseded_2026-04/mlpbn_other_datasets.json": (
        "final_ablations/mlpbn_other_datasets.json"
    ),
    "phase1/superseded_2026-05-04/faithful_tent_results.json": (
        "phase1/faithful_tent_results.json"
    ),
    "ntufi_tta/method_summary.json": "core_regen/ntufi_har_lame_sar_cotta.json",
    "signfi_tta/method_summary.json": "core_regen/signfi_top10_lame_sar_cotta.json",
}
# Same quantity, different key name across writers.
KEY_ALIASES = {"neg_rate": "negative_adaptation_rate"}
IGNORED_KEYS = {
    "time_seconds",
    "elapsed_seconds",
    "seconds",
    "adaptation_time_seconds",
    "timestamp",
    "datePublished",
}
ABS_TOL = 1e-6


def flatten(node: object, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(node, dict):
        for key, value in node.items():
            if key in IGNORED_KEYS:
                continue
            name = KEY_ALIASES.get(key, key)
            out.update(flatten(value, f"{prefix}/{name}"))
    elif isinstance(node, list):
        for index, value in enumerate(node):
            out.update(flatten(value, f"{prefix}[{index}]"))
    elif isinstance(node, bool):
        return out
    elif isinstance(node, (int, float)):
        out[prefix] = float(node)
    return out


def compare_file(base: Path, repro: Path) -> dict[str, object]:
    a = flatten(json.loads(base.read_text()))
    b = flatten(json.loads(repro.read_text()))
    shared = sorted(set(a) & set(b))
    diffs = []
    for key in shared:
        x, y = a[key], b[key]
        if math.isnan(x) and math.isnan(y):
            continue
        delta = abs(x - y)
        if delta > ABS_TOL:
            rel = delta / max(abs(x), 1e-12)
            diffs.append((key, x, y, delta, rel))
    diffs.sort(key=lambda d: -d[3])
    return {
        "compared": len(shared),
        "only_in_baseline": len(set(a) - set(b)),
        "only_in_reproduction": len(set(b) - set(a)),
        "differing": len(diffs),
        "max_abs_diff": diffs[0][3] if diffs else 0.0,
        "largest": [
            {"field": k, "baseline": x, "reproduction": y, "abs_diff": d}
            for k, x, y, d, _ in diffs[:8]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("reproduction", type=Path)
    parser.add_argument("--files", nargs="*", help="Baseline-relative paths to check.")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    files = args.files or sorted(
        str(p.relative_to(args.baseline)) for p in args.baseline.rglob("*.json")
    )
    results: dict[str, object] = {}
    for rel in files:
        base = args.baseline / rel
        repro = args.reproduction / RENAMED.get(rel, rel)
        if not base.exists():
            results[rel] = {"status": "missing-in-baseline"}
        elif not repro.exists():
            results[rel] = {"status": "not-reproduced"}
        else:
            comp = compare_file(base, repro)
            comp["status"] = "identical" if comp["differing"] == 0 else "differs"
            comp["reproduction_path"] = str(RENAMED.get(rel, rel))
            results[rel] = comp
    args.report.parent.mkdir(parents=True, exist_ok=True)
    save_json_file(results, args.report)
    for rel, r in results.items():
        status = r["status"]  # type: ignore[index]
        extra = ""
        if status == "differs":
            extra = "{}/{} fields, max |d|={:.4g}".format(
                r["differing"],  # type: ignore[index]
                r["compared"],  # type: ignore[index]
                r["max_abs_diff"],  # type: ignore[index]
            )
        print(f"{status:20s} {rel} {extra}")


if __name__ == "__main__":
    main()
