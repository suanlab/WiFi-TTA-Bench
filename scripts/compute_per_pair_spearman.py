# pyright: basic, reportMissingImports=false

"""Per-dataset-pair Spearman rank correlation of TTA method gains.

Reviewer 1 asked for per-pair rho values rather than a single aggregate.
This script reads the canonical method_summary JSONs for the three real
datasets, intersects the methods evaluated on all three, ranks them by mean
gain on each dataset, and reports pairwise Spearman rho with bootstrap 95% CI.

Output:
    outputs/coverage/cross_dataset_spearman.json
    outputs/coverage/cross_dataset_spearman.csv

Usage:
    python scripts/compute_per_pair_spearman.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

OUTPUT_DIR = Path("outputs/coverage")
DATASETS = {
    "widar_bvp": [
        "outputs/widar_full_tta/method_summary.json",
        "outputs/new_methods/widar_bvp_lame_sar_cotta.json",
    ],
    "ntufi_har": [
        "outputs/ntufi_tta/method_summary.json",
        "outputs/new_methods/ntufi_har_lame_sar_cotta.json",
    ],
    "signfi_top10": [
        "outputs/signfi_tta/method_summary.json",
        "outputs/new_methods/signfi_top10_lame_sar_cotta.json",
    ],
}


def load_method_gains(paths: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in paths:
        if not Path(p).exists():
            continue
        with open(p) as f:
            d = json.load(f)
        for method, payload in d.items():
            if not isinstance(payload, dict):
                continue
            mean_gain = payload.get("mean_gain", payload.get("mean"))
            if mean_gain is None:
                continue
            out[method] = float(mean_gain)
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation; ties broken by argsort."""
    if a.size != b.size or a.size < 2:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    if denom == 0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def bootstrap_rho_ci(
    a: np.ndarray, b: np.ndarray, n_resamples: int = 10000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = a.size
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        samples.append(spearman(a[idx], b[idx]))
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    gains = {ds: load_method_gains(paths) for ds, paths in DATASETS.items()}

    # Methods evaluated on all three datasets
    common = set.intersection(*[set(g.keys()) for g in gains.values()])
    common.discard("no_adapt")  # zero by construction; uninformative for ranking
    methods = sorted(common)

    print(f"Methods evaluated on all three datasets ({len(methods)}):")
    for m in methods:
        row = ", ".join(f"{ds}={gains[ds][m]:+.4f}" for ds in DATASETS)
        print(f"  {m:<26s} {row}")

    pairs = [
        ("widar_bvp", "ntufi_har"),
        ("widar_bvp", "signfi_top10"),
        ("ntufi_har", "signfi_top10"),
    ]
    pair_rows: list[dict[str, object]] = []
    print("\nPer-pair Spearman rho (with 95% bootstrap CI, 10K resamples):")
    print(f"{'pair':<28s} {'rho':>8s} {'CI_lo':>8s} {'CI_hi':>8s}")
    for a, b in pairs:
        va = np.array([gains[a][m] for m in methods])
        vb = np.array([gains[b][m] for m in methods])
        rho = spearman(va, vb)
        lo, hi = bootstrap_rho_ci(va, vb)
        print(f"{a + ' x ' + b:<28s} {rho:>+.4f} {lo:>+.4f} {hi:>+.4f}")
        pair_rows.append(
            {
                "pair": f"{a} x {b}",
                "rho": rho,
                "ci_lower": lo,
                "ci_upper": hi,
                "n_methods": len(methods),
            }
        )

    rhos = [r["rho"] for r in pair_rows]
    mean_rho = float(np.mean(rhos))
    print(f"\nMean rho across {len(pair_rows)} pairs: {mean_rho:+.4f}")

    payload = {
        "methods_in_intersection": methods,
        "n_methods": len(methods),
        "pairwise": pair_rows,
        "mean_rho": mean_rho,
        "method_gains": {ds: {m: gains[ds][m] for m in methods} for ds in DATASETS},
    }
    out_json = OUTPUT_DIR / "cross_dataset_spearman.json"
    out_csv = OUTPUT_DIR / "cross_dataset_spearman.csv"
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2)
    with open(out_csv, "w") as f:
        f.write("pair,rho,ci_lower,ci_upper,n_methods\n")
        for row in pair_rows:
            f.write(
                f"{row['pair']},{row['rho']:.4f},"
                f"{row['ci_lower']:.4f},{row['ci_upper']:.4f},{row['n_methods']}\n"
            )
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
