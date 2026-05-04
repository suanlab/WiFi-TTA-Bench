# pyright: basic, reportMissingImports=false

"""Regenerate Figure 1 with a 3-panel layout: Widar | NTU-Fi | SignFi.

The original Fig 1 shows only Widar + NTU-Fi and therefore omits the paper's
central SignFi counterexample. This script reads the canonical method summaries
for all three real datasets and emits a 3-column bar chart.

Output:
    manuscript/paper2/figures/fig1_method_comparison.pdf (+ .png)

Usage:
    python scripts/regenerate_fig1_with_signfi.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_PDF = Path("manuscript/paper2/figures/fig1_method_comparison.pdf")
OUTPUT_PNG = Path("manuscript/paper2/figures/fig1_method_comparison.png")

DATASETS = [
    (
        "Widar_BVP (cross-room)",
        "outputs/widar_full_tta/method_summary.json",
        "outputs/new_methods/widar_lame_sar_cotta.json",
    ),
    (
        "NTU-Fi (cross-session)",
        "outputs/ntufi_tta/method_summary.json",
        "outputs/new_methods/ntufi_har_lame_sar_cotta.json",
    ),
    (
        "SignFi-10 (cross-split)",
        "outputs/signfi_tta/method_summary.json",
        "outputs/new_methods/signfi_top10_lame_sar_cotta.json",
    ),
]

METHOD_ORDER = [
    "no_adapt",
    "tent",
    "entropy_tta",
    "physics_tta",
    "selective_physics_tta",
    "shot",
    "t3a",
    "sar",
    "cotta",
    "lame",
]
METHOD_LABELS = {
    "no_adapt": "no_adapt",
    "tent": "tent-proj",
    "entropy_tta": "entropy",
    "physics_tta": "physics",
    "selective_physics_tta": "sel.phys",
    "shot": "im",
    "t3a": "t3a",
    "sar": "sar",
    "cotta": "cotta",
    "lame": "lame",
}


def load_summary(primary: str, extras: str | None) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with open(primary) as f:
        d = json.load(f)
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = {
                "mean_gain": float(v.get("mean_gain", v.get("mean", 0.0))),
                "ci_lower": float(v.get("ci_lower", 0.0)),
                "ci_upper": float(v.get("ci_upper", 0.0)),
            }
    if extras and Path(extras).exists():
        with open(extras) as f:
            d = json.load(f)
        for k, v in d.items():
            if isinstance(v, dict) and k not in out:
                out[k] = {
                    "mean_gain": float(v.get("mean_gain", 0.0)),
                    "ci_lower": float(v.get("ci_lower", 0.0)),
                    "ci_upper": float(v.get("ci_upper", 0.0)),
                }
    return out


def main() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), sharey=False)
    for ax, (title, primary, extras) in zip(axes, DATASETS, strict=True):
        summary = load_summary(primary, extras)
        methods = [m for m in METHOD_ORDER if m in summary]
        values = np.array([summary[m]["mean_gain"] * 100.0 for m in methods])
        errs_low = np.array(
            [
                (summary[m]["mean_gain"] - summary[m]["ci_lower"]) * 100.0
                for m in methods
            ]
        )
        errs_high = np.array(
            [
                (summary[m]["ci_upper"] - summary[m]["mean_gain"]) * 100.0
                for m in methods
            ]
        )
        xs = np.arange(len(methods))
        colors = ["#4a90e2" if v >= 0 else "#e07b7b" for v in values]
        ax.bar(
            xs,
            values,
            color=colors,
            edgecolor="black",
            linewidth=0.6,
            yerr=[errs_low, errs_high],
            capsize=3,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [METHOD_LABELS.get(m, m) for m in methods],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_ylabel("Target gain (pp)" if ax is axes[0] else "")

    fig.suptitle(
        "Mean target-accuracy gain vs. no_adapt (95% bootstrap CI, n=15); "
        "dashed line = no_adapt baseline.",
        y=1.02,
        fontsize=10,
    )
    fig.tight_layout()
    OUTPUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", dpi=180)
    fig.savefig(OUTPUT_PNG, bbox_inches="tight", dpi=180)
    print(f"Saved: {OUTPUT_PDF}")
    print(f"Saved: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
