# pyright: basic, reportMissingImports=false

"""Generate the collapse-guard figures from released artifacts.

Figure A (dose-response): mean target gain and negative-adaptation rate as the
diversity floor tightens, showing that guard strictness controls harm smoothly and
that the strictest setting reduces exactly to no_adapt.

Figure B (asymmetry): the guard removes most of the harm where adaptation hurts
(Widar, hard cross-room shift) and leaves the gain untouched where it helps
(SignFi, milder temporal shift).

Outputs: manuscript/paper2/figures/fig6_collapse_guard_sweep.{pdf,png}
         manuscript/paper2/figures/fig7_collapse_guard_asymmetry.{pdf,png}
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FIGDIR = Path("manuscript/paper2/figures")
SWEEP_DIR = Path("outputs/collapse_guard_sweep")
WIDAR_PATH = Path("outputs/final_ablations/collapse_guard_widar.json")
SIGNFI_PATH = Path(
    "outputs/collapse_guard_signfi_tent/signfi_top10_lame_sar_cotta.json"
)

# Kept consistent with the existing figures in this manuscript.
COLOR_GUARDED = "#2A9D8F"
COLOR_UNGUARDED = "#E07B7B"
COLOR_NEUTRAL = "#4A4A4A"

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.18,
    }
)


def _pp(value: float) -> float:
    """Fraction to percentage points."""
    return value * 100.0


def _asymmetric_error(mean: float, lower: float, upper: float) -> tuple[float, float]:
    """Convert a CI to (lower_delta, upper_delta), clamped non-negative."""
    return (max(mean - lower, 0.0), max(upper - mean, 0.0))


def load_sweep() -> tuple[
    list[float], list[float], list[float], list[float], list[float]
]:
    """Read the diversity-floor sweep, sorted by floor."""
    floors: list[float] = []
    gains: list[float] = []
    lowers: list[float] = []
    uppers: list[float] = []
    negrates: list[float] = []
    paths = sorted(
        SWEEP_DIR.glob("ratio_*.json"),
        key=lambda p: float(re.search(r"ratio_([0-9.]+)\.json", p.name).group(1)),
    )
    if not paths:
        raise FileNotFoundError(f"No sweep artifacts under {SWEEP_DIR}")
    for path in paths:
        floor = float(re.search(r"ratio_([0-9.]+)\.json", path.name).group(1))
        summary = json.loads(path.read_text())["summary"]["mlp"]
        entry = summary.get("collapse_guarded_entropy_tta")
        if entry is None:
            continue
        floors.append(floor)
        gains.append(_pp(entry["mean_gain"]))
        lowers.append(_pp(entry["ci_lower"]))
        uppers.append(_pp(entry["ci_upper"]))
        negrates.append(entry["negative_adaptation_rate"])
    return floors, gains, lowers, uppers, negrates


def figure_sweep() -> None:
    """Figure A: dose-response of the diversity floor."""
    floors, gains, lowers, uppers, negrates = load_sweep()
    widar = json.loads(WIDAR_PATH.read_text())["summary"]["mlp"]
    unguarded = widar["entropy_tta"]

    fig, (ax_gain, ax_rate) = plt.subplots(1, 2, figsize=(6.8, 2.6))

    ax_gain.axhline(0.0, color="black", linewidth=0.8, linestyle="-", alpha=0.5)
    ax_gain.axhline(
        _pp(unguarded["mean_gain"]),
        color=COLOR_UNGUARDED,
        linewidth=1.2,
        linestyle="--",
        label=f"unguarded ({_pp(unguarded['mean_gain']):.1f} pp)",
    )
    ax_gain.fill_between(floors, lowers, uppers, color=COLOR_GUARDED, alpha=0.16)
    ax_gain.plot(
        floors, gains, color=COLOR_GUARDED, marker="o", markersize=4, label="guarded"
    )
    ax_gain.set_xlabel(r"Diversity floor $\tau$")
    ax_gain.set_ylabel("Mean target gain (pp)")
    ax_gain.set_title("Harm shrinks as the floor tightens")
    ax_gain.legend(loc="lower right")
    ax_gain.annotate(
        "reduces to\nno_adapt",
        xy=(1.0, 0.0),
        xytext=(0.73, -1.4),
        fontsize=7,
        color=COLOR_NEUTRAL,
        ha="center",
        arrowprops={"arrowstyle": "->", "color": COLOR_NEUTRAL, "linewidth": 0.7},
    )

    ax_rate.axhline(
        unguarded["negative_adaptation_rate"],
        color=COLOR_UNGUARDED,
        linewidth=1.2,
        linestyle="--",
        label="unguarded",
    )
    ax_rate.plot(
        floors,
        negrates,
        color=COLOR_GUARDED,
        marker="s",
        markersize=4,
        label="guarded",
    )
    ax_rate.set_xlabel(r"Diversity floor $\tau$")
    ax_rate.set_ylabel("Negative-adaptation rate")
    ax_rate.set_ylim(-0.05, 1.05)
    ax_rate.set_title("Failures become no-ops only past 0.9")
    ax_rate.legend(loc="lower left")

    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / "fig6_collapse_guard_sweep.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig6_collapse_guard_sweep.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info("wrote fig6_collapse_guard_sweep")


def figure_asymmetry() -> None:
    """Figure B: harm removed on Widar, gain preserved on SignFi."""
    widar = json.loads(WIDAR_PATH.read_text())["summary"]["mlp"]
    signfi = json.loads(SIGNFI_PATH.read_text())

    panels = [
        (
            "Widar_BVP (hard cross-room)\nadaptation is harmful",
            widar["entropy_tta"],
            widar["collapse_guarded_entropy_tta"],
            "entropy TTA",
        ),
        (
            "SignFi-10 (milder temporal)\nadaptation helps",
            signfi["tent"],
            signfi["collapse_guarded_tent"],
            "TENT",
        ),
    ]

    # Share one y-axis across panels: the whole point is that the harmful pair
    # collapses toward zero while the beneficial pair does not move, which a
    # per-panel scale would hide.
    extremes = [
        _pp(entry[key])
        for _, off, on, _ in panels
        for entry in (off, on)
        for key in ("ci_lower", "ci_upper")
    ]
    span = max(extremes) - min(extremes)
    y_low = min(extremes) - 0.18 * span
    y_high = max(extremes) + 0.12 * span

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9), sharey=True)
    for ax, (title, off, on, method) in zip(axes, panels, strict=True):
        means = [_pp(off["mean_gain"]), _pp(on["mean_gain"])]
        errors = np.array(
            [
                _asymmetric_error(
                    _pp(e["mean_gain"]), _pp(e["ci_lower"]), _pp(e["ci_upper"])
                )
                for e in (off, on)
            ]
        ).T
        ax.set_ylim(y_low, y_high)
        bars = ax.bar(
            [0, 1],
            means,
            width=0.55,
            color=[COLOR_UNGUARDED, COLOR_GUARDED],
            edgecolor="white",
            linewidth=0.6,
        )
        ax.errorbar(
            [0, 1],
            means,
            yerr=errors,
            fmt="none",
            ecolor=COLOR_NEUTRAL,
            elinewidth=0.9,
            capsize=3,
        )
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{method}", f"{method}\n+ guard"])
        if ax is axes[0]:
            ax.set_ylabel("Mean target gain (pp)")
        ax.set_title(title)

        # Place each value label clear of its own confidence interval.
        for idx, (bar, value) in enumerate(zip(bars, means, strict=True)):
            if value >= 0:
                label_y = value + errors[1][idx] + 0.03 * span
                vertical = "bottom"
            else:
                label_y = value - errors[0][idx] - 0.03 * span
                vertical = "top"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                label_y,
                f"{value:+.2f}",
                ha="center",
                va=vertical,
                fontsize=8,
            )

        change = means[1] - means[0]
        if means[0] < 0:
            note = f"{(1 - means[1] / means[0]) * 100:.0f}% of harm removed"
        else:
            note = (
                "gain fully retained"
                if abs(change) < 1e-9
                else f"{means[1] / means[0] * 100:.0f}% of gain retained"
            )
        ax.text(
            0.5,
            0.04,
            note,
            transform=ax.transAxes,
            ha="center",
            fontsize=8,
            color=COLOR_NEUTRAL,
            style="italic",
        )

    fig.tight_layout()
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGDIR / "fig7_collapse_guard_asymmetry.pdf", bbox_inches="tight", dpi=300
    )
    fig.savefig(
        FIGDIR / "fig7_collapse_guard_asymmetry.png", bbox_inches="tight", dpi=150
    )
    plt.close(fig)
    logger.info("wrote fig7_collapse_guard_asymmetry")


def main() -> None:
    figure_sweep()
    figure_asymmetry()
    logger.info("figures written to %s", FIGDIR)


if __name__ == "__main__":
    main()
