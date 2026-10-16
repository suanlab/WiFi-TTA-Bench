# pyright: basic, reportMissingImports=false

"""Generate all figures for the WiFi-TTA-Bench paper."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

FIGDIR = Path("manuscript/paper2/figures")
FIGDIR.mkdir(parents=True, exist_ok=True)

_SISYPHUS_PREFIX = ".sisyphus/evidence/"
_OUTPUTS_PREFIX = "outputs/"


def _canonical(path: str) -> str:
    """Prefer outputs/ over .sisyphus/evidence/ for the same subpath."""
    if path.startswith(_SISYPHUS_PREFIX):
        subpath = path[len(_SISYPHUS_PREFIX) :]
        canonical = _OUTPUTS_PREFIX + subpath
        if Path(canonical).exists():
            logger.debug("canonical: %s -> %s", path, canonical)
            return canonical
        # TODO(provenance): regenerate under outputs/ so this fallback can be removed
        logger.warning(
            "canonical path missing, falling back to .sisyphus/evidence: %s", canonical
        )
    return path


def load_json(path: str) -> dict | list:
    resolved = _canonical(path)
    with open(resolved) as f:
        return json.load(f)


FIG1_DATASETS = [
    (
        "Widar_BVP (cross-room)",
        "outputs/widar_full_tta/method_summary.json",
        "outputs/new_methods/widar_bvp_lame_sar_cotta.json",
    ),
    (
        "NTU-Fi (ordered partitions)",
        "outputs/core_regen/ntufi_har_lame_sar_cotta.json",
        "outputs/new_methods/ntufi_har_lame_sar_cotta.json",
    ),
    (
        "SignFi-10 (ordered partitions)",
        "outputs/core_regen/signfi_top10_lame_sar_cotta.json",
        "outputs/new_methods/signfi_top10_lame_sar_cotta.json",
    ),
]
FIG1_METHOD_ORDER = [
    "no_adapt",
    "tent",
    "entropy_tta",
    "physics_tta",
    "selective_physics_tta",
    "shot",
    "t3a",
    "sar",
    "cotta",
]
FIG1_METHOD_LABELS = {
    "no_adapt": "no_adapt",
    "tent": "tent-proj",
    "entropy_tta": "entropy",
    "physics_tta": "physics",
    "selective_physics_tta": "sel.phys",
    "shot": "im",
    "t3a": "t3a-ent",
    "sar": "sar-lite",
    "cotta": "cotta-lite",
}


def _load_fig1_summary(primary: str, extras: str | None) -> dict:
    out: dict = {}
    with open(_canonical(primary)) as f:
        d = json.load(f)
    for k, v in d.items():
        if isinstance(v, dict):
            out[k] = {
                "mean_gain": float(v.get("mean_gain", v.get("mean", 0.0))),
                "ci_lower": float(v.get("ci_lower", 0.0)),
                "ci_upper": float(v.get("ci_upper", 0.0)),
            }
    if extras and Path(_canonical(extras)).exists():
        with open(_canonical(extras)) as f:
            d = json.load(f)
        for k, v in d.items():
            if isinstance(v, dict) and k not in out:
                out[k] = {
                    "mean_gain": float(v.get("mean_gain", 0.0)),
                    "ci_lower": float(v.get("ci_lower", 0.0)),
                    "ci_upper": float(v.get("ci_upper", 0.0)),
                }
    return out


def fig1_method_comparison() -> None:
    """3-panel bar plot: Widar | NTU-Fi | SignFi, with 95% bootstrap CIs.

    Supersedes the old 2-panel Widar+NTU-Fi figure; the SignFi panel is the
    paper's central counterexample and must appear in Figure 1.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), sharey=False)
    for ax, (title, primary, extras) in zip(axes, FIG1_DATASETS, strict=True):
        summary = _load_fig1_summary(primary, extras)
        methods = [m for m in FIG1_METHOD_ORDER if m in summary]
        values = np.array([summary[m]["mean_gain"] * 100.0 for m in methods])
        errs_lo = np.array(
            [
                (summary[m]["mean_gain"] - summary[m]["ci_lower"]) * 100.0
                for m in methods
            ]
        )
        errs_hi = np.array(
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
            yerr=[errs_lo, errs_hi],
            capsize=3,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [FIG1_METHOD_LABELS.get(m, m) for m in methods],
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
    )
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig1_method_comparison.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig1_method_comparison.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Figure 1 saved")


def fig2_synthetic_real_gap() -> None:
    """Grouped bar: synthetic vs real gain for key methods."""
    synthetic = load_json("outputs/synthetic_shift/soft_selective_comparison.json")
    widar = load_json("outputs/widar_full_tta/method_summary.json")

    methods = [
        "physics_tta",
        "safe_physics_tta",
        "selective_physics_tta",
        "tent",
        "entropy_tta",
        "t3a",
    ]
    labels = [
        "Physics",
        "Safe\nPhysics",
        "Selective\nPhysics",
        "TENT",
        "Entropy",
        "T3A",
    ]

    syn_gains = [synthetic[m]["mean_gain"] * 100 for m in methods]
    real_gains = [widar[m]["mean_gain"] * 100 if m in widar else 0 for m in methods]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.bar(
        x - width / 2,
        syn_gains,
        width,
        label="Synthetic",
        color="#3498db",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        real_gains,
        width,
        label="Real (Widar)",
        color="#e67e22",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axhline(y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Mean Target Gain (pp)", fontsize=10)
    ax.set_title(
        "Synthetic vs. real-Widar gain for physics/entropy TTA families",
        fontsize=11,
        fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.set_ylim(-12, 10)

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig2_synthetic_real_gap.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig2_synthetic_real_gap.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Figure 2 saved")


def fig3_step_budget() -> None:
    """Line plot: gain vs adaptation steps for key methods."""
    data = load_json("outputs/db_ablations/step_budget.json")
    methods = ["entropy_tta", "tent", "physics_tta", "selective_physics_tta"]
    colors = {
        "entropy_tta": "#e74c3c",
        "tent": "#9b59b6",
        "physics_tta": "#3498db",
        "selective_physics_tta": "#2ecc71",
    }
    labels = {
        "entropy_tta": "Entropy",
        "tent": "TENT",
        "physics_tta": "Physics",
        "selective_physics_tta": "Selective Physics",
    }
    steps = [1, 3, 5, 10]

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        gains = []
        for s in steps:
            g = [r["gain"] * 100 for r in data if r["steps"] == s and r["method"] == m]
            gains.append(np.mean(g) if g else 0)
        ax.plot(
            steps,
            gains,
            "o-",
            color=colors[m],
            label=labels[m],
            linewidth=2,
            markersize=6,
        )

    ax.axhline(
        y=0, color="black", linestyle="--", linewidth=0.8, alpha=0.5, label="no_adapt"
    )
    ax.set_xlabel("Adaptation Steps", fontsize=11)
    ax.set_ylabel("Mean Target Gain (pp)", fontsize=11)
    ax.set_title("Step-Budget Sensitivity (Widar_BVP)", fontsize=12, fontweight="bold")
    ax.set_xticks(steps)
    ax.legend(fontsize=9)
    ax.set_ylim(-18, 2)

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig3_step_budget.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig3_step_budget.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Figure 3 saved")


def fig4_tsne() -> None:
    """t-SNE scatter of source/target before/after adaptation."""
    data = load_json("outputs/synthetic_shift/analysis/tsne_embeddings.json")

    fig, ax = plt.subplots(figsize=(7, 6))
    markers = {"source": "o", "target": "s"}
    phase_colors = {"before": "#3498db", "after": "#e74c3c"}
    domain_phase = {}
    for pt in data:
        key = (pt["domain"], pt["phase"])
        domain_phase.setdefault(key, {"x": [], "y": [], "label": []})
        domain_phase[key]["x"].append(pt["x"])
        domain_phase[key]["y"].append(pt["y"])
        domain_phase[key]["label"].append(pt["label"])

    legend_labels = {
        ("source", "before"): "Source",
        ("target", "before"): "Target (before)",
        ("target", "after"): "Target (after)",
    }

    for (domain, phase), pts in domain_phase.items():
        color = "#2ecc71" if domain == "source" else phase_colors.get(phase, "#95a5a6")
        marker = markers.get(domain, "o")
        label = legend_labels.get((domain, phase), f"{domain}/{phase}")
        ax.scatter(
            pts["x"],
            pts["y"],
            c=color,
            marker=marker,
            s=30,
            alpha=0.6,
            label=label,
            edgecolors="white",
            linewidth=0.3,
        )

    ax.set_xlabel("t-SNE 1", fontsize=10)
    ax.set_ylabel("t-SNE 2", fontsize=10)
    ax.set_title(
        "Latent Space: Source vs Target (Synthetic)", fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    fig.savefig(FIGDIR / "fig4_tsne.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig4_tsne.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Figure 4 saved")


COLLAPSE_JSON = "outputs/collapse_trajectories/collapse_trajectories.json"
WIDAR_CLASSES = ("slide", "draw-zigzag", "push&pull", "sweep", "clap", "draw-circle")


def fig5_confusion() -> None:
    """Widar room-2 confusion matrices before and after entropy TTA.

    Counts are summed over the five seeds of the held-out room-2 fold and
    row-normalised, from ``scripts/analyze_collapse_trajectories.py``.
    """
    payload = load_json(COLLAPSE_JSON)
    runs = [
        r
        for r in payload["runs"]  # type: ignore[index]
        if r["dataset"] == "widar_bvp"
        and r["backbone"] == "mlp"
        and r["fold"] == "env2"
    ]
    steps = runs[0]["entropy_trajectory"][-1]["steps"]
    panels = []
    for k, title in ((0, "Source only"), (-1, f"After entropy TTA ({steps} steps)")):
        counts = np.sum(
            [np.array(r["entropy_trajectory"][k]["confusion"]) for r in runs], axis=0
        )
        acc = np.trace(counts) / counts.sum()
        used = counts.sum(axis=0) / counts.sum()
        panels.append((title, counts / counts.sum(axis=1, keepdims=True), acc, used))

    n = len(WIDAR_CLASSES)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6))
    for ax, (title, matrix, acc, used) in zip(axes, panels, strict=True):
        im = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(WIDAR_CLASSES, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(WIDAR_CLASSES, fontsize=8)
        ax.set_xlabel("Predicted class", fontsize=9)
        if ax is axes[0]:
            ax.set_ylabel("True class", fontsize=9)
        else:
            ax.set_yticklabels([])
        for i in range(n):
            for j in range(n):
                ax.text(
                    j,
                    i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if matrix[i, j] > 0.45 else "black",
                    fontsize=7,
                )
        share = ", ".join(f"{100 * u:.0f}" for u in used)
        ax.set_title(
            f"{title}: accuracy {100 * acc:.1f}%\npredicted-class share (%): {share}",
            fontsize=9,
        )
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    fig.savefig(FIGDIR / "fig5_confusion.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig5_confusion.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Figure 5 saved")


def fig8_collapse_trajectories() -> None:
    """Class-usage diversity and accuracy change along unguarded entropy TTA."""
    payload = load_json(COLLAPSE_JSON)
    cells: dict[str, list[dict]] = {}
    for r in payload["runs"]:  # type: ignore[index]
        name = {"widar_bvp": "Widar", "ntufi_har": "NTU-Fi", "signfi_top10": "SignFi"}
        net = {"mlp": "MLP", "resnet2d": "ResNet-BN", "transformer": "Transformer"}
        label = f"{name[r['dataset']]} / {net[r['backbone']]}"
        cells.setdefault(label, []).append(r)
    fig, (ax_d, ax_a) = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for label, runs in cells.items():
        traj = [r["entropy_trajectory"] for r in runs]
        steps = [t["steps"] for t in traj[0]]
        div = np.array([[t["diversity"] for t in tr] for tr in traj])
        gain = np.array(
            [[100 * (t["accuracy"] - tr[0]["accuracy"]) for t in tr] for tr in traj]
        )
        (line,) = ax_d.plot(steps, div.mean(axis=0), marker="o", label=label)
        ax_a.plot(steps, gain.mean(axis=0), marker="o", color=line.get_color())
    ax_d.set_xlabel("Adaptation steps")
    ax_d.set_ylabel("Predicted-class diversity $D$")
    ax_d.set_title("Class usage on the evaluation half", fontsize=10)
    ax_a.axhline(0.0, color="grey", lw=0.8, ls="--")
    ax_a.set_xlabel("Adaptation steps")
    ax_a.set_ylabel("Accuracy change (pp)")
    ax_a.set_title("Target accuracy relative to source", fontsize=10)
    ax_d.legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    fig.savefig(FIGDIR / "fig8_collapse_trajectories.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(FIGDIR / "fig8_collapse_trajectories.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Figure 8 saved")


if __name__ == "__main__":
    fig1_method_comparison()
    fig2_synthetic_real_gap()
    fig3_step_budget()
    fig4_tsne()
    fig5_confusion()
    fig8_collapse_trajectories()
    print(f"\nAll figures saved to {FIGDIR}/")
